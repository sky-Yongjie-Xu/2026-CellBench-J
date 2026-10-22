__all__ = ["scFoundation"]

##### 模型加载模块
import os
from bio.io.paths import weight_path
import torch
import torch.nn as nn
from torch import optim
from bio.vendor.scfoundation.load import load_model_frommmf, gatherData
from bio.models.heads import Decoder, PerturbDecoder, Pooler_classify


class ModelLoader:
    @staticmethod
    def load_pretrained_model(ckpt_path, model_key='cell'):
        """加载预训练的scFoundation模型"""
        model, pretrain_config = load_model_frommmf(ckpt_path, model_key)
        model.eval()
        return model, pretrain_config


##### 数据预处理模块
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse
import scanpy as sc
import pickle
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    @staticmethod
    def preprocess_adata(adata):
        """预处理AnnData对象"""
        # 确保有gene_name列
        if 'gene_name' not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index.tolist()

        # 计算细胞计数
        adata.obs['n_counts'] = adata.X.sum(axis=1)
        adata.obs['filter_pass'] = (adata.obs['n_counts'] > 0)

        return adata

    @staticmethod
    def convert_adata_to_df(adata):
        """将AnnData转换为DataFrame"""
        if issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = adata.X

        idx = adata.obs_names.tolist()
        try:
            col = adata.var_names.tolist()
        except:
            col = adata.var.gene_name.tolist()

        return pd.DataFrame(data, index=idx, columns=col)

    @staticmethod
    def main_gene_selection(X_df, gene_list):
        """筛选目标基因并填充缺失基因"""
        to_fill_columns = list(set(gene_list) - set(X_df.columns))
        padding_df = pd.DataFrame(
            np.zeros((X_df.shape[0], len(to_fill_columns))),
            columns=to_fill_columns,
            index=X_df.index
        )

        X_df = pd.concat([X_df, padding_df], axis=1)[gene_list]
        var = pd.DataFrame(index=X_df.columns)
        var['mask'] = [1 if i in to_fill_columns else 0 for i in var.index]

        return X_df, to_fill_columns, var

    @staticmethod
    def prepare_input_tensor(data_df):
        """准备模型输入张量"""
        counts = data_df.to_numpy(dtype=np.float32)
        totalcount = counts.sum(axis=1, keepdims=True)
        if np.any(totalcount <= 0):
            raise ValueError('scFoundation has cells with no genes in the 19264-gene vocabulary')
        normalized = np.log1p(counts / totalcount * 1e4)
        data_total = np.hstack([normalized, np.full_like(totalcount, 4.), np.log10(totalcount)])
        return torch.tensor(data_total, dtype=torch.float32)


    @staticmethod
    def encode_labels(adata, label_field='celltype'):
        """将标签编码为数字"""
        le = LabelEncoder()
        labels = le.fit_transform(adata.obs[label_field])
        adata.obs['label'] = labels
        return le

##### 数据定义模块
from torch.utils.data import Dataset, DataLoader, Subset

class GeneExpressionDataset(Dataset):
    """自定义基因表达数据集"""
    def __init__(self, data_tensor, adata, labels=None):
        self.features = data_tensor
        self.adata = adata
        self.labels = labels.long() if labels is not None else None
        self.cell_ids = torch.arange(len(self.features), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def _get_expr_row(self, idx):
        row = self.adata.X[idx]
        if sp.issparse(row):
            row = row.toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = {
            "input": self.features[idx],
            "original": self._get_expr_row(idx),
            "cell_id": self.cell_ids[idx]
        }
        if self.labels is not None:
            sample["label"] = self.labels[idx]
        return sample

##### 数据定义模块
class DataLoaderBuilder:
    @staticmethod
    def build_dataloader(dataset, batch_size=32, shuffle=True):
        """构建标准数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True
        )

##### 主类模块
from tqdm import tqdm
from bio.models.adapter import FoundationAdapter

class scFoundation(FoundationAdapter):
    """整合所有模块的主类"""

    def __init__(self, cfgs, *args, **kargs):
        self._load_gene_list(cfgs)
        super(scFoundation, self).__init__(cfgs, *args, **kargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_gene_list(self, cfgs):
        """加载基因列表"""
        from bio.io.paths import project_path, weight_path
        from bio.io.paths import model_resource
        gene_list_path = model_resource(cfgs['model_cfg'], 'scFoundation', 'OS_scRNA_gene_index.19264.tsv', 'gene_list_path')
        if not hasattr(self, 'gene_list'):
            self.gene_list = pd.read_csv(
                gene_list_path,
                header=0,
                delimiter='\t'
            )['gene_name'].tolist()

    def load_pretrained(self, model_cfg):
        """构建模型网络"""
        from bio.io.paths import model_resource
        self.model, self.pretrain_config = ModelLoader.load_pretrained_model(
            str(model_resource(model_cfg, 'scFoundation', 'models.ckpt', 'ckpt_path'))
        )
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.setup_task_heads(3072)

    def prepare_inputs(self, data_cfg, train=True):
        """获取数据加载器"""
        if self.cfgs["model_cfg"].get("inference_batch_size", 1) != 1:
            raise ValueError("scFoundation native four-way pooling requires inference_batch_size=1")
        self.adata_all = self.read_adata(data_cfg["dataset_name"])

        adata_all = self.read_model_input()
        adata_all = DataPreprocessor.preprocess_adata(adata_all)
        DataPreprocessor.encode_labels(adata_all)
        data_df = DataPreprocessor.convert_adata_to_df(adata_all)
        data_df = DataPreprocessor.main_gene_selection(data_df, self.gene_list)[0]
        data_tensor = DataPreprocessor.prepare_input_tensor(data_df)
        label = torch.from_numpy(adata_all.obs['label'].values)

        dataset = GeneExpressionDataset(data_tensor, self.adata_all, labels=label)

        self.all_loader = DataLoaderBuilder.build_dataloader(
            dataset,
            batch_size=1,
            shuffle=False
        )

    def encode_batch(self, data_dict):
        """前向传播"""
        data = data_dict['input'].to(self.device)
        label = data_dict['label'].to(self.device)

        data_gene_ids = torch.arange(19266).repeat(data.shape[0], 1).to(self.device)
        value_labels = data > 0

        x, x_padding = gatherData(data, value_labels, self.pretrain_config['pad_token_id'])
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels, self.pretrain_config['pad_token_id'])

        x = self.model.token_emb(torch.unsqueeze(x, 2).float().to(self.device), output_weight=0)
        x += self.model.pos_emb(position_gene_ids.to(self.device))
        geneemb = self.model.encoder(x, x_padding.to(self.device))

        geneemb1 = geneemb[:,-1,:]
        geneemb2 = geneemb[:,-2,:]
        geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
        geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
        geneembmerge = torch.concat([geneemb1, geneemb2, geneemb3, geneemb4],axis=1)
        return data_dict['original'], geneembmerge, label

    def forward_annotation(self, cell_emb):
        logit = self.classify(cell_emb)
        return logit

    def forward_reconstruction(self, cell_emb):
        pred = self.restruct(cell_emb)
        return pred

    def forward_perturbation(self, ctrl_emb, pert_id, ctrl_expr):
        delta_expr = self.pert_decoder(ctrl_emb, pert_id)
        pred_expr = (ctrl_expr + delta_expr).clamp_min(0)
        return pred_expr
