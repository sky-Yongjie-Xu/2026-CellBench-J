__all__ = ["scMulan"]

##### 模型加载模块
import os
from bio.io.paths import weight_path, project_path, model_resource
import torch
import json
from torch import optim
from bio.vendor.scmulan.model.model import MulanConfig, scMulanModel
from bio.vendor.scmulan.utils.hf_tokenizer import scMulanTokenizer
from bio.models.heads import Decoder, PerturbDecoder, Pooler_classify


class ModelLoader:
    @staticmethod
    def load_pretrained_model(ckpt_path):
        """加载预训练的scMulan模型"""
        # 加载模型配置和权重
        ckp = torch.load(ckpt_path, map_location='cpu')
        gptconf = MulanConfig(**ckp['model_args'])

        # 初始化模型
        model = scMulanModel(gptconf)
        model.load_state_dict(ckp['model'])
        model.eval()

        return model, ckp['model_args']['expression_level']


##### 数据预处理模块
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import scanpy as sc
from anndata import AnnData
from bio.vendor.scmulan.reference.GeneSymbolUniform.pyGSUni import GeneSymbolUniform

class DataPreprocessor:
    def __init__(self, mulan_gene_set, weights_dir=None):
        self.mulan_gene_set = mulan_gene_set
        self.weights_dir = project_path(weights_dir or "weights/scMulan")

    def preprocess_adata(self, adata):
        """预处理AnnData对象"""

        adata = GeneSymbolUniform(
            input_adata=adata,
            ref_table_path=str(self.weights_dir / 'GeneSymbolRef_upd0731.csv'),
            gene_list_path=str(self.weights_dir / 'total_gene_list_42117.txt'),
            output_dir="./",
            output_prefix='ms'
        )

        adata_select = adata[:, adata.var.index.isin(self.mulan_gene_set)]

        row_sums = np.array(adata_select.X.sum(axis=1)).flatten()
        non_zero_rows_mask = row_sums > 0
        filtered_adata = adata_select[non_zero_rows_mask, :]

        return filtered_adata

    def prepare_dataframe(self, adata):
        """准备DataFrame格式的表达矩阵"""
        cellDFHVG = pd.DataFrame(
            adata.X.toarray() if sp.issparse(adata.X) else adata.X,
            columns=adata.var_names,
            index=adata.obs.index.tolist()
        )
        return cellDFHVG

##### 数据定义模块
from torch.utils.data import Dataset, DataLoader, Subset

class CellExpressionDataset(Dataset):
    """自定义基因表达数据集"""
    def __init__(self, df, label=None, adata=None):
        self.df = df
        self.label = label
        self.cell_ids = df.index.tolist()
        self.expression = torch.tensor(df.values, dtype=torch.float32)

        X = adata.X
        self.original_X = X.toarray().astype("float32") if sp.issparse(X) else X.astype("float32")
        self.cell_idx = torch.arange(len(self.expression))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        genes_series = self.df.loc[cell_id]
        expressed_genes = genes_series[genes_series > 0].index.tolist()
        expr_values = genes_series[expressed_genes].values

        sample = {
            "gene_dict": dict(zip(expressed_genes, expr_values)),
            "original": self.original_X[idx],
            "cell_id": self.cell_idx[idx]
        }

        if self.label is not None:
            sample["label"] = self.label[idx]

        return sample

##### 数据加载模块
import pickle

class DataLoaderBuilder:
    @staticmethod
    def load_or_process_data(adata, preprocessor):
        """加载或处理数据"""
        adata_origin = adata.copy()
        adata_filter = preprocessor.preprocess_adata(adata)

        _, label = np.unique(np.array(adata_filter.obs['celltype']), return_inverse=True)
        label = torch.from_numpy(label)

        adata_matrix = preprocessor.prepare_dataframe(adata_filter)
        return adata_matrix, label, adata_origin, adata_filter  # 返回过滤后的 adata

##### 主类模块
from tqdm import tqdm
from bio.models.adapter import FoundationAdapter

class scMulan(FoundationAdapter):
    """整合所有模块的主类"""

    def __init__(self, cfgs, *args, **kargs):
        meta_path = model_resource(cfgs['model_cfg'], 'scMulan', 'meta_info.pt', 'meta_info_path')
        meta_info = torch.load(meta_path, map_location='cpu')
        self.mulan_gene_set = meta_info['gene_set']
        super(scMulan, self).__init__(cfgs, *args, **kargs)
        self.tokenizer = scMulanTokenizer(chars=meta_info['token_set'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, model_cfg):
        """构建模型网络"""
        ckpt_path = str(model_resource(model_cfg, 'scMulan', 'ckpt_scMulan.pt', 'ckpt_path'))
        self.model, self.n_express_level = ModelLoader.load_pretrained_model(ckpt_path)

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.setup_task_heads(self.model.config.n_embd)

    def prepare_inputs(self, data_cfg, train=True):
        """获取数据加载器"""
        self.adata_all = self.read_adata(data_cfg["dataset_name"])

        adata_all = self.read_model_input()
        preprocessor = DataPreprocessor(self.mulan_gene_set, self.cfgs['model_cfg'].get('weights_dir'))
        adata_matrix, label, adata_origin, adata_filtered = DataLoaderBuilder.load_or_process_data(adata_all, preprocessor)
        if list(adata_filtered.obs_names) != list(self.adata_all.obs_names):
            raise ValueError("scMulan native preprocessing dropped or reordered cells")
        adata_origin = self.adata_all.copy()

        # 保存过滤后的 adata，用于 correction 任务
        # 确保过滤后的 adata 有正确的 batch 和 celltype 信息（如果预处理过程中丢失了）
        if 'batch' in self.adata_all.obs.columns and 'batch' not in adata_filtered.obs.columns:
            # 使用过滤后的 adata 的索引来获取对应的 batch
            filtered_indices = adata_filtered.obs.index
            adata_filtered.obs['batch'] = self.adata_all.obs.loc[filtered_indices, 'batch'].values
        if 'celltype' in self.adata_all.obs.columns and 'celltype' not in adata_filtered.obs.columns:
            # 使用过滤后的 adata 的索引来获取对应的 celltype
            filtered_indices = adata_filtered.obs.index
            adata_filtered.obs['celltype'] = self.adata_all.obs.loc[filtered_indices, 'celltype'].values
        self.adata_filtered = adata_filtered  # 保存过滤后的 adata

        dataset = CellExpressionDataset(adata_matrix, label, adata_origin)
        self.all_loader = DataLoader(
            dataset,
            batch_size=self.cfgs['model_cfg'].get('inference_batch_size', 8),
            collate_fn=lambda x: x,
            shuffle=False,
            pin_memory=True
        )

    def encode_batch(self, data_dict_batch):
        """前向传播"""
        with torch.no_grad():
            raw_exprs = []
            cellembs = []
            labels = []

            for data_dict in data_dict_batch:
                cell_expression_dict = data_dict['gene_dict']
                labels.append(data_dict['label'])
                raw_exprs.append(data_dict['original'])

                expressed_genes = list(cell_expression_dict.keys())[::-1]
                expression_values = list(cell_expression_dict.values())[::-1]

                max_expression = np.max(expression_values)
                bins = np.linspace(0, max_expression, self.n_express_level+1)
                binned_expr = np.digitize(expression_values, bins, right=True)

                annotation_task_token = '<PCT>'
                prefix = expressed_genes + [annotation_task_token]
                ec_binned_expr = np.append(binned_expr, [0])
                ec_prefix = self.tokenizer.encode(prefix)

                prompt_entities = torch.tensor(ec_prefix).unsqueeze(0).to(self.device)
                prompt_values = torch.tensor(ec_binned_expr).unsqueeze(0).to(self.device)

                _, _, hidden = self.model.generate_cellGenesis(
                    prompt_entities, prompt_values,
                    max_new_tokens=len(ec_prefix) + 3,
                    top_k=1,
                    return_hidden=True
                )

                cellembs.append(hidden[-1][0, -2, :])

        raw_exprs = torch.tensor(raw_exprs)
        cellemb = torch.stack(cellembs, dim=0)
        label = torch.tensor(labels).to(self.device)
        return raw_exprs, cellemb, label

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
