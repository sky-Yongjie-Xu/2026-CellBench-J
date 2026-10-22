__all__ = ["LangCell"]

##### 模型加载模块
from transformers import BertConfig, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ModelLoader:
    @staticmethod
    def load_pretrained_model(ckpt_path, proj_dim=256):
        """加载预训练的LangCell模型"""
        model, loading = BertModel.from_pretrained(os.path.join(ckpt_path, 'cell_bert'),
                                                  local_files_only=True, output_loading_info=True)
        missing = [k for k in loading['missing_keys'] if not k.startswith('pooler.')]
        if missing or loading.get('mismatched_keys'):
            raise ValueError(f'LangCell checkpoint mismatch: {loading}')
        proj_path = os.path.join(ckpt_path, "cell_proj.bin")
        model.pooler = Pooler(model.config, pretrained_proj=proj_path, proj_dim=proj_dim)
        return model

class Pooler(nn.Module):
    """自定义池化层"""
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj, map_location="cpu"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        return F.normalize(self.proj(pooled_output), dim=-1)

##### 数据预处理模块
import numpy as np
import scipy.sparse as sp
import pandas as pd
import scanpy as sc
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    @staticmethod
    def preprocess_adata(adata):
        """预处理AnnData对象"""
        if 'gene_name' not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['n_counts'] = adata.X.sum(axis=1)
        adata.obs['filter_pass'] = (adata.obs['n_counts'] > 0)
        return adata

    @staticmethod
    def preprocess_dataset(dataset):
        """预处理tokenized数据集"""
        if "celltype" not in dataset.column_names:
            for label_name in ["cell_type", "str_labels", "labels"]:
                if label_name in dataset.column_names:
                    dataset = dataset.rename_column(label_name, "celltype")
                    break
        return dataset


##### 数据标记化模块
from bio.vendor.langcell.utils import LangCellTranscriptomeTokenizer

class DataTokenizer:
    @staticmethod
    def symbol_to_ensembl(gene_list):
        """基因符号转ENSEMBL ID"""
        import mygene
        mg = mygene.MyGeneInfo()
        return mg.querymany(
            gene_list,
            scopes='symbol',
            fields='ensembl.gene',
            as_dataframe=True,
            species='human'
        ).reset_index().drop_duplicates(
            subset='query'
        )['ensembl.gene'].fillna('0').tolist()

    @staticmethod
    def tokenize_data(adata, model_cfg=None):
        """单细胞数据tokenization"""
        if 'gene_name' not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index.tolist()

        if 'gene_ids' not in adata.var.columns:
            try:
                adata.var['gene_ids'] = DataTokenizer.symbol_to_ensembl(adata.var["gene_name"].tolist())
                valid_genes = adata.var[adata.var['gene_ids'].astype(str).str.startswith('ENS', na=False)].index
                adata = adata[:, valid_genes]
            except Exception as e:
                print(f"基因ID转换失败: {str(e)}")
                adata.var['gene_ids'] = adata.var["gene_name"]

        adata.obs['n_counts'] = adata.X.sum(axis=1)
        adata.var['ensembl_id'] = adata.var['gene_ids']

        from geneformer import TranscriptomeTokenizer
        from bio.models.compat.tokenizer import tokenizer_options
        tk = LangCellTranscriptomeTokenizer(
            dict([(k, k) for k in adata.obs.keys()]), nproc=int((model_cfg or {}).get("tokenizer_workers", 1)),
            **tokenizer_options(model_cfg or {}, TranscriptomeTokenizer))
        tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)

        cleaned_metadata = {}
        for key, values in cell_metadata.items():
            cleaned_values = [str(value) if not pd.isna(value) else "" for value in values]
            cleaned_metadata[key] = cleaned_values

        tokenized_dataset = tk.create_dataset(tokenized_cells, cleaned_metadata)
        return tokenized_dataset

##### 数据定义模块
from torch.utils.data import DataLoader
from bio.vendor.langcell.utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification

class CollatorWithExpr:
    """Official LangCell CLS prefix/padding without version-specific HF helpers."""
    def __init__(self, token_dictionary, max_length=2048):
        self.tokens = token_dictionary
        self.max_length = max_length

    def __call__(self, features):
        sequences = [([self.tokens['<cls>']] + list(f['input_ids']))[:self.max_length] for f in features]
        length = max(map(len, sequences))
        ids = torch.full((len(features), length), self.tokens['<pad>'], dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, :len(sequence)] = torch.tensor(sequence)
            mask[i, :len(sequence)] = 1
        return dict(input_ids=ids, attention_mask=mask,
                    labels=torch.tensor([f['label'] for f in features], dtype=torch.long),
                    expression_matrix=torch.tensor(np.asarray([f['expression_matrix'] for f in features]), dtype=torch.float32),
                    cell_id=torch.tensor([int(f['cell_id']) for f in features], dtype=torch.long))


##### 主类模块
from tqdm import tqdm
import torch.optim as optim
from bio.models.adapter import FoundationAdapter
from bio.models.heads import Decoder, PerturbDecoder, Pooler_classify

class LangCell(FoundationAdapter):
    """整合所有模块的主类"""

    def __init__(self, *args, **kwargs):
        super(LangCell, self).__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, model_cfg):
        """构建模型网络"""
        from bio.io.paths import model_resource
        self.model = ModelLoader.load_pretrained_model(
            ckpt_path=str(model_resource(model_cfg, 'LangCell', legacy_key='ckpt')),
            proj_dim=256
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.setup_task_heads(256)

    def prepare_inputs(self, data_cfg, train=True):
        """完全内存中的数据处理流程"""
        self.adata_all = self.read_adata(data_cfg["dataset_name"])

        adata_all = self.read_model_input()
        adata_all = DataPreprocessor.preprocess_adata(adata_all)

        dataset = DataTokenizer.tokenize_data(adata_all, self.cfgs['model_cfg'])
        dataset = DataPreprocessor.preprocess_dataset(dataset)

        source_ids = np.asarray(dataset['cell_id'], dtype=int)
        if set(source_ids) != set(range(self.adata_all.n_obs)) or len(source_ids) != self.adata_all.n_obs:
            raise ValueError('LangCell tokenizer changed source cells; refuse truncation')
        expression_matrix = self.evaluation_expression()[source_ids]

        if "label" not in dataset.column_names:
            types = sorted(list(set(dataset["celltype"])))
            type2num = {t: i for i, t in enumerate(types)}
            dataset = dataset.map(
                lambda x: {"label": type2num[x["celltype"]]},
                num_proc=1
            )

        dataset = dataset.add_column("expression_matrix", list(expression_matrix.astype(np.float32)))
        if "cell_id" not in dataset.column_names:
            dataset = dataset.add_column("cell_id", list(range(len(dataset))))

        keep_cols = {"input_ids", "label", "expression_matrix", "cell_id"}
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in keep_cols]
        )

        import pickle
        from bio.io.paths import project_path, weight_path
        from bio.io.paths import model_resource
        dictionary_path = model_resource(self.cfgs['model_cfg'], 'LangCell', 'token_dictionary_gc30M.pkl', 'token_dictionary_file')
        with dictionary_path.open('rb') as f:
            token_dictionary = pickle.load(f)
        token_dictionary['<cls>'] = len(token_dictionary)
        data_collator = CollatorWithExpr(token_dictionary=token_dictionary)
        self.all_loader = DataLoader(
            dataset,
            batch_size=self.cfgs['model_cfg'].get('inference_batch_size', 8),
            shuffle=False,
            collate_fn=data_collator,
            pin_memory=True
        )

    def encode_batch(self, data_list):
        """前向传播"""
        input_ids = data_list['input_ids'].to(self.device)
        if input_ids.min() < 0 or input_ids.max() >= self.model.config.vocab_size:
            raise ValueError('LangCell token IDs are outside the checkpoint vocabulary')
        attention_mask = data_list['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask)
        cell_emb = outputs.pooler_output
        label = data_list["labels"].to(self.device)
        return data_list['expression_matrix'], cell_emb, label

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
