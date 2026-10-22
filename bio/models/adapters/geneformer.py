__all__ = ["Geneformer"]

##### 模型加载模块
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch


class ModelLoader:
    @staticmethod
    def load_pretrained_model(model_type, num_classes, model_directory, mode="eval", quantize=False):
        """加载预训练模型"""
        # 模型类映射
        model_classes = {
            "Pretrained": BertForMaskedLM,
            "GeneClassifier": BertForTokenClassification,
            "CellClassifier": BertForSequenceClassification,
            "MTLCellClassifier": BertForMaskedLM
        }
        model_class = model_classes.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")

        # 模型参数配置
        model_args = {
            "pretrained_model_name_or_path": model_directory,
            "output_attentions": False,
            "output_hidden_states": True,
        }
        if model_type != "Pretrained":
            model_args["num_labels"] = num_classes

        # 加载模型
        model, loading = model_class.from_pretrained(**model_args, local_files_only=True, output_loading_info=True)
        if loading['missing_keys'] or loading.get('mismatched_keys'):
            raise ValueError(f'Geneformer checkpoint mismatch: {loading}')

        if mode == "eval":
            model.eval()

        return model


##### 数据标记化模块
import pandas as pd
import scanpy as sc
from bio.vendor.geneformer.tokenizer import GeneformerTranscriptomeTokenizer

class DataTokenizer:
    @staticmethod
    def symbol_to_ensembl(gene_list):
        """将基因符号转换为ENSEMBL ID"""
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
        """将单细胞数据token化为模型输入格式"""
        if 'gene_name' not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index.tolist()
        else:
            adata.var.set_index(adata.var["gene_name"], inplace=True)

        # 处理基因ID
        if 'gene_ids' not in adata.var.columns:
            try:
                adata.var['gene_ids'] = DataTokenizer.symbol_to_ensembl(adata.var["gene_name"].tolist())
                valid_genes = adata.var[adata.var['gene_ids'].astype(str).str.startswith('ENS', na=False)].index
                adata = adata[:, valid_genes]
            except Exception as e:
                print(f"警告: 基因ID转换失败: {str(e)}")
                adata.var['gene_ids'] = adata.var["gene_name"]
        else:
            print("'gene_ids' 已存在于 adata.var 中，跳过添加步骤。")

        # 计算细胞计数
        adata.obs['n_counts'] = adata.X.sum(axis=1)
        adata.var['ensembl_id'] = adata.var['gene_ids']
        adata.obs['filter_pass'] = (adata.obs['n_counts'] > 0)

        # 创建tokenizer
        from geneformer import TranscriptomeTokenizer
        from bio.models.compat.tokenizer import tokenizer_options
        tk = GeneformerTranscriptomeTokenizer(
            dict([(k, k) for k in adata.obs.keys()]), nproc=int((model_cfg or {}).get("tokenizer_workers", 1)),
            **tokenizer_options(model_cfg or {}, TranscriptomeTokenizer))

        # 对数据进行tokenization
        tokenized_cells, cell_metadata = tk.tokenize_anndata(adata)

        # 清理元数据
        cleaned_metadata = {}
        for key, values in cell_metadata.items():
            cleaned_values = []
            for value in values:
                if isinstance(value, (int, float)):
                    cleaned_values.append(str(value))
                elif pd.isna(value):
                    cleaned_values.append("")
                else:
                    cleaned_values.append(value)
            cleaned_metadata[key] = cleaned_values

        # 创建并保存数据集
        tokenized_dataset = tk.create_dataset(tokenized_cells, cleaned_metadata)
        return tokenized_dataset


##### 嵌入提取模块
import torch

class EmbeddingExtractor:
    @staticmethod
    def extract_embeddings(model, input_dict, device):
        """从模型中提取细胞嵌入表示"""
        expression_matrix = input_dict["expression_matrix"]
        input_dict.pop("expression_matrix", None)
        input_dict.pop("cell_id", None)
        # 将数据移动到指定设备
        for k in input_dict:
            input_dict[k] = input_dict[k].to(device=device)

        input_dict["attention_mask"] = (input_dict["attention_mask"]!=0)
        label = input_dict["labels"]

        # 前向传播
        outputs = model(input_ids=input_dict["input_ids"], attention_mask=input_dict["attention_mask"])
        logits = outputs.logits
        hidden_state = outputs.hidden_states[-2]

        # 计算细胞嵌入
        hidden_state_mask = hidden_state * input_dict["attention_mask"].unsqueeze(-1)
        cell_emb = hidden_state_mask.sum(dim=1) / input_dict["attention_mask"].sum(dim=1).unsqueeze(-1)

        return expression_matrix, cell_emb, label


##### 数据定义模块
from torch.utils.data import DataLoader
from bio.vendor.geneformer.collator_for_classification import DataCollatorForCellClassification

class CollatorWithOriginal(DataCollatorForCellClassification):
    """自定义数据collator，添加原始表达矩阵和细胞ID"""
    def __call__(self, features):
        expression_matrix = torch.stack([
            torch.tensor(f.pop("expression_matrix"), dtype=torch.float32)
            for f in features
        ])

        cell_ids = torch.tensor([int(f.pop("cell_id")) for f in features])

        batch = super().__call__(features)
        batch["expression_matrix"] = expression_matrix
        batch["cell_id"] = cell_ids
        return batch


##### 主类模块
import os
import pickle
import scipy.sparse as sparse
import numpy as np
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk
from bio.models.adapter import FoundationAdapter
from bio.models.heads import Decoder, PerturbDecoder, Pooler_classify
import torch.optim as optim
from bio.vendor.geneformer.perturber_utils import xu_load_and_filter, TOKEN_DICTIONARY_FILE
from tqdm import tqdm

class Geneformer(FoundationAdapter):
    """整合所有模块的主类"""

    def __init__(self, *args, **kargs):
        super(Geneformer, self).__init__(*args, **kargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, model_cfg):
        """构建模型网络"""
        model_type = "Pretrained"
        from bio.io.paths import weight_path, project_path
        from bio.io.paths import model_resource
        model_ckpt = str(model_resource(model_cfg, 'Geneformer', 'gf-12L-30M-i2048', 'pretrain_directory'))

        self.model = ModelLoader.load_pretrained_model(
            model_type=model_type,
            num_classes=self.cell_type_num,
            model_directory=model_ckpt,
            mode="eval"
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.setup_task_heads(self.model.config.hidden_size)

    def prepare_inputs(self, data_cfg, train=True):
        """获取数据加载器"""
        self.adata_all = self.read_adata(data_cfg["dataset_name"])

        adata_all = self.read_model_input()
        from bio.io.paths import weight_path
        from bio.io.paths import model_resource
        with model_resource(self.cfgs['model_cfg'], 'Geneformer', 'token_dictionary_gc30M.pkl', 'token_dictionary_file').open('rb') as f:
            self.gene_token_dict = pickle.load(f)

        dataset = DataTokenizer.tokenize_data(adata_all, self.cfgs['model_cfg'])

        dataset = xu_load_and_filter(None, 1, dataset)

        if "label" not in dataset.column_names:
            types = sorted(list(set(dataset["celltype"])))
            type2num = {t: i for i, t in enumerate(types)}
            dataset = dataset.map(
                lambda x: {"label": type2num[x["celltype"]]},
                num_proc=1
            )

        exper_martix = self.evaluation_expression()
        exper_martix = exper_martix.astype(np.float32)
        source_ids = np.asarray(dataset["cell_id"], dtype=int)
        if set(source_ids) != set(range(self.adata_all.n_obs)):
            raise ValueError("Geneformer tokenizer dropped or duplicated source cells")
        expr_list = list(exper_martix[source_ids])
        dataset = dataset.add_column("expression_matrix", expr_list)
        dataset = dataset.remove_columns(["cell_id"]).add_column("cell_id", source_ids.tolist())

        data_collator = CollatorWithOriginal(token_dictionary=self.gene_token_dict)
        self.all_loader = DataLoader(
            dataset,
            batch_size=self.cfgs['model_cfg'].get('inference_batch_size', 8),
            shuffle=False,
            collate_fn=data_collator,
            pin_memory=True
        )

    def encode_batch(self, input_dict):
        """前向传播"""
        expression_matrix, cell_emb, label = EmbeddingExtractor.extract_embeddings(
            model=self.model,
            input_dict=input_dict,
            device=self.device
        )
        return expression_matrix, cell_emb, label

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
