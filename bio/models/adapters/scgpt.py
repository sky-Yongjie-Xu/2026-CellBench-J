__all__ = ["scGPT"]

##### 模型加载模块
import os
from bio.io.paths import weight_path
import json
import torch
import torch.nn as nn
from torch import optim
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import load_pretrained
from bio.models.heads import Decoder, PerturbDecoder, Pooler_classify


class ModelLoader:
    @staticmethod
    def load_pretrained_model(model_dir, cell_type_num):
        """加载预训练的scGPT模型"""
        # 加载模型配置
        with open(os.path.join(model_dir, "args.json"), "r") as f:
            model_configs = json.load(f)

        # 初始化词汇表
        vocab = GeneVocab.from_file(os.path.join(model_dir, "vocab.json"))
        for token in ["<pad>", "<cls>", "<eoc>"]:
            if token not in vocab:
                vocab.append_token(token)

        # 创建模型
        model = TransformerModel(
            len(vocab),  # ntokens
            model_configs["embsize"],  # embsize
            model_configs["nheads"],  # nhead
            model_configs["d_hid"],  # d_hid
            model_configs["nlayers"],  # nlayers
            nlayers_cls=3,
            n_cls=cell_type_num,
            vocab=vocab,
            dropout=0.2,
            pad_token="<pad>",
            pad_value=-2,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=False,
            num_batch_labels=2,
            domain_spec_batchnorm=False,
            input_emb_style="continuous",
            n_input_bins=51,
            cell_emb_style="cls",
            mvc_decoder_style="inner product",
            ecs_threshold=0.0,
            explicit_zero_prob=False,
            # The standard attention backend also runs on Turing GPUs (e.g. RTX 2070).
            use_fast_transformer=False,
            fast_transformer_backend="flash",
            pre_norm=False,
        )

        model_file = os.path.join(model_dir, "best_model.pt")
        # Upstream maps fused-attention checkpoint keys to native PyTorch attention.
        # Filtering keys directly would otherwise discard the pretrained Q/K/V weights.
        state = torch.load(model_file, map_location='cpu')
        load_pretrained(model, state, strict=False, verbose=False)
        import re
        from bio.models.loading import verify_backbone
        mapped = {}
        for key, value in state.items():
            key = re.sub(r'self_attn\.(?:_impl\.)?Wqkv\.', 'self_attn.in_proj_', key)
            key = key.replace('self_attn._impl.out_proj.', 'self_attn.out_proj.')
            mapped[key] = value
        verify_backbone(model, mapped, ('encoder.', 'value_encoder.', 'transformer_encoder.'))

        return model, vocab


##### 数据预处理模块
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse
import scanpy as sc
from scgpt.preprocess import Preprocessor

class DataPreprocessor:
    def __init__(self, vocab):
        self.vocab = vocab
        self.preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=False,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            subset_hvg=False,
            hvg_flavor="cell_ranger",
            binning=51,
            result_binned_key="X_binned",
        )

    def process_adata(self, adata):
        """预处理AnnData对象"""
        if 'gene_name' not in adata.var.columns:
            adata.var["gene_name"] = adata.var.index.tolist()


        adata.var["id_in_vocab"] = [
            1 if gene in self.vocab else -1 for gene in adata.var["gene_name"]
        ]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        self.preprocessor(adata, batch_key=None)
        return adata


##### 数据标记化模块
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value

class DataTokenizer:
    def __init__(self, vocab, max_length=1200):
        self.vocab = vocab
        self.max_length = max_length

    def prepare_model_input(self, adata):
        """准备模型输入张量"""
        input_layer_key = "X_binned"
        all_counts = (
            adata.layers[input_layer_key].toarray()
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )

        celltypes_labels = adata.obs["celltype_id"].values
        batch_ids = adata.obs["batch_id"].values.astype(int)
        genes = adata.var["gene_name"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)

        tokenized = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=self.max_length,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=False,
        )

        input_values = random_mask_value(
            tokenized["values"],
            mask_ratio=0.0,
            mask_value=-1,
            pad_value=-2,
        )

        return {
            "gene_ids": tokenized["genes"],
            "values": input_values,
            "target_values": tokenized["values"],
            "batch_labels": torch.from_numpy(batch_ids).long(),
            "celltype_labels": torch.from_numpy(celltypes_labels).long(),
        }


##### 数据定义模块
from torch.utils.data import Dataset, DataLoader, Subset

class SeqWithOriginalDataset(Dataset):
    """自定义数据集类，包含原始表达矩阵"""
    def __init__(self, data_dict, adata):
        self.data = data_dict
        self.adata = adata
        self.cell_ids = list(range(len(self)))

        X = adata.X
        self.original_X = X.toarray().astype("float32") if sp.issparse(X) else X.astype("float32")

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        sample = {
            "gene_ids": self.data["gene_ids"][idx],
            "values": self.data["values"][idx],
            "target_values": self.data["target_values"][idx],
            "batch_labels": self.data["batch_labels"][idx],
            "celltype_labels": self.data["celltype_labels"][idx],
            "original": torch.tensor(self.original_X[idx], dtype=torch.float32),
            "cell_id": torch.tensor(self.cell_ids[idx], dtype=torch.long)
        }
        return sample


##### 主类模块
from tqdm import tqdm
import pickle
from bio.models.adapter import FoundationAdapter

class scGPT(FoundationAdapter):
    """整合所有模块的主类"""

    def __init__(self, *args, **kargs):
        super(scGPT, self).__init__(*args, **kargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, model_cfg):
        """构建模型网络"""
        from bio.io.paths import model_resource
        model_dir = str(model_resource(model_cfg, 'scGPT', 'human', 'ckpt_path'))
        self.model, self.vocab = ModelLoader.load_pretrained_model(
            model_dir, self.cell_type_num
        )

        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.setup_task_heads(512)

        self.preprocessor = DataPreprocessor(self.vocab)
        self.tokenizer = DataTokenizer(self.vocab, model_cfg.get("max_length", 1200))

    def prepare_inputs(self, data_cfg, train=True):
        """获取数据加载器"""
        self.adata_all = self.read_adata(data_cfg["dataset_name"])

        adata_all = self.read_model_input()
        _, label = np.unique(np.array(adata_all.obs['celltype']), return_inverse=True)
        adata_all.obs['celltype_id'] = label
        adata_all.obs["batch_id"] = adata_all.obs["str_batch"] = "1"
        adata_origin = self.adata_all.copy()

        adata_all = self.preprocessor.process_adata(adata_all)
        data_pt = self.tokenizer.prepare_model_input(adata_all)

        dataset = SeqWithOriginalDataset(data_pt, adata_origin)
        self.all_loader = DataLoader(
            dataset,
            batch_size=self.cfgs['model_cfg'].get('inference_batch_size', 8),
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )

    def encode_batch(self, data_dict):
        """前向传播"""
        input_gene_ids = data_dict["gene_ids"].to(self.device)
        input_values = data_dict["values"].to(self.device)
        target_values = data_dict["target_values"].to(self.device)
        batch_labels = data_dict["batch_labels"].to(self.device)
        celltype_labels = data_dict["celltype_labels"].to(self.device)

        src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])
        output_dict = self.model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )

        cellemb = torch.nn.functional.normalize(output_dict['cell_emb'], dim=-1)
        return data_dict["original"], cellemb, celltype_labels

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
