__all__ = ["Nicheformer"]

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from bio.models.adapter import FoundationAdapter
from bio.models.heads import Decoder, PerturbDecoder, Pooler_classify


def _safe_to_csr(x):
    import scipy.sparse as sp

    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def _remap_csr_columns(x_csr, col_map: np.ndarray, out_n_cols: int):
    import scipy.sparse as sp

    x_csr = x_csr.tocsr()
    n_rows = x_csr.shape[0]

    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    indices_parts = []
    data_parts = []

    nnz_total = 0
    for i in range(n_rows):
        start, end = x_csr.indptr[i], x_csr.indptr[i + 1]
        cols = x_csr.indices[start:end]
        mapped = col_map[cols]
        keep = mapped >= 0

        if keep.any():
            new_cols = mapped[keep]
            new_data = x_csr.data[start:end][keep]
            indices_parts.append(new_cols)
            data_parts.append(new_data)
            nnz_total += new_cols.shape[0]

        indptr[i + 1] = nnz_total

    if nnz_total == 0:
        return sp.csr_matrix((n_rows, out_n_cols), dtype=x_csr.dtype)

    indices = np.concatenate(indices_parts).astype(np.int32, copy=False)
    data = np.concatenate(data_parts)

    out = sp.csr_matrix((data, indices, indptr), shape=(n_rows, out_n_cols), dtype=x_csr.dtype)
    out.sum_duplicates()
    out.sort_indices()
    return out


def _sf_normalize_dense(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    counts = x.sum(axis=1)
    counts = counts + (counts == 0.0)
    scaling = 10000.0 / counts
    x *= scaling.reshape((-1, 1))
    return x


def _tokenize_dense_topk(x: np.ndarray, max_len: int, aux_tokens: int) -> np.ndarray:
    # x: (n_cells, n_genes)
    n_cells = x.shape[0]
    out = np.zeros((n_cells, max_len), dtype=np.int32)
    for i in range(n_cells):
        row = x[i]
        nz = np.nonzero(row)[0]
        if nz.size == 0:
            continue
        nz_sorted = nz[np.argsort(-row[nz])][:max_len]
        out[i, : nz_sorted.size] = (nz_sorted + aux_tokens).astype(np.int32)
    return out


@dataclass
class NicheformerPaths:
    hf_dir: str
    technology_mean_path: Optional[str]


class _NicheformerSeqDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, original_x: np.ndarray, labels: np.ndarray):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.original_x = original_x
        self.labels = labels

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'original': torch.tensor(self.original_x[idx], dtype=torch.float32),
            'celltype_labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'cell_id': torch.tensor(idx, dtype=torch.long),
        }


class Nicheformer(FoundationAdapter):
    def __init__(self, *args, **kwargs):
        super(Nicheformer, self).__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_vocab(self, vocab_path: str) -> Tuple[Dict[str, int], list[str]]:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)

        gene_items = [(k, v) for k, v in vocab.items() if isinstance(v, int) and v >= 30 and k.startswith('ENS')]
        gene_items.sort(key=lambda kv: kv[1])
        gene_ids = [k for k, _ in gene_items]
        return vocab, gene_ids

    def load_pretrained(self, model_cfg: Dict[str, Any]):
        from bio.io.paths import model_resource
        hf_dir = str(model_resource(model_cfg, 'Nicheformer', legacy_key='hf_dir'))

        vocab_path = os.path.join(hf_dir, 'vocab.json')
        self.vocab, self.gene_ids = self._load_vocab(vocab_path)
        self.aux_tokens = 30

        # context_length is total length; we will always prefix 3 metadata tokens
        with open(os.path.join(hf_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        self.context_length = int(config.get('context_length', 1500))
        self.gene_context_length = max(1, self.context_length - 3)

        self.default_specie_token = int(model_cfg.get('specie_token', self.vocab.get('[SPECIES_human]', 5)))
        self.default_assay_token = int(model_cfg.get('assay_token', self.vocab.get('[TECH_10x_3_tp]', 15)))
        self.default_modality_token = int(model_cfg.get('modality_token', self.vocab.get('[MODALITY_dissociated]', 3)))

        technology_mean_path = str(model_resource(model_cfg, 'Nicheformer', 'dissociated_mean_script.npy', 'technology_mean_path'))
        self.technology_mean = None
        if technology_mean_path and os.path.exists(technology_mean_path):
            self.technology_mean = np.load(technology_mean_path)

        if self.technology_mean is None or len(self.technology_mean) != len(self.gene_ids):
            raise ValueError("Nicheformer requires its complete technology mean in vocabulary order")

        from transformers import AutoModel

        self.model, loading = AutoModel.from_pretrained(
            hf_dir,
            trust_remote_code=True,
            local_files_only=True, output_loading_info=True,
        )
        if loading["missing_keys"] or loading.get("mismatched_keys"):
            raise ValueError(f"Nicheformer checkpoint mismatch: {loading}")
        self.model.to(self.device)
        self.model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.setup_task_heads(512)






    def _tokenize_adata(self, adata, chunk_size=256):
        """Port of native size-factor/technology-mean ranking, without log guessing."""
        genes = adata.var['ensembl_id'].astype(str).tolist() if 'ensembl_id' in adata.var else adata.var_names.tolist()
        mapping = {gene: i for i, gene in enumerate(self.gene_ids)}
        if sum(g in mapping for g in genes) < 100:
            raise ValueError('Insufficient Nicheformer Ensembl vocabulary overlap')
        col_map = np.asarray([mapping.get(g, -1) for g in genes], dtype=np.int32)
        counts = _remap_csr_columns(_safe_to_csr(adata.X), col_map, len(self.gene_ids))
        values = counts.data
        if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values, np.rint(values)):
            raise ValueError('Nicheformer requires genuine nonnegative integer counts')
        mean = self.technology_mean.copy()
        mean[mean == 0] = 1
        tokens = np.zeros((adata.n_obs, self.context_length), dtype=np.int64)
        tokens[:, :3] = [self.default_specie_token, self.default_assay_token, self.default_modality_token]
        for start in range(0, adata.n_obs, chunk_size):
            end = min(start + chunk_size, adata.n_obs)
            block = _sf_normalize_dense(counts[start:end].toarray())
            block = block / mean.reshape(1, -1)
            tokens[start:end, 3:] = _tokenize_dense_topk(block, self.gene_context_length, self.aux_tokens)
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(tokens != self.vocab['[PAD]'])

    def prepare_inputs(self, data_cfg, train=True):
        self.adata_all = self.read_adata(data_cfg['dataset_name'])

        adata_all = self.read_model_input()
        _, label = np.unique(np.array(adata_all.obs['celltype']), return_inverse=True)
        adata_all.obs['celltype_id'] = label

        # 对于correction任务，保存batch和celltype标签
        if self.task == "correction":
            from sklearn.preprocessing import LabelEncoder
            # 保存celltype标签
            self.cell_type_labels = label.copy()
            # 提取并保存batch标签
            if 'batch' in adata_all.obs.columns:
                batch_labels = adata_all.obs['batch'].values
                if not hasattr(self, 'batch_encoder') or self.batch_encoder is None:
                    self.batch_encoder = LabelEncoder()
                if len(batch_labels) > 0 and isinstance(batch_labels[0], str):
                    self.batch_labels = self.batch_encoder.fit_transform(batch_labels.astype(str))
                else:
                    self.batch_labels = batch_labels.astype(int)
            else:
                # 如果没有batch信息，创建默认batch
                self.batch_labels = np.zeros(adata_all.shape[0], dtype=int)
                if not hasattr(self, 'batch_encoder') or self.batch_encoder is None:
                    self.batch_encoder = LabelEncoder()
                    self.batch_encoder.fit(['batch_0'])

        # Keep original expression for downstream tasks
        x_original = self.evaluation_expression()

        input_ids, attention_mask = self._tokenize_adata(adata_all)

        dataset = _NicheformerSeqDataset(
            input_ids=input_ids,
            attention_mask=attention_mask,
            original_x=x_original,
            labels=label,
        )

        self.all_loader = DataLoader(
            dataset,
            batch_size=self.cfgs['model_cfg'].get('inference_batch_size', 8),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def encode_batch(self, input_dict):
        """前向传播，根据任务类型返回不同的值

        对于extract_cell_embeddings等任务：返回 (raw_expr, cell_emb, label)
        对于full_finetuning_annotation任务：返回 (logit, label, cell_emb)
        """
        # 检测是否在annotation任务中（通过检查是否有classify head且任务类型）
        is_annotation_mode = (
            hasattr(self, 'task') and
            self.task == "annotation" and
            hasattr(self, 'classify') and
            self.classify is not None and
            hasattr(self, '_full_finetuning_mode') and
            self._full_finetuning_mode
        )

        # 对于Nicheformer，需要特殊处理input_ids和attention_mask
        if 'input_ids' in input_dict and 'attention_mask' in input_dict:
            input_ids = input_dict['input_ids'].to(self.device)
            attention_mask = input_dict['attention_mask'].to(self.device)
            labels = input_dict.get('celltype_labels', input_dict.get('label', None))
            if labels is not None:
                labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else labels

            # 获取原始表达量
            raw_expr = input_dict.get('original', None)
            if raw_expr is None:
                raw_expr = input_dict.get('cell', None)

            with torch.no_grad():
                if hasattr(self.model, 'get_embeddings'):
                    cell_emb = self.model.get_embeddings(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        layer=-1,
                        with_context=False,
                    )
                else:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    if isinstance(out, (tuple, list)):
                        hidden = out[0]
                    else:
                        hidden = out.last_hidden_state
                    hidden = hidden[:, 3:, :].mean(dim=1)
                    cell_emb = hidden

            if is_annotation_mode:
                logit = self.forward_annotation(cell_emb)
                return logit, labels, cell_emb
            else:
                return raw_expr, cell_emb, labels
        else:
            raise ValueError('Nicheformer batch requires input_ids and attention_mask')

    def forward_annotation(self, cell_emb):
        """注释任务前向传播"""
        logit = self.classify(cell_emb)
        return logit

    def forward_reconstruction(self, cell_emb):
        """重建任务前向传播"""
        pred = self.restruct(cell_emb)
        return pred

    def forward_perturbation(self, ctrl_emb, pert_id, ctrl_expr):
        """扰动任务前向传播"""
        delta_expr = self.pert_decoder(ctrl_emb, pert_id)
        pred_expr = (ctrl_expr + delta_expr).clamp_min(0)
        return pred_expr

    def full_finetuning_annotation(self, cfgs):
        """重写 full_finetuning_annotation 以适配 Nicheformer"""
        print("start full finetuning annotation")

        self._full_finetuning_mode = True

        result = super().full_finetuning_annotation(cfgs)

        self._full_finetuning_mode = False

        return result
