"""Control matching and aligned perturbation datasets."""
import numpy as np
import torch
import anndata as ad
import scipy.sparse as sp
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

def create_evaluation_adata(pert_indices, control_map, raw_exprs, pred_exprs, labels, cell_ids, conditions, controls):
    """创建用于评估的完整测试集adata"""

    # 获取测试集的所有细胞索引
    pert_indices = np.asarray(pert_indices)
    all_pert_indices = list(pert_indices)           # 测试扰动细胞
    all_control_indices = np.flatnonzero((controls == 1) & np.isin(labels, labels[pert_indices])).tolist()
    all_indices = np.unique(all_pert_indices + all_control_indices)  # 测试集所有细胞

    # 创建真实数据的测试集完整adata
    adata_real = ad.AnnData(X=raw_exprs[all_indices])  # 只包含测试集细胞！
    adata_real.obs['control'] = controls[all_indices]    # 标识control/扰动
    adata_real.obs['condition'] = conditions[all_indices]
    adata_real.obs['celltype'] = labels[all_indices]
    adata_real.obs['cell_id'] = cell_ids[all_indices]

    # 创建预测数据的测试集完整adata
    adata_pred = adata_real.copy()

    # 只替换测试集扰动细胞的表达量
    for i, orig_idx in enumerate(all_indices):
        if orig_idx in pert_indices:  # 如果是测试集的扰动细胞
            # 用模型预测值替换
            pred_idx = np.where(pert_indices == orig_idx)[0][0]
            adata_pred.X[i] = pred_exprs[pred_idx]
        # control细胞保持真实值

    return adata_pred, adata_real

def match_controls(pert_indices, all_control_indices, labels, conditions, match_by_celltype=True):
    """为扰动细胞匹配相同细胞类型的control细胞，并返回匹配关系字典"""
    pert_to_control = {}
    rng = np.random.RandomState(42)

    for i, pert_idx in enumerate(pert_indices):
        if match_by_celltype:
            celltype = labels[pert_idx]
            available_controls = [
                ctrl for ctrl in all_control_indices
                if labels[ctrl] == celltype
            ]
        else:
            available_controls = all_control_indices

        if len(available_controls) == 0:
            raise ValueError(f'No matched control cells for perturbation row {pert_idx}')

        matched_control = rng.choice(available_controls)
        pert_to_control[pert_idx] = matched_control

    return pert_to_control

def create_perturbation_adata(pert_indices, control_map, raw_exprs, cell_embs, labels, cell_ids, conditions, controls):
    """创建只包含扰动细胞的adata，control数据以数组形式存储在obsm中"""

    # 只包含扰动细胞
    adata_pert = ad.AnnData(X=raw_exprs[pert_indices])
    adata_pert.obsm['cell_emb'] = cell_embs[pert_indices]
    adata_pert.obs['label'] = labels[pert_indices]
    adata_pert.obs['cell_id'] = cell_ids[pert_indices]
    adata_pert.obs['condition'] = conditions[pert_indices]
    adata_pert.obs['control'] = controls[pert_indices]
    adata_pert.obs['original_index'] = pert_indices

    # 准备control数据的数组形式（而不是字典）
    n_pert = len(pert_indices)
    n_genes = raw_exprs.shape[1]
    emb_dim = cell_embs.shape[1]

    # 创建数组来存储匹配的control数据
    control_embs = np.zeros((n_pert, emb_dim))
    control_exprs = np.zeros((n_pert, n_genes))
    control_indices = np.zeros(n_pert, dtype=int)

    for i, pert_idx in enumerate(pert_indices):
        ctrl_idx = control_map[pert_idx]
        control_embs[i] = cell_embs[ctrl_idx]
        control_exprs[i] = raw_exprs[ctrl_idx]
        control_indices[i] = ctrl_idx

    # 以数组形式存储在obsm中
    adata_pert.obsm['control_embeddings'] = control_embs
    adata_pert.obsm['control_expression'] = control_exprs
    adata_pert.obsm['control_original_indices'] = control_indices.reshape(-1, 1)
    adata_pert.obsm['perturbation_matching'] = np.array([(pert_idx, control_map[pert_idx])
                                                        for pert_idx in pert_indices])

    return adata_pert

class PerturbationMatchingDataset(Dataset):
    def __init__(self, adata):
        self.adata = adata  # 这个adata只包含扰动细胞

        # 检查必要的obsm字段
        required_obsm = ['control_embeddings', 'control_expression', 'perturbation_matching']
        for key in required_obsm:
            if key not in adata.obsm:
                raise ValueError(f"adata.obsm中缺少必要的字段: {key}")

        # 准备扰动细胞数据
        self.pert_expr = torch.tensor(
            adata.X.toarray() if sp.issparse(adata.X) else adata.X,
            dtype=torch.float32
        )
        self.pert_emb = torch.tensor(adata.obsm['cell_emb'], dtype=torch.float32)

        # 准备control数据（从obsm中获取数组）
        self.ctrl_emb = torch.tensor(adata.obsm['control_embeddings'], dtype=torch.float32)
        self.ctrl_expr = torch.tensor(adata.obsm['control_expression'], dtype=torch.float32)

        # 编码扰动条件
        if "pert_id" not in adata.obs.columns:
            encoder = LabelEncoder()
            adata.obs["pert_id"] = encoder.fit_transform(adata.obs["condition"])
        self.pert_ids = torch.tensor(adata.obs["pert_id"].values).long()

    def __len__(self):
        return len(self.adata)  # 只有扰动细胞的数量

    def __getitem__(self, idx):
        # 直接通过索引获取对应的control数据
        ctrl_emb = self.ctrl_emb[idx]
        ctrl_expr = self.ctrl_expr[idx]
        pert_emb = self.pert_emb[idx]
        pert_expr = self.pert_expr[idx]
        pert_id = self.pert_ids[idx]

        return (ctrl_emb, pert_emb, pert_id, pert_expr, ctrl_expr)

def collate_fn(batch):
    ctrl_emb   = torch.stack([b[0] for b in batch])
    pert_emb   = torch.stack([b[1] for b in batch])
    pert_id    = torch.stack([b[2] for b in batch])
    pert_expr  = torch.stack([b[3] for b in batch])
    ctrl_expr  = torch.stack([b[4] for b in batch])
    return ctrl_emb, pert_emb, pert_id, pert_expr, ctrl_expr
