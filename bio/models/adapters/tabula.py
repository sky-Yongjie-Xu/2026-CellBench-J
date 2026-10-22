__all__ = ["Tabula"]

##### 模型加载模块
import os
from bio.io.paths import weight_path
import sys
import json
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import issparse
import scanpy as sc
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 添加 Tabula 路径到 sys.path
from bio.io.paths import project_path
tabula_path = str(project_path('public_code/Tabula'))
if tabula_path not in sys.path:
    sys.path.insert(0, tabula_path)

from tabula.finetune.preprocessor import check_vocab, Preprocessor, get_pretrained_model
from tabula.finetune.tokenizer import GeneVocab
from tabula.finetune.utils import FinetuneConfig
# binning 函数将从 dataloader 模块导入或自己实现

from bio.models.adapter import FoundationAdapter
from bio.models.heads import Pooler_classify, Decoder, PerturbDecoder




##### 数据预处理模块
class TabulaDataPreprocessor:
    def __init__(self, vocab_file, n_bins=51):
        self.vocab_file = vocab_file
        self.n_bins = n_bins
        self.preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=False,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            subset_hvg=False,
            binning=n_bins,
            result_binned_key="X_binned",
        )

    def process_adata(self, adata):
        """预处理AnnData对象"""
        # 确保 gene_name 存在
        if 'gene_name' not in adata.var.columns:
            adata.var["gene_name"] = adata.var_names.tolist()

        # 尝试转换 Ensembl ID 到 Gene Symbol

        # 检查并过滤 vocab 中的基因
        adata_filtered, removed_genes = check_vocab(adata, self.vocab_file)
        if len(removed_genes) > 0:
            print(f"[Tabula] 移除了 {len(removed_genes)} 个不在 vocab 中的基因")

        # 应用预处理器
        self.preprocessor(adata_filtered, batch_key=None)

        return adata_filtered


##### 数据集定义模块
class TabulaDataset(Dataset):
    """Tabula 数据集，返回模型所需的数据格式"""
    def __init__(self, adata, vocab_file, in_feature=1200, n_bins=51):
        self.adata = adata
        self.in_feature = in_feature
        self.n_bins = n_bins
        self.cell_ids = list(range(len(adata)))

        # 加载 vocab
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)

        # 获取基因 ID 列表
        self.gene_ids = adata.var['gene_name'].tolist()

        # 获取 binned 数据
        if 'X_binned' in adata.layers:
            self.binned_data = adata.layers['X_binned']
        else:
            # 如果没有 binned 数据，从 X_log1p 创建
            if 'X_log1p' in adata.layers:
                data = adata.layers['X_log1p']
            else:
                data = adata.X

            if issparse(data):
                data = data.toarray()

            # 对每行进行 binning（使用与 Tabula 教程相同的方法）
            binned_rows = []
            for row in data:
                # Tabula 的 binning 方法：对非零值进行分箱
                if len(row.shape) == 1:
                    non_zero_ids = row.nonzero()[0]
                    if len(non_zero_ids) > 0:
                        non_zero_row = row[non_zero_ids]
                        bins = np.quantile(non_zero_row, np.linspace(0, 1, self.n_bins - 1))
                        non_zero_digits = np.digitize(non_zero_row, bins)
                        binned_row = np.zeros_like(row, dtype=np.int64)
                        binned_row[non_zero_ids] = non_zero_digits
                    else:
                        binned_row = np.zeros_like(row, dtype=np.int64)
                else:
                    binned_row = np.zeros_like(row, dtype=np.int64)
                binned_rows.append(binned_row)
            self.binned_data = np.stack(binned_rows)

        # 获取原始表达数据
        X = adata.X
        self.original_X = X.toarray().astype("float32") if issparse(X) else X.astype("float32")

        # 获取标签
        if 'celltype' in adata.obs:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            self.labels = le.fit_transform(adata.obs['celltype'].astype(str))
            self.label_encoder = le
        else:
            self.labels = np.zeros(len(adata))
            self.label_encoder = None

        # 获取 batch 信息
        if 'str_batch' in adata.obs:
            batch_strings = adata.obs['str_batch'].astype(str).tolist()
            unique_batches = sorted(set(batch_strings))
            self.batch_mapping = {b: i for i, b in enumerate(unique_batches)}
            self.batch_ids = [self.batch_mapping[b] for b in batch_strings]
        else:
            self.batch_ids = [0] * len(adata)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        # 获取 binned 值
        binned_values = self.binned_data[idx]

        # 获取非零基因索引
        non_zero_indices = [i for i, value in enumerate(binned_values) if value != 0]

        # 如果非零基因不够，补充零值位置
        if len(non_zero_indices) < self.in_feature:
            zero_indices = [i for i in range(len(binned_values)) if i not in non_zero_indices]
            import random
            random.shuffle(zero_indices)
            non_zero_indices += zero_indices[:self.in_feature - len(non_zero_indices)]

        # 随机打乱
        import random
        random.shuffle(non_zero_indices)
        non_zero_indices = non_zero_indices[:self.in_feature]

        # 获取基因 ID 和值
        gene_names = [self.gene_ids[i] for i in non_zero_indices]
        genes = torch.tensor(
            [self.vocab.get(g.upper(), self.vocab.get('<pad>', 0)) for g in gene_names],
            dtype=torch.long
        )
        values = torch.tensor(binned_values[non_zero_indices], dtype=torch.float32)

        # 其他信息
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        batch = torch.tensor(self.batch_ids[idx], dtype=torch.long)
        original = torch.tensor(self.original_X[idx], dtype=torch.float32)
        cell_id = torch.tensor(self.cell_ids[idx], dtype=torch.long)

        return {
            "gene_ids": genes,
            "values": values,
            "label": label,
            "batch": batch,
            "original": original,
            "cell_id": cell_id
        }


##### 主类模块
class Tabula(FoundationAdapter):
    """Tabula 模型集成类"""

    def __init__(self, *args, **kwargs):
        super(Tabula, self).__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(self, model_cfg):
        """构建模型网络"""
        # 从配置中获取路径
        from bio.io.paths import model_resource
        model_path = str(model_resource(model_cfg, 'Tabula', 'pancreas.pth', 'ckpt_path'))
        vocab_file = str(model_resource(model_cfg, 'Tabula', 'vocab.json', 'vocab_file'))
        config_file = str(model_resource(model_cfg, 'Tabula', 'finetune_framework_annotation.yaml', 'config_file'))
        n_bins = model_cfg.get('n_bins', 51)
        in_feature = model_cfg.get('in_feature', 1200)

        # 确保路径是绝对路径
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), '../../', model_path)
        if not os.path.isabs(vocab_file):
            vocab_file = os.path.join(os.path.dirname(__file__), '../../', vocab_file)
        if not os.path.isabs(config_file):
            config_file = os.path.join(os.path.dirname(__file__), '../../', config_file)

        self.vocab_file = vocab_file
        self.n_bins = n_bins
        self.in_feature = in_feature

        # 加载配置
        finetune_config = FinetuneConfig(seed=self.seed, config_path=config_file)

        # 允许通过 model_cfg 覆盖 pretrained_backbone（支持 FlashAttention）
        requested_backbone = model_cfg.get('pretrained_backbone', 'FastFormer')

        if requested_backbone.lower() == 'flashattention':
            import tabula.model.transfomer.transformer as transformer_module
            from bio.models.compat.tabula_attention import NativeFlashAttention
            transformer_module.FlashAttention = NativeFlashAttention

        finetune_config.set_finetune_param('pretrained_backbone', requested_backbone)
        print(f"[Tabula] Using backbone: {requested_backbone}")

        # 设置 supervised_out_feature（cell type 数量）
        finetune_config.set_model_param('supervised_out_feature', self.cell_type_num)

        # 临时降低 Tabula 日志级别，只隐藏 INFO 级别的详细参数加载信息（保留 WARNING 和 ERROR）
        import logging
        tabula_logger = logging.getLogger('Tabula')

        # 保存原始级别
        original_logger_level = tabula_logger.level
        original_handlers_levels = {}
        # 只隐藏 INFO，保留 WARNING 和 ERROR
        for handler in tabula_logger.handlers:
            original_handlers_levels[handler] = handler.level
            handler.setLevel(logging.WARNING)  # 只显示 WARNING 及以上级别，隐藏 INFO

        tabula_logger.setLevel(logging.WARNING)  # 只显示 WARNING 及以上级别的日志

        try:
            # 加载预训练模型
            # 注意：Tabula 的 get_pretrained_model 会先尝试完整加载，如果失败（通常是因为预训练模型的结构与当前配置不完全匹配），
            # 会自动切换到部分加载模式。这是正常的流程，不是错误。
            # 预训练的权重可能是用 FlashAttention 训练的，但当前使用 FastFormer，或者 supervised_head 的类别数不同等。
            print(f"[Tabula] Loading pretrained model from {model_path}...")
            print(f"[Tabula] Note: If full load fails, partial loading will be attempted automatically.")

            self.backbone = get_pretrained_model(
                finetune_config=finetune_config,
                model_path=model_path,
                device=str(self.device)
            )
            print(f"[Tabula] Model loaded successfully (may be partial load if structures differ).")
        except Exception as e:
            # 如果加载完全失败，打印详细错误信息
            print(f"[Tabula] CRITICAL ERROR: Failed to load model completely: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # 恢复原始日志级别
            tabula_logger.setLevel(original_logger_level)
            for handler, level in original_handlers_levels.items():
                handler.setLevel(level)
        from bio.models.loading import verify_backbone
        state = torch.load(model_path, map_location='cpu')
        state = {('transformer.' + k if 'transformer.' + k in self.backbone.state_dict() else k): v
                 for k, v in state.items()}
        verify_backbone(self.backbone, state, ('feature_tokenizer.', 'transformer.', 'bn.', 'cls_token.'))
        self.backbone.eval()  # 设置为评估模式

        # 创建任务头
        # Tabula 的 embedding 维度是 d_token (通常是 192)
        emb_dim = finetune_config.get_model_param('d_token')

        self.setup_task_heads(emb_dim)

        # 创建数据预处理器
        self.preprocessor = TabulaDataPreprocessor(vocab_file, n_bins)

    def prepare_inputs(self, data_cfg, train=True):
        """获取数据加载器"""
        if hasattr(self, 'all_loader') and self.all_loader is not None:
            return

        print(f"[Tabula] Loading dataset: {data_cfg['dataset_name']}")

        # 使用 BaseModel 的 read_adata 方法读取数据
        adata_loaded = self.read_adata(data_cfg['dataset_name'])

        # 确保 train 列存在
        if 'train' not in adata_loaded.obs.columns:
            n_cells = adata_loaded.shape[0]
            np.random.seed(self.seed)
            train_indices = np.random.choice(n_cells, size=int(0.8 * n_cells), replace=False)
            adata_loaded.obs['train'] = 2  # 默认测试
            adata_loaded.obs.loc[adata_loaded.obs.index[train_indices], 'train'] = 0  # 训练

        # 确保 batch 信息存在
        if 'str_batch' not in adata_loaded.obs.columns:
            adata_loaded.obs['str_batch'] = '1'

        # 保存原始 adata（用于后续使用）
        self.adata_all = adata_loaded.copy()

        # 临时降低 Tabula 日志级别，避免输出过多的预处理信息（但保留 ERROR）
        import logging
        tabula_logger = logging.getLogger('Tabula')
        original_logger_level = tabula_logger.level
        original_handlers_levels = {}
        for handler in tabula_logger.handlers:
            original_handlers_levels[handler] = handler.level
            handler.setLevel(logging.ERROR)  # 设置为 ERROR 隐藏 INFO/WARNING，但保留 ERROR
        tabula_logger.setLevel(logging.ERROR)

        try:
            # 预处理数据
            adata_processed = self.preprocessor.process_adata(self.read_model_input())
        finally:
            # 恢复原始日志级别
            tabula_logger.setLevel(original_logger_level)
            for handler, level in original_handlers_levels.items():
                handler.setLevel(level)

        # 创建数据集
        dataset = TabulaDataset(adata_processed, self.vocab_file, self.in_feature, self.n_bins)
        dataset.original_X = self.evaluation_expression()

        # 创建数据加载器
        batch_size = self.cfgs['model_cfg'].get('inference_batch_size', data_cfg.get('batch_size', 32))
        self.all_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=0  # 避免多进程问题
        )
        self.train_loader = self.all_loader  # 兼容性设置

        print(f"[Tabula] Data prepared: {len(dataset)} cells, {len(adata_processed.var)} genes")

    def encode_batch(self, data_dict):
        """前向传播"""
        gene_ids = data_dict["gene_ids"].to(self.device)
        values = data_dict["values"].to(self.device)
        batch_info = data_dict.get("batch", None)
        if batch_info is not None:
            batch_info = batch_info.to(self.device)

        # 调用模型获取输出
        output = self.backbone(
            genes=gene_ids,
            values=values,
            batch_info=batch_info,
            head=None  # 不指定 head，只获取 embedding
        )

        # 提取 cell embedding
        cell_emb = output['cell_embed']  # shape: [batch_size, d_token]

        # 获取原始表达和标签
        raw_expr = data_dict["original"].cpu().numpy() if "original" in data_dict else None
        label = data_dict.get("label", None)
        if label is not None:
            label = label.cpu().numpy()

        return raw_expr, cell_emb.cpu().numpy(), label

    def forward_annotation(self, cell_emb):
        """前向传播 - 分类任务"""
        if isinstance(cell_emb, np.ndarray):
            cell_emb = torch.from_numpy(cell_emb).to(self.device)
        logit = self.classify(cell_emb)
        return logit

    def forward_reconstruction(self, cell_emb):
        """前向传播 - 重建任务"""
        if isinstance(cell_emb, np.ndarray):
            cell_emb = torch.from_numpy(cell_emb).to(self.device)
        pred = self.restruct(cell_emb)
        return pred

    def forward_perturbation(self, ctrl_emb, pert_id, ctrl_expr):
        """前向传播 - 扰动预测任务"""
        if isinstance(ctrl_emb, np.ndarray):
            ctrl_emb = torch.from_numpy(ctrl_emb).to(self.device)
        if isinstance(pert_id, np.ndarray):
            pert_id = torch.from_numpy(pert_id).to(self.device)
        if isinstance(ctrl_expr, np.ndarray):
            ctrl_expr = torch.from_numpy(ctrl_expr).to(self.device)
        delta_expr = self.pert_decoder(ctrl_emb, pert_id)
        pred_expr = (ctrl_expr + delta_expr).clamp_min(0)
        return pred_expr
