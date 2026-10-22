"""Task orchestration; model encoders and metrics are supplied by the adapter."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class EmbeddingTaskMixin:
    def perform_clustering(self, cell_embs, labels):
        """执行聚类任务"""
        input_dict = {"cell_emb": cell_embs, "label": labels}
        metric = self.metric(input_dict, "clustering")
        return metric

    def embedding_extraction(self, cfgs):
        """提取嵌入"""
        raw_exprs, cell_embs, labels, cell_ids = self.extract_cell_embeddings()
        return cell_embs, labels

    def no_finetuning_cluster(self, cfgs):
        """提取嵌入 + 聚类"""
        raw_exprs, cell_embs, labels, cell_ids = self.extract_cell_embeddings()

        has_nan = np.isnan(cell_embs).any()
        if has_nan:
            mask = ~np.isnan(cell_embs).any(axis=1)
            cell_embs = cell_embs[mask]
            labels = labels[mask]

        pca_model = PCA(n_components=min(256, cell_embs.shape[0], cell_embs.shape[1]))
        cell_embs = pca_model.fit_transform(cell_embs)

        metric = self.perform_clustering(cell_embs, labels)
        return cell_embs, labels

    def no_finetuning_correction(self, cfgs):
        """提取批次校正后的嵌入并计算correction指标"""
        raw_exprs, cell_embs, labels, cell_ids = self.extract_cell_embeddings()

        has_nan = np.isnan(cell_embs).any()
        if has_nan:
            mask = ~np.isnan(cell_embs).any(axis=1)
            cell_embs = cell_embs[mask]
            labels = labels[mask]

        pca_model = PCA(n_components=min(256, cell_embs.shape[0], cell_embs.shape[1]))
        cell_embs = pca_model.fit_transform(cell_embs)

        # 确保 batch_labels 的长度与 cell_embs 一致
        # 优先使用过滤后的 adata（如果存在，说明数据被预处理过滤过）
        if hasattr(self, 'adata_filtered') and self.adata_filtered is not None:
            # 使用过滤后的 adata，其细胞数量应该与 cell_embs 一致
            if len(self.adata_filtered) == len(cell_embs):
                batch_labels = pd.Categorical(self.adata_filtered.obs['batch'].values)
            else:
                raise ValueError(
                    f"Batch labels mismatch! Embeddings count ({len(cell_embs)}) "
                    f"does not match adata_filtered ({len(self.adata_filtered)})."
                )
        elif len(cell_ids) == len(cell_embs) and len(cell_ids) <= len(self.adata_all):
            # 检查 cell_ids 是否是有效的索引
            if np.all(cell_ids < len(self.adata_all)) and np.all(cell_ids >= 0):
                # cell_ids 是有效的索引，使用它们来索引 batch_labels
                batch_labels = pd.Categorical(self.adata_all.obs['batch'].values[cell_ids])
            else:
                raise ValueError(
                    f"Batch labels mismatch! cell_ids are not valid indices for adata_all."
                )
        else:
            # 长度不匹配
            raise ValueError(
                f"Batch labels mismatch! Embeddings count ({len(cell_embs)}) "
                f"does not match adata_filtered ({len(self.adata_filtered) if hasattr(self, 'adata_filtered') and self.adata_filtered is not None else 0}) "
                f"or provided cell_ids."
            )

        mode = cfgs.get('evaluation_cfg', {}).get('correction_mode', 'harmony')
        if mode == 'harmony':
            if len(np.unique(batch_labels)) < 2:
                raise ValueError('Harmony correction requires at least two batches')
            import harmonypy
            ho = harmonypy.run_harmony(cell_embs, pd.DataFrame({'batch': batch_labels}), ['batch'], random_state=self.seed)
            cell_embs = ho.Z_corr.T
        elif mode != 'native':
            raise ValueError('correction_mode must be harmony or native')

        # 计算correction指标
        input_dict = {
            'embedding': cell_embs,
            'batch_labels': batch_labels,
            'cell_type_labels': labels
        }
        metrics = self.metric(input_dict, task="correction")

        return cell_embs, labels, batch_labels
