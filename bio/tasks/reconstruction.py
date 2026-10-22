"""Task orchestration; model encoders and metrics are supplied by the adapter."""
import numpy as np
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
import scanpy as sc
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error


class ReconstructionTaskMixin:
    def adapter_finetuning_reconstruction(self, cfgs):
        """提取原始数据和嵌入 + 训练验证测试集划分 + 微调 + 重建"""
        raw_exprs, cell_embs, labels, cell_ids = self.extract_cell_embeddings()

        has_nan = np.isnan(cell_embs).any()
        if has_nan:
            mask = ~np.isnan(cell_embs).any(axis=1)
            cell_embs = cell_embs[mask]
            raw_exprs = raw_exprs[mask]
            labels = labels[mask]
            cell_ids = cell_ids[mask]

        X_train, X_temp, y_train, y_temp = train_test_split(
            np.arange(len(cell_embs)), labels,
            test_size=0.2, random_state=cfgs['seed']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5, random_state=cfgs['seed']
        )

        scaler = StandardScaler()
        cell_embs_train_scaled = scaler.fit_transform(cell_embs[X_train])
        cell_embs_val_scaled = scaler.transform(cell_embs[X_val])
        cell_embs_test_scaled = scaler.transform(cell_embs[X_test])

        cell_embs_scaled = np.zeros_like(cell_embs)
        cell_embs_scaled[X_train] = cell_embs_train_scaled
        cell_embs_scaled[X_val] = cell_embs_val_scaled
        cell_embs_scaled[X_test] = cell_embs_test_scaled

        adata_train = ad.AnnData(X=raw_exprs[X_train])
        adata_train.obs['celltype'] = labels[X_train]
        adata_train.obsm['cell_emb'] = cell_embs_scaled[X_train]

        fewshot_idx = self.get_fewshot_adata(adata_train, self.topk, self.seed)
        X_fewshot = X_train[fewshot_idx]

        total_genes = raw_exprs.shape[1]
        hvg_train = self.adata_all[X_train].copy()
        sc.pp.highly_variable_genes(hvg_train, n_top_genes=min(400, hvg_train.n_vars))
        selected_gene_indices = np.flatnonzero(hvg_train.var['highly_variable'].to_numpy()).tolist()
        # if total_genes < 1000000000000000:
        #     selected_gene_indices = list(range(total_genes))
        # else:
        #     selected_gene_indices, cluster_mask = self.build_fixed_topk_cluster_mask(adata_train, cluster_key="celltype", topk_per_cluster=50)

        full_expr_selected = np.zeros((raw_exprs.shape[0], total_genes))
        full_expr_selected[:, selected_gene_indices] = raw_exprs[:, selected_gene_indices]

        fewshot_mask = torch.zeros((len(X_fewshot), total_genes), dtype=torch.bool)
        fewshot_mask[:, selected_gene_indices] = 1

        val_mask = torch.zeros((len(X_val), total_genes), dtype=torch.bool)
        val_mask[:, selected_gene_indices] = 1

        test_mask = torch.zeros((len(X_test), total_genes), dtype=torch.bool)
        test_mask[:, selected_gene_indices] = 1

        cell_embs_tensor = torch.FloatTensor(cell_embs_scaled)
        full_expr_tensor = torch.FloatTensor(full_expr_selected)

        fewshot_embs = cell_embs_tensor[X_fewshot]
        fewshot_expr = full_expr_tensor[X_fewshot]

        self.fewshot_loader = DataLoader(
            TensorDataset(fewshot_embs, fewshot_expr, fewshot_mask),
            batch_size=64, shuffle=True
        )

        val_embs = cell_embs_tensor[X_val]
        val_expr = full_expr_tensor[X_val]

        self.valid_loader = DataLoader(
            TensorDataset(val_embs, val_expr, val_mask),
            batch_size=64, shuffle=False
        )

        test_embs = cell_embs_tensor[X_test]
        test_expr = full_expr_tensor[X_test]

        self.test_loader = DataLoader(
            TensorDataset(test_embs, test_expr, test_mask),
            batch_size=64, shuffle=False
        )

        best_metric = float("inf")
        no_improve = 0
        patience = 5
        delta = 1e-4
        best_state_dict = None

        for epoch in range(500):
            self.restruct.train()
            running_loss = 0.0
            batch_count = 0
            for cell_emb, expr, mask in self.fewshot_loader:
                cell_emb = cell_emb.to(self.device)
                expr = expr.to(self.device)
                mask = mask.to(self.device)

                pred = self.forward_reconstruction(cell_emb)
                loss = F.mse_loss(pred[mask], expr[mask])

                self.optimizer_restruct.zero_grad()
                loss.backward()
                self.optimizer_restruct.step()

                running_loss += loss.item()
                batch_count += 1

            if (epoch + 1) % 20 == 0:
                self.restruct.eval()
                recon_list, truth_list = [], []

                with torch.no_grad():
                    for cell_emb, expr, mask in self.valid_loader:
                        cell_emb = cell_emb.to(self.device)
                        expr = expr.to(self.device)
                        mask = mask.to(self.device)

                        pred = self.forward_reconstruction(cell_emb)

                        original_shape = (expr.shape[0], len(selected_gene_indices))
                        pred = pred[mask].reshape(original_shape)
                        expr = expr[mask].reshape(original_shape)
                        recon_list.append(pred.cpu().numpy())
                        truth_list.append(expr.cpu().numpy())

                valid_input = {
                    "recon": np.concatenate(recon_list, 0),
                    "truth": np.concatenate(truth_list, 0),
                }

                val_metric = mean_squared_error(valid_input["truth"], valid_input["recon"])

                if val_metric < best_metric - delta:
                    best_metric = val_metric
                    best_state_dict = copy.deepcopy(self.restruct.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    break

        if best_state_dict is not None:
            self.restruct.load_state_dict(best_state_dict)

        self.restruct.eval()
        recon_list, truth_list = [], []

        with torch.no_grad():
            for cell_emb, expr, mask in self.test_loader:
                cell_emb = cell_emb.to(self.device)
                expr = expr.to(self.device)

                pred = self.forward_reconstruction(cell_emb)

                original_shape = (expr.shape[0], len(selected_gene_indices))
                pred = pred[mask].reshape(original_shape)
                expr = expr[mask].reshape(original_shape)
                recon_list.append(pred.cpu().numpy())
                truth_list.append(expr.cpu().numpy())

        recon = np.concatenate(recon_list, 0)
        truth = np.concatenate(truth_list, 0)
        test_input = {
            "recon": recon,
            "truth": truth,
        }
        final_score = self.metric(test_input, task="reconstruction")
        return recon, truth
