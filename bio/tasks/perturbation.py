"""Task orchestration; model encoders and metrics are supplied by the adapter."""
import numpy as np
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import pickle
import copy
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from bio.io.results import log_metrics
from bio.data.perturbation import create_evaluation_adata, match_controls, create_perturbation_adata, PerturbationMatchingDataset, collate_fn


class PerturbationTaskMixin:
    def get_control_map_path(self, dataset_name, split_type):
        """生成control_map的存储路径"""
        from bio.io.artifacts import prediction_directory
        save_dir = str(prediction_directory(self.cfgs) / 'control_maps')
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{split_type}_control_map.pkl")

    def save_control_map(self, control_map, dataset_name, split_type):
        """保存control_map到文件"""
        filename = self.get_control_map_path(dataset_name, split_type)
        with open(filename, 'wb') as f:
            pickle.dump(control_map, f)
        print(f"Control map saved to {filename}")

    def load_control_map(self, dataset_name, split_type):
        """从文件加载control_map"""
        filename = self.get_control_map_path(dataset_name, split_type)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                control_map = pickle.load(f)
            print(f"Control map loaded from {filename}")
            return control_map
        return None

    def adapter_finetuning_perturbation(self, cfgs, eps=1e-3):
        """提取原始数据和嵌入 + 对照组匹配 + 固定测试集 + 训练验证划分 + 微调 + 扰动"""
        from bio.io.artifacts import prediction_directory
        save_dir = str(prediction_directory(self.cfgs))
        os.makedirs(save_dir, exist_ok=True)

        filename = f"{self.dataset_name}_{self.model_name}_seed{cfgs['seed']}_topk{self.topk}.pkl"
        save_path = os.path.join(save_dir, filename)

        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                test_input = pickle.load(f)
            if test_input.get("metric_version") != 2:
                raise ValueError("Incompatible perturbation prediction protocol")
            log_metrics(self.cfgs, test_input["metrics"], task="perturbation")
            return test_input["recon"].X, test_input["truth"].X

        raw_exprs, cell_embs, labels, cell_ids = self.extract_cell_embeddings()
        conditions = self.adata_all.obs['condition'].values
        controls = self.adata_all.obs['control'].values

        has_nan = np.isnan(cell_embs).any()
        if has_nan:
            mask = ~np.isnan(cell_embs).any(axis=1)
            cell_embs = cell_embs[mask]
            raw_exprs = raw_exprs[mask]
            labels = labels[mask]
            cell_ids = cell_ids[mask]
            conditions = conditions[mask]
            controls = controls[mask]

        is_perturbation = controls != 1
        perturb_indices = np.where(is_perturbation)[0]
        control_indices = np.where(~is_perturbation)[0]

        all_control_map = self.load_control_map(self.dataset_name, "all")
        if all_control_map is None:
            all_control_map = match_controls(perturb_indices, control_indices, labels, conditions)
            self.save_control_map(all_control_map, self.dataset_name, "all")

        perturb_indices_list = list(perturb_indices)
        perturb_train_val, perturb_test = train_test_split(
            perturb_indices_list, test_size=0.1, stratify=conditions[perturb_indices_list], random_state=42  # 固定10%作为测试集
        )
        perturb_train, perturb_val = train_test_split(
            perturb_train_val, test_size=0.1111, stratify=conditions[perturb_train_val], random_state=cfgs['seed']  # 训练:验证=8:1
        )

        train_control_map = {idx: all_control_map[idx] for idx in perturb_train if idx in all_control_map}
        val_control_map = {idx: all_control_map[idx] for idx in perturb_val if idx in all_control_map}
        test_control_map = {idx: all_control_map[idx] for idx in perturb_test if idx in all_control_map}

        condition_vocab = {name: i for i, name in enumerate(sorted(set(map(str, conditions))))}

        adata_train = create_perturbation_adata(
            perturb_train, train_control_map, raw_exprs, cell_embs, labels, cell_ids, conditions, controls
        )

        adata_val = create_perturbation_adata(
            perturb_val, val_control_map, raw_exprs, cell_embs, labels, cell_ids, conditions, controls
        )

        adata_test = create_perturbation_adata(
            perturb_test, test_control_map, raw_exprs, cell_embs, labels, cell_ids, conditions, controls
        )

        for part in (adata_train, adata_val, adata_test):
            part.obs['pert_id'] = part.obs['condition'].astype(str).map(condition_vocab).astype(int)

        X_train_fewshot = self.get_fewshot_adata(adata_train, self.topk, self.seed, label_key="condition")
        adata_fewshot = adata_train[X_train_fewshot].copy()

        self.pert_fewshotloader = DataLoader(
            PerturbationMatchingDataset(adata_fewshot),
            batch_size=64,
            collate_fn=collate_fn,
            shuffle=True
        )

        self.pert_validloader = DataLoader(
            PerturbationMatchingDataset(adata_val),
            batch_size=64,
            collate_fn=collate_fn,
            shuffle=False
        )

        self.pert_testloader = DataLoader(
            PerturbationMatchingDataset(adata_test),
            batch_size=64,
            collate_fn=collate_fn,
            shuffle=False
        )

        best_metric = float("inf")
        no_improve = 0
        patience = 5
        delta = 1e-4
        best_state_dict = None

        for epoch in range(500):
            self.pert_decoder.train()
            running_loss = 0.0

            for batch in self.pert_fewshotloader:
                ctrl_emb, pert_emb, pert_id, pert_expr, ctrl_expr = batch
                ctrl_emb, pert_emb, pert_id = ctrl_emb.to(self.device), pert_emb.to(self.device), pert_id.to(self.device)
                pert_expr, ctrl_expr = pert_expr.to(self.device), ctrl_expr.to(self.device)

                pred_expr = self.forward_perturbation(ctrl_emb, pert_id, ctrl_expr)

                loss_expr = F.mse_loss(pred_expr, pert_expr)

                loss = loss_expr

                self.optimizer_pert.zero_grad()
                loss.backward()
                self.optimizer_pert.step()

                running_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                self.pert_decoder.eval()
                all_pred, all_true, all_ctrl = [], [], []

                with torch.no_grad():
                    for batch in self.pert_validloader:
                        ctrl_emb, pert_emb, pert_id, pert_expr, ctrl_expr = batch
                        ctrl_emb, pert_emb, pert_id = ctrl_emb.to(self.device), pert_emb.to(self.device), pert_id.to(self.device)
                        pert_expr, ctrl_expr = pert_expr.to(self.device), ctrl_expr.to(self.device)

                        pred_expr = self.forward_perturbation(ctrl_emb, pert_id, ctrl_expr)

                        all_pred.append(pred_expr.cpu().numpy())
                        all_true.append(pert_expr.cpu().numpy())
                        all_ctrl.append(ctrl_expr.cpu().numpy())

                valid_input = {
                    "recon": np.concatenate(all_pred, 0),
                    "truth": np.concatenate(all_true, 0),
                    "ctrl":  np.concatenate(all_ctrl, 0),
                }

                val_metric = mean_squared_error(valid_input["truth"], valid_input["recon"])

                if val_metric < best_metric - delta:
                    best_metric = val_metric
                    best_state_dict = copy.deepcopy(self.pert_decoder.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    break

        if best_state_dict is not None:
            self.pert_decoder.load_state_dict(best_state_dict)

        self.pert_decoder.eval()
        all_pred, all_true, all_ctrl, all_pert_id = [], [], [], []

        with torch.no_grad():
            for batch in self.pert_testloader:
                ctrl_emb, pert_emb, pert_id, pert_expr, ctrl_expr = batch
                ctrl_emb, pert_emb, pert_id = ctrl_emb.to(self.device), pert_emb.to(self.device), pert_id.to(self.device)
                pert_expr, ctrl_expr = pert_expr.to(self.device), ctrl_expr.to(self.device)

                pred_expr = self.forward_perturbation(ctrl_emb, pert_id, ctrl_expr)

                all_pred.append(pred_expr.cpu().numpy())
                all_true.append(pert_expr.cpu().numpy())
                all_ctrl.append(ctrl_expr.cpu().numpy())

        pred_exprs = np.concatenate(all_pred, 0)
        true_exprs = np.concatenate(all_true, 0)
        ctrl_exprs = np.concatenate(all_ctrl, 0)

        test_input = {
            "recon": pred_exprs,
            "truth": true_exprs,
            "ctrl": ctrl_exprs,
        }

        import json
        test_input['groups'] = np.array([json.dumps([str(conditions[i]), str(labels[i])]) for i in perturb_test])
        references = {}
        for label in np.unique(labels[perturb_test]):
            eligible = control_indices[labels[control_indices] == label]
            if not len(eligible):
                raise ValueError(f'No same-cell-type control reference for label {label}')
            references[label] = raw_exprs[eligible].mean(axis=0)
        test_input['ctrl'] = np.stack([references[labels[i]] for i in perturb_test])
        final_score = self.metric(test_input, task="perturbation")

        adata_pred, adata_real = create_evaluation_adata(
            pert_indices=perturb_test,
            control_map=test_control_map,
            raw_exprs=raw_exprs,  # 所有细胞的原始表达量
            pred_exprs=pred_exprs,  # 模型预测的扰动表达量
            labels=labels,
            cell_ids=cell_ids,
            conditions=conditions,
            controls=controls
        )

        adata_pred.var_names = self.adata_all.var_names.copy()
        adata_real.var_names = self.adata_all.var_names.copy()

        test_input_vcc = {
            "recon": adata_pred,
            "truth": adata_real,
            "metric_version": 2,
            "metrics": final_score,
        }

        print(f"Saving test_input to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(test_input_vcc, f)
        return adata_pred.X, adata_real.X
