"""Task orchestration; model encoders and metrics are supplied by the adapter."""
import numpy as np
import torch
import anndata as ad
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AnnotationTaskMixin:
    def adapter_finetuning_annotation(self, cfgs):
        """提取嵌入 + 训练测试集划分 + 数据标准化 + 微调 + 注释"""
        raw_exprs, cell_embs, labels, cell_ids = self.extract_cell_embeddings()

        has_nan = np.isnan(cell_embs).any()
        if has_nan:
            mask = ~np.isnan(cell_embs).any(axis=1)
            cell_embs = cell_embs[mask]
            labels = labels[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            cell_embs, labels, test_size=0.2, random_state=cfgs['seed'], stratify=labels
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        adata_train = ad.AnnData(X=X_train_scaled)
        adata_train.obs['celltype'] = y_train
        fewshot_idx = self.get_fewshot_adata(adata_train, self.topk, self.seed)
        adata_fewshot = adata_train[fewshot_idx]
        fewshot_dataset = TensorDataset(torch.FloatTensor(adata_fewshot.X), torch.LongTensor(adata_fewshot.obs['celltype']))
        self.fewshot_loader = DataLoader(fewshot_dataset, batch_size=64, shuffle=True)

        test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        for epoch in range(500):
            running_loss = 0.0
            correct = 0
            total = 0
            self.classify.train()
            for inputs, label in self.fewshot_loader:
                if inputs.size(0) == 1:
                    continue
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                logit = self.forward_annotation(inputs)
                loss = self.loss(logit, label)

                self.optimizer_classify.zero_grad()
                loss.backward()
                self.optimizer_classify.step()

                running_loss += loss.item()
                _, predicted = torch.max(logit.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        test_preds = []
        test_probabilities = []
        test_labels = []
        self.classify.eval()
        with torch.no_grad():
            for inputs, label in self.test_loader:
                inputs = inputs.to(self.device)
                label = label.to(self.device)
                logit = self.forward_annotation(inputs)
                pred = torch.softmax(logit, dim=1).argmax(dim=1)
                test_probabilities.append(torch.softmax(logit, dim=1).cpu().numpy())
                test_preds.append(pred.cpu().numpy())
                test_labels.append(label.cpu().numpy())

        preds = np.concatenate([x.ravel() for x in test_preds])
        labels = np.concatenate([x.ravel() for x in test_labels])
        input_dict = {"pred":preds, "label":labels, "probabilities": np.concatenate(test_probabilities),
                      "classes": np.arange(self.cell_type_num)}

        metric = self.metric(input_dict, "annotation")
        return labels, preds

    def full_finetuning_annotation(self, cfgs):
        print("start full finetuning annotation")

        for epoch in range(500):
            embeddings = []
            logits = []
            labels = []
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs in self.fewshot_loader:
                logit, label, cell_emb = self.forward(inputs)
                loss = self.loss(logit,label)
                loss.backward()

                if hasattr(self, 'classify') and self.classify is not None:
                    self.optimizer_classify.step()
                    self.optimizer_classify.zero_grad()

                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()
                _, predicted = torch.max(logit.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        preds = []
        labels = []
        cell_embs = []
        with torch.no_grad():
            if self.model_name not in ["PCA_model","UMAP","scVI"]:
                self.model.eval()
            for inputs in self.test_loader:
                logit, label, cell_emb = self.forward(inputs)
                pred = torch.softmax(logit, dim=1).argmax(dim=1)
                preds.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())
                cell_embs.append(cell_emb.cpu().numpy())

        preds = np.concatenate([x.ravel() for x in preds])
        labels = np.concatenate([x.ravel() for x in labels])
        cell_emb = np.concatenate(cell_embs,0)
        input_dict = {"pred":preds, "label":labels, "cell_emb":cell_emb}
        metric = self.metric(input_dict, "annotation")

        return np.vstack(cell_embs), np.hstack(labels), np.hstack(preds)
