"""Task orchestration; model encoders and metrics are supplied by the adapter."""
import numpy as np


class TaskSupportMixin:
    def get_fewshot_adata(self, adata, nshot=1, seed=None, label_key='celltype'):
        original_state = np.random.get_state()

        if seed is not None:
            np.random.seed(seed)

        labels = adata.obs[label_key]
        unique_labels = labels.unique()
        label_to_indices = {}
        train_idx = []

        if self.task == "perturbation":
            for label in unique_labels:
                indices = np.where(labels == label)[0]

                if len(indices) == 0:
                    continue

                nshot_actual = min(nshot, len(indices))
                selected_indices = np.random.choice(indices, nshot_actual, replace=False)
                train_idx.extend(selected_indices)

                train_idx = np.random.permutation(train_idx).tolist()
        else:
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                shuffled_indices = np.random.permutation(indices)
                label_to_indices[label] = shuffled_indices

            for label in unique_labels:
                idxs = label_to_indices[label][:nshot]
                train_idx.extend(idxs)

        np.random.set_state(original_state)
        return train_idx
