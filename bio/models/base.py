"""Shared experiment context; task algorithms live in bio.tasks."""
from typing import Dict, Any
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from bio.tasks.common import TaskSupportMixin
from bio.tasks.embedding import EmbeddingTaskMixin
from bio.tasks.annotation import AnnotationTaskMixin
from bio.tasks.reconstruction import ReconstructionTaskMixin
from bio.tasks.perturbation import PerturbationTaskMixin
from bio.tasks.evaluation import MetricMixin


class BaseModel(TaskSupportMixin, EmbeddingTaskMixin, AnnotationTaskMixin,
                ReconstructionTaskMixin, PerturbationTaskMixin, MetricMixin, nn.Module):
    def __init__(self, cfgs: Dict[str, Any], training: bool = True):
        super().__init__()
        self.cfgs = cfgs
        self.training = training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.task = cfgs['model_cfg']['tasks']
        self.dataset_name = cfgs['data_cfg']["dataset_name"]
        self.model_name = cfgs['model_cfg']["model"]
        self.topk = 1 if cfgs['topk'] is None else cfgs['topk']
        self.seed = 1 if cfgs['seed'] is None else cfgs['seed']
        self.run_name =  self.dataset_name + "_" + self.model_name + f"_top{self.topk}_{self.seed}"

        self.label_encoder = LabelEncoder()

        self.cell_type_descriptions = {}
        adata = self.read_adata(cfgs['data_cfg']['dataset_name'])

        if self.task == "perturbation":
            self.pert_num = len(set(adata.obs['condition']))

        self.cell_type_num = len(set(adata.obs['celltype']))
        self.cell_dim_num = adata.shape[1]

        self.data_sub = None
        self.fitted = False
        self.downstream_tasks = cfgs['model_cfg']["tasks"]
        self.to(self.device)

        self.build_network(cfgs['model_cfg'])
        if self.task == "embedding_extraction":
            self.get_loader(cfgs['data_cfg'], train=True)
        else:
            self.get_loader(cfgs['data_cfg'], train=False)
        self.loss = nn.CrossEntropyLoss()

    def read_adata(self, dataset_name):
        from bio.data import datasets
        from bio.data.catalog import CURATED, read_curated
        if dataset_name in CURATED:
            return read_curated(dataset_name, self.cfgs['data_cfg'])
        return getattr(datasets, 'read_' + dataset_name)(self.cfgs['data_cfg'])

    def build_network(self, model_cfg):
        raise NotImplementedError

    def get_loader(self, data_cfg, train=True):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError('Use the adapter encode_batch implementation')
