"""Frozen embedding adapters; upstream inference runs in a separate interpreter."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from torch import nn

from bio.data.foundation import align_embeddings, check_resources, prepare_input
from bio.io.paths import PROJECT_ROOT, project_path
from bio.models.adapter import FoundationAdapter

__all__ = ['UCE', 'scPRINT2', 'TranscriptFormer', 'CellFM', 'GeneCompass', 'scLong']


class FrozenExternalModel(FoundationAdapter):
    def load_pretrained(self, model_cfg):
        variants = {'UCE': '4layer', 'scPRINT2': 'small-v2', 'TranscriptFormer': 'tf_sapiens',
                    'CellFM': 'base', 'GeneCompass': 'Small', 'scLong': '1b'}
        if model_cfg.get('variant') != variants[self.model_name]:
            raise ValueError(f'Unsupported {self.model_name} variant; this adapter implements {variants[self.model_name]}')
        if model_cfg.get('organism', 'NCBITaxon:9606') != 'NCBITaxon:9606':
            raise NotImplementedError('The initial six-model benchmark configurations support human datasets')
        self.model = nn.Identity()
        self._head_dim = None

    def prepare_inputs(self, data_cfg, train=True):
        self.adata_all = self.read_adata(data_cfg['dataset_name'])
        directory, _ = self.get_embedding_dir_and_filename()
        path = Path(directory) / 'foundation_input.h5ad'
        expected = prepare_input(self.cfgs, self.adata_all, path)
        self.all_loader = [{'input_path': path, 'expected': expected,
                            'cell_id': np.arange(self.adata_all.n_obs)}]

    def _heads(self, dim):
        if self._head_dim != dim:
            self.setup_task_heads(dim)
            self._head_dim = dim

    def encode_batch(self, batch):
        # Loading and native preprocessing execute in the model's isolated worker.
        config = self.cfgs['model_cfg']
        weights = project_path(config.get('weights_dir', f'weights/{self.model_name}'))
        missing = check_resources(self.model_name, weights)
        if missing:
            raise FileNotFoundError(f'{self.model_name}: missing {missing}')
        directory = batch['input_path'].parent
        python = project_path(os.environ.get(f'CELLBENCH_{self.model_name.upper()}_PYTHON',
                              config.get('backend_python', sys.executable)))
        python = python.parent.resolve() / python.name
        if not python.is_file():
            raise FileNotFoundError(f'Backend Python is missing: {python}')
        settings = directory / 'backend_settings.json'
        settings.write_text(json.dumps(config, indent=2))
        command = [str(python), str(PROJECT_ROOT / 'scripts/extract_frontier_embeddings.py'),
                   '--model', self.model_name, '--input', str(batch['input_path']),
                   '--output', str(directory / 'foundation_embeddings.npz'), '--weights', str(weights),
                   '--source', str(project_path(config.get('source_dir', f'public_code/{self.model_name}'))),
                   '--settings', str(settings), '--seed', str(self.seed)]
        env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)
        with np.load(directory / 'foundation_embeddings.npz', allow_pickle=False) as result:
            embeddings = align_embeddings(result['embeddings'], result['cell_ids'], batch['expected'])
        _, labels = np.unique(self.adata_all.obs['celltype'].astype(str), return_inverse=True)
        return self.evaluation_expression(), embeddings, labels

    def forward_annotation(self, cell_emb):
        return self.classify(cell_emb)

    def forward_reconstruction(self, cell_emb):
        return self.restruct(cell_emb)

    def forward_perturbation(self, ctrl_emb, pert_id, ctrl_expr):
        return (ctrl_expr + self.pert_decoder(ctrl_emb, pert_id)).clamp_min(0)

    def full_finetuning_annotation(self, cfgs):
        raise NotImplementedError('These adapters evaluate frozen pretrained embeddings; full backbone finetuning is not implemented')


class UCE(FrozenExternalModel):
    pass


class scPRINT2(FrozenExternalModel):
    pass


class TranscriptFormer(FrozenExternalModel):
    pass


class CellFM(FrozenExternalModel):
    pass


class GeneCompass(FrozenExternalModel):
    pass


class scLong(FrozenExternalModel):
    pass
