"""CellPLM adapter using the pinned official zero-shot embedding pipeline."""
import sys
from pathlib import Path

import numpy as np
from torch import optim

from bio.io.paths import project_path, model_resource
from bio.models.adapter import FoundationAdapter

__all__ = ['CellPLM']


class CellPLM(FoundationAdapter):
    def load_pretrained(self, model_cfg):
        source = project_path(model_cfg.get('source_dir', 'public_code/CellPLM'))
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
        import CellPLM.pipeline as pipeline_module
        from CellPLM.model import OmicsFormer
        import json
        import torch
        from unittest.mock import patch
        from bio.models.loading import verify_backbone
        def cpu_load_pretrain(pretrain_prefix, overwrite_config=None, pretrain_directory='./ckpt'):
            path = Path(pretrain_directory)
            parameters = json.loads((path / f'{pretrain_prefix}.config.json').read_text())
            parameters.update(overwrite_config or {})
            model = OmicsFormer(**parameters)
            state = torch.load(str(path / f'{pretrain_prefix}.best.ckpt'), map_location='cpu')['model_state_dict']
            expected = model.state_dict()
            matched = {k: v for k, v in state.items() if k in expected and v.shape == expected[k].shape}
            model.load_state_dict(matched, strict=False)
            verify_backbone(model, state, ('embedder.', 'encoder.', 'latent.', 'pre_latent_norm.'))
            return model
        with patch.object(pipeline_module, 'load_pretrain', cpu_load_pretrain):
            self.pipeline = CellEmbeddingPipeline(
                pretrain_prefix=model_cfg.get('pretrain_prefix', '20230926_85M'),
                pretrain_directory=str(model_resource(model_cfg, 'CellPLM', legacy_key='pretrain_directory')))
        self.model = self.pipeline.model.to(self.device).eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # Official embedding head returns the complete pretrained latent representation.
        self.setup_task_heads(self.model.head.out_dim if hasattr(self.model.head, 'out_dim') else 512)

    def prepare_inputs(self, data_cfg, train=True):
        self.adata_all = self.read_adata(data_cfg['dataset_name'])
        data = self.read_model_input()
        data.obs = data.obs[[]].copy()  # No labels or batch IDs enter the encoder.
        self.all_loader = [{'adata': data, 'cell_id': np.arange(data.n_obs)}]

    def encode_batch(self, batch):
        embedding = self.pipeline.predict(
            batch['adata'],
            inference_config={'batch_size': self.cfgs['model_cfg'].get('inference_batch_size', 32)},
            ensembl_auto_conversion=False, device=self.device)
        _, labels = np.unique(self.adata_all.obs['celltype'].astype(str), return_inverse=True)
        return self.evaluation_expression(), embedding, labels

    def forward_annotation(self, cell_emb):
        return self.classify(cell_emb)

    def forward_reconstruction(self, cell_emb):
        return self.restruct(cell_emb)

    def forward_perturbation(self, ctrl_emb, pert_id, ctrl_expr):
        return (ctrl_expr + self.pert_decoder(ctrl_emb, pert_id)).clamp_min(0)

    def full_finetuning_annotation(self, cfgs):
        raise NotImplementedError('The official CellEmbeddingPipeline supports frozen zero-shot embeddings')
