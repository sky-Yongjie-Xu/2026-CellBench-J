"""Statistical baselines with the same extraction and cache contract as FMs.

These methods fit the supplied dataset; they do not load foundation checkpoints.
"""
import numpy as np

from bio.models.adapter import FoundationAdapter

__all__ = ['PCA_model', 'UMAP', 'scVI']


class StatisticalAdapter(FoundationAdapter):
    def load_pretrained(self, model_cfg):
        # There is no pretrained checkpoint for these dataset-fitted baselines.
        self._head_dim = None

    def prepare_inputs(self, data_cfg, train=True):
        self.adata_all = self.read_adata(data_cfg['dataset_name'])
        x = self.evaluation_expression()
        dim = int(self.cfgs['model_cfg'].get('embedding_dim', 256))
        if self.model_name == 'PCA_model':
            from sklearn.decomposition import PCA
            dim = min(dim, *x.shape)
            self.model = PCA(n_components=dim, random_state=self.seed)
            embedding = self.model.fit_transform(x)
        elif self.model_name == 'UMAP':
            import umap
            if len(x) < 4:
                raise ValueError('UMAP baseline requires at least four cells')
            dim = min(dim, x.shape[1], len(x) - 2)
            self.model = umap.UMAP(n_components=dim, random_state=self.seed,
                                   n_neighbors=min(15, len(x) - 1), init='random')
            embedding = self.model.fit_transform(x)
        else:
            import scvi
            data = self.read_model_input()
            # Never relabel normalized expression as counts.
            scvi.settings.seed = self.seed
            scvi.model.SCVI.setup_anndata(data)
            self.scvi_model = scvi.model.SCVI(data, n_layers=2, n_latent=dim, gene_likelihood='nb')
            self.scvi_model.train(max_epochs=self.cfgs['model_cfg'].get('max_epochs', 400),
                                  accelerator='gpu' if self.device == 'cuda' else 'cpu',
                                  devices=1, logger=False, enable_checkpointing=False)
            embedding = self.scvi_model.get_latent_representation(data)
        _, labels = np.unique(self.adata_all.obs['celltype'].astype(str), return_inverse=True)
        self._heads(embedding.shape[1])
        self.all_loader = [dict(embedding=embedding, original=x, label=labels,
                                cell_id=np.arange(len(x)))]

    def _heads(self, dim):
        if self._head_dim != dim:
            self.setup_task_heads(dim)
            self._head_dim = dim

    def encode_batch(self, batch):
        return batch['original'], batch['embedding'], batch['label']

    def forward_annotation(self, cell_emb):
        return self.classify(cell_emb)

    def forward_reconstruction(self, cell_emb):
        return self.restruct(cell_emb)

    def forward_perturbation(self, ctrl_emb, pert_id, ctrl_expr):
        return (ctrl_expr + self.pert_decoder(ctrl_emb, pert_id)).clamp_min(0)

    def full_finetuning_annotation(self, cfgs):
        raise NotImplementedError('Statistical baselines have no pretrained backbone to finetune')


class PCA_model(StatisticalAdapter):
    pass


class UMAP(StatisticalAdapter):
    pass


class scVI(StatisticalAdapter):
    pass
