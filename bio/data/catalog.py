"""Curated benchmark datasets; raw downloads are separate from runnable matrices."""
CURATED = {
    'bmmc': {'filename': 'bmmc.h5ad', 'batch': True},
    'hlca': {'filename': 'hlca.h5ad', 'batch': True},
    'heart': {'filename': 'heart.h5ad', 'batch': True},
    'kidney': {'filename': 'kidney.h5ad', 'batch': True},
    'seaad_mtg': {'filename': 'seaad_mtg.h5ad', 'batch': True},
    'replogleK562': {'filename': 'replogleK562.h5ad', 'perturbation': True},
    'replogleRPE1': {'filename': 'replogleRPE1.h5ad', 'perturbation': True},
}


def read_curated(name, data_cfg):
    from bio.data.datasets import _read_flat_dataset
    result = _read_flat_dataset(data_cfg, CURATED[name]['filename'],
                                batch_fields=('batch',), n_top_genes=data_cfg.get('hvg', 3000))
    if result is None:
        raise FileNotFoundError(f"Curated dataset {name} missing: run dataset download/preparation first")
    return result
