"""integration metrics; no model training or result persistence."""
import numpy as np
from .clustering import clustering


def correction(embedding, batch, labels, options=None, seed=41):
    """Use official scIB implementations; record optional dependency failures explicitly."""
    import pandas as pd
    import scib
    from bio.runtime.scib import lisi
    options = options or {}
    scores, a = clustering(embedding, labels, options, seed)
    batch = np.asarray(batch)
    if batch.shape != (a.n_obs,):
        raise ValueError('Batch labels must align to embedding rows')
    if pd.isna(batch).any() or pd.isna(labels).any():
        raise ValueError('Batch/cell-type annotations cannot be missing')
    a.obs['batch'] = pd.Categorical(batch)
    result = dict(ARI=scores['ari'], NMI=scores['nmi'], ASW_celltype=scores['asw'],
                  clustering_mode=scores['clustering_mode'], clustering_resolution=scores['clustering_resolution'],
                  metric_implementation='scib', scib_version=scib.__version__, metric_status={})
    tasks = {
        'Batch_ASW': lambda: scib.metrics.silhouette_batch(a, 'batch', 'celltype', 'X_emb', verbose=False),
        'iLISI_scaled': lambda: lisi(scib.metrics.ilisi_graph, a, 'batch', type_='knn', k0=min(90,a.n_obs-1), n_cores=1, scale=True),
        'cLISI_scaled': lambda: lisi(scib.metrics.clisi_graph, a, 'celltype', type_='knn', k0=min(90,a.n_obs-1), n_cores=1, scale=True),
        'Graph_connectivity': lambda: scib.metrics.graph_connectivity(a, 'celltype'),
        'kBET': lambda: scib.metrics.kBET(a, 'batch', 'celltype', type_='knn', scaled=True),
    }
    for name, fn in tasks.items():
        if name in ('Batch_ASW', 'iLISI_scaled', 'kBET') and a.obs.batch.nunique() < 2:
            result[name] = None
            result['metric_status'][name] = 'undefined: fewer than two batches'
            continue
        if name == 'cLISI_scaled' and a.obs.celltype.nunique() < 2:
            result[name] = None
            result['metric_status'][name] = 'undefined: fewer than two cell types'
            continue
        if name == 'kBET' and not options.get('compute_kbet', True):
            result[name] = None
            result['metric_status'][name] = 'disabled by configuration'
            continue
        try:
            value = float(fn())
            result[name] = value if np.isfinite(value) else None
            result['metric_status'][name] = 'ok' if np.isfinite(value) else 'undefined: no eligible neighborhoods/groups'
        except Exception as exc:
            if options.get('strict_optional_metrics', False):
                raise
            result[name] = None
            result['metric_status'][name] = f'unavailable: {type(exc).__name__}: {exc}'
    return result
