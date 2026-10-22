"""clustering metrics; no model training or result persistence."""
import numpy as np
from sklearn import metrics as skm


def clustering(embedding, labels, options=None, seed=41):
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    from scipy.optimize import linear_sum_assignment
    options = options or {}
    x = np.asarray(embedding, dtype=float)
    y = np.asarray(labels)
    if x.ndim != 2 or len(x) != len(y) or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('Clustering requires finite embeddings for at least three aligned cells')
    a = ad.AnnData(x)
    a.obs['celltype'] = pd.Categorical(y)
    a.obsm['X_emb'] = x
    sc.pp.neighbors(a, n_neighbors=min(int(options.get('n_neighbors', 15)), len(x)-1),
                    use_rep='X_emb', random_state=seed)
    mode = options.get('clustering_mode', 'fixed')
    if mode == 'oracle_nmi':
        import scib
        resolution, _, _ = scib.metrics.clustering.opt_louvain(
            a, label_key='celltype', cluster_key='cluster', use_rep='X_emb',
            function=scib.metrics.nmi, verbose=False, force=True)
    elif mode == 'fixed':
        resolution = float(options.get('clustering_resolution', 1.0))
        sc.tl.louvain(a, resolution=resolution, random_state=seed, key_added='cluster')
    else:
        raise ValueError('clustering_mode must be fixed or oracle_nmi')
    pred = a.obs['cluster'].to_numpy()
    table = skm.cluster.contingency_matrix(y, pred)
    row, col = linear_sum_assignment(-table)
    asw = (float(skm.silhouette_score(x, y))+1)/2 if 1 < len(np.unique(y)) < len(y) else None
    result = dict(acc=float(table[row, col].sum()/len(y)), ari=float(skm.adjusted_rand_score(y, pred)),
                  nmi=float(skm.normalized_mutual_info_score(y, pred)), asw=asw,
                  clustering_mode=mode, clustering_resolution=float(resolution), n_clusters=len(np.unique(pred)))
    result['AvgCC'] = float(np.mean([result['ari'], result['nmi'], asw])) if asw is not None else None
    return result, a
