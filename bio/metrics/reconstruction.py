"""reconstruction metrics; no model training or result persistence."""
import numpy as np
from ._common import matrices, correlations


def reconstruction(pred, truth, top_k=50):
    p, t = matrices(pred, truth)
    k = min(int(top_k), p.shape[1])
    if k < 1:
        raise ValueError('reconstruction_top_k must be positive')
    result = dict(MSE_expr=float(np.mean((p-t)**2)), MAE_expr=float(np.mean(np.abs(p-t))),
                  n_cells=len(p), n_genes=p.shape[1], top_k=k)
    for axis, suffix in [(1, 'expr'), (0, 'gene')]:
        for method in ['pearson', 'spearman']:
            value, n = correlations(p, t, method, axis)
            result[f'{method.title()}_{suffix}'] = value
            result[f'{method.title()}_{suffix}_valid_n'] = n
    result['TopK_expression_overlap'] = float(np.mean([
        len(set(np.argsort(-a, kind='stable')[:k]) & set(np.argsort(-b, kind='stable')[:k])) / k
        for a, b in zip(p, t)]))
    return result
