import numpy as np
from scipy.stats import pearsonr, spearmanr

def matrices(pred, truth):
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    if pred.ndim != 2 or pred.shape != truth.shape or min(pred.shape) == 0:
        raise ValueError('Predictions and targets must be nonempty aligned matrices')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite predictions/targets must not be converted into valid scores')
    return pred, truth

def correlations(pred, truth, method='pearson', axis=1):
    p, t = (pred, truth) if axis == 1 else (pred.T, truth.T)
    fn = pearsonr if method == 'pearson' else spearmanr
    values = [float(fn(a, b)[0]) for a, b in zip(p, t)
              if len(a) > 1 and np.ptp(a) > 0 and np.ptp(b) > 0]
    values = [x for x in values if np.isfinite(x)]
    return (float(np.mean(values)) if values else None), len(values)
