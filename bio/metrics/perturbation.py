"""perturbation metrics; no model training or result persistence."""
import numpy as np
from ._common import matrices, correlations


def perturbation(pred, truth, control, groups, top_k=50, min_effect=0.0):
    """Equal-weight condition × cell-type means in the supplied expression space.

    Delta is a difference in that space, NOT an inferred log fold change.
    Control must be a biologically matched reference, aligned to each target cell.
    """
    p, t = matrices(pred, truth)
    c, _ = matrices(control, truth)
    g = np.asarray(groups).astype(str)
    if g.shape != (len(p),):
        raise ValueError('Require one condition × cell-type group per predicted cell')
    names = np.unique(g)
    aggregate = lambda x: np.stack([x[g == key].mean(0) for key in names])
    p, t, c = map(aggregate, (p, t, c))
    dp, dt = p-c, t-c
    result = dict(MSE_expr=float(np.mean((p-t)**2)), MAE_expr=float(np.mean(np.abs(p-t))),
                  MSE_delta=float(np.mean((dp-dt)**2)), MAE_delta=float(np.mean(np.abs(dp-dt))),
                  control_baseline_MSE=float(np.mean(dt**2)),
                  control_baseline_MAE=float(np.mean(np.abs(dt))), n_groups=len(names),
                  aggregation='equal_weight_condition_celltype', delta_definition='difference_in_evaluation_expression_space')
    baseline = result['control_baseline_MSE']
    result['MSE_skill_vs_control'] = 1-result['MSE_expr']/baseline if baseline > 0 else None
    for method in ['pearson', 'spearman']:
        result[f'{method.title()}_delta'], result[f'{method.title()}_delta_valid_n'] = correlations(dp, dt, method)
    if min_effect < 0 or int(top_k) < 1:
        raise ValueError('Require nonnegative min_effect and positive perturbation_top_k')
    sign_scores, overlaps, per_group = [], [], {}
    k = min(int(top_k), p.shape[1])
    for key, a, b, pr, tr in zip(names, dp, dt, p, t):
        mask = np.abs(b) > min_effect
        sign = float(np.mean(np.sign(a[mask]) == np.sign(b[mask]))) if mask.any() else None
        if sign is not None:
            sign_scores.append(sign)
        directional = []
        for sign_dir in [1, -1]:
            true_idx = np.flatnonzero(sign_dir*b > min_effect)
            pred_idx = np.flatnonzero(sign_dir*a > min_effect)
            true_idx = true_idx[np.argsort(-sign_dir*b[true_idx], kind='stable')[:k]]
            pred_idx = pred_idx[np.argsort(-sign_dir*a[pred_idx], kind='stable')[:k]]
            if len(true_idx):
                directional.append(len(set(true_idx) & set(pred_idx))/len(true_idx))
        overlap = float(np.mean(directional)) if directional else None
        if overlap is not None:
            overlaps.append(overlap)
        per_group[key] = dict(MAE_expr=float(np.mean(np.abs(pr-tr))), MSE_expr=float(np.mean((pr-tr)**2)),
                              sign_agreement=sign, top_k_directional_recall=overlap,
                              changed_genes=int(mask.sum()))
    result.update(Sign_agree_changed=float(np.mean(sign_scores)) if sign_scores else None,
                  TopK_directional_recall=float(np.mean(overlaps)) if overlaps else None,
                  top_k=k, min_effect=min_effect, sign_valid_groups=len(sign_scores), per_group=per_group)
    return result
