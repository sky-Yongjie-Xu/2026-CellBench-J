"""PDS/DES/MAE from cell_eval, equal weighting over condition × cell type."""
import tempfile
import numpy as np
from scipy import sparse


def evaluate_vcc(pred, truth, expression_scale, num_threads=1):
    from cell_eval import MetricsEvaluator
    if expression_scale != 'log1p_normalized':
        raise ValueError('VCC DE analysis requires declared log1p-normalized nonnegative expression. '
                         'Set data_cfg.evaluation_expression_scale=log1p_normalized only after verifying the dataset.')
    if pred.shape != truth.shape or not pred.obs_names.equals(truth.obs_names) or not pred.var_names.equals(truth.var_names):
        raise ValueError('VCC predictions and targets must have aligned cells and genes')
    for data in (pred, truth):
        x = data.X.data if sparse.issparse(data.X) else np.asarray(data.X)
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError('VCC expression must be finite and nonnegative')
    for key in ('condition', 'celltype'):
        if not pred.obs[key].astype(str).equals(truth.obs[key].astype(str)):
            raise ValueError(f'VCC {key} labels are misaligned')
    records = []
    mapping = {'PDS': 'discrimination_score_l1', 'DES': 'overlap_at_N', 'MAE': 'mae'}
    for celltype in sorted(truth.obs.celltype.astype(str).unique()):
        mask = truth.obs.celltype.astype(str).to_numpy() == celltype
        p, t = pred[mask].copy(), truth[mask].copy()
        if 'ctrl' not in set(t.obs.condition.astype(str)):
            raise ValueError(f'No control population for cell type {celltype}')
        if len(set(t.obs.condition.astype(str)) - {'ctrl'}) < 2:
            raise ValueError(f'PDS needs at least two perturbation candidates for {celltype}')
        with tempfile.TemporaryDirectory(prefix='cellbench-vcc-') as outdir:
            evaluator = MetricsEvaluator(adata_pred=p, adata_real=t, control_pert='ctrl',
                                         pert_col='condition', num_threads=num_threads,
                                         outdir=outdir, pdex_kwargs={'is_log1p': True})
            results, _ = evaluator.compute(profile='vcc')
        # Never index describe() by row number: its layout depends on library version.
        for row in results.to_dicts():
            if row['perturbation'] == 'ctrl':
                continue
            record = dict(celltype=celltype, condition=str(row['perturbation']))
            for name, column in mapping.items():
                value = row.get(column)
                record[name] = float(value) if value is not None and np.isfinite(value) else None
            records.append(record)
    if not records:
        raise ValueError('No eligible perturbation groups')
    summary = dict(n_groups=len(records), per_group=records,
                   aggregation='equal_weight_condition_celltype',
                   PDS_definition='cell_eval.discrimination_score_l1',
                   DES_definition='cell_eval.overlap_at_N', expression_scale=expression_scale)
    for name in mapping:
        values = [row[name] for row in records if row[name] is not None]
        summary[name] = float(np.mean(values)) if values else None
        summary[name+'_valid_groups'] = len(values)
    return summary
