"""Export v2 scores and scale-invariant ranks on complete paired benchmark blocks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bio.metrics import METRIC_VERSION
from bio.io.paths import project_path

PRIMARY = {
    'clustering': {'ari': True, 'nmi': True, 'asw': True},
    'annotation': {'acc': True, 'f1_m': True},
    'reconstruction': {'Pearson_expr': True, 'MSE_expr': False},
    'perturbation': {'Pearson_delta': True, 'MAE_expr': False},
    'perturbation_vcc': {'PDS': True, 'DES': True, 'MAE': False},
    'correction': {'ARI': True, 'NMI': True, 'ASW_celltype': True,
                   'Batch_ASW': True, 'iLISI_scaled': True, 'kBET': True},
}


def paired_ranks(frame, primary):
    """Rank only complete dataset/seed blocks, then average seeds before datasets.

    Missing methods/metrics exclude a whole block, never reward missing failures.
    Different shot budgets, phases and metric protocols are never pooled.
    """
    keys = ['topk', 'phase', 'protocol']
    ranks, coverage = [], []
    for protocol, part in frame.groupby(keys, dropna=False):
        models = set(part['model'])
        for (dataset, seed), block in part.groupby(['dataset','seed'], dropna=False):
            complete = len(models) >= 2 and set(block.model) == models
            complete &= not block.model.duplicated().any()
            complete &= all(key in block and np.isfinite(pd.to_numeric(block[key], errors='coerce')).all() for key in primary)
            coverage.append(dict(zip(keys,protocol), dataset=dataset, seed=seed, complete=bool(complete),
                                 expected_models=len(models), observed_models=len(set(block.model))))
            if not complete:
                continue
            for metric, higher in primary.items():
                values=block[metric].rank(method='average',ascending=not higher)
                for model, rank in zip(block.model,values):
                    ranks.append(dict(zip(keys,protocol),dataset=dataset,seed=seed,model=model,metric=metric,rank=rank))
    if not ranks:
        return pd.DataFrame(), pd.DataFrame(coverage)
    # One contribution per dataset, irrespective of the number of random seeds.
    r=pd.DataFrame(ranks).groupby(keys+['dataset','model','metric'],dropna=False)['rank'].mean().reset_index()
    r=r.groupby(keys+['model','metric'],dropna=False)['rank'].mean().reset_index()
    overall=r.groupby(keys+['model'],dropna=False)['rank'].mean().reset_index(name='mean_metric_rank')
    return overall.sort_values(keys+['mean_metric_rank']), pd.DataFrame(coverage)


def export_task(task, root='output'):
    if task not in PRIMARY:
        raise ValueError(f'Unknown task {task}')
    root=project_path(root)
    rows=[]
    for path in sorted((root/'metrics'/task).glob('*.json')):
        record=json.loads(path.read_text())
        if record.get('metric_version',1) != METRIC_VERSION:
            continue
        cfg=record['config']
        protocol=dict(evaluation=cfg.get('evaluation_cfg',{}), scale=cfg.get('data_cfg',{}).get('evaluation_expression_scale','dataset_defined'))
        row={k:record.get(k) for k in ['dataset','model','seed','topk','phase','updated_at','run_id']}
        row['phase']=row['phase'] or 'unspecified'
        row['protocol']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()[:16]
        row['metric_version']=METRIC_VERSION
        row.update({k:v for k,v in record['summary'].items() if isinstance(v,(int,float)) or v is None})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No v{METRIC_VERSION} {task} results in {root}')
    frame=pd.DataFrame(rows).sort_values('updated_at')
    # Repeated executions are not independent seeds. Keep latest exact repeat.
    frame=frame.drop_duplicates(['dataset','model','seed','topk','phase','protocol'],keep='last')
    destination=root/'exports'/'v2'; destination.mkdir(parents=True,exist_ok=True)
    frame.to_csv(destination/f'{task}_scores.csv',index=False)
    rank,coverage=paired_ranks(frame,PRIMARY[task])
    rank.to_csv(destination/f'{task}_ranks.csv',index=False)
    coverage.to_csv(destination/f'{task}_coverage.csv',index=False)
    print(f'Exported {len(frame)} runs to {destination}; {int(coverage.complete.sum()) if len(coverage) else 0} complete comparison blocks')
    return frame,rank,coverage
