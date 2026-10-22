"""Store benchmark metrics locally without an experiment tracking service."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid

from bio.io.paths import output_path, project_path
from bio.metrics import METRIC_VERSION


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, 'detach'):
        return _json_value(value.detach().cpu().tolist())
    if hasattr(value, 'tolist'):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot save metric/config value of type {type(value).__name__}')


def start_run(cfgs):
    cfgs['_local_run_id'] = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ_') + uuid.uuid4().hex[:12]
    return cfgs['_local_run_id']


def log_metrics(cfgs, metrics, task=None):
    if int(os.environ.get('RANK', '0')) != 0:
        return None
    run_id = cfgs.get('_local_run_id') or start_run(cfgs)
    task = task or cfgs['model_cfg']['tasks']
    path = output_path(cfgs, 'metrics', task, run_id + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads(path.read_text())['summary'] if path.exists() else {}
    summary.update(_json_value(metrics))
    dataset = cfgs['data_cfg']['dataset_name']
    model = cfgs['model_cfg']['model']
    record = dict(metric_version=METRIC_VERSION, run_id=run_id, task=task, dataset=dataset, model=model,
                  seed=cfgs.get('seed', 41), topk=cfgs.get('topk', 1),
                  phase=cfgs.get('phase'), summary=summary,
                  config=_json_value(cfgs), updated_at=datetime.now(timezone.utc).isoformat())
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    print(f'Metrics saved: {path}', flush=True)
    return path


@dataclass
class LocalRun:
    name: str
    summary: dict
    dataset: str
    model: str
    topk: int
    seed: int


def load_runs(task, root='output', metric_version=METRIC_VERSION):
    """Load records for existing plotting/export scripts, newest repeat last."""
    folder = project_path(root) / 'metrics' / task
    runs = []
    for path in sorted(folder.glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('metric_version', 1) != metric_version:
            continue
        name = f"{record['dataset']}_{record['model']}_top{record['topk']}_{record['seed']}"
        runs.append(LocalRun(name, record['summary'], record['dataset'], record['model'],
                             record['topk'], record['seed']))
    if not runs:
        raise FileNotFoundError(f'No local {task} metrics in {folder}; run the benchmark first')
    return runs
