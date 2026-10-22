#!/usr/bin/env python3
"""Report source/resources/environment/inference separately for the six new models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from bio.io.paths import PROJECT_ROOT as ROOT
from bio.data.foundation import MODELS, check_resources
from bio.io.paths import project_path

CONFIGS = dict(zip(MODELS, ('uce', 'scprint2', 'transcriptformer', 'cellfm', 'genecompass', 'sclong')))
IMPORTS = {'UCE': 'import torch, accelerate, scanpy',
           'scPRINT2': 'import scprint2; from scprint2.tasks import Embedder',
           'TranscriptFormer': 'from transcriptformer.model.model import Transcriptformer',
           'CellFM': 'import mindspore; from mindspore import nn, ops',
           'GeneCompass': 'import transformers, torch, datasets; from transformers import Trainer; assert transformers.__version__ == "4.30.0"',
           'scLong': 'import torch, torch_geometric, local_attention'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/frontier_status.json')
    args = parser.parse_args()
    sources = {r['model']: r for r in json.loads((ROOT / 'public_code/sources.json').read_text())}
    result = {}
    for model in MODELS:
        cfg = yaml.safe_load((ROOT / f'configs/{CONFIGS[model]}.yaml').read_text())['model_cfg']
        python = project_path(os.environ.get(f'CELLBENCH_{model.upper()}_PYTHON', cfg['backend_python']))
        python = python.parent.resolve() / python.name
        entry = dict(source_ready=sources.get(model, {}).get('status') == 'complete',
                     commit=sources.get(model, {}).get('commit'), missing_resources=check_resources(model),
                     interpreter=str(python), environment_ready=False, inference_verified=False)
        if python.is_file():
            env = dict(os.environ, PYTHONNOUSERSITE='1', MPLCONFIGDIR=str(ROOT / 'output/runtime_cache/matplotlib'),
                       NUMBA_CACHE_DIR=str(ROOT / 'output/runtime_cache/numba'))
            if model == 'scPRINT2':
                env.update(LAMIN_SETTINGS_DIR=str(ROOT / 'weights/scPRINT2/ontology_settings'),
                           LAMIN_CACHE_DIR=str(ROOT / 'weights/scPRINT2/ontology_cache'))
            try:
                p = subprocess.run([str(python), '-c', IMPORTS[model]], env=env,
                                   capture_output=True, text=True, timeout=120)
                entry['environment_ready'] = p.returncode == 0
                if p.returncode:
                    entry['environment_error'] = p.stderr[-3000:]
            except Exception as exc:
                entry['environment_error'] = str(exc)
        else:
            entry['environment_error'] = 'Backend interpreter is not installed'
        smoke = ROOT / f'output/frontier_smoke/{CONFIGS[model]}/embeddings.npz'
        if smoke.is_file():
            import numpy as np
            with np.load(smoke, allow_pickle=False) as data:
                entry['inference_verified'] = bool(data['embeddings'].ndim == 2
                                                   and np.isfinite(data['embeddings']).all()
                                                   and len(data['cell_ids']) == len(data['embeddings']))
                entry['smoke_shape'] = list(data['embeddings'].shape)
        result[model] = entry
        print(f"{model}: source={entry['source_ready']} resources={not entry['missing_resources']} "
              f"environment={entry['environment_ready']} inference={entry['inference_verified']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return int(any(r['missing_resources'] or not r['environment_ready'] for r in result.values()))


if __name__ == '__main__':
    raise SystemExit(main())
