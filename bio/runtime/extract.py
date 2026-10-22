#!/usr/bin/env python3
"""Execute exactly one upstream model in its own environment and validate its output."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
sys.dont_write_bytecode = True

import numpy as np

from bio.io.paths import PROJECT_ROOT
from bio.data.foundation import MODELS, align_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()
    args.input, args.output = args.input.resolve(), args.output.resolve()
    args.weights, args.source = args.weights.resolve(), args.source.resolve()
    cache = PROJECT_ROOT / 'output/runtime_cache' / args.model
    for key, subdir in [('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib'),
                        ('CUDA_CACHE_PATH', 'cuda'), ('HF_HOME', 'huggingface')]:
        path = cache / subdir
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(path))
    config = json.loads(args.settings.read_text())
    if args.model == 'scPRINT2':
        os.environ.setdefault('LAMIN_SETTINGS_DIR', str(args.weights / 'ontology_settings'))
        os.environ.setdefault('LAMIN_CACHE_DIR', str(args.weights / 'ontology_cache'))
    os.environ.update({str(k): str(v) for k, v in config.get('backend_env', {}).items()})
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.model != 'CellFM':
        import torch
        torch.set_num_threads(int(config.get('cpu_threads', 4)))
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    import anndata as ad
    original = ad.read_h5ad(args.input, backed='r')
    expected = original.obs_names.to_numpy(dtype=str)
    original.file.close()
    from bio.runtime.frontier_backends import BACKENDS
    x, cell_ids = BACKENDS[args.model](args, config)
    x = align_embeddings(x, cell_ids, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + '.part')
    with partial.open('wb') as f:
        np.savez_compressed(f, embeddings=x, cell_ids=expected)
    partial.replace(args.output)
    print(f'{args.model}: saved {x.shape} to {args.output}', flush=True)


if __name__ == '__main__':
    main()
