#!/usr/bin/env python3
"""Import the actual model adapters and exercise small CPU/GPU operations."""
import importlib
import json
from pathlib import Path
import sys
import site
import traceback

from bio.io.paths import PROJECT_ROOT as ROOT


def main():
    results = {'python': sys.version, 'prefix': sys.prefix, 'checks': {}}
    results['checks']['isolated_user_site'] = {'ok': not site.ENABLE_USER_SITE}
    modules = ['torch', 'torchtext', 'torch_geometric', 'transformers', 'peft',
               'scanpy', 'anndata', 'scvi', 'scgpt', 'geneformer', 'scib',
               'harmonypy', 'cell_eval', 'bio.models']
    for name in modules:
        try:
            module = importlib.import_module(name)
            results['checks'][name] = {'ok': True, 'version': str(getattr(module, '__version__', ''))}
            print(f'OK {name}', flush=True)
        except Exception:
            results['checks'][name] = {'ok': False, 'error': traceback.format_exc()}
            print(f'FAILED {name}\n{traceback.format_exc()}', flush=True)
    from bio.models import get_model, list_models
    for name in list_models():
        try:
            get_model(name)
            results['checks']['adapter:' + name] = {'ok': True}
        except Exception:
            results['checks']['adapter:' + name] = {'ok': False, 'error': traceback.format_exc()}
    try:
        import torch
        results['cuda_available'] = torch.cuda.is_available()
        results['cuda_runtime'] = torch.version.cuda
        if torch.cuda.is_available():
            results['gpu'] = torch.cuda.get_device_name(0)
            results['compute_capability'] = torch.cuda.get_device_capability(0)
            x = torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4)
            actual = (x @ x.T).cpu()
            expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            torch.testing.assert_close(actual, expected @ expected.T)
            torch.cuda.synchronize()
            results['checks']['cuda_matrix_multiply'] = {'ok': True}
            print(f"OK CUDA computation: {results['gpu']}", flush=True)
        else:
            results['checks']['cuda_matrix_multiply'] = {'ok': False, 'error': 'CUDA is unavailable'}
    except Exception:
        results['checks']['cuda_matrix_multiply'] = {'ok': False, 'error': traceback.format_exc()}
    output = ROOT / 'output/environment_check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(f'Report: {output}', flush=True)
    return int(any(not r['ok'] for r in results['checks'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
