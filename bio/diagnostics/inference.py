#!/usr/bin/env python3
"""Exercise a complete adapter, identity checks and cache on real cells."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from bio.io.paths import PROJECT_ROOT as ROOT
sys.dont_write_bytecode = True
CONFIGS = dict(CellPLM='cellplm', Geneformer='geneformer', LangCell='langcell',
               scFoundation='scfoundation', scGPT='scgpt', scMulan='scmulan',
               Nicheformer='nicheformer', Tabula='tabula', GeneCompass='genecompass',
               UCE='uce', scPRINT2='scprint2', TranscriptFormer='transcriptformer',
               CellFM='cellfm', scLong='sclong', PCA_model='pca', UMAP='umap', scVI='scvi')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=CONFIGS, required=True)
    parser.add_argument('--input', type=Path, default=ROOT / 'datasets/pbmc_12k.h5ad')
    parser.add_argument('--cells', type=int, default=2)
    parser.add_argument('--target-genes', type=int, default=100)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.cells < 2 or args.target_genes < 1:
        parser.error('Use at least two cells and one target gene')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('PYTHONNOUSERSITE', '1')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'output/runtime_cache/matplotlib'))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(ROOT / 'output/runtime_cache/numba'))
    import anndata as ad
    import numpy as np
    import torch
    from bio.config import config_loader
    from bio import models as modeling
    key = CONFIGS[args.model]
    output = ROOT / 'output/adapter_smoke' / key
    output.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(4)
        source = ad.read_h5ad(args.input, backed='r')
        data = source[:args.cells].to_memory()
        source.file.close()
        label_key = next((k for k in ('celltype', 'cell_type', 'str_labels', 'labels') if k in data.obs), None)
        if label_key is None:
            raise ValueError('Input needs cell type labels to verify evaluation alignment')
        data.obs['celltype'] = data.obs[label_key].astype(str)
        evaluation = data[:, :min(args.target_genes, data.n_vars)].copy()
        fixture = output / 'source.h5ad'
        data.write_h5ad(fixture)
        config = config_loader(str(ROOT / f'configs/{key}.yaml'))
        config.update(seed=41, topk=1, output_dir=str(output))
        config['model_cfg'].update(tasks='embedding_extraction', backend_device=args.device)
        config['data_cfg'].update(dataset_name='pbmc12k', foundation_input_path=str(fixture))
        model_class = getattr(modeling, args.model)

        class SmokeAdapter(model_class):
            def read_adata(self, *unused):
                return evaluation.copy()

        torch.manual_seed(41)
        np.random.seed(41)
        model = SmokeAdapter(config, False)
        expression, embedding, labels, ids = model.extract_cell_embeddings()
        assert expression.shape == evaluation.shape
        assert embedding.shape[0] == evaluation.n_obs and np.isfinite(embedding).all()
        cached = model.extract_cell_embeddings()
        assert np.array_equal(embedding, cached[1])
        report = dict(model=args.model, ok=True, shape=list(embedding.shape), raw_shape=list(expression.shape),
                      cache_repeat_identical=True, cell_ids=ids.tolist(), device=str(model.device),
                      input=str(args.input), cells=evaluation.n_obs)
    except Exception:
        report = dict(model=args.model, ok=False, error=traceback.format_exc())
        print(report['error'], flush=True)
    (output / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return int(not report['ok'])


if __name__ == '__main__':
    raise SystemExit(main())
