#!/usr/bin/env python3
"""Check dataset metadata and required model resources without importing PyTorch."""
import argparse
import json
from pathlib import Path
import sys

from bio.io.paths import PROJECT_ROOT as ROOT
from bio.data.foundation import REQUIRED as FRONTIER_REQUIRED
REQUIRED = {
    "CellPLM": ["20230926_85M.best.ckpt", "20230926_85M.config.json"],
    "Geneformer": ["gf-12L-30M-i2048/config.json", "gf-12L-30M-i2048/pytorch_model.bin",
                   "token_dictionary_gc30M.pkl", "gene_median_dictionary_gc30M.pkl"],
    "LangCell": ["cell_bert/config.json", "cell_bert/pytorch_model.bin", "cell_proj.bin",
                 "token_dictionary_gc30M.pkl", "gene_median_dictionary_gc30M.pkl"],
    "scFoundation": ["models.ckpt", "OS_scRNA_gene_index.19264.tsv"],
    "scGPT": ["human/args.json", "human/vocab.json", "human/best_model.pt"],
    "scMulan": ["ckpt_scMulan.pt", "meta_info.pt", "GeneSymbolRef_upd0731.csv", "total_gene_list_42117.txt"],
    "Nicheformer": ["config.json", "vocab.json", "model.safetensors", "configuration_nicheformer.py",
                   "modeling_nicheformer.py", "masking.py", "dissociated_mean_script.npy"],
    "Tabula": ["pancreas.pth", "vocab.json", "finetune_framework_annotation.yaml",
               "source/tabula/finetune/preprocessor.py"],
}
from bio.data.foundation import DATASETS
REQUIRED.update(FRONTIER_REQUIRED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", action="store_true", help="Inspect h5ad metadata (requires h5py)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"models": {}, "datasets": {}}
    for model, filenames in REQUIRED.items():
        missing = [n for n in filenames if not (ROOT / 'weights' / model / n).is_file()
                   or (ROOT / 'weights' / model / n).stat().st_size == 0]
        report['models'][model] = {"ready": not missing, "missing": missing}
        print(f"{model}: {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
    if args.datasets:
        import h5py
        for name, filename in DATASETS.items():
            path = ROOT / 'datasets' / filename
            if not path.is_file():
                report['datasets'][name] = {"ready": False, "missing": filename}
                continue
            with h5py.File(path, 'r') as f:
                obs = f['obs']
                var = f['var']
                index = obs.attrs.get('_index', '_index')
                var_index = var.attrs.get('_index', '_index')
                cells, genes = len(obs[index]), len(var[var_index])
                entry = {"path": str(path.relative_to(ROOT)), "cells": cells, "genes": genes,
                         "obs_columns": list(obs), "var_columns": list(var), "ready": True}
                if name in ('uc_epi', 'uc_imm'):
                    for suffix, count in (('cell', cells), ('celltype', cells), ('gene', genes),
                                          ('batch_health', cells), ('batch_location', cells), ('batch_patient', cells)):
                        sidecar = path.parent / f'{name}_{suffix}.tsv'
                        if not sidecar.is_file():
                            entry.setdefault('errors', []).append(f'Missing {sidecar.name}')
                        else:
                            with sidecar.open() as stream:
                                actual = sum(1 for _ in stream)
                            if actual != count:
                                entry.setdefault('errors', []).append(f'{sidecar.name}: {actual} rows, expected {count}')
                    entry['ready'] = not entry.get('errors')
                report['datasets'][name] = entry
                print(f"{name}: {cells:,} cells x {genes:,} genes ({'OK' if entry['ready'] else 'ERROR'})")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return int(any(not r['ready'] for section in report.values() for r in section.values()))


if __name__ == '__main__':
    raise SystemExit(main())
