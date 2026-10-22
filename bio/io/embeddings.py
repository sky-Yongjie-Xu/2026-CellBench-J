"""Identity, provenance and atomic storage for every foundation adapter."""
import hashlib
import json
from pathlib import Path
import pickle
import uuid

import numpy as np

from bio.data.foundation import input_path
from bio.io.paths import PROJECT_ROOT, project_path

SCHEMA = 2


def fingerprint(cfgs):
    model = cfgs['model_cfg']['model']
    cfg = {k: v for k, v in cfgs['model_cfg'].items() if k != 'tasks'}
    inputs = [input_path(cfgs)]
    if cfgs['data_cfg']['dataset_name'] in ('uc_epi', 'uc_imm'):
        inputs += sorted(inputs[0].parent.glob(cfgs['data_cfg']['dataset_name'] + '_*.tsv'))
    resources = [project_path(cfg.get('weights_dir', f'weights/{model}'))]
    for key in ('pretrain_directory', 'ckpt_path', 'ckpt', 'hf_dir', 'vocab_file',
                'config_file', 'technology_mean_path', 'token_dictionary_file',
                'gene_median_file', 'gene_list_path', 'meta_info_path'):
        if cfg.get(key):
            resources.append(project_path(cfg[key]))
    def stat(path):
        return [str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns] if path.is_file() else [str(path), 'missing']
    files = set()
    for path in resources:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(p for p in path.rglob('*') if p.is_file()
                         and '.git' not in p.parts and '__pycache__' not in p.parts
                         and not p.name.endswith(('.part', '.pyc')))
    code = list((PROJECT_ROOT / 'bio').rglob('*.py'))
    code += list((PROJECT_ROOT / 'scripts').glob('*frontier*.py'))
    source = project_path(cfg.get('source_dir', f'public_code/{model}'))
    code += list(source.rglob('*.py'))
    payload = dict(schema=SCHEMA, model=cfg, data=cfgs['data_cfg'], seed=cfgs.get('seed', 41),
                   inputs=[stat(p) for p in inputs], resources=[stat(p) for p in sorted(files)],
                   code=[[str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p),
                          hashlib.sha256(p.read_bytes()).hexdigest()] for p in sorted(set(code)) if p.is_file()])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest(), payload


def validate_result(raw, embeddings, labels, cell_ids, expected_count):
    raw, embeddings, labels = np.asarray(raw), np.asarray(embeddings), np.asarray(labels)
    ids = np.asarray(cell_ids)
    if ids.ndim != 1 or len(ids) != expected_count or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Embedding output must preserve integer source-row IDs for every cell')
    if set(ids.tolist()) != set(range(expected_count)):
        raise ValueError('Embedding output dropped, duplicated or added source cells')
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_count or embeddings.shape[1] == 0:
        raise ValueError('Invalid embedding matrix shape')
    if raw.ndim != 2 or raw.shape[0] != expected_count or labels.shape != (expected_count,):
        raise ValueError('Expression/label rows do not match embedding rows')
    if not np.isfinite(embeddings).all() or not np.isfinite(raw).all():
        raise ValueError('Nonfinite model output; refuse zero replacement or silent row removal')
    order = np.argsort(ids)
    return raw[order], embeddings[order].astype(np.float32), labels[order], ids[order]


def atomic_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
