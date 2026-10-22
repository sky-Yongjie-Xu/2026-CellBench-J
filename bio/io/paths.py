"""Resolve project resources independently of the process working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def weight_path(path):
    return str(project_path(Path("weights") / path))


def output_path(cfgs, *parts):
    return project_path(cfgs.get('output_dir', 'output')).joinpath(*parts)


def model_resource(model_cfg, model, filename='', legacy_key=None):
    """Resolve the canonical weights_dir, with old config keys as fallbacks."""
    if model_cfg.get('weights_dir'):
        return project_path(model_cfg['weights_dir']) / filename
    if legacy_key and model_cfg.get(legacy_key):
        return project_path(model_cfg[legacy_key])
    return project_path('weights') / model / filename
