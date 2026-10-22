"""Model registry and lazy access to configured adapter classes."""
import os
from bio.io.paths import PROJECT_ROOT
from .registry import get_model, list_models

os.environ.setdefault('HF_MODULES_CACHE', str(PROJECT_ROOT / 'output/runtime_cache/huggingface/modules'))
__all__ = list(list_models())


def __getattr__(name):
    if name in __all__:
        value = get_model(name)
        globals()[name] = value
        return value
    raise AttributeError(name)
