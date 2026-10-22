"""Explicit, lazy registry. Selecting one model never imports unrelated adapters."""
from importlib import import_module
from types import MappingProxyType

MODEL_MODULES = MappingProxyType({
    'CellPLM': 'cellplm', 'Geneformer': 'geneformer', 'LangCell': 'langcell',
    'scFoundation': 'scfoundation', 'scGPT': 'scgpt', 'scMulan': 'scmulan',
    'Nicheformer': 'nicheformer', 'Tabula': 'tabula',
    'UCE': 'external', 'scPRINT2': 'external', 'TranscriptFormer': 'external',
    'CellFM': 'external', 'GeneCompass': 'external', 'scLong': 'external',
    'PCA_model': 'baselines', 'UMAP': 'baselines', 'scVI': 'baselines',
})


def get_model(name):
    if name not in MODEL_MODULES:
        raise ValueError(f'Unknown model {name!r}; available: {", ".join(MODEL_MODULES)}')
    module = import_module(f'bio.models.adapters.{MODEL_MODULES[name]}')
    return getattr(module, name)


def list_models():
    return tuple(MODEL_MODULES)
