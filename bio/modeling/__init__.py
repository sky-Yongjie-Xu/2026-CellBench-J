"""Compatibility namespace. New code should use bio.models.get_model."""
from bio.models import __all__, __getattr__, get_model, list_models
