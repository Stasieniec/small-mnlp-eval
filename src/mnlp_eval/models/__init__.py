"""Translation backends and the loader registry."""

from mnlp_eval.models.base import SegmentOutput, Translator, TranslatorInfo
from mnlp_eval.models.registry import available_loaders, build_translator

__all__ = [
    "SegmentOutput",
    "Translator",
    "TranslatorInfo",
    "available_loaders",
    "build_translator",
]
