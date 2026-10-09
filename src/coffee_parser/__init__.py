"""Shared code for the coffee order parser."""

from . import generator
from .menu import DRINK_CODES, MILKS, SIZES, SYRUPS, UNCLEAR
from .ollama import PROMPT, STOPS, clean, generate
from .parsing import parse_compact, to_json
from .testset import TEST_SET, category

__all__ = [
    "DRINK_CODES",
    "MILKS",
    "PROMPT",
    "SIZES",
    "STOPS",
    "SYRUPS",
    "TEST_SET",
    "UNCLEAR",
    "category",
    "clean",
    "generate",
    "generator",
    "parse_compact",
    "to_json",
]
