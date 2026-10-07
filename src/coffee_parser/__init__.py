"""Shared code for the coffee order parser."""

from . import generator
from .menu import DRINK_CODES, MILKS, SIZES, SYRUPS, UNCLEAR
from .ollama import PROMPT, STOPS, clean, generate
from .parsing import parse_compact, to_json
from .testset import TEST_SET, category

# coffee_parser.nora and coffee_parser.experiment need torch, so they are imported
# explicitly by the research notebook rather than re-exported here.

__all__ = [
    "DRINK_CODES", "MILKS", "SIZES", "SYRUPS", "UNCLEAR",
    "PROMPT", "STOPS", "clean", "generate",
    "parse_compact", "to_json",
    "TEST_SET", "category",
    "generator",
]
