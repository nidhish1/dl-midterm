"""Strict SVG validation + safe repairs for competition submissions."""

from post_processing.hard_validate import process_svg, FALLBACK_SVG

__all__ = ["process_svg", "FALLBACK_SVG"]
