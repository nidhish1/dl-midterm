"""
Competition-style SVG cleanup from highestnotebook.ipynb (Kaggle baseline).

Forces viewBox=\"0 0 256 256\", width/height 256, xmlns; strips script; caps length at 8000.
Does not parse XML — same regex behavior as the notebook.
"""
from __future__ import annotations

import re


def enforce_svg_constraints(svg: str) -> str:
    if not svg:
        return ""
    svg = str(svg).strip()
    svg = re.sub(r"<script.*?>.*?</script>", "", svg, flags=re.DOTALL | re.IGNORECASE)
    if 'viewBox="0 0 256 256"' not in svg:
        svg = re.sub(r'viewBox="[^"]*"', 'viewBox="0 0 256 256"', svg)
        if 'viewBox="' not in svg:
            svg = re.sub(r"<svg", '<svg viewBox="0 0 256 256"', svg)
    if 'width="256"' not in svg:
        svg = re.sub(r"<svg", '<svg width="256" height="256"', svg)
    if 'xmlns="http://www.w3.org/2000/svg"' not in svg:
        svg = re.sub(r"<svg", '<svg xmlns="http://www.w3.org/2000/svg"', svg)
    if len(svg) > 8000:
        svg = svg[:8000]
        if "</svg>" not in svg[-100:]:
            svg = svg + "</svg>"
    return svg


# Same fallback SVG as highestnotebook.ipynb when enforce + validation fail.
NOTEBOOK_PLACEHOLDER_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">'
    '<rect width="256" height="256" fill="#F5F5F5"/>'
    '<circle cx="128" cy="128" r="80" fill="#4CAF50"/>'
    '<text x="128" y="200" text-anchor="middle" fill="#333" font-size="14">Generated</text>'
    "</svg>"
)


def is_valid_svg(svg: str) -> bool:
    if not svg:
        return False
    if not svg.startswith("<svg"):
        return False
    if len(svg) > 8000:
        return False
    if svg.count("<path") > 256:
        return False
    return True
