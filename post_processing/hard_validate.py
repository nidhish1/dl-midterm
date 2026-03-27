"""
Hard validation + safe repairs for raw model SVG (submission CSV).

Safe: trim outer junk, collapse whitespace between tags, drop script/metadata/foreignObject,
unwrap other disallowed wrappers (preserve child geometry), add missing xmlns, valid XML serialize.

Not done here: path coordinate rewriting, viewBox changes, transform flattening, canonicalizer numeric rounding.
"""
from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from svg_canonicalizer import ALLOWED_TAGS, local_name, count_paths

FALLBACK_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"></svg>'
)

MAX_SVG_LEN = 16_000
MAX_PATHS = 256

# Filter primitives not listed in ALLOWED_TAGS but often valid inside <filter>.
_FILTER_PRIMITIVES = {
    "feBlend",
    "feColorMatrix",
    "feComponentTransfer",
    "feComposite",
    "feConvolveMatrix",
    "feDiffuseLighting",
    "feDisplacementMap",
    "feDistantLight",
    "feDropShadow",
    "feFlood",
    "feFuncA",
    "feFuncB",
    "feFuncG",
    "feFuncR",
    "feGaussianBlur",
    "feImage",
    "feMerge",
    "feMergeNode",
    "feMorphology",
    "feOffset",
    "fePointLight",
    "feSpecularLighting",
    "feSpotLight",
    "feTile",
    "feTurbulence",
}

ALLOWED_ALL: set[str] = set(ALLOWED_TAGS) | _FILTER_PRIMITIVES

# Entire subtree removed (no drawing impact or unsafe to keep).
SAFE_DROP_SUBTREE = {
    "script",
    "metadata",
    "foreignObject",
    "animate",
    "animateTransform",
    "animateMotion",
    "set",
    "mpath",
}

# If present, prefer full-row fallback (losing these changes semantics / raster).
FORBID_TAG_FALLBACK = {"image", "video", "audio", "iframe"}

_SVG_BLOCK_RE = re.compile(r"<svg[\s\S]*?</svg>", re.IGNORECASE)


def extract_first_svg_block(text: str) -> str | None:
    m = _SVG_BLOCK_RE.search(text or "")
    if not m:
        return None
    return m.group(0).strip()


def collapse_tags_whitespace(s: str) -> str:
    return re.sub(r">\s+<", "><", s.strip())


def ensure_svg_xmlns(root: ET.Element) -> None:
    if local_name(root.tag) != "svg":
        return
    # Honour existing default namespace via attrib keys
    if root.tag.startswith("{"):
        return
    if not any(local_name(k) == "xmlns" for k in root.attrib):
        root.set("xmlns", "http://www.w3.org/2000/svg")


def _needs_fallback_for_disallowed(tag: str) -> bool:
    t = tag.lower()
    return t in {x.lower() for x in FORBID_TAG_FALLBACK}


def repair_tree(root: ET.Element) -> str | None:
    """
    Mutate tree in place. Returns 'fallback' if a forbidden tag is seen,
    None if repairs applied successfully.
    """
    changed = True
    safety = 0
    while changed and safety < 10_000:
        safety += 1
        changed = False
        for parent in root.iter():
            for child in list(parent):
                tag = local_name(child.tag)
                if tag in ALLOWED_ALL:
                    continue
                if _needs_fallback_for_disallowed(tag):
                    return "fallback"
                if tag in SAFE_DROP_SUBTREE:
                    parent.remove(child)
                    changed = True
                    continue
                # Unwrap: replace disallowed wrapper with its children
                idx = list(parent).index(child)
                parent.remove(child)
                for j, gc in enumerate(list(child)):
                    parent.insert(idx + j, gc)
                changed = True
                break
            if changed:
                break
    return None


def tree_tags_valid(root: ET.Element) -> bool:
    for el in root.iter():
        if local_name(el.tag) not in ALLOWED_ALL:
            return False
    return True


def serialize_svg(root: ET.Element) -> str:
    try:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
    except Exception:
        pass
    out = ET.tostring(root, encoding="unicode", method="xml", default_namespace=None)
    out = collapse_tags_whitespace(out)
    return out


@dataclass
class RowResult:
    id: str
    ok: bool
    used_fallback: bool
    reason: str
    path_count: int
    out_len: int


def process_svg(svg: str) -> tuple[str, dict[str, Any]]:
    """
    Return (output_svg, info dict).
    """
    info: dict[str, Any] = {"used_fallback": False, "reason": "ok", "path_count": 0, "out_len": 0}

    block = extract_first_svg_block(svg or "")
    if block is None:
        info["used_fallback"] = True
        info["reason"] = "no_svg_block"
        info["out_len"] = len(FALLBACK_SVG)
        return FALLBACK_SVG, info

    block = collapse_tags_whitespace(block)
    try:
        root = ET.fromstring(block)
    except ET.ParseError as e:
        info["used_fallback"] = True
        info["reason"] = f"xml_parse_error:{e}"
        info["out_len"] = len(FALLBACK_SVG)
        return FALLBACK_SVG, info

    if local_name(root.tag) != "svg":
        info["used_fallback"] = True
        info["reason"] = "root_not_svg"
        info["out_len"] = len(FALLBACK_SVG)
        return FALLBACK_SVG, info

    ensure_svg_xmlns(root)

    fb = repair_tree(root)
    if fb == "fallback":
        info["used_fallback"] = True
        info["reason"] = "forbidden_tag_image_or_media"
        info["out_len"] = len(FALLBACK_SVG)
        return FALLBACK_SVG, info

    if not tree_tags_valid(root):
        info["used_fallback"] = True
        info["reason"] = "disallowed_tags_remain"
        info["out_len"] = len(FALLBACK_SVG)
        return FALLBACK_SVG, info

    n_paths = count_paths(root)
    info["path_count"] = n_paths
    if n_paths > MAX_PATHS:
        info["used_fallback"] = True
        info["reason"] = f"path_count_{n_paths}"
        info["out_len"] = len(FALLBACK_SVG)
        return FALLBACK_SVG, info

    try:
        out = serialize_svg(root)
    except Exception as e:
        info["used_fallback"] = True
        info["reason"] = f"serialize_error:{e}"
        info["out_len"] = len(FALLBACK_SVG)
        return FALLBACK_SVG, info

    info["out_len"] = len(out)
    if len(out) > MAX_SVG_LEN:
        info["used_fallback"] = True
        info["reason"] = f"length_{len(out)}"
        info["out_len"] = len(FALLBACK_SVG)
        return FALLBACK_SVG, info

    return out, info


def post_process_csv(in_csv: Path, out_csv: Path, report_json: Path | None) -> list[RowResult]:
    results: list[RowResult] = []
    with in_csv.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None or "id" not in reader.fieldnames or "svg" not in reader.fieldnames:
            raise ValueError("CSV must have id, svg columns")
        rows = list(reader)

    with out_csv.open("w", newline="", encoding="utf-8") as f_out:
        w = csv.DictWriter(f_out, fieldnames=["id", "svg"])
        w.writeheader()
        for row in rows:
            rid = row.get("id", "")
            out_svg, info = process_svg(row.get("svg", "") or "")
            used_fb = bool(info.get("used_fallback"))
            w.writerow({"id": rid, "svg": out_svg})
            results.append(
                RowResult(
                    id=str(rid),
                    ok=not used_fb,
                    used_fallback=used_fb,
                    reason=str(info.get("reason", "")),
                    path_count=int(info.get("path_count", 0)),
                    out_len=int(info.get("out_len", len(out_svg))),
                )
            )

    if report_json is not None:
        summary = {
            "rows": len(results),
            "fallbacks": sum(1 for r in results if r.used_fallback),
            "by_reason": {},
        }
        for r in results:
            if r.used_fallback:
                summary["by_reason"][r.reason] = summary["by_reason"].get(r.reason, 0) + 1
        summary["details"] = [asdict(r) for r in results]
        report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return results


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Hard-validate + safely repair submission CSV SVG column.")
    ap.add_argument("--in_csv", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--report_json", type=Path, default=None, help="Optional full row-level report.")
    args = ap.parse_args()

    res = post_process_csv(args.in_csv, args.out_csv, args.report_json)
    n_fb = sum(1 for r in res if r.used_fallback)
    print(f"Wrote {args.out_csv}  rows={len(res)}  fallbacks={n_fb}")
    if args.report_json:
        print(f"Report: {args.report_json}")


if __name__ == "__main__":
    main()
