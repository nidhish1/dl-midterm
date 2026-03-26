import csv
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

ALLOWED_TAGS = {
    "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon",
    "defs", "use", "symbol", "clipPath", "mask", "linearGradient", "radialGradient",
    "stop", "text", "tspan", "title", "desc", "style", "pattern", "marker", "filter",
}

# Allowed attributes per tag
COMMON_ATTRS = {
    "fill", "stroke", "stroke-width",
    "fill-opacity", "stroke-opacity", "opacity",
    "transform", "stroke-linecap", "stroke-linejoin",
    "stroke-dasharray"
}

TAG_ATTRS = {
    "svg": {"width", "height", "viewBox", "xmlns"},
    "g": set(),
    "path": {"d"},
    "rect": {"x", "y", "width", "height", "rx", "ry"},
    "circle": {"cx", "cy", "r"},
    "ellipse": {"cx", "cy", "rx", "ry"},
    "line": {"x1", "y1", "x2", "y2"},
    "polyline": {"points"},
    "polygon": {"points"},
    "defs": set(),
    "use": {"href", "x", "y", "width", "height"},
    "symbol": {"viewBox", "width", "height", "preserveAspectRatio"},
    "clipPath": {"clipPathUnits", "transform"},
    "mask": {"x", "y", "width", "height", "maskUnits", "maskContentUnits"},
    "linearGradient": {
        "x1", "y1", "x2", "y2", "gradientUnits", "spreadMethod",
        "href", "gradientTransform",
    },
    "radialGradient": {
        "cx", "cy", "r", "fx", "fy", "gradientUnits", "href", "gradientTransform",
    },
    "stop": {"offset", "stop-color", "stop-opacity"},
    "text": {
        "x", "y", "dx", "dy", "rotate", "textLength", "lengthAdjust",
        "text-anchor", "dominant-baseline", "font-family", "font-size", "font-weight",
    },
    "tspan": {
        "x", "y", "dx", "dy", "rotate", "textLength", "lengthAdjust",
        "text-anchor", "dominant-baseline", "font-family", "font-size", "font-weight",
    },
    "title": set(),
    "desc": set(),
    "style": {"type", "media"},
    "pattern": {
        "x", "y", "width", "height", "patternUnits", "patternContentUnits",
        "patternTransform", "viewBox", "href", "preserveAspectRatio",
    },
    "marker": {
        "refX", "refY", "markerWidth", "markerHeight", "orient",
        "markerUnits", "viewBox", "preserveAspectRatio",
    },
    "filter": {
        "x", "y", "width", "height", "filterUnits", "primitiveUnits",
        "color-interpolation-filters",
    },
}

NUMERIC_ATTRS = {
    "x", "y", "x1", "y1", "x2", "y2",
    "cx", "cy", "r", "rx", "ry",
    "fx", "fy",
    "width", "height", "stroke-width",
    "fill-opacity", "stroke-opacity", "opacity",
    "stop-opacity",
    "offset",
    "font-size",
    "refX", "refY", "markerWidth", "markerHeight",
}

STYLE_WHITELIST = {
    "fill", "stroke", "stroke-width",
    "fill-opacity", "stroke-opacity", "opacity",
    "stroke-linecap", "stroke-linejoin", "stroke-dasharray"
}


def local_name(tag: str) -> str:
    """Strip XML namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def trim_to_svg(svg_text: str) -> str:
    """Trim junk before <svg and after </svg>."""
    start = svg_text.find("<svg")
    end = svg_text.rfind("</svg>")
    if start == -1 or end == -1:
        raise ValueError("Could not find complete <svg>...</svg> block.")
    return svg_text[start:end + len("</svg>")].strip()


def parse_style(style_value: str) -> Dict[str, str]:
    """
    Convert style='fill:red; stroke:black' into dict.
    Only keep whitelisted properties.
    """
    out = {}
    for part in style_value.split(";"):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k in STYLE_WHITELIST and v:
            out[k] = v
    return out


def format_number(x: float, decimals: int = 2) -> str:
    """Round and remove trailing zeros."""
    if math.isfinite(x):
        x = round(x, decimals)
        if abs(x) < 1e-12:
            x = 0.0
        s = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(x)


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def normalize_number_string(value: str, decimals: int = 2) -> str:
    try:
        return format_number(float(value), decimals)
    except Exception:
        return value.strip()


def normalize_color(value: str) -> str:
    value = value.strip()

    # Lowercase named colors / keywords like none, currentColor
    if re.fullmatch(r"[a-zA-Z]+", value):
        return value.lower()

    # rgb(255,0,0) -> #ff0000
    m = re.fullmatch(
        r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)",
        value,
        flags=re.IGNORECASE
    )
    if m:
        r, g, b = map(int, m.groups())
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    # Normalize hex color to lowercase
    if re.fullmatch(r"#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?", value):
        return value.lower()

    return value


def normalize_attr(tag: str, key: str, value: str, decimals: int = 2) -> str:
    value = value.strip()

    if key in {"fill", "stroke", "stop-color"}:
        return normalize_color(value)

    if key in NUMERIC_ATTRS:
        return normalize_number_string(value, decimals)

    if key == "d":
        return re.sub(r"\s+", " ", value).strip()

    if key == "points":
        return re.sub(r"\s+", " ", value).strip()

    if key == "viewBox":
        nums = _NUMBER_RE.findall(value)
        if len(nums) == 4:
            nums = [format_number(float(x), decimals) for x in nums]
            return " ".join(nums)
        return "0 0 256 256"

    if key == "transform":
        # Keep transform, just collapse whitespace
        return re.sub(r"\s+", " ", value).strip()

    return re.sub(r"\s+", " ", value).strip()


def merged_attrs_for_tag(tag: str) -> List[str]:
    """
    Return canonical attribute order for this tag.
    """
    order = []

    if tag == "svg":
        order.extend(["xmlns", "width", "height", "viewBox"])

    geometry_order = {
        "path": ["d"],
        "rect": ["x", "y", "width", "height", "rx", "ry"],
        "circle": ["cx", "cy", "r"],
        "ellipse": ["cx", "cy", "rx", "ry"],
        "line": ["x1", "y1", "x2", "y2"],
        "polyline": ["points"],
        "polygon": ["points"],
        "g": [],
        "svg": [],
    }

    order.extend(geometry_order.get(tag, []))
    order.extend([
        "fill", "stroke", "stroke-width",
        "fill-opacity", "stroke-opacity", "opacity",
        "stroke-linecap", "stroke-linejoin", "stroke-dasharray",
        "transform",
    ])
    return order


def sanitize_element(elem: ET.Element, decimals: int = 2) -> Optional[ET.Element]:
    """
    Return cleaned element if allowed.
    Unsupported elements are dropped (children are not hoisted).
    """
    tag = local_name(elem.tag)

    if tag not in ALLOWED_TAGS:
        return None

    new_elem = ET.Element(tag)

    # Parse style attr into explicit attrs
    raw_attrs = {}
    for k, v in elem.attrib.items():
        raw_attrs[local_name(k)] = v

    if "style" in raw_attrs:
        style_attrs = parse_style(raw_attrs["style"])
        for k, v in style_attrs.items():
            raw_attrs.setdefault(k, v)

    allowed_attrs = TAG_ATTRS[tag] | COMMON_ATTRS

    cleaned_attrs = {}
    for key, value in raw_attrs.items():
        if key in allowed_attrs:
            cleaned_attrs[key] = normalize_attr(tag, key, value, decimals)

    if tag == "svg":
        if "xmlns" not in cleaned_attrs:
            cleaned_attrs["xmlns"] = "http://www.w3.org/2000/svg"

    # Apply attrs in fixed order
    for key in merged_attrs_for_tag(tag):
        if key in cleaned_attrs and cleaned_attrs[key] != "":
            new_elem.set(key, cleaned_attrs[key])

    return new_elem


def sanitize_children(src_elem: ET.Element, dst_elem: ET.Element, decimals: int = 2) -> None:
    """
    Recursively sanitize children.
    Unsupported child tags are skipped entirely (no hoisting).
    """
    for child in list(src_elem):
        cleaned = sanitize_element(child, decimals=decimals)

        if cleaned is not None:
            sanitize_children(child, cleaned, decimals=decimals)

            if local_name(cleaned.tag) == "g" and len(cleaned) == 0 and len(cleaned.attrib) == 0:
                continue

            dst_elem.append(cleaned)
        else:
            continue


def count_paths(elem: ET.Element) -> int:
    total = 1 if local_name(elem.tag) == "path" else 0
    for child in list(elem):
        total += count_paths(child)
    return total


def canonicalize_svg(svg_text: str, decimals: int = 2, strict: bool = True) -> str:
    """
    Convert raw SVG text into canonical SVG text.
    """
    svg_text = trim_to_svg(svg_text)
    root = ET.fromstring(svg_text)

    cleaned_root = sanitize_element(root, decimals=decimals)
    if cleaned_root is None or local_name(cleaned_root.tag) != "svg":
        raise ValueError("Root is not a supported <svg> element.")

    sanitize_children(root, cleaned_root, decimals=decimals)

    # Path count constraint
    if strict:
        n_paths = count_paths(cleaned_root)
        if n_paths > 256:
            raise ValueError(f"SVG has {n_paths} <path> elements; exceeds limit of 256.")

    out = ET.tostring(cleaned_root, encoding="unicode", method="xml")

    # Final whitespace cleanup
    out = re.sub(r">\s+<", "><", out).strip()

    if strict and len(out) > 16000:
        raise ValueError(f"Canonical SVG length {len(out)} exceeds 16000 chars.")

    if not out.startswith("<svg"):
        raise ValueError("Canonical SVG does not start with <svg.")

    return out


def canonicalize_row_svg(svg_text: str, decimals: int = 2) -> str:
    """Prefer strict rules; fall back to non-strict if path count or length fails."""
    try:
        return canonicalize_svg(svg_text, decimals=decimals, strict=True)
    except ValueError:
        return canonicalize_svg(svg_text, decimals=decimals, strict=False)


def process_train_csv(
    input_path: Path,
    output_path: Path,
    decimals: int = 2,
) -> None:
    with input_path.open(newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None or "svg" not in reader.fieldnames:
            raise ValueError("Input CSV must include an 'svg' column.")

        fieldnames = list(reader.fieldnames)
        rows_out = []
        errors = 0
        for row in reader:
            raw = row.get("svg", "") or ""
            try:
                row["svg"] = canonicalize_row_svg(raw, decimals=decimals)
            except Exception:
                row["svg"] = raw
                errors += 1
            rows_out.append(row)

    with output_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    if errors:
        print(f"Warning: {errors} row(s) left unchanged due to errors.", file=sys.stderr)


if __name__ == "__main__":
    # Script lives in repo root; dataset CSVs are under dl-spring-2026-svg-generation/
    here = Path(__file__).resolve().parent
    data_dir = here / "dl-spring-2026-svg-generation"
    default_in = data_dir / "train.csv"
    default_out = data_dir / "train_canonicsed.csv"

    in_csv = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default_in
    out_csv = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else default_out

    process_train_csv(in_csv, out_csv)
    print(f"Wrote {out_csv}")
