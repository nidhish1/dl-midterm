#!/usr/bin/env python3
"""CLI entry: run from repo root, or invoke as `python3 -m post_processing.hard_validate`."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from post_processing.hard_validate import main

if __name__ == "__main__":
    main()
