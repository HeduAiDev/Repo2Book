"""Pytest config for Ch06 — make `implementation` importable from this dir."""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CHAPTER_ROOT = _HERE.parent
if str(_CHAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAPTER_ROOT))
