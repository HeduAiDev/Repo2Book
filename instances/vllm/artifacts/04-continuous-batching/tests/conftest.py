"""Pytest config: make `implementation` importable when running pytest from
this directory (`pytest tests/`) or from repo root (`pytest instances/.../tests`).

Mirrors the pattern in `_legacy/test_scheduler.py` but lives in conftest.py so
test files can `from implementation.scheduler import Scheduler` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CHAPTER_ROOT = _HERE.parent  # artifacts/04-continuous-batching/
if str(_CHAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_CHAPTER_ROOT))
