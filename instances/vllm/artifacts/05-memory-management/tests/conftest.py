"""Pytest configuration: add implementation/ to sys.path for in-function imports."""
import sys
from pathlib import Path

# The kv_cache_manager.py has in-function imports like `from block_pool import ...`
# that work when running the module directly from implementation/ but fail when
# called from tests/. Add the implementation directory to sys.path.
impl_dir = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(impl_dir))
