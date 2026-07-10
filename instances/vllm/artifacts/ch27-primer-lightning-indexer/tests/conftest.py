import pathlib
import sys

IMPL_DIR = pathlib.Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))
