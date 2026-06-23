import sys
from pathlib import Path

# Make the chapter dir importable as a package root so
# `import implementation.xxx` resolves.
CH_DIR = Path(__file__).resolve().parent.parent
if str(CH_DIR) not in sys.path:
    sys.path.insert(0, str(CH_DIR))
