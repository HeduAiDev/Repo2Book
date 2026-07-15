"""Driver — m07 SPMD launch grid + cdiv. The 01-vector-add benchmark size
N=98432, BLOCK=1024: cdiv=97 = 96 full tiles + 1 ragged tail tile of 512
real elements. Also a couple of contrast sizes to show the ceiling behaviour.
Runs implementation/spmd_tile.py's cdiv / spmd_grid."""
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))
import spmd_tile as st  # noqa: E402

BLOCK = 1024
cases = [98432, 98304, 1024, 1025]
out = {"BLOCK": BLOCK, "cases": []}
for N in cases:
    g = st.cdiv(N, BLOCK)
    full = N // BLOCK
    remainder = N - full * BLOCK
    out["cases"].append({
        "N": N,
        "cdiv": g,
        "full_tiles": full,
        "tail_elements": remainder,
        "is_exact": remainder == 0,
    })

print(json.dumps(out, indent=2))
for c in out["cases"]:
    print(f"N={c['N']}  cdiv(N,1024)={c['cdiv']}  full_tiles={c['full_tiles']}  "
          f"tail_elements={c['tail_elements']}  exact={c['is_exact']}")
