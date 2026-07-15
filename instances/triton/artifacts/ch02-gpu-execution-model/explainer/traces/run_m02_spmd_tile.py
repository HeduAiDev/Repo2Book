"""Driver — m02 SPMD tile model. Small case N=256, BLOCK=64 (the exact
example in 01-vector-add.py's L40 comment): 4 programs, each owns a contiguous
64-element tile. Runs the reference model in implementation/spmd_tile.py."""
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))
import spmd_tile as st  # noqa: E402

N = 256
BLOCK = 64

grid = st.spmd_grid(N, BLOCK)
out = {"N": N, "BLOCK": BLOCK, "grid_size": len(grid), "programs": []}
for pid in grid:
    offs = st.tile_offsets(pid, BLOCK)
    out["programs"].append({
        "program_id": pid,
        "block_start": pid * BLOCK,
        "first_offset": offs[0],
        "last_offset": offs[-1],
        "tile_len": len(offs),
    })

print(json.dumps(out, indent=2))
print("\n# per-program tile [first, last] (half-open upper = last+1):")
for p in out["programs"]:
    print(f"pid={p['program_id']}  block_start={p['block_start']}  "
          f"offsets=[{p['first_offset']}:{p['last_offset']+1}]  len={p['tile_len']}")
