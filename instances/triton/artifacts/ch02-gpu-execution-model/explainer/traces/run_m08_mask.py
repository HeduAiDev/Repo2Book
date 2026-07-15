"""Driver — m08 boundary mask. The ragged tail tile from N=98432, BLOCK=1024:
pid=96, block_start=98304, offsets 98304..99327; every offset >= 98432 is
masked off. Shows 896 lanes masked, 128 kept. Runs
implementation/spmd_tile.py's tile_offsets / boundary_mask."""
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))
import spmd_tile as st  # noqa: E402

N = 98432
BLOCK = 1024
PID = 96

offs = st.tile_offsets(PID, BLOCK)
mask = st.boundary_mask(offs, N)
kept = sum(mask)
masked = len(mask) - kept
# index of first masked lane
first_masked = next(i for i, m in enumerate(mask) if not m)

out = {
    "N": N, "BLOCK": BLOCK, "pid": PID,
    "block_start": PID * BLOCK,
    "first_offset": offs[0], "last_offset": offs[-1],
    "tile_len": len(offs),
    "kept_lanes": kept, "masked_lanes": masked,
    "first_masked_lane_index": first_masked,
    "first_masked_offset": offs[first_masked],
}
print(json.dumps(out, indent=2))
print(f"pid={PID} block_start={out['block_start']} offsets=[{offs[0]}:{offs[-1]+1}]")
print(f"lane {first_masked} is offset {offs[first_masked]} == N={N} -> first masked")
print(f"kept={kept} masked={masked} (of {len(mask)})")
# spot-check a kept lane and a masked lane
print(f"lane 127: offset={offs[127]} mask={mask[127]}")
print(f"lane 128: offset={offs[128]} mask={mask[128]}")
