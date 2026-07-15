"""Driver — m01 SIMT hierarchy (figure backing). BLOCK_SIZE=1024 logical
threads in a block are cut into 32 warps of 32 lanes each. Also a small
BLOCK=64 case (2 warps) for a legible figure. Runs
implementation/simt_hierarchy.py."""
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))
import simt_hierarchy as sh  # noqa: E402

out = {"warp_size": sh.WARP_SIZE, "cases": []}
for block in (64, 1024):
    warps = sh.num_warps_per_block(block)
    parts = sh.partition_into_warps(block)
    out["cases"].append({
        "threads_per_block": block,
        "num_warps": warps,
        "warp0_lanes": [parts[0][0], parts[0][-1]],
        "warp1_lanes": [parts[1][0], parts[1][-1]],
    })
print(json.dumps(out, indent=2))
for c in out["cases"]:
    print(f"BLOCK={c['threads_per_block']} -> {c['num_warps']} warps of "
          f"{sh.WARP_SIZE} lanes; warp0 lanes {c['warp0_lanes'][0]}..{c['warp0_lanes'][1]}, "
          f"warp1 lanes {c['warp1_lanes'][0]}..{c['warp1_lanes'][1]}")
