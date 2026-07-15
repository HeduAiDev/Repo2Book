"""Driver — m05 occupancy. Ampere-class SM (64K regs, 2048 threads = 64 warps).
Register sweep 32/64/128 regs/thread -> 100%/50%/25% occupancy. Plus one
shared-memory-bound launch to show the min-of-two-gates rule. Runs
implementation/occupancy.py."""
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))
import occupancy as oc  # noqa: E402

max_warps = oc.max_warps_per_sm()  # 64
out = {
    "max_registers_per_sm": oc.MAX_REGISTERS_PER_SM,
    "max_threads_per_sm": oc.MAX_THREADS_PER_SM,
    "max_warps_per_sm": max_warps,
    "register_sweep": [],
    "shared_mem_bound_case": {},
}

for regs in (32, 64, 128):
    occ = oc.occupancy_by_registers(regs)
    resident_threads = int(occ * oc.MAX_THREADS_PER_SM)
    out["register_sweep"].append({
        "regs_per_thread": regs,
        "resident_threads": resident_threads,
        "resident_warps": resident_threads // oc.WARP_SIZE,
        "occupancy_fraction": occ,
        "occupancy_percent": int(round(occ * 100)),
    })

# shared-memory-bound: 32 regs/thread (reg gate = 100%), but 48 KiB smem/block
# with 256 threads/block -> only 3 blocks fit by smem -> 768 threads -> 37.5%.
regs = 32
threads_per_block = 256
smem = 48 * 1024
res = oc.occupancy(regs, threads_per_block, smem)
out["shared_mem_bound_case"] = {
    "regs_per_thread": regs,
    "threads_per_block": threads_per_block,
    "shared_mem_per_block_bytes": smem,
    "occupancy_by_registers": res["occupancy_by_registers"],
    "occupancy_by_shared_memory": res["occupancy_by_shared_memory"],
    "occupancy": res["occupancy"],
    "occupancy_percent": int(round(res["occupancy"] * 100)),
    "limiting_factor": res["limiting_factor"],
}

print(json.dumps(out, indent=2))
for r in out["register_sweep"]:
    print(f"{r['regs_per_thread']} regs/thread -> {r['resident_threads']} threads "
          f"= {r['resident_warps']} warps = {r['occupancy_percent']}% occupancy")
s = out["shared_mem_bound_case"]
print(f"smem-bound: reg-gate={int(s['occupancy_by_registers']*100)}% "
      f"smem-gate={int(s['occupancy_by_shared_memory']*100)}% "
      f"-> min = {s['occupancy_percent']}% (limited by {s['limiting_factor']})")
