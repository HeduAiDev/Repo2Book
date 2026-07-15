"""Driver — m06 register spill. A kernel wanting 40 regs/thread under a 32-reg
budget spills 8; a 32-reg kernel spills 0. Spilled accesses fall from the
register tier (~1 cycle) to the global/DRAM tier (~400-800 cycles). Runs
implementation/register_spill.py (which reads the latency ladder from
implementation/memory_hierarchy.py)."""
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))
import register_spill as rs  # noqa: E402
import memory_hierarchy as mh  # noqa: E402

BUDGET = 32
out = {"register_budget_per_thread": BUDGET,
       "reg_latency_cycles": mh.LATENCY_CYCLES["register"],
       "global_latency_cycles": mh.LATENCY_CYCLES["global"],
       "cases": []}

for needed in (32, 40, 64):
    r = rs.effective_access_cycles(needed, BUDGET)
    out["cases"].append({
        "regs_needed": needed,
        "spilled_registers": r["spilled_registers"],
        "resident_registers": r["resident_registers"],
        "resident_access_cycles": r["resident_access_cycles"],
        "spilled_access_cycles": r["spilled_access_cycles"],
        "spill_latency_multiplier": r["spill_latency_multiplier"],
    })

print(json.dumps(out, indent=2))
for c in out["cases"]:
    print(f"needs {c['regs_needed']} regs (budget {BUDGET}): "
          f"spill={c['spilled_registers']} resident={c['resident_registers']} "
          f"spilled_access={c['spilled_access_cycles']} cycles "
          f"(x{c['spill_latency_multiplier']:g} vs register ~1 cycle)")
