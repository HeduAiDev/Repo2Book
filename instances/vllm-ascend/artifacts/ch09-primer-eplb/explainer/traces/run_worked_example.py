#!/usr/bin/env python3
"""Driver for ch34 worked example (2 rank x 8 expert, 1 layer, 2 redundant slots).

Runs the chapter's primer reference implementation
(instances/vllm-ascend/artifacts/ch34-primer-eplb/implementation/eplb.py) — the
host-runnable NumPy port of vllm_ascend's DefaultEplb global load-balancing
planner — and prints every scalar the explainer tables cite, so lint_explainer
can trace each number back to a real run.

Run:
  python3 run_worked_example.py > worked_example_trace.txt
(host-runnable, pure NumPy — no NPU/CANN/vLLM needed).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.abspath(os.path.join(HERE, "..", "..", "implementation"))
sys.path.insert(0, IMPL)
import eplb  # noqa: E402


def rule(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


# ---------------------------------------------------------------------------
# Scenario: 1 MoE layer, 2 NPUs (cards), 5 physical slots per card, 8 logical
# experts (ids 0..7). The naive starting placement co-locates the two genuinely
# hot experts (3: heat 60, 4: heat 55) on card0, and wastes both redundant
# copies on already-cold experts (1 and 5).
# ---------------------------------------------------------------------------
placement = [[[3, 4, 1, 2, 1], [0, 5, 6, 7, 5]]]
workload = [[[60, 55, 10, 10, 0], [10, 10, 10, 10, 0]]]
hot = [10, 10, 10, 60, 55, 10, 10, 10]  # folded per-logical-expert heat

placement_arr = np.array(placement)
workload_arr = np.array(workload)

rule("INPUT")
print("num_layers=1  num_cards=2  slots_per_card=5  num_logical_experts=8")
print("card0 slots =", placement[0][0], " raw heat =", workload[0][0],
      " sum =", int(sum(workload[0][0])))
print("card1 slots =", placement[0][1], " raw heat =", workload[0][1],
      " sum =", int(sum(workload[0][1])))

# --- fold physical heat -> logical experts -----------------------------------
first_layer = placement_arr[0]
expert_ids, counts = np.unique(first_layer, return_counts=True)
num_original = len(expert_ids)
num_redundant = eplb.count_redundant_slots(first_layer)
rule("STEP 0  fold heat + read redundancy budget")
print("unique expert ids =", list(expert_ids.tolist()), " copy counts =", list(counts.tolist()))
print("num_original_experts =", num_original, " num_redundant =", num_redundant)
folded = eplb.fold_physical_heat_to_experts(placement_arr, workload_arr, num_original)
print("folded per-expert heat =", [float(x) for x in folded[0]])
total_heat = float(sum(folded[0]))
print("total heat =", total_heat, " ideal per-card lower bound = total/num_cards =", total_heat / 2)
heat_before = eplb.max_heat_per_layer(workload_arr)
print("max_heat_per_layer BEFORE (heaviest card raw load) =", heat_before[0])

# --- mechanism: redundant-experts-replication --------------------------------
rule("MECH redundant-experts-replication  (replicate_hot_experts greedy rounds)")
weights = [(e, float(folded[0][e])) for e in range(num_original)]
n = len(weights)
replicas_of = [[] for _ in range(n)]
w = list(weights)
next_id = n
for r in range(num_redundant):
    order = np.argsort([x for _, x in w], kind="stable")[::-1]
    w = [w[i] for i in order]
    top_id, top_w = w[0]
    k = len(replicas_of[top_id])
    raw_total = top_w * (k + 1)
    replicas_of[top_id].append(next_id)
    next_id += 1
    new_avg = raw_total / (k + 2)
    remaining = num_redundant - (r + 1)
    print(f"round {r + 1}: pick expert {top_id}  avg_before={top_w}  k {k}->{k + 1}"
          f"  avg_after={new_avg}  remaining_budget {num_redundant - r}->{remaining}")
    w[0] = (top_id, new_avg)
final_weights = w
print("replicas_of =", replicas_of)
fw_by_id = {e: val for e, val in final_weights}
print("final diluted per-expert avg =", {e: fw_by_id[e] for e in range(n)})

# --- mechanism: greedy-balanced-packing --------------------------------------
rule("MECH greedy-balanced-packing  (pack_replicated_experts LPT)")
total_slots = num_original + num_redundant
items_per_box = total_slots // 2
remaining_items = total_slots % 2
print(f"total_slots={total_slots}  items_per_box={items_per_box}  remainder={remaining_items}")
boxes = [[], []]
box_weights = [0.0, 0.0]
box_counts = [0, 0]
idx = 0
for e in range(num_original):
    for _ in replicas_of[e]:
        cw = fw_by_id[e]
        boxes[idx].append(e)
        box_weights[idx] += cw
        box_counts[idx] += 1
        print(f"seed: replica of expert {e} (w={cw}) -> card{idx}  card_load={box_weights[idx]}")
        idx += 1
order = np.argsort([x for _, x in final_weights], kind="stable")[::-1]
ordered = [final_weights[i] for i in order]
for item_id, weight in ordered:
    mb = -1
    for i in range(2):
        if item_id in boxes[i]:
            continue
        if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
            if mb == -1 or box_weights[i] < box_weights[mb]:
                mb = i
    cand = [round(x, 2) for x in box_weights]
    boxes[mb].append(item_id)
    box_weights[mb] += weight
    box_counts[mb] += 1
    if box_counts[mb] == items_per_box + 1 and remaining_items > 0:
        remaining_items -= 1
    print(f"place expert {item_id} (w={weight}): cand card loads {cand} -> card{mb}"
          f"  new card_load={box_weights[mb]}")
print("boxes =", boxes)
print("box_weights =", box_weights)
heat_after = max(box_weights)
print("max_heat_per_layer AFTER (heaviest card) =", heat_after)

# --- mechanism: constraint-local-exchange ------------------------------------
rule("MECH constraint-local-exchange  (local_exchange keep-in-place)")
new_layer = eplb.local_exchange(placement_arr[0], boxes)
for card_id in range(2):
    current = list(placement_arr[0][card_id])
    newset = boxes[card_id]
    kept = [e for e in newset if e in current]
    arrivals = [e for e in newset if e not in current]
    print(f"card{card_id}: old slots {current}  new set {newset}"
          f"  kept(in place)={kept}  new arrivals={arrivals}"
          f"  -> final slots {new_layer[card_id]}  D2D copies={len(arrivals)}")

# --- mechanism: rebalance-orchestration-change-gate --------------------------
rule("MECH rebalance-orchestration-change-gate  (end-to-end + 0.95 gate)")
change, priority, new_placement = eplb.rebalance_experts(placement, workload)
total_before = float(heat_before[0])
total_after = float(heat_after)
threshold = round(0.95 * total_before, 2)
ratio = round(total_after / total_before, 3)
print("total_heat_before =", total_before, " total_heat_after =", total_after)
print("improvement_ratio (after/before) =", ratio)
print("gate threshold 0.95 * before =", threshold)
print("gate fires (after < threshold)?", total_after < 0.95 * total_before, " => change =", change)
print("per_layer_priority =", priority)
print("new_placement =", new_placement)

# --- metric: par = max/mean --------------------------------------------------
rule("METRIC par = heaviest-card-load / mean-card-load")
loads_before, par_before = eplb.compute_imbalance(placement[0], hot)
loads_after, par_after = eplb.compute_imbalance(new_placement[0], hot)
print("BEFORE card loads =", [float(x) for x in loads_before],
      " mean =", float(loads_before.mean()), " par_before =", round(par_before, 3))
print("AFTER  card loads =", [float(x) for x in loads_after],
      " mean =", float(loads_after.mean()), " par_after =", round(par_after, 3))
print("par reduced from", round(par_before, 3), "to", round(par_after, 3),
      "(1.0 = perfect balance)")
