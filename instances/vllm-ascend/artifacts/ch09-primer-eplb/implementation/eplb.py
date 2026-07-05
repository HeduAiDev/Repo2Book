"""Reference implementation — EPLB (Expert-Parallelism Load Balancer) planning
algorithm.

Faithful to:
  - DeepSeek-V3 Technical Report, arXiv:2412.19437, §3.4 "Inference and
    Deployment" (the redundant-experts deployment strategy: duplicate hot
    experts, then rearrange to balance load "without increasing the cross-node
    all-to-all communication overhead").
  - deepseek-ai/EPLB README, "Global Load Balancing" policy (paper pack
    appendix, book/papers/ch34-primer-eplb/paper.md:L91-L95): "replicates the
    experts globally regardless of expert groups, and pack[s] the replicated
    experts to individual GPUs".

This module is a small, dependency-free NumPy port of the *global* policy — the
same policy that ships as vllm_ascend's default EPLB planner (policy_type=1,
vllm_ascend/eplb/core/policy/policy_default_eplb.py, `DefaultEplb`). The book
chapter walks through that landing code with real source excerpts; this module
keeps the identical algorithm (same greedy-replication recurrence, same LPT-style
packing, same 0.95 change gate) flattened out of its class/`DynamicTable`
plumbing so a reader can step through it — and reproduce the chapter's
2-rank x 8-expert worked example — without an NPU.

Not implemented: the *hierarchical* policy (group -> node packing) from the same
README appendix. The landing code does not implement it either (see the chapter
narrative for why vllm_ascend's DefaultEplb only ever exercises the global
policy) — inventing it here would go beyond what either the paper prose or the
landing code actually does.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


def fold_physical_heat_to_experts(placement: np.ndarray, workload: np.ndarray, num_original_experts: int) -> np.ndarray:
    # PAPER: §3.4 (statistics collected during the online deployment)
    """Sum raw per-physical-slot heat back onto its logical expert id, per layer.

    A logical expert may occupy more than one physical slot (its base copy plus
    any redundant replicas); the planner must reason about *logical* expert
    load, so the first step is always to fold the physical measurements back.

    PAPER: §3.4 "high-load experts are detected based on statistics collected
    during the online deployment" — the statistics are collected per physical
    slot, but the deployment decision below operates on the folded, per-expert
    totals.
    """
    layer_num, npu_num, experts_per_npu = workload.shape
    folded = np.zeros((layer_num, num_original_experts))
    for layer_idx in range(layer_num):
        heat_by_expert: dict = defaultdict(float)
        placement_layer = placement[layer_idx]
        workload_layer = workload[layer_idx]
        for npu_idx in range(npu_num):
            for slot_idx in range(experts_per_npu):
                heat_by_expert[int(placement_layer[npu_idx][slot_idx])] += float(workload_layer[npu_idx][slot_idx])
        for expert_id in range(num_original_experts):
            folded[layer_idx][expert_id] = heat_by_expert[expert_id]
    return folded


def max_heat_per_layer(workload: np.ndarray) -> list:
    # PAPER: §3.4 (ensure that each GPU processes approximately the same number of tokens)
    """Per layer, the heaviest card's *raw* physical load — the quantity the
    planner is ultimately trying to push down (wall-clock latency of an MoE
    layer is set by its slowest, i.e. most heavily loaded, card).

    PAPER: §3.4 "we need to ensure that each GPU processes approximately the
    same number of tokens".
    """
    return [float(np.max(np.sum(workload[layer_idx], axis=1))) for layer_idx in range(workload.shape[0])]


def count_redundant_slots(physical_expert_ids: np.ndarray) -> int:
    # PAPER: §3.4 (besides the original experts it hosts, each GPU also hosts redundant experts)
    """Read back how many *extra* physical copies the current placement already
    carries, from however many times each logical expert id repeats.

    PAPER: §3.4 "For each GPU, besides the original 8 experts it hosts, it will
    also host one additional redundant expert" — the redundancy budget is a
    property of the deployment, re-derived from the placement rather than
    invented fresh at every rebalance.
    """
    _, counts = np.unique(np.asarray(physical_expert_ids), return_counts=True)
    return int(np.sum(counts - 1))


def replicate_hot_experts(expert_weights: Sequence, num_redundant: int):
    # PAPER: §3.4 (redundant experts) + Appendix README §Global Load Balancing
    """Greedily hand out `num_redundant` replicas, one at a time, always to
    whichever logical expert currently has the highest *per-replica average*
    load — then re-price that expert's average down as it gains a copy:
    `new_avg = old_avg * (k + 1) / (k + 2)` for an expert that already holds
    `k` extra replicas (so `k=0` on its first split simply halves its load).

    This is a greedy descent on the single largest per-replica load: whichever
    expert is currently the bottleneck gets diluted, and the next-largest
    afterwards is free to become the new target if the dilution drops the
    first expert below it.

    PAPER: §3.4 "we introduce a deployment strategy of redundant experts, which
    duplicates high-load experts and deploys them redundantly"; Appendix
    README §Global Load Balancing ("replicates the experts globally ...
    regardless of expert groups").

    Returns:
        replicas_of: list indexed by expert id -> list of fresh replica ids
            handed to that expert (empty if it received none).
        final_weights: the expert_weights list with each split expert's weight
            entry updated to its final diluted average (same list length as
            the input — one entry per *logical* expert, covering both its
            base copy and any replicas).
    """
    n = len(expert_weights)
    replicas_of: list = [[] for _ in range(n)]
    weights = list(expert_weights)
    next_replica_id = n
    for _ in range(num_redundant):
        order = np.argsort([w for _, w in weights], kind="stable")[::-1]
        weights = [weights[i] for i in order]
        top_id, top_w = weights[0]
        k = len(replicas_of[top_id])
        raw_total = top_w * (k + 1)
        replicas_of[top_id].append(next_replica_id)
        next_replica_id += 1
        new_avg = raw_total / (k + 2)
        weights[0] = (top_id, new_avg)
    return replicas_of, weights


def pack_replicated_experts(final_weights: Sequence, replicas_of: Sequence, num_cards: int):
    # PAPER: Appendix README §Global Load Balancing (pack replicated experts to GPUs)
    """LPT-style greedy bin packing over the (base copy + replicas) of every
    logical expert.

    Step A seeds one card per replica slot (walking experts in id order), each
    carrying that expert's final diluted average weight — this is what spreads
    an expert's redundant copies across *different* cards.
    Step B then places each expert's own base copy into whichever eligible
    card (not already holding that expert, and with room under
    `total_slots // num_cards`, plus one each for the `total_slots % num_cards`
    remainder) currently carries the least total weight — descending by
    weight, so the heaviest items settle first (classic LPT / "longest
    processing time first" makespan-minimization heuristic).

    PAPER: Appendix README §Global Load Balancing ("pack the replicated experts
    to individual GPUs to ensure different GPUs are load-balanced").
    """
    route_expert_num = len(final_weights)
    num_redundant = sum(len(r) for r in replicas_of)
    total_slots = route_expert_num + num_redundant
    items_per_box = total_slots // num_cards
    remaining_items = total_slots % num_cards

    boxes: list = [[] for _ in range(num_cards)]
    boxes_weights: list = [[] for _ in range(num_cards)]
    box_weights = [0.0] * num_cards
    box_counts = [0] * num_cards

    weight_by_expert = {expert_id: w for expert_id, w in final_weights}

    # Step A: seed one card per replica slot, in expert-id order.
    index = 0
    for expert_id in range(route_expert_num):
        for _ in replicas_of[expert_id]:
            cur_weight = weight_by_expert[expert_id]
            boxes[index].append(expert_id)
            boxes_weights[index].append(cur_weight)
            box_weights[index] += cur_weight
            box_counts[index] += 1
            index += 1

    # Step B: place each expert's own base copy, heaviest first.
    order = np.argsort([w for _, w in final_weights], kind="stable")[::-1]
    ordered = [final_weights[i] for i in order]
    for item_id, weight in ordered:
        min_box_index = -1
        for i in range(num_cards):
            if item_id in boxes[i]:
                continue
            if box_counts[i] < items_per_box or (box_counts[i] == items_per_box and remaining_items > 0):
                if min_box_index == -1 or box_weights[i] < box_weights[min_box_index]:
                    min_box_index = i
        boxes[min_box_index].append(item_id)
        boxes_weights[min_box_index].append(weight)
        box_weights[min_box_index] += weight
        box_counts[min_box_index] += 1
        if box_counts[min_box_index] == items_per_box + 1 and remaining_items > 0:
            remaining_items -= 1

    result = [
        {
            "card": i,
            "experts": boxes[i],
            "weights": boxes_weights[i],
            "total_weight": box_weights[i],
        }
        for i in range(num_cards)
    ]
    return result, boxes


def balanced_pack_redundancy(expert_weights: Sequence, num_cards: int, num_redundant: int):
    # PAPER: §3.4 (redundant experts) + Appendix README §Global Load Balancing
    """One layer's full placement decision: replicate, then pack.

    Composed exactly as the landing code's
    `original_compute_balanced_pack_redundancy` interleaves them, just split
    here into the two named steps above for a readable trace.
    """
    replicas_of, final_weights = replicate_hot_experts(expert_weights, num_redundant)
    return pack_replicated_experts(final_weights, replicas_of, num_cards)


def local_exchange(old_placement_layer: Sequence, new_deployment_layer: Sequence):
    # PAPER: §3.4 (rearrange without increasing cross-node all-to-all overhead)
    """Per card, keep whichever newly-assigned experts were *already* resident
    on that card in whatever slot they already occupied; only genuinely new
    arrivals get shuffled into the leftover slots.

    Because the packer above only decides *which card* each expert lands on
    (not which physical slot on that card), this step is what keeps a
    rebalance from moving weights that did not actually need to move.

    PAPER: §3.4 "we carefully rearrange experts among GPUs within a node based
    on the observed loads, striving to balance the load ... without increasing
    the cross-node all-to-all communication overhead."
    """
    result = []
    for card_id in range(len(new_deployment_layer)):
        current = list(old_placement_layer[card_id])
        new_list = list(new_deployment_layer[card_id])
        num = len(new_list)

        used_current = [False] * len(current)
        placed = [None] * num  # indexed by *old* slot position, not new-list order
        leftovers = []

        for expert_id in new_list:
            matched = False
            for j, cur in enumerate(current):
                if not used_current[j] and cur == expert_id:
                    used_current[j] = True
                    placed[j] = expert_id
                    matched = True
                    break
            if not matched:
                leftovers.append(expert_id)

        it = iter(leftovers)
        result.append([p if p is not None else next(it) for p in placed])
    return result


def rebalance_experts(old_placement: Sequence, workload: Sequence):
    # PAPER: §3.4 (adjusted periodically based on observed loads)
    """Top-level planning entry point: fold heat -> replicate + pack per layer
    -> keep-in-place where possible -> rank layers by improvement -> gate.

    PAPER: §3.4 "the high-load experts are detected based on statistics
    collected during the online deployment and are adjusted periodically
    (e.g., every 10 minutes)".

    Returns:
        change: 1 if the rebalance is worth applying (see 5% deadband below),
            else 0.
        priority: per-layer indices, ordered so the layer with the largest
            improvement (smallest after/before ratio) comes first — used to
            sequence a multi-step migration (see chapter 9's D2D transfer
            machinery, which this planner feeds).
        new_placement: the new [layer][card][slot] physical expert ids.
    """
    old_placement_arr = np.array(old_placement)
    workload_arr = np.array(workload)
    layer_num, num_npus, experts_per_npu = workload_arr.shape

    first_layer_ids = old_placement_arr[0]
    expert_ids, _counts = np.unique(first_layer_ids, return_counts=True)
    num_redundant = count_redundant_slots(first_layer_ids)
    num_original_experts = len(expert_ids)

    if num_npus < num_redundant:
        raise ValueError(f"num_npus ({num_npus}) must be >= num_redundant ({num_redundant})")

    folded = fold_physical_heat_to_experts(old_placement_arr, workload_arr, num_original_experts)
    heat_before = max_heat_per_layer(workload_arr)
    total_heat_before = sum(heat_before)

    new_placement = [None] * layer_num
    heat_after = [0.0] * layer_num
    for layer in range(layer_num):
        expert_weights = [(e, float(folded[layer][e])) for e in range(num_original_experts)]
        result, layer_deployment = balanced_pack_redundancy(expert_weights, num_npus, num_redundant)
        new_placement[layer] = layer_deployment
        heat_after[layer] = max(r["total_weight"] for r in result)

    new_placement = [
        local_exchange(old_placement_arr[layer], new_placement[layer]) for layer in range(layer_num)
    ]

    improvement_ratio = [heat_after[layer] / heat_before[layer] for layer in range(layer_num)]
    priority = list(np.argsort(improvement_ratio))
    total_heat_after = sum(heat_after)

    # 5% deadband: rebalancing (D2D copies + consistency risk) has a cost, so a
    # marginal improvement is not worth acting on.
    # PAPER: §3.4 "adjusted periodically" (implying: not on every fluctuation).
    change = 1 if total_heat_after < 0.95 * total_heat_before else 0
    return change, priority, new_placement


def compute_imbalance(placement_layer: Sequence, expert_hotness: np.ndarray):
    # PAPER: §3.4 (ensure each GPU processes approximately the same number of tokens)
    """The measure the planner is proxying for: par = heaviest-card-load /
    mean-card-load. par == 1 is perfect balance; par > 1 says the slowest card
    is running `par` times the average load (i.e. the wall-clock slowdown a
    reader should expect from that placement).

    A replicated expert's hotness is split evenly across however many physical
    cards currently hold a copy of it.

    PAPER: §3.4 (this is the wall-clock-facing readout of "ensure that each GPU
    processes approximately the same number of tokens"; the paper does not
    name the ratio, but it is exactly what the landing code's
    `EplbWorker._compute_imbalance` reports as `par` for observability).
    """
    hotness = np.asarray(expert_hotness, dtype=float)
    flat_ids = np.array([e for card in placement_layer for e in card])
    counts = np.bincount(flat_ids, minlength=len(hotness))
    unit_hotness = np.divide(hotness, counts, out=np.zeros_like(hotness), where=counts != 0)
    card_loads = np.array([sum(unit_hotness[e] for e in card) for card in placement_layer])
    par = float(card_loads.max() / card_loads.mean())
    return card_loads, par
