# Tests for ch34's paper-faithful reference implementation of the EPLB planning
# algorithm (arXiv:2412.19437 §3.4 + deepseek-ai/EPLB README "Global Load
# Balancing", as landed in vllm_ascend/eplb/core/policy/policy_default_eplb.py).
#
# TDD note: these assert the *observable behavior* the dossier records for the
# landing code (redundant-copy dilution recurrence, LPT packing invariants, the
# 0.95 change gate, the max/mean imbalance metric) — including a fully
# hand-verified 2-rank x 8-expert worked example matching the chapter's planned
# "numeric walkthrough" section.
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementation"))

import eplb  # noqa: E402


# ---------------------------------------------------------------------------
# fold_physical_heat_to_experts (add_redundant analogue)
# ---------------------------------------------------------------------------
def test_fold_sums_duplicate_physical_slots_back_to_one_logical_expert():
    # layer0, 2 npus, 3 slots each; expert 1 has two physical copies (npu0 slot0,
    # npu1 slot2) whose raw heat must sum back into one logical-expert entry.
    placement = np.array([[[1, 2, 0], [3, 4, 1]]])
    workload = np.array([[[7.0, 5.0, 9.0], [2.0, 1.0, 3.0]]])
    folded = eplb.fold_physical_heat_to_experts(placement, workload, num_original_experts=5)
    # expert1 = 7 (npu0 slot0) + 3 (npu1 slot2) = 10
    assert folded[0].tolist() == [9.0, 10.0, 5.0, 2.0, 1.0]


# ---------------------------------------------------------------------------
# max_heat_per_layer / count_redundant_slots
# ---------------------------------------------------------------------------
def test_max_heat_per_layer_takes_the_heaviest_card_each_layer():
    workload = np.array([[[10.0, 5.0], [3.0, 3.0]], [[1.0, 1.0], [9.0, 9.0]]])
    assert eplb.max_heat_per_layer(workload) == [15.0, 18.0]


def test_count_redundant_slots_reads_back_extra_copies():
    # ids 0,1,2,3 each once, id 1 duplicated once -> 1 redundant slot total
    ids = np.array([[0, 1, 2], [3, 1, 4]])
    assert eplb.count_redundant_slots(ids) == 1


# ---------------------------------------------------------------------------
# replicate_hot_experts: dilution recurrence new_avg = old_avg * (k+1) / (k+2)
# ---------------------------------------------------------------------------
def test_replicate_hot_experts_always_targets_current_max_and_dilutes():
    # expert 2 (weight 100) totally dominates -> both redundant slots go to it.
    weights = [(0, 10.0), (1, 10.0), (2, 100.0), (3, 10.0)]
    replicas_of, final = eplb.replicate_hot_experts(weights, num_redundant=2)
    assert replicas_of[2] == [4, 5]  # two fresh replica ids, in order
    assert replicas_of[0] == replicas_of[1] == replicas_of[3] == []
    final_by_id = dict(final)
    # 1st split: 100/2 = 50 ; 2nd split (still the max at 50 > all others'10): 50*2/3 = 33.33...
    assert final_by_id[2] == pytest.approx(100.0 / 3.0)
    assert final_by_id[0] == 10.0
    assert final_by_id[1] == 10.0
    assert final_by_id[3] == 10.0


def test_replicate_hot_experts_switches_target_once_diluted_below_runner_up():
    # expert A=60 splits once -> 30, which then falls below expert B=55, so the
    # second redundant slot must go to B instead of A again.
    weights = [(0, 10.0), (1, 10.0), (2, 60.0), (3, 55.0), (4, 10.0), (5, 10.0), (6, 10.0), (7, 10.0)]
    replicas_of, final = eplb.replicate_hot_experts(weights, num_redundant=2)
    assert len(replicas_of[2]) == 1
    assert len(replicas_of[3]) == 1
    final_by_id = dict(final)
    assert final_by_id[2] == pytest.approx(30.0)
    assert final_by_id[3] == pytest.approx(27.5)


# ---------------------------------------------------------------------------
# pack_replicated_experts / balanced_pack_redundancy: LPT-style greedy packing
# ---------------------------------------------------------------------------
def test_pack_respects_capacity_and_never_double_books_same_expert_on_one_card():
    weights = [(0, 40.0), (1, 30.0), (2, 20.0), (3, 10.0)]
    replicas_of = [[4], [], [], []]  # expert 0 gets one redundant copy
    result, boxes = eplb.pack_replicated_experts(weights, replicas_of, num_cards=2)
    assert sum(len(b) for b in boxes) == 5  # 4 experts + 1 redundant slot
    for box in boxes:
        assert len(box) == len(set(box)), "same expert placed twice on one card"
    counts = [len(b) for b in boxes]
    assert max(counts) - min(counts) <= 1  # 5 items over 2 boxes -> 3/2 split


def test_balanced_pack_redundancy_is_replicate_then_pack_composed():
    weights = [(0, 10.0), (1, 10.0), (2, 100.0), (3, 10.0)]
    result, boxes = eplb.balanced_pack_redundancy(weights, num_cards=2, num_redundant=2)
    # expert 2 dominates every redundant slot -> it must appear on both cards
    assert all(2 in box for box in boxes)


# ---------------------------------------------------------------------------
# local_exchange: keep whatever already sits on a card, only shuffle new arrivals
# ---------------------------------------------------------------------------
def test_local_exchange_keeps_overlap_in_place_and_fills_gaps_with_newcomers():
    old_card = [5, 1, 9]
    new_card = [1, 7, 9]  # 1 and 9 already resident; 7 is a genuine newcomer
    fixed = eplb.local_exchange([old_card], [new_card])[0]
    # 1 and 9 must land in whatever slot they already occupied
    assert fixed[old_card.index(1)] == 1
    assert fixed[old_card.index(9)] == 9
    assert set(fixed) == {1, 7, 9}


# ---------------------------------------------------------------------------
# compute_imbalance: par = heaviest-card-load / mean-card-load
# ---------------------------------------------------------------------------
def test_compute_imbalance_perfect_balance_is_one():
    placement = [[0, 1], [2, 3]]
    hotness = np.array([10.0, 10.0, 10.0, 10.0])
    loads, par = eplb.compute_imbalance(placement, hotness)
    assert loads.tolist() == [20.0, 20.0]
    assert par == pytest.approx(1.0)


def test_compute_imbalance_splits_a_replicated_experts_hotness_across_its_cards():
    # expert 0 has two physical copies, one per card -> its hotness (100) is
    # halved between the two cards it occupies.
    placement = [[0, 1], [0, 2]]
    hotness = np.array([100.0, 20.0, 20.0])
    loads, par = eplb.compute_imbalance(placement, hotness)
    assert loads.tolist() == [70.0, 70.0]
    assert par == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rebalance_experts: end-to-end orchestration + the 0.95 change gate
# ---------------------------------------------------------------------------
def test_rebalance_experts_suppresses_small_improvements_below_the_5pct_deadband():
    # Construct a case where packing can only shave off a sliver -> change=0.
    old_placement = [[[0, 1], [2, 3]]]
    workload = [[[100.0, 99.0], [98.0, 100.0]]]
    change, priority, new_placement = eplb.rebalance_experts(old_placement, workload)
    assert change == 0


def test_worked_example_2rank_8expert_matches_hand_computation():
    """The chapter's numeric walkthrough: 2 NPUs, 8 routed experts, one redundant
    slot per NPU (2 total) -- but the *naive* starting placement wastes both
    redundant copies on already-cold experts (1 and 5) while co-locating the two
    genuinely hot experts (3, weight 60; 4, weight 55) on the same card (card0).

    card0 = [3, 4, 1, 2, 1(dup)]  raw heat = 60+55+10+10+0   = 135
    card1 = [0, 5, 6, 7, 5(dup)]  raw heat = 10+10+10+10+0   =  40
    total = 175, ideal mean/card = 87.5, par_before = 135/87.5 = 1.5428...

    rebalance_experts must: read back num_redundant=2 from the placement, hand
    the 2 redundant slots to the *actual* hot experts (3 and 4, diluting them to
    30 and 27.5), and pack to an exact 87.5 / 87.5 split -> par_after = 1.0,
    with a >5% total-heat improvement (135 -> 87.5) so change must fire.
    """
    old_placement = [[
        [3, 4, 1, 2, 1],
        [0, 5, 6, 7, 5],
    ]]
    workload = [[
        [60.0, 55.0, 10.0, 10.0, 0.0],
        [10.0, 10.0, 10.0, 10.0, 0.0],
    ]]
    folded = eplb.fold_physical_heat_to_experts(np.array(old_placement), np.array(workload), num_original_experts=8)
    loads_before, par_before = eplb.compute_imbalance(old_placement[0], folded[0])
    assert loads_before.tolist() == [135.0, 40.0]
    assert par_before == pytest.approx(135.0 / 87.5)

    change, priority, new_placement = eplb.rebalance_experts(old_placement, workload)
    assert change == 1

    new_layer0 = new_placement[0]
    assert len(new_layer0) == 2
    # every original expert id must still be present exactly as many times as
    # its total physical-copy count (8 originals + 2 redundant copies = 10 slots)
    flat = [e for card in new_layer0 for e in card]
    assert len(flat) == 10
    assert sorted(set(flat)) == list(range(8))
    # the two previously-co-located hot experts must now be split one-per-card
    assert sum(card.count(3) for card in new_layer0) == 2  # 1 base + 1 replica
    assert sum(card.count(4) for card in new_layer0) == 2
    for card in new_layer0:
        assert 3 in card and 4 in card  # each card ends up hosting one copy of each

    loads_after, par_after = eplb.compute_imbalance(new_layer0, folded[0])
    assert loads_after.tolist() == pytest.approx([87.5, 87.5])
    assert par_after == pytest.approx(1.0)
    assert par_after < par_before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
