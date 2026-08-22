"""m10 figure-numbers driver — FlatLogprobs vs nested list[dict[int, Logprob]]:
object-count/GC account (logprobs.py:L30-L135). Empirically counts GC-tracked
container objects (gc.get_objects delta around the build): nested = L dicts +
L*K Logprob dataclass instances + 1 list; flat = 6 primitive lists (start/end
+ 4 data lists), elements are untracked primitives. Also demonstrates the
read-side price: __getitem__ rebuilds a fresh dict per access, and the slice
face rebuilds a shifted FlatLogprobs (what DELTA tail-slicing consumes).
"""
import gc
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import logprobs_lane as lane  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def R(x, nd=4):
    return round(float(x), nd)


def build_nested(L, K):
    c = lane.create_sample_logprobs(flat_logprobs=False)
    for i in range(L):
        # sampled == top1 -> K unique entries per position (dedup eats one)
        lane.append_logprobs_for_next_position(
            c, [10 + i, 20 + i], [-0.1, -0.2], ["a", "b"], 1, K
        )
    return c


def build_flat(L, K):
    c = lane.create_sample_logprobs(flat_logprobs=True)
    for i in range(L):
        lane.append_logprobs_for_next_position(
            c, [10 + i, 20 + i], [-0.1, -0.2], ["a", "b"], 1, K
        )
    return c


def tracked_delta(builder, *args):
    gc.collect()
    before = len(gc.get_objects())
    obj = builder(*args)
    gc.collect()
    after = len(gc.get_objects())
    return obj, after - before


def main():
    L, K = 100, 2  # 100 positions, k+1=3 columns, dedup -> 2 entries/pos

    nested, n_delta = tracked_delta(build_nested, L, K)
    flat, f_delta = tracked_delta(build_flat, L, K)

    nested_dicts = sum(1 for pos in nested if isinstance(pos, dict))
    nested_logprobs = sum(len(pos) for pos in nested)
    flat_inner_lists = [k for k in vars(flat) if isinstance(getattr(flat, k), list)]

    # scale probe: 10x positions
    nested2, n2_delta = tracked_delta(build_nested, 10 * L, K)
    flat2, f2_delta = tracked_delta(build_flat, 10 * L, K)

    # read side: __getitem__ rebuilds a fresh dict each access
    e1 = flat[42]
    e2 = flat[42]
    # slice face (DELTA tail-slicing consumes this)
    tail = flat[90:]

    trace = {
        "mechanism": "m10",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "params": {
            "positions": L,
            "k": K,
            "entries_per_position_after_dedup": 2,
            "counting": "gc-tracked container objects alive after build (dicts, dataclass instances, lists; CPython does not GC-track plain ints/floats)",
        },
        "nested": {
            "positions": len(nested),
            "dict_objects": nested_dicts,
            "logprob_objects": nested_logprobs,
            "container_list_objects": 1,
            "tracked_delta_measured": n_delta,
            "arithmetic_total": nested_dicts + nested_logprobs + 1,
        },
        "flat": {
            "positions": len(flat),
            "inner_list_names": flat_inner_lists,
            "inner_list_count": len(flat_inner_lists),
            "tracked_delta_measured": f_delta,
            "token_ids_len": len(flat.token_ids),
            "start_indices_head": [int(v) for v in flat.start_indices[:3]],
            "end_indices_head": [int(v) for v in flat.end_indices[:3]],
        },
        "scale_10x": {
            "positions": 10 * L,
            "nested_tracked_delta": n2_delta,
            "nested_dict_objects": 10 * L,
            "nested_logprob_objects": 2 * 10 * L,
            "flat_tracked_delta": f2_delta,
            "ratio_nested_over_flat": R(n2_delta / f2_delta, 1),
        },
        "read_side": {
            "getitem_position": 42,
            "getitem_returns_dict": isinstance(e1, dict),
            "getitem_rebuilds_each_call": e1 == e2 and e1 is not e2,
            "getitem_entries": len(e1),
            "slice_type": type(tail).__name__,
            "slice_positions": len(tail),
            "slice_start_indices": [int(v) for v in tail.start_indices[:3]],
            "note": "slice __getitem__ rebuilds a shifted FlatLogprobs — the DELTA tail slice logprobs[-len(token_ids):] rides this face",
        },
    }

    p = Path(__file__).resolve().parent / "m10_flat_vs_nested.json"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
