#!/usr/bin/env python3
"""Host-side cross-check for the argmax worked example (mechanism reduce-with-indices-argmax).

This is NOT the Triton runtime — it is a faithful pure-Python re-enactment of the
combine logic that `_argmax_combine` (python/triton/language/standard.py:L140) drives inside
the reduce region, so the explainer's per-step table numbers are checked by an actual run
rather than by hand only. The real path compiles `_argmax_combine`'s AST into a create_reduce
region (see ch08); on GPU the pairwise combine runs as a parallel tree. Here we mirror one
valid pairwise schedule and print every combine step.
"""


def argmax_combine(v1, i1, v2, i2, tie_break_left=True):
    # mirrors standard.py _argmax_combine (L140-L150): core.where(gt, ...) on (value,index) pairs
    tie = (v1 == v2 and i1 < i2) if tie_break_left else False
    gt = (v1 > v2) or tie
    v_ret = v1 if gt else v2
    i_ret = i1 if gt else i2
    return gt, v_ret, i_ret


def trace(name, values):
    print(f"== scenario {name}: input={values} ==")
    pairs = [(v, i) for i, v in enumerate(values)]
    # one valid tree schedule: combine adjacent pairs, then combine the partials
    step = 0
    # level 1: (0,1) and (2,3)
    (v1, i1), (v2, i2) = pairs[0], pairs[1]
    gt, v, i = argmax_combine(v1, i1, v2, i2)
    step += 1
    print(f"  step {step}: ({v1},{i1}) x ({v2},{i2})  gt={gt}  -> ({v},{i})")
    left = (v, i)
    (v1, i1), (v2, i2) = pairs[2], pairs[3]
    gt, v, i = argmax_combine(v1, i1, v2, i2)
    step += 1
    print(f"  step {step}: ({v1},{i1}) x ({v2},{i2})  gt={gt}  -> ({v},{i})")
    right = (v, i)
    # level 2: combine the two partials
    (v1, i1), (v2, i2) = left, right
    gt, v, i = argmax_combine(v1, i1, v2, i2)
    step += 1
    print(f"  step {step}: ({v1},{i1}) x ({v2},{i2})  gt={gt}  -> ({v},{i})")
    print(f"  RESULT: value={v} index={i}  (argmax={i})")
    print()


if __name__ == "__main__":
    trace("distinct", [3, 1, 4, 1])
    # tie scenario collapses to 2 elements to show tie_break_left picks the smaller index
    print("== scenario tie: input=[4, 4] ==")
    gt, v, i = argmax_combine(4, 0, 4, 1)
    print(f"  step 1: (4,0) x (4,1)  gt={gt} (tie: 4==4 and 0<1)  -> ({v},{i})")
    print(f"  RESULT: value={v} index={i}  (left index wins)")
