"""M3 driver: al.copy address-space checks (UB -> UB/L1).

Drives every branch of the reduced copy / copy_from_ub_to_l1 and records the
outcome: either the create_copy_buffer op that got built, or the exact error
message the language layer raised. Output -> m3_copy.json.
"""
import json
from pathlib import Path

from _env import load_env


def run():
    env, FakeBuilder = load_env()
    tl, bl, al = env.tl, env.bl, env.al
    AS = al.ascend_address_space
    rows = []

    def mkbuf(b, space, shape=(16, 16), dtype=None):
        return bl.alloc(dtype or tl.float32, list(shape), _address_space=space, _builder=b)

    def attempt(fn_name, src_space, dst_space, is_910=True, src_shape=(16, 16), dst_shape=(16, 16), use_tensor_src=False):
        b = FakeBuilder(is_910_95=is_910)
        if use_tensor_src:
            src = tl.tensor("ub-handle", tl.block_type(tl.float32, list(src_shape)))
        else:
            src = mkbuf(b, src_space, src_shape)
        dst = mkbuf(b, dst_space, dst_shape)
        fn = getattr(al, fn_name)
        outcome, err = None, None
        try:
            fn(src, dst, _builder=b)
            ops = [_flat(c) for c in b.calls if c[0] == "create_copy_buffer"]
            outcome = "ok" if ops else "no-op"
        except Exception as e:
            outcome = type(e).__name__
            err = str(e)
        return {
            "fn": fn_name,
            "src_space": src_space.name if hasattr(src_space, "name") else str(src_space),
            "dst_space": dst_space.name if hasattr(dst_space, "name") else str(dst_space),
            "is_910_95": is_910,
            "src_shape": list(src_shape), "dst_shape": list(dst_shape),
            "tensor_src": use_tensor_src,
            "outcome": outcome, "error": err,
            "copy_ops": [_flat(c) for c in b.calls if c[0] == "create_copy_buffer"],
        }

    rows.append(attempt("copy", AS.UB, AS.UB))
    rows.append(attempt("copy", AS.UB, AS.L1))
    rows.append(attempt("copy", AS.UB, AS.L0C))       # dst not UB/L1 -> reject
    rows.append(attempt("copy", AS.L1, AS.L1))         # src not UB -> reject
    rows.append(attempt("copy", AS.UB, AS.UB, use_tensor_src=True))  # tensor -> reject
    rows.append(attempt("copy", AS.UB, AS.UB, src_shape=(16, 16), dst_shape=(8, 8)))  # shape mismatch
    rows.append(attempt("copy", AS.UB, AS.L1, is_910=False))  # chip gate
    rows.append(attempt("copy_from_ub_to_l1", AS.UB, AS.L1))   # legacy: ok
    rows.append(attempt("copy_from_ub_to_l1", AS.UB, AS.UB))   # legacy: dst must be L1 -> reject

    return {"mechanism": "M3", "cases": rows}


def _flat(c):
    return [repr(x) if not isinstance(x, (str, int, float, bool, type(None), tuple, list)) else x for x in c]


if __name__ == "__main__":
    out = run()
    p = Path(__file__).parent / "m3_copy.json"
    p.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
