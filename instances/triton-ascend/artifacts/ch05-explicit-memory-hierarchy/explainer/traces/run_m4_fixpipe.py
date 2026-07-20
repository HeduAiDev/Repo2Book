"""M4 driver: al.fixpipe L0C(tensor) -> UB(buffer) happy path + structural gates.

Records the six-arg create_fixpipe op that a valid NZ2ND fixpipe builds, plus
the structural rejections (src not tensor / dst not buffer / dst not UB / chip
gate). Alignment arithmetic is exercised separately in run_m7_align.py.
Output -> m4_fixpipe.json.
"""
import json
from pathlib import Path

from _env import load_env


def run():
    env, FakeBuilder = load_env()
    tl, bl, al = env.tl, env.bl, env.al
    AS = al.ascend_address_space
    rows = []

    def src_tensor(shape, dtype=None):
        return tl.tensor("l0c-handle", tl.block_type(dtype or tl.float32, list(shape)))

    def dst_buf(b, shape, space=None, dtype=None):
        return bl.alloc(dtype or tl.float32, list(shape), _address_space=space or AS.UB, _builder=b)

    # Happy path: fp32 [64,128] NZ2ND
    b = FakeBuilder()
    src = src_tensor([64, 128])
    dst = dst_buf(b, [64, 128])
    al.fixpipe(src, dst, dma_mode=al.FixpipeDMAMode.NZ2ND, _builder=b)
    fx = [_flat(c) for c in b.calls if c[0] == "create_fixpipe"]
    rows.append({
        "case": "fp32 [64,128] NZ2ND (valid)",
        "src_handle": src.handle, "dst_handle": dst.handle,
        "create_fixpipe": fx,
    })

    # Happy path: fp16 [64,128] NZ2ND
    b2 = FakeBuilder()
    src2 = src_tensor([64, 128], tl.float16)
    dst2 = dst_buf(b2, [64, 128], dtype=tl.float16)
    al.fixpipe(src2, dst2, dma_mode=al.FixpipeDMAMode.NZ2ND, _builder=b2)
    rows.append({
        "case": "fp16 [64,128] NZ2ND (valid)",
        "create_fixpipe": [_flat(c) for c in b2.calls if c[0] == "create_fixpipe"],
    })

    # Structural rejections
    def reject(desc, thunk):
        try:
            thunk()
            return {"case": desc, "outcome": "NO-RAISE"}
        except Exception as e:
            return {"case": desc, "outcome": type(e).__name__, "error": str(e)}

    b3 = FakeBuilder()
    d3 = dst_buf(b3, [64, 128])
    rows.append(reject("src not tensor (buffer as src)", lambda: al.fixpipe(d3, d3, _builder=b3)))

    b4 = FakeBuilder()
    s4 = src_tensor([64, 128])
    rows.append(reject("dst not buffer (tensor as dst)", lambda: al.fixpipe(s4, s4, _builder=b4)))

    b5 = FakeBuilder()
    s5 = src_tensor([64, 128])
    d5 = dst_buf(b5, [64, 128], space=AS.L1)
    rows.append(reject("dst on L1 (not UB)", lambda: al.fixpipe(s5, d5, _builder=b5)))

    b6 = FakeBuilder(is_910_95=False)
    s6 = src_tensor([64, 128])
    d6 = dst_buf(b6, [64, 128])
    rows.append(reject("chip gate is_910_95=False", lambda: al.fixpipe(s6, d6, _builder=b6)))

    return {"mechanism": "M4", "cases": rows}


def _flat(c):
    return [repr(x) if not isinstance(x, (str, int, float, bool, type(None), tuple, list)) else x for x in c]


if __name__ == "__main__":
    out = run()
    p = Path(__file__).parent / "m4_fixpipe.json"
    p.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
