"""M2 driver: bl.alloc on an explicit address space.

Allocates fp32 [64,128] on UB and fp16 [16,32] on L1, records what the reduced
alloc() pushes onto the builder, plus the observable fields of the returned
bl.buffer. Output -> m2_alloc.json.
"""
import json
from pathlib import Path

from _env import load_env


def snap(buf):
    return {
        "type_name": buf.type.name,
        "shape": list(buf.shape),
        "dtype": str(buf.dtype),
        "space": repr(buf.space),
    }


def run():
    env, FakeBuilder = load_env()
    tl, bl, al = env.tl, env.bl, env.al
    rows = []

    # Case A: fp32 [64,128] on UB
    b = FakeBuilder()
    buf = bl.alloc(tl.float32, [64, 128], _address_space=al.ascend_address_space.UB, _builder=b)
    rows.append({"case": "fp32 [64,128] @UB", "buffer": snap(buf), "calls": [list(c) if not isinstance(c, tuple) else _flat(c) for c in b.calls]})

    # Case B: fp16 [16,32] on L1
    b2 = FakeBuilder()
    buf2 = bl.alloc(tl.float16, [16, 32], _address_space=al.ascend_address_space.L1, _builder=b2)
    rows.append({"case": "fp16 [16,32] @L1", "buffer": snap(buf2), "calls": [_flat(c) for c in b2.calls]})

    # Case C: is_mem_unique=True adds an extra annotation
    b3 = FakeBuilder()
    buf3 = bl.alloc(tl.float32, [8, 8], _address_space=al.ascend_address_space.UB, is_mem_unique=True, _builder=b3)
    rows.append({"case": "fp32 [8,8] @UB is_mem_unique", "buffer": snap(buf3), "calls": [_flat(c) for c in b3.calls]})

    # Case D: int1 rejected
    b4 = FakeBuilder()
    try:
        bl.alloc(tl.int1, [8], _address_space=al.ascend_address_space.UB, _builder=b4)
        err = None
    except TypeError as e:
        err = str(e)
    rows.append({"case": "int1 [8] @UB (rejected)", "error": err, "calls": [_flat(c) for c in b4.calls]})

    return {"mechanism": "M2", "cases": rows}


def _flat(c):
    return [repr(x) if not isinstance(x, (str, int, float, bool, type(None), tuple, list)) else x for x in c]


if __name__ == "__main__":
    out = run()
    p = Path(__file__).parent / "m2_alloc.json"
    p.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
