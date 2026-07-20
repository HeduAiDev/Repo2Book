"""M7 driver: fixpipe alignment arithmetic sweep.

Sweeps (dtype, dma_mode, dual_dst_mode, M, N) chosen so each alignment gate is
isolated (a value that clears every earlier gate but trips exactly the one under
test), and records the real outcome: 'pass' (create_fixpipe built) or the exact
ValueError message. Output -> m7_align.json.
"""
import json
from pathlib import Path

from _env import load_env


def run():
    env, FakeBuilder = load_env()
    tl, bl, al = env.tl, env.bl, env.al
    AS = al.ascend_address_space
    DMA = al.FixpipeDMAMode
    DUAL = al.FixpipeDualDstMode

    dtypes = {"fp32": tl.float32, "fp16": tl.float16, "bf16": tl.bfloat16}

    def one(dtype_name, dma, dual, M, N):
        b = FakeBuilder()
        dt = dtypes[dtype_name]
        src = tl.tensor("l0c-handle", tl.block_type(dt, [M, N]))
        dst = bl.alloc(dt, [M, N], _address_space=AS.UB, _builder=b)
        try:
            al.fixpipe(src, dst, dma_mode=dma, dual_dst_mode=dual, _builder=b)
            built = any(c[0] == "create_fixpipe" for c in b.calls)
            return {"outcome": "pass" if built else "no-op", "error": None}
        except ValueError as e:
            return {"outcome": "ValueError", "error": str(e)}

    sweep = [
        ("fp32", DMA.NZ2ND, DUAL.NO_DUAL, 64, 128, "N=128 %8=0 -> pass"),
        ("fp32", DMA.NZ2ND, DUAL.NO_DUAL, 64, 100, "N=100 %8=4 -> gate #1 (align 8)"),
        ("fp32", DMA.NZ2NZ, DUAL.NO_DUAL, 64, 136, "N=136 %8=0 but %16=8, non-NZ2ND -> gate #2 (align 16)"),
        ("fp32", DMA.NZ2ND, DUAL.COLUMN_SPLIT, 64, 104, "N=104 %8=0 but %32=8, column-split -> gate #3 (align 32)"),
        ("fp32", DMA.NZ2DN, DUAL.NO_DUAL, 100, 128, "N=128 ok, M=100 %8=4, NZ2DN -> gate #4 (first dim 8)"),
        ("fp32", DMA.NZ2DN, DUAL.NO_DUAL, 64, 128, "M=64 %8=0, N=128 ok -> pass"),
        ("fp16", DMA.NZ2ND, DUAL.NO_DUAL, 64, 128, "N=128 %16=0 -> pass"),
        ("fp16", DMA.NZ2ND, DUAL.NO_DUAL, 64, 100, "N=100 %16=4 -> gate #5 (16b align 16)"),
        ("bf16", DMA.NZ2DN, DUAL.NO_DUAL, 100, 128, "N=128 ok, M=100 %16=4, NZ2DN -> gate #6 (16b first dim 16)"),
        ("bf16", DMA.NZ2DN, DUAL.NO_DUAL, 64, 128, "M=64 %16=0, N=128 ok -> pass"),
    ]

    rows = []
    for dtype_name, dma, dual, M, N, note in sweep:
        res = one(dtype_name, dma, dual, M, N)
        rows.append({
            "dtype": dtype_name,
            "dma_mode": dma if isinstance(dma, str) else str(dma),
            "dual_dst": dual if isinstance(dual, str) else str(dual),
            "M": M, "N": N,
            "N%8": N % 8, "N%16": N % 16, "N%32": N % 32, "M%8": M % 8, "M%16": M % 16,
            "note": note,
            "outcome": res["outcome"], "error": res["error"],
        })
    return {"mechanism": "M7", "cases": rows}


if __name__ == "__main__":
    out = run()
    p = Path(__file__).parent / "m7_align.json"
    p.write_text(json.dumps(out, indent=2, default=str, ensure_ascii=False))
    print(json.dumps(out, indent=2, default=str, ensure_ascii=False))
