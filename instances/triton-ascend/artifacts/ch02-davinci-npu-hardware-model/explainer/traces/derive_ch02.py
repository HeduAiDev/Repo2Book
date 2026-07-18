#!/usr/bin/env python3
"""ch02 primer 数值推演驱动脚本（host 纯算术，无 NPU/CANN）。

本章 skip_impl、无精简版：所有数字要么是源码常量（标 file:Lxxx），要么是
对齐/tiling/grid 规则对这些常量的**纯算术推导**。本脚本把这些推导逐条算出来，
产出 derive_ch02.json，让 explainer.json 逐轮表里的每个派生数字可复现。
trace_source 仍记为 manual（未在真机运行 kernel），本脚本只做算术复核。
"""
import json

out = {}

# ---- 源码常量（逐字复核，见 file:Lxxx）----
UB_BYTES = 192 * 1024                     # programming_guide.md:180,272  192 KB
UB_BITS = 1_572_864                       # programming_guide.md:272      1,572,864 bits
OVF_REQ_BITS = 3_072_256                  # programming_guide.md:268      overflow requires
OVF_AVAIL_BITS = 1_572_864                # programming_guide.md:268      available
F32 = 4                                   # f32 = 4 bytes（标准）
ALIGN_VV = 32                             # programming_guide.md:111      32B 末轴对齐(VV)
ALIGN_CV = 512                            # programming_guide.md:111      512B 末轴对齐(CV)

out["constants"] = {
    "UB_bytes": UB_BYTES, "UB_KB": UB_BYTES // 1024, "UB_bits": UB_BITS,
    "UB_bits_check_192KB": 192 * 1024 * 8,
    "ovf_requires_bits": OVF_REQ_BITS, "ovf_available_bits": OVF_AVAIL_BITS,
    "ovf_ratio": round(OVF_REQ_BITS / OVF_AVAIL_BITS, 3),
}

# ---- double-buffer 减半 ----
usable_ub_bytes = UB_BYTES // 2           # programming_guide.md:180 doublebuffer halves
out["double_buffer"] = {"full_KB": UB_BYTES // 1024, "usable_KB": usable_ub_bytes // 1024}

# ---- 三级 tiling: triton_better_kernel (architecture_difference.md:107-110) ----
ncore = 32
xblock = 32768
xblock_sub = 8192
xnumel = ncore * xblock
# intra-core loop: range(0, XBLOCK, XBLOCK_SUB)
sub_offsets = list(range(0, xblock, xblock_sub))
num_sub_blocks = -(-xblock // xblock_sub)  # cdiv
sub_ub_bytes = xblock_sub * F32
whole_xblock_bytes = xblock * F32
out["tiling_3level"] = {
    "ncore": ncore, "xblock": xblock, "xblock_sub": xblock_sub,
    "xnumel_total": xnumel,
    "sub_offsets": sub_offsets, "num_sub_blocks": num_sub_blocks,
    "sub_block_bytes": sub_ub_bytes, "sub_block_KB": sub_ub_bytes // 1024,
    "sub_fits_usable_UB": sub_ub_bytes <= usable_ub_bytes,
    "whole_xblock_bytes": whole_xblock_bytes, "whole_xblock_KB": whole_xblock_bytes // 1024,
    "whole_xblock_overflows_usable_UB": whole_xblock_bytes > usable_ub_bytes,
}

# ---- masked_fill_kernel 核内 sub-block 循环 (migrate_from_gpu.md:280-282) ----
BLOCK_SIZE = 32768
BLOCK_SIZE_SUB = 8192
mf_num_sub = -(-BLOCK_SIZE // BLOCK_SIZE_SUB)  # tl.cdiv
out["masked_fill"] = {
    "BLOCK_SIZE": BLOCK_SIZE, "BLOCK_SIZE_SUB": BLOCK_SIZE_SUB,
    "num_sub_blocks": mf_num_sub,
    "sub_offsets": [i * BLOCK_SIZE_SUB for i in range(mf_num_sub)],
}

# ---- 显式搬运: add_kernel 单核逐块 (programming_guide.md:92-106) ----
x = [10, 20, 30, 40, 50, 60, 70, 80]
y = [1, 2, 3, 4, 5, 6, 7, 8]
n = len(x)
BS = 4
num_blocks = -(-n // BS)  # cdiv
blocks = []
for b in range(num_blocks):
    offs = list(range(b * BS, min((b + 1) * BS, n)))
    xs = [x[i] for i in offs]
    ys = [y[i] for i in offs]
    outs = [a + c for a, c in zip(xs, ys)]
    blocks.append({"block": b, "offsets": offs, "x": xs, "y": ys, "out": outs})
out["explicit_move"] = {"n": n, "BLOCK_SIZE": BS, "num_blocks": num_blocks, "blocks": blocks}

# ---- grid 强绑物理核: step 循环 range(pid, NUM_BLOCKS, NUM_CORE) ----
NUM_CORE = 2
NUM_BLOCKS = 5
per_core = {}
for pid in range(NUM_CORE):
    per_core[pid] = list(range(pid, NUM_BLOCKS, NUM_CORE))
covered = sorted(b for v in per_core.values() for b in v)
max_per_core = max(len(v) for v in per_core.values())
out["grid_step"] = {
    "NUM_CORE": NUM_CORE, "NUM_BLOCKS": NUM_BLOCKS,
    "per_core": {str(k): v for k, v in per_core.items()},
    "covered_blocks": covered,
    "covers_all_once": covered == list(range(NUM_BLOCKS)),
    "max_blocks_per_core": max_per_core,
}

# ---- 末轴对齐膨胀 ----
def pad_up(nbytes, align):
    return -(-nbytes // align) * align

align_rows = []
for shape, tail, mode in [((2048, 1), 1, "VV"), ((2048, 3), 3, "VV"),
                          ((2048, 8), 8, "VV"), ((2048, 1), 1, "CV")]:
    align = ALIGN_VV if mode == "VV" else ALIGN_CV
    tail_bytes = tail * F32
    padded_bytes = pad_up(tail_bytes, align)
    padded_elems = padded_bytes // F32
    align_rows.append({
        "shape": list(shape), "tail_elems": tail, "mode": mode, "align_B": align,
        "tail_bytes": tail_bytes, "aligned": tail_bytes % align == 0,
        "padded_elems": padded_elems, "padded_bytes": padded_bytes,
        "inflation": round(padded_elems / tail, 3),
    })
out["tail_align"] = {"f32_bytes": F32, "align_VV_elems": ALIGN_VV // F32,
                     "align_CV_elems": ALIGN_CV // F32, "rows": align_rows}

print(json.dumps(out, ensure_ascii=False, indent=2))
