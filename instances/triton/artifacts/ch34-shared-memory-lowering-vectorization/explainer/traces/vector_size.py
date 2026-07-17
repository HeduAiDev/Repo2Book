#!/usr/bin/env python3
"""m5 replica: getVectorSize = min(128/pointeeBitWidth, getPtrContiguity), then
clamped by mask alignment.  Source: LoadStoreOpToLLVM.cpp:L142 (min) and L191 (mask clamp).
Pure integer arithmetic from source constants (128-bit HW ceiling); no CUDA needed.
"""

HW_BITS = 128  # "The maximum vector size is 128 bits on NVIDIA GPUs." (L141)

scenarios = [
    # name, dtype, bitwidth, contiguity, maskAlign(None=no mask)
    ("A full-width fp16",   "fp16", 16, 16, None),
    ("B contiguity-limited", "fp16", 16,  4, None),
    ("C mask-clamped",      "fp16", 16, 16,    2),
    ("D scalar+remark",     "fp16", 16, 16,    1),
    ("E fp32 full",         "fp32", 32,  8, None),
    ("F i8 full",           "i8",    8, 32, None),
]

print("scenario | dtype | bitwidth | 128/bw | contig | maskAlign | vec | bytes | bits | remark?")
for name, dt, bw, contig, mask in scenarios:
    hw = HW_BITS // bw
    vec = min(hw, contig)
    vecOrig = vec
    if mask is not None:
        vec = min(vec, mask)
    numElems = max(contig, 2)  # numElems>1 assumed so the remark branch is testable
    remark = (vec == 1 and numElems > 1)
    by = vec * bw // 8
    bits = vec * bw
    print(f"{name:20s} | {dt:5s} | {bw:8d} | {hw:6d} | {contig:6d} | "
          f"{str(mask):9s} | {vec:3d} | {by:5d} | {bits:4d} | {remark}")
