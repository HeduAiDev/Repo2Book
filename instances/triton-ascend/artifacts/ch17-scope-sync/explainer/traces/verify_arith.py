#!/usr/bin/env python3
"""ch17 arithmetic verification — mirrors the exact integer formulas in
DAGSync.cpp so the manual worked-example tables (m7 CBUF nz shape, m15 flag pool)
can be cross-checked on host. This is NOT a pass dump: it only replays the
integer math from the pinned source (see line refs), which is deterministic and
host-independent.
"""

# --- m7: getBlockElemsFor32BAlign + newCbubAllocShape (DAGSync.cpp L386-420) ---
K_ALIGN = 32  # kAlignBytes, DAGSync.cpp:L387

def elem_bytes(width_bits):            # getElemBytesForAlign, L374-380
    return (width_bits + 7) // 8

def blk_for_32b(width_bits):           # getBlockElemsFor32BAlign, L386-396
    eb = elem_bytes(width_bits)
    if eb >= K_ALIGN:
        return 1
    assert K_ALIGN % eb == 0
    return K_ALIGN // eb

def cbuf_nz_shape(M, N, width_bits):   # newCbubAllocShape, L398-419
    assert M % 16 == 0                 # M % 16 != 0 -> nullopt
    blk = blk_for_32b(width_bits)
    return (N // blk, M // 16, 16, blk)

print("=== m7: CBUF 32B-aligned nz shape, memref [M=32, N=64] ===")
for name, w in [("fp16", 16), ("fp32", 32), ("int8", 8)]:
    blk = blk_for_32b(w)
    shp = cbuf_nz_shape(32, 64, w)
    innermost_bytes = blk * elem_bytes(w)
    total = shp[0] * shp[1] * shp[2] * shp[3]
    print(f"{name}: elem_bytes={elem_bytes(w)} blk=32/{elem_bytes(w)}={blk} "
          f"nz_shape=(N/blk,M/16,16,blk)=({64}//{blk},{32}//16,16,{blk})={shp} "
          f"innermost={blk}*{elem_bytes(w)}B={innermost_bytes}B total={total} (=M*N={32*64})")

# --- m15: flag = syncFlag % 14 (DAGSync.cpp L1116/L1241/L1281) ---
FLAG_POOL = 14
print("\n=== m15: flag pool syncFlag % 14 ===")
for syncFlag in [0, 1, 13, 14, 15]:
    print(f"syncFlag={syncFlag} -> flag={syncFlag % FLAG_POOL}")
