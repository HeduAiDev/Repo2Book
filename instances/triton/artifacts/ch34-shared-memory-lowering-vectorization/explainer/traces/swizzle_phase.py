#!/usr/bin/env python3
"""m2 replica: swizzle phase(r)=(r//perPhase)%maxPhase compiled to physical offset.

Two truthful paths from the pinned source (triton v3.2.0 @ 9641643):
 (1) ldmatrix loader  — SharedToDotOperandMMAv2.cpp:L162,L181
       phase = urem(udiv(rowInMat, perPhase), maxPhase)
       swizzledCol = xor_(contiguousIndex, phase)
 (2) LinearLayout base — LinearLayoutConversions.cpp:L369 (sharedToLinearLayoutNoLeadingOffset)
       base_col(row=2^i) = (vec * ((row//perPhase) % maxPhase)) % numCols
       physical col-offset for a general row r = XOR of bases for set bits of r
This is pure integer arithmetic extracted from source — no CUDA needed. It reproduces
exactly the constants the compiler emits; the actual PTX/LLVM offsets need make_llir
in-container (see manual_reason).
"""

vec, perPhase, maxPhase = 8, 2, 4
numCols = 64          # element columns of the swizzle unit
contiguousIndex = 3   # a fixed logical matrix-column index for the ldmatrix path


def phase(r):
    return (r // perPhase) % maxPhase


# --- path (2): GF(2) linear bases for power-of-two rows ---
bases = {}
logRows = 3  # numRows = 8 -> log2 = 3 (rows 1,2,4)
for i in range(logRows):
    row = 1 << i
    bases[row] = (vec * ((row // perPhase) % maxPhase)) % numCols
print("LinearLayout row-bases {row: colshift}:", bases)


def ll_colshift(r):
    """XOR-combine the stored bases for the set bits of r (applyLinearLayout)."""
    out = 0
    for i in range(logRows):
        if r & (1 << i):
            out ^= bases[1 << i]
    return out


print("\nrow | floor=r//pP | phase | ldmx col=idx^phase | vec*phase | LL(XOR bases) | linear-match")
for r in range(0, 9):
    ph = phase(r)
    ldmx = contiguousIndex ^ ph
    arith_shift = (vec * ph) % numCols
    ll = ll_colshift(r) if r < 8 else "(period wrap)"
    match = (ll == arith_shift) if r < 8 else "(row8==row0)"
    print(f"{r:3d} | {r//perPhase:11d} | {ph:5d} | {ldmx:18d} | {arith_shift:9d} | {str(ll):13s} | {match}")

print("\nperiod (rows) = perPhase*maxPhase =", perPhase * maxPhase)
print("distinct phases =", maxPhase, "-> up to", maxPhase, "-way bank spread")
# bijection check: for fixed row, col -> col^phase is a permutation of columns
for r in [0, 2, 4, 6]:
    cols = list(range(8))
    swz = [c ^ phase(r) for c in cols]
    print(f"row {r}: cols {cols} -> swizzled {swz}  bijection={sorted(swz)==cols}")
