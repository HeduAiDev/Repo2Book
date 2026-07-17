#!/usr/bin/env python3
"""Hand-derivation aid (NOT the pinned C++; trace_source=manual).

Re-implements the *deterministic* core of triton v3.2.0
lib/Analysis/Allocation.cpp::computeOffsets in pure Python to check the
arithmetic of the ch26 worked example. Mirrors:
  - buildInterferenceGraph  (Allocation.cpp:L641-L681): edge iff
      liveness(time) ranges intersect AND address ranges intersect.
  - allocate                (Allocation.cpp:L684-L725): first-fit graph
      coloring; color-0 buffers keep offset, others bumped to
      max_{neighbor}(offset+size); sharedMemorySize = max(offset+size).
  - computeOffsets do-while (Allocation.cpp:L560-L565): rerun to fixed point.
Initial offsets = 0 for all (calculateStarts' loose lower bound,
L580-L637; the fixed point corrects any real overlap by bumping).
Interval is half-open [start,end); intersects: s1<e2 and s2<e1
(Allocation.h:L71-73). alignTo omitted: example sizes are 128-aligned so
alignTo is identity for alignment=128.
"""

def intersects(a, b):
    return a[0] < b[1] and b[0] < a[1]

# buffer: name -> (size_bytes, liveness_interval[start,end))
# 3 explicit buffers; A and C are time-disjoint so they can share an offset.
bufs = {
    "A": {"size": 1024, "live": (1, 4)},
    "B": {"size": 512,  "live": (3, 6)},
    "C": {"size": 1024, "live": (5, 8)},
}
order = ["A", "B", "C"]  # coloring / iteration order
for n in bufs:
    bufs[n]["offset"] = 0  # calculateStarts loose init

def build_interference():
    edges = {n: set() for n in bufs}
    for x in order:
        for y in order:
            if x == y:
                continue
            bx, by = bufs[x], bufs[y]
            xaddr = (bx["offset"], bx["offset"] + bx["size"])
            yaddr = (by["offset"], by["offset"] + by["size"])
            if intersects(bx["live"], by["live"]) and intersects(xaddr, yaddr):
                edges[x].add(y)
    return edges

def allocate(edges):
    colors = {n: (0 if n == order[0] else -1) for n in order}
    for x in order:
        used = {colors[y] for y in edges[x] if colors[y] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[x] = c
    shared = 0
    for x in order:
        new_off = 0
        for y in edges[x]:
            new_off = max(new_off, bufs[y]["offset"] + bufs[y]["size"])
        if colors[x] != 0:
            bufs[x]["offset"] = new_off  # aligned (identity here)
        shared = max(shared, bufs[x]["offset"] + bufs[x]["size"])
    return colors, shared

rounds = []
# initial interference (before do-while), from calculateStarts all-zero offsets
edges0 = build_interference()
rounds.append(("initial (all offset 0)",
               {n: bufs[n]["offset"] for n in order},
               {n: sorted(edges0[n]) for n in order}))
edges = edges0
i = 0
while any(edges[n] for n in order):
    i += 1
    colors, shared = allocate(edges)
    edges = build_interference()
    rounds.append((f"after allocate round {i}",
                   {n: bufs[n]["offset"] for n in order},
                   {n: sorted(edges[n]) for n in order},
                   dict(colors), shared))

print("=== ch26 first-fit allocation derivation (v3.2.0 Allocation.cpp) ===")
for r in rounds:
    if len(r) == 3:
        label, offs, e = r
        print(f"\n[{label}]")
        print(f"  offsets = {offs}")
        print(f"  interference = {e}")
    else:
        label, offs, e, colors, shared = r
        print(f"\n[{label}]")
        print(f"  colors  = {colors}")
        print(f"  offsets = {offs}")
        print(f"  interference (rebuilt) = {e}")
        print(f"  sharedMemorySize = max(offset+size) = {shared} bytes")

sum_sizes = sum(b["size"] for b in bufs.values())
final_shared = max(b["offset"] + b["size"] for b in bufs.values())
print("\n=== summary ===")
for n in order:
    b = bufs[n]
    print(f"  {n}: size={b['size']}B live={b['live']} offset={b['offset']} "
          f"addr=[{b['offset']},{b['offset']+b['size']})")
print(f"  sum of sizes (no reuse) = {sum_sizes} bytes")
print(f"  sharedMemorySize (with reuse) = {final_shared} bytes")
print(f"  saved by reuse (A,C share offset 0) = {sum_sizes - final_shared} bytes")

# --- call-graph two-level max (Allocation.h:L268-275 getSharedMemorySize) ---
print("\n=== call-graph two-level max ===")
bar_shared = 2048  # callee bar's own sharedMemorySize (bytes)
foo_own = final_shared  # foo's own explicit/scratch buffers max = 1536
# virtual buffer for the call = callee sharedMemorySize; placed at the call op,
# live only there. Say the call happens after foo's buffers die -> virtual
# reuses offset 0 -> foo total = max(foo_own, bar_shared).
foo_shared = max(foo_own, bar_shared)
module_shared = max([foo_shared])  # single root foo
print(f"  bar.sharedMemorySize = {bar_shared} B")
print(f"  foo own buffers max  = {foo_own} B")
print(f"  foo virtual buffer (=bar) = {bar_shared} B, reuses offset 0")
print(f"  foo.sharedMemorySize = max({foo_own},{bar_shared}) = {foo_shared} B")
print(f"  module.getSharedMemorySize = max over roots = {module_shared} B")

# occupancy illustration (hardware constant illustrative)
smem_per_sm = 49152  # 48 KiB, illustrative per-SM configurable smem
for need in (module_shared, 8192, 24576):
    blocks = smem_per_sm // need
    print(f"  smem/block={need}B -> blocks/SM by smem = {smem_per_sm}//{need} = {blocks}")
