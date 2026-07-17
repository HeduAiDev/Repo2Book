#!/usr/bin/env python3
"""Hand-derivation aid (NOT the pinned C++; trace_source=manual).

Re-implements the *deterministic* core of triton v3.2.0
lib/Analysis/Membar.cpp::update + include/triton/Analysis/Membar.h::BlockInfo
in pure Python to check the ch26 membar worked example. Mirrors:
  - BlockInfo.isIntersected (Membar.h:L38-45): RAW = syncWrite vs other.read,
      WAR = syncRead vs other.write, WAW = syncWrite vs other.write; RAR is
      never queried (no write -> no sync).
  - BlockInfo.sync (): clears both interval sets after a barrier.
  - BlockInfo.join(): unions cur into running blockInfo.
  - update non-scratch branch (Membar.cpp:L101-155,L178-184): classify op's
      shared-memory effect (Write/Read) on the buffer's allocated interval;
      if isIntersected(blockInfo, cur) -> insert barrier before op, then sync.
All ops touch one explicit buffer b with allocated interval [0,512).
"""

def intersects(a, b):
    return a[0] < b[1] and b[0] < a[1]

INTERVAL = (0, 512)  # allocation->getAllocatedInterval(b)

class BlockInfo:
    def __init__(self):
        self.write = {}  # interval -> set(op)
        self.read = {}

    def is_intersected(self, other):
        def hit(m1, m2):
            for i1 in m1:
                for i2 in m2:
                    if intersects(i1, i2):
                        return True
            return False
        raw = hit(self.write, other.read)   # RAW
        war = hit(self.read, other.write)   # WAR
        waw = hit(self.write, other.write)  # WAW
        return raw or war or waw, raw, war, waw

    def sync(self):
        self.write.clear()
        self.read.clear()

    def join(self, other):
        for i, s in other.write.items():
            self.write.setdefault(i, set()).update(s)
        for i, s in other.read.items():
            self.read.setdefault(i, set()).update(s)

# (op name, effect) on buffer b. effect: 'W' write, 'R' read.
ops = [
    ("op1 local_store b", "W"),
    ("op2 local_load  b", "R"),
    ("op3 local_store b", "W"),
    ("op4 local_load  b", "R"),
    ("op5 local_load  b", "R"),
]

block = BlockInfo()
barriers = []
print("=== ch26 membar insertion derivation (v3.2.0 Membar.cpp) ===")
print(f"buffer b allocated interval = {INTERVAL}\n")
for name, eff in ops:
    cur = BlockInfo()
    if eff == "W":
        cur.write[INTERVAL] = {name}
    else:
        cur.read[INTERVAL] = {name}
    hit, raw, war, waw = block.is_intersected(cur)
    kind = "RAW" if raw else "WAR" if war else "WAW" if waw else "-"
    inserted = ""
    if hit:
        barriers.append(name)
        inserted = f"  --> insert gpu.barrier BEFORE {name} ({kind}); then sync()"
        block.sync()
    block.join(cur)
    bw = {k: sorted(v) for k, v in block.write.items()}
    br = {k: sorted(v) for k, v in block.read.items()}
    print(f"{name} [{eff}] : hit={hit} ({kind}){inserted}")
    print(f"    blockInfo after: write={bw} read={br}")

print(f"\ntotal barriers inserted = {len(barriers)} at: {barriers}")
print("note: op5 is Read-after-Read (RAR) vs op4 -> no write in either side "
      "-> isIntersected false -> NO barrier (RAR never needs sync).")
