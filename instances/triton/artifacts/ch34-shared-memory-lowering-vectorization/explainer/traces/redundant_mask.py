#!/usr/bin/env python3
"""m6 replica: redundantDataMask — in a broadcast layout only the unique holder writes.
Source: LoadStoreOpToLLVM.cpp:L52-L62.
Per dim: if shape[dim] >= shapePerCTATile[dim]: no mask (no replication).
Else mask &= icmp_slt(threadDim*sizePerThread[dim], shape[dim]),
  threadDim = warpId[dim]*threadsPerWarp[dim] + laneId[dim].
Pure integer/boolean arithmetic; no CUDA needed.
1-D blocked layout example.
"""

shape = 64
sizePerThread = 1
threadsPerWarp = 32
warpsPerCTA = 4
shapePerCTATile = sizePerThread * threadsPerWarp * warpsPerCTA  # = 128
replication = shapePerCTATile // shape

print(f"shape={shape} sizePerThread={sizePerThread} threadsPerWarp={threadsPerWarp} "
      f"warpsPerCTA={warpsPerCTA} shapePerCTATile={shapePerCTATile} "
      f"replication={replication}x (shape<tile -> replicated)")

print("\nwarp | lane | threadDim | threadDim*spt | < shape? | action")
writers = 0
covered = set()
for warp in range(warpsPerCTA):
    for lane in range(threadsPerWarp):
        threadDim = warp * threadsPerWarp + lane
        idx = threadDim * sizePerThread
        write = idx < shape
        if write:
            writers += 1
            for k in range(sizePerThread):
                covered.add(idx + k)
        if lane in (0, 31):  # print warp boundaries only
            print(f"{warp:4d} | {lane:4d} | {threadDim:9d} | {idx:13d} | "
                  f"{str(write):8s} | {'WRITE' if write else 'masked'}")

total_threads = warpsPerCTA * threadsPerWarp
print(f"\nwriters={writers} of {total_threads} threads "
      f"({total_threads - writers} masked) -> store traffic x{writers/total_threads:.2f}")
print(f"covered global indices = {len(covered)} of {shape} "
      f"(each written exactly once: {len(covered)==shape and writers*sizePerThread==shape})")
