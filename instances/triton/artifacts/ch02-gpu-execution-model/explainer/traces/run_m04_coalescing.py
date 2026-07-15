"""Driver — m04 coalescing. One warp (32 lanes), fp32 (4 bytes each).
Contiguous offsets (vector-add's pattern) -> 128 bytes -> 1 transaction.
Strided/gather (stride=32 elements) -> 32 scattered segments -> 32
transactions -> 1/32 the effective bandwidth. Runs
implementation/coalescing.py."""
import json
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))
import coalescing as co  # noqa: E402

WARP = co.WARP_SIZE          # 32
FP32 = 4                     # bytes
TXN = co.TRANSACTION_BYTES   # 128
lanes = list(range(WARP))

# vector-add: offsets = block_start + arange(0, BLOCK); take one warp of it.
block_start = 0
contig = co.warp_offsets_bytes(block_start, lanes, FP32)
n_contig = co.count_transactions(contig)

# counter-example: strided gather, stride = 32 elements
stride = 32
strided = co.strided_offsets_bytes(0, lanes, stride, FP32)
n_strided = co.count_transactions(strided)

out = {
    "warp_size": WARP, "element_bytes": FP32, "transaction_bytes": TXN,
    "contiguous": {
        "bytes_touched": WARP * FP32,
        "first_addr": contig[0], "last_addr": contig[-1],
        "transactions": n_contig,
    },
    "strided": {
        "stride_elements": stride,
        "first_addr": strided[0], "last_addr": strided[-1],
        "transactions": n_strided,
    },
    "bandwidth_ratio": n_contig / n_strided,  # effective BW of strided vs contig
    "slowdown_factor": n_strided // n_contig,
}
print(json.dumps(out, indent=2))
print(f"contiguous: {WARP} lanes x {FP32}B = {WARP*FP32}B, addrs "
      f"{contig[0]}..{contig[-1]} -> {n_contig} transaction")
print(f"strided(stride={stride}): addrs {strided[0]}..{strided[-1]} -> "
      f"{n_strided} transactions -> effective bandwidth 1/{n_strided}")
