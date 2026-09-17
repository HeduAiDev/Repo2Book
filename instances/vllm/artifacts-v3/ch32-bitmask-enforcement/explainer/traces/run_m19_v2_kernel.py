# ch32 m19 驱动：V2 落地——StructuredOutputsWorker（GPU 常驻缓冲+copy_stream 双
# H2D+cu_num_logits 前缀和行映射）+ 自写 Triton kernel 分块几何（真 CUDA 执行）。
# grid/BLOCK_SIZE 经 kernel 发射 spy 记录；词表尾谓词用最后一个 token 50256 检验；
# 与 V1 路径（xgr.apply_token_bitmask_inplace）逐元素交叉核验。
import sys

import numpy as np
import torch

from _ch32_common import VOCAB, dump

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))
import vllm.v1.worker.gpu.structured_outputs as so_mod
from vllm.utils.math_utils import cdiv

out = {"env": {"vocab": VOCAB, "cuda_device": torch.cuda.get_device_name(0)}}

# ── kernel 发射 spy（记录 grid 与 BLOCK_SIZE，随后调真 kernel）────────────────
launches = []
real_kernel = so_mod._apply_grammar_bitmask_kernel


class KernelSpy:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            launches.append({
                "grid": list(grid),
                "num_kernel_args": len(args),
                "BLOCK_SIZE": kwargs.get("BLOCK_SIZE"),
                "logits_shape": list(args[0].shape),
                "logits_stride": args[1],
                "bitmask_shape": list(args[3].shape),
                "bitmask_stride": args[4],
                "vocab_size_arg": args[5],
            })
            return real_kernel[grid](*args, **kwargs)
        return launch


so_mod._apply_grammar_bitmask_kernel = KernelSpy()

from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker  # noqa: E402

MAX_NUM_LOGITS = 8
worker = StructuredOutputsWorker(MAX_NUM_LOGITS, VOCAB, torch.device("cuda"))


class FakeInputBatch:
    def __init__(self, req_ids, cu_num_logits):
        self.req_ids = list(req_ids)
        self.cu_num_logits_np = np.asarray(cu_num_logits, dtype=np.int32)


# 4 个请求的 logits 行区间：r0=[0,2) r1=[2,5) r2=[5,7) r3=[7,8)——cu 前缀和 [0,2,5,7,8]
ib = FakeInputBatch(["r0", "r1", "r2", "r3"], [0, 2, 5, 7, 8])
grammar_req_ids = ["r1", "r3"]  # 只有 r1、r3 受语法约束（调度序）

# 紧凑掩码 4 行（r1 的 3 行 + r3 的 1 行），每行允许一个可区分 token：
# 行 0→100、行 1→200、行 2→300（r1 三个 spec 位+bonus？r1 区间 3 行=2 spec+bonus）、
# 行 3→50256（词表最后一个 token——检验尾块谓词）


def make_row(allowed):
    row = np.zeros(cdiv(VOCAB, 32), dtype=np.uint32)
    for t in allowed:
        row[t // 32] |= np.uint32(1 << (t % 32))
    return row.view(np.int32)


compact = np.stack([make_row([100]), make_row([200]), make_row([300]),
                    make_row([50256])])

logits = torch.zeros((8, VOCAB), device="cuda", dtype=torch.float32)
worker.apply_grammar_bitmask(logits, ib, grammar_req_ids, compact)
torch.cuda.synchronize()

allowed_per_row = []
for r in range(8):
    keep = [t for t in range(VOCAB) if logits[r, t].item() != float("-inf")]
    allowed_per_row.append(keep)

# 行映射推演（与源码 L55-L63 同循环）：mapping = r1 区间 [2,3,4] + r3 区间 [7]
expected_mapping = list(range(2, 5)) + [7]
out["apply"] = {
    "cu_num_logits": [0, 2, 5, 7, 8],
    "grammar_req_ids": grammar_req_ids,
    "worker_req_ids": ib.req_ids,
    "mapping_expected": expected_mapping,
    "num_masks": int(compact.shape[0]),
    "assert_num_masks_eq_mapping": int(compact.shape[0]) == len(expected_mapping),
    "logits_rows_constrained": [2, 3, 4, 7],
    "logits_rows_untouched": [0, 1, 5, 6],
    "allowed_by_constrained_row": {"2": allowed_per_row[2], "3": allowed_per_row[3],
                                   "4": allowed_per_row[4], "7": allowed_per_row[7]},
    "untouched_rows_allowed_count": {str(r): len(allowed_per_row[r])
                                     for r in (0, 1, 5, 6)},
    "untouched_rows_all_zero": all(
        logits[r].abs().sum().item() == 0.0 for r in (0, 1, 5, 6)),
    "kernel_launch": launches[0],
    "note": "logits 行 2/3/4（r1）分别允许 100/200/300；行 7（r3）允许 "
            "50256（词表尾 token——最后一块的部分越界由 block_offset<vocab_size 谓词"
            "遮蔽，核验通过）；行 0/1/5/6 非语法行不被触碰（保持 0.0，全 50257 位"
            "无一 -inf）",
}

# ── 分块几何账（m19 主表数据）─────────────────────────────────────────────────
grid = launches[0]["grid"]
BLOCK = launches[0]["BLOCK_SIZE"]
out["geometry"] = {
    "grid": grid,
    "grid_dim0_num_masks": grid[0],
    "grid_dim1_vocab_blocks": grid[1],
    "BLOCK_SIZE": BLOCK,
    "vocab": VOCAB,
    "cdiv_vocab_over_block": cdiv(VOCAB, BLOCK),
    "tokens_covered_dim1": grid[1] * BLOCK,
    "tail_masked_positions": grid[1] * BLOCK - VOCAB,
    "int32_per_row": cdiv(VOCAB, 32),
    "packed_int32_per_program": BLOCK // 32,
    "unpack_shape": [BLOCK // 32, 32],
    "vocab_tail_token": 50256,
    "vocab_tail_int32_col": 50256 // 32,
    "vocab_tail_bit_in_col": 50256 % 32,
    "note": "第二维 7 个 program × 8192 token = 57344 覆盖 50257——尾块 7087 个位置"
            "由双谓词（位=0 且 block_offset<vocab_size）遮蔽；[256,1]>>[1,32] 广播"
            "解包 256 个 int32 → 8192 个布尔",
}

# ── 交叉核验：同一行掩码走 V1 路径（xgr）与 V2 kernel 结果逐元素一致 ──────────
import xgrammar as xgr

logits_v1 = torch.zeros((1, VOCAB), device="cuda", dtype=torch.float32)
full_sorted = np.stack([make_row([100])] + [np.full(cdiv(VOCAB, 32), -1, np.int32)] * 0)
xgr.apply_token_bitmask_inplace(logits_v1, torch.from_numpy(compact[:1]).to("cuda"))
torch.cuda.synchronize()
keep_v1 = [t for t in range(VOCAB) if logits_v1[0, t].item() != float("-inf")]
out["cross_check_v1_xgr"] = {
    "v1_allowed": keep_v1,
    "v2_allowed_row2": allowed_per_row[2],
    "equal": keep_v1 == allowed_per_row[2],
    "note": "同一行掩码（允许 100）：V1 路径 xgr.apply_token_bitmask_inplace 与 "
            "V2 自写 Triton kernel 结果逐元素一致——同一 bit 语义（1=允许/0=禁→-inf）"
            "的两条落地路径",
}

# V2 常驻缓冲账（每步只拷活跃前缀）
out["resident_buffer"] = {
    "buffer_shape": list(worker.grammar_bitmask.shape),
    "buffer_bytes": int(np.prod(worker.grammar_bitmask.shape)) * 4,
    "per_step_copied_rows": int(compact.shape[0]),
    "per_step_copied_bytes": int(compact.shape[0] * compact.shape[1]) * 4,
    "indices_tensor_rows": len(expected_mapping),
    "note": "GPU 常驻 [8,1571] int32 缓冲一次分配；每步 copy_stream 只异步搬紧凑掩码"
            "前缀 4 行 + 一张 int32 行索引张量——与 V1 重排整表后全量搬运的对照账",
}

dump("trace_m19_v2_kernel.json", out)
print("grid:", out["geometry"]["grid"], "BLOCK:", BLOCK)
print("rows kept:", {r: allowed_per_row[r] for r in (2, 3, 4, 7)})
print("untouched ok:", out["apply"]["untouched_rows_all_zero"])
print("cross-check:", out["cross_check_v1_xgr"]["equal"])
