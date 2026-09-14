# SOURCE: vllm/_custom_ops.py
# HOST SEAM（B1 CUDA kernel 镜像·精确数学）：本章消费的 indexer 自定义算子族。
# 真实实现是 csrc CUDA kernel（torch.ops._C/_C_cache_ops 面）；host 镜像承载
# 同一函数签名与数学：
#   indexer_k_quant_and_cache    K 的逐 token-group FP8 量化+缓存插入融合
#     （132B/条 = 128B fp8 值 + 4B fp32 scale；ue8m0 幂次 scale 与
#     _fused_indexer_q_rope_quant_kernel L176-L178 同式）——真实锚
#     vllm/_custom_ops.py:L2989-L2999 + csrc/indexer_kernels（quant+scatter）
#   cp_gather_indexer_k_quant_cache  分页 IndexCache → 连续 workspace 收集
#     （block_table 寻址）——真实锚 vllm/_custom_ops.py:L3045-L3056
#   top_k_per_row_prefill  因果窗 [cu_ks, cu_ke) 内降序 top-k、tie 取小
#     index、输出相对窗口起点、不足 topk 留 -1——真实锚
#     vllm/_custom_ops.py:L3001-L3021 +
#     csrc/libtorch_stable/sampler.cu:L545-.../L725-...（插入排序的
#     `logit < other || (== && i < j)` tie-break 与 `outIndices[i]-rowStart`
#     相对化逐条对齐）
#   top_k_per_row_decode   同语义、窗口 [0, seq_len)；1D seq_lens 时核内自算
#     rowEnd = seq_len - next_n + (row % next_n) + 1（sampler.cu:L572-L576）
#   cooperative_topk / persistent_topk  三核分派的另两核——同 top-k 语义的
#     专用核（小批合作式 / topk∈{512,1024,2048} 持久式；csrc radix topk 族），
#     host 经 torch.library 挂进 torch.ops._C 使分派代码逐字可跑
#   concat_mla_q                ql_nope ‖ q_pe 沿尾维（真实锚 L2984-L2988）
#   gather_and_maybe_dequant_cache / cp_gather_and_upconvert_fp8_kv_cache
#     sparse MLA 消费侧的分页 gather（656B 布局 upconvert）——ch25 同款 seam
from __future__ import annotations

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.utils.torch_utils import vllm_lib

_C_lib = None


def _ensure_C_ops():
    """torch.ops._C.cooperative/persistent_topk 的真实注册位。"""
    # SOURCE: vllm/_custom_ops.py torch.ops._C 面 —— HOST SEAM 注册
    global _C_lib
    if _C_lib is None:
        _C_lib = torch.library.Library("_C", "FRAGMENT")
    return _C_lib


# SOURCE: csrc indexer_kernels 的 ue8m0 幂次 scale 同式（HOST SEAM；锚点双置）
def _ue8m0_scale(x: torch.Tensor) -> torch.Tensor:
    """逐 token 128 组 ue8m0 幂次 scale（fused 核同式）。"""
    _, fp8_max = get_fp8_min_max()
    amax = x.float().abs().amax(dim=-1, keepdim=True)
    return torch.exp2(
        torch.ceil(torch.log2(torch.clamp(amax, min=1e-10) / fp8_max)))


# SOURCE: vllm/_custom_ops.py:L2989-L2999 indexer_k_quant_and_cache —— HOST
#   SEAM 参考数学
def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    kv_cache_dtype: str,
) -> None:
    # SOURCE: vllm/_custom_ops.py:L2989-L2999 —— HOST SEAM
    scale = _ue8m0_scale(k)
    q = torch.clamp(
        k.float() / scale, *get_fp8_min_max()).to(torch.float8_e4m3fn)
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    head = kv_cache.shape[-1] - 4  # 132B 布局：128B fp8 + 4B fp32 scale
    for t in range(k.shape[0]):
        slot = int(slot_mapping[t].item())
        if slot < 0:
            continue
        flat[slot, :head] = q[t].view(torch.uint8)
        flat[slot, head:] = scale[t].view(torch.uint8).reshape(-1)


# SOURCE: vllm/_custom_ops.py:L3045-L3056 cp_gather_indexer_k_quant_cache
#   —— HOST SEAM 参考数学（分页 gather：block_table[req, pos//bs]*bs+pos%bs）
def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    # SOURCE: vllm/_custom_opts.py:L3045-L3056 —— HOST SEAM（字节搬运：值区经
    #   uint8 视图写入（dst 为 fp8 张量时按位 reinterpret——真实核语义），
    #   scale 区 4B 原样）
    block_size = kv_cache.shape[1]
    flat = kv_cache.view(-1, kv_cache.shape[-1])
    head = kv_cache.shape[-1] - 4
    dst_k_u8 = dst_k.view(torch.uint8)
    cu = cu_seq_lens.tolist()
    for b in range(len(cu) - 1):
        for pos in range(cu[b + 1] - cu[b]):
            blk = int(block_table[b, pos // block_size].item())
            src = blk * block_size + pos % block_size
            dst_k_u8[cu[b] + pos] = flat[src, :head]
            dst_scale[cu[b] + pos] = flat[src, head:]


# SOURCE: csrc/libtorch_stable/sampler.cu 插入排序 tie-break 同构（HOST SEAM；锚点双置）
def _topk_desc(vals: list, k: int, base: int = 0):
    """降序 top-k、tie 取小 index（插入排序 tie-break 同构）。"""
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
    return [base + i for i in order[:k]]


# SOURCE: vllm/_custom_ops.py:L3001-L3021 top_k_per_row_prefill —— HOST SEAM
#   参考数学
def top_k_per_row_prefill(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    # SOURCE: vllm/_custom_ops.py:L3001-L3021 + sampler.cu topKPerRowPrefill
    #   —— HOST SEAM（输出相对 rowStart）
    ks = cu_seqlen_ks.tolist()
    ke = cu_seqlen_ke.tolist()
    for r in range(num_rows):
        vals = logits[r, ks[r]:ke[r]].float().tolist()
        sel = _topk_desc(vals, topk_tokens)
        for i, rel in enumerate(sel):
            raw_topk_indices[r, i] = rel  # 相对窗口起点（核内 -rowStart）
        # 窗口不足 topk 的尾部留 -1（buffer 预清哨兵不被覆盖）


# SOURCE: vllm/_custom_ops.py:L3023-L3043 top_k_per_row_decode —— HOST SEAM
#   参考数学
def top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    # SOURCE: vllm/_custom_ops.py:L3023-L3043 + sampler.cu topKPerRowDecode
    #   —— HOST SEAM（1D seq_lens 核内自算 rowEnd）
    is2d = seq_lens.dim() == 2
    for r in range(num_rows):
        if is2d:
            end = max(0, int(seq_lens.reshape(-1)[r].item()))
        else:
            b, j = r // next_n, r % next_n
            end = max(0, int(seq_lens[b].item()) - next_n + j + 1)
        vals = logits[r, :end].float().tolist()
        sel = _topk_desc(vals, topk_tokens)
        for i, idx in enumerate(sel):
            raw_topk_indices[r, i] = idx
        # 不足 topk 的尾部留 -1


# SOURCE: vllm/_custom_ops.py concat_mla_q —— HOST SEAM（真实 L2984-L2988）
def concat_mla_q(ql_nope: torch.Tensor, q_pe: torch.Tensor,
                 q_out: torch.Tensor) -> None:
    # SOURCE: vllm/_custom_ops.py:L2984-L2988 —— HOST SEAM
    torch.cat((ql_nope, q_pe), dim=-1, out=q_out)


# SOURCE: vllm/_custom_ops.py gather_and_maybe_dequant_cache —— HOST SEAM
#   （ch25 同款：bf16/auto 路径分页 gather；量化变体不触达）
def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str = "auto",
    scale=None,
    seq_starts=None,
) -> None:
    # SOURCE: vllm/_custom_ops.py gather_and_maybe_dequant_cache —— HOST SEAM
    block_size = src_cache.shape[1]
    cu = cu_seq_lens.tolist()
    # 逐请求把分页行收集进连续 workspace（token_to_seq 的全局行号布局）
    row = 0
    for b in range(len(cu) - 1):
        for pos in range(cu[b + 1] - cu[b]):
            blk = int(block_table[b, pos // block_size].item())
            src = blk * block_size + pos % block_size
            if row < dst.shape[0]:
                dst[row] = src_cache.reshape(-1, src_cache.shape[-1])[src]
            row += 1


# SOURCE: vllm/_custom_ops.py cp_gather_and_upconvert_fp8_kv_cache —— HOST
#   SEAM（656B 布局：512B fp8 nope + 16B scale + 128B rope bf16 → bf16 行）
def cp_gather_and_upconvert_fp8_kv_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    workspace_starts: torch.Tensor,
    num_reqs: int,
) -> None:
    # SOURCE: vllm/_custom_ops.py —— HOST SEAM（fp8_ds_mla upconvert 数学）
    # 656B = 512B fp8(=128 nope × 4?? 逐块 scale 布局按 flashmla_sparse.py
    # 模块 docstring：512 fp8 + 16B fp32×4 scale + 128B bf16 rope）
    block_size = src_cache.shape[1]
    width = src_cache.shape[-1]
    flat = src_cache.view(-1, width)
    starts = workspace_starts.tolist()
    for b in range(num_reqs):
        n = int(dst.shape[0] - starts[b]) if b == num_reqs - 1 else (
            starts[b + 1] - starts[b])
        for pos in range(max(0, n)):
            blk = int(block_table[b, pos // block_size].item())
            src = blk * block_size + pos % block_size
            row = flat[src]
            nope_q = row[:512].view(torch.float8_e4m3fn).float()
            scales = row[512:528].view(torch.float32).float()  # 4×128 块 scale
            rope = row[528:656].view(torch.bfloat16).float()
            deq = (nope_q.view(4, 128) * scales.view(4, 1)).reshape(-1)
            dst[starts[b] + pos] = torch.cat([deq, rope])


# ── torch.ops._C 三核分派中的 cooperative/persistent 两核（host 注册位） ──
def _cooperative_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
    topk_tokens: int,
    max_seq_len: int,
) -> None:
    # SOURCE: csrc radix topk 家族 cooperative_topk —— HOST SEAM（同
    #   top_k_per_row_decode 语义：小批合作式核）
    next_n = 1
    top_k_per_row_decode(logits, next_n, seq_lens, topk_indices,
                         logits.shape[0], logits.stride(0), logits.stride(1),
                         topk_tokens)


def _persistent_topk(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
    topk_tokens: int,
    num_columns: int,
) -> None:
    # SOURCE: csrc radix topk 家族 persistent_topk —— HOST SEAM（topk∈
    #   {512,1024,2048} 持久式核；语义同 top_k_per_row_decode）
    next_n = 1
    top_k_per_row_decode(logits, next_n, seq_lens, topk_indices,
                         logits.shape[0], logits.stride(0), logits.stride(1),
                         topk_tokens)


def _register_C_topk_ops() -> None:
    # SOURCE: torch.ops._C 注册面 —— HOST SEAM
    lib = _ensure_C_ops()
    if hasattr(torch.ops._C, "cooperative_topk"):
        return
    lib.define(
        "cooperative_topk(Tensor logits, Tensor seq_lens, Tensor(a!) "
        "topk_indices, Tensor topk_workspace, int topk_tokens, "
        "int max_seq_len) -> ()")
    lib.impl("cooperative_topk", _cooperative_topk, "CPU")
    lib.define(
        "persistent_topk(Tensor logits, Tensor seq_lens, Tensor(a!) "
        "topk_indices, Tensor topk_workspace, int topk_tokens, "
        "int num_columns) -> ()")
    lib.impl("persistent_topk", _persistent_topk, "CPU")


_register_C_topk_ops()
