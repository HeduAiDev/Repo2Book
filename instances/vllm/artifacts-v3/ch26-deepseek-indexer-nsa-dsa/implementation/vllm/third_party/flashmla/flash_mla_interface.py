# SOURCE: vllm/third_party/flashmla/flash_mla_interface.py
# HOST SEAM（B1 CUDA kernel 镜像·精确数学）：FlashMLA 稀疏核族的 host 参考
# 镜像。真实实现是 FlashMLA sm90/sm100 CUDA kernel；本镜像承载同一函数签名
# 与注意力数学（在 576/512 维潜向量上只对选中条目真算——O(Lk)）：
#   flash_mla_sparse_fwd(q, kv, indices, sm_scale, topk_length)
#     → per (t,h)：softmax(q[t,h]·K_selᵀ·scale)·V_sel；V = K[:, :head_dim_v]
#   flash_mla_with_kvcache(..., indices, topk_length, extra_k_cache,
#     extra_indices_in_kvcache, extra_topk_length, attn_sink, ...)
#     → K = SWA 缓存选中行 ∪ extra 缓存选中行（一核双源）；attn_sink 为每头
#     sink logit——进 softmax 分母、不进 value 分子（FlashMLA attn sink 语义）
#   get_mla_metadata → (FlashMLASchedMeta 容器, num_splits)（真实为核内
#     planner 元数据；host 为契约容器）
from __future__ import annotations

from dataclasses import dataclass, field

import torch


# SOURCE: vllm/third_party/flashmla/flash_mla_interface.py FlashMLASchedMeta
#   —— HOST SEAM：契约容器
@dataclass
class FlashMLASchedMeta:
    # SOURCE: vllm/third_party/flashmla —— HOST SEAM 容器位
    tile_scheduler_metadata: torch.Tensor = field(default_factory=lambda: torch.zeros(1))
    num_splits: torch.Tensor = field(default_factory=lambda: torch.zeros(1))


# SOURCE: vllm/third_party/flashmla/flash_mla_interface.py get_mla_metadata
#   —— HOST SEAM
def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int = 1,
    topk: int | None = None,
    num_heads_q: int = 64,
    is_fp8_kvcache: bool = False,
    **kwargs,
) -> tuple[FlashMLASchedMeta, torch.Tensor]:
    # SOURCE: vllm/third_party/flashmla —— HOST SEAM 契约容器
    return FlashMLASchedMeta(), torch.zeros(
        cache_seqlens.shape[0] + 1, dtype=torch.int32)


# SOURCE: vllm/third_party/flashmla —— HOST SEAM（锚点双置）
def _selected_rows(cache: torch.Tensor, indices: torch.Tensor,
                   lengths) -> list[torch.Tensor]:
    """cache [num_blocks, block, 1?, D] → 逐 (b, s) 行 gather（slot id 寻址）。

    indices 形态：[B, S, K]（decode 一核双源）或 [B, K]；lengths 相应
    [B, S]/[B]。
    """
    flat = cache.reshape(-1, cache.shape[-1])
    out = []
    B = indices.shape[0]
    S = indices.shape[1] if indices.dim() == 3 else 1
    lens = lengths.tolist() if torch.is_tensor(lengths) else lengths
    for b in range(B):
        for s in range(S):
            row = indices[b, s] if indices.dim() == 3 else indices[b]
            if S > 1:
                n = int(lens[b][s])
            else:
                n = int(lens[b]) if not isinstance(lens[b], list) else int(lens[b][0])
            ids = [int(i) for i in row.tolist()[:n] if i >= 0]
            out.append(flat[ids] if ids else flat[:0])
    return out


# SOURCE: vllm/third_party/flashmla/flash_mla_interface.py flash_mla_with_
#   kvcache —— HOST SEAM 参考数学（一核双源）
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None = None,
    head_dim_v: int = 512,
    tile_scheduler_metadata=None,
    cache_seqlens=None,
    num_splits=None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    topk_length=None,
    attn_sink=None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
    out: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SOURCE: vllm/third_party/flashmla —— HOST SEAM 参考数学
    scale = softmax_scale if softmax_scale is not None else 1.0
    B, S, H, D = q.shape
    swa_rows = _selected_rows(k_cache, indices, topk_length)  # [B*S][K, D]
    if extra_k_cache is not None:
        extra_rows = _selected_rows(extra_k_cache, extra_indices_in_kvcache,
                                    extra_topk_length)
        keys_all = [torch.cat([a, b]) for a, b in zip(swa_rows, extra_rows)]
    else:
        keys_all = swa_rows
    output = out if out is not None else torch.empty(
        B, S, H, head_dim_v, dtype=q.dtype)
    lse = torch.empty(H, B * S, dtype=torch.float32)
    for b in range(B):
        for s in range(S):
            keys = keys_all[b * S + s]
            if keys.shape[0] == 0:
                output[b, s] = 0
                lse[:, b * S + s] = float("-inf")
                continue
            scores = (q[b, s].float() @ keys.T.float()) * scale  # [H, K]
            if attn_sink is not None:
                # sink 按头数对齐（真实路径 q 已 pad 到 padded_heads——
                # 直调测试可传未 pad 的 q，取前 H 个 sink 位）
                sink = attn_sink.float()[:H].unsqueeze(-1)
                scores = torch.cat([scores, sink], dim=-1)
            probs = torch.softmax(scores, dim=-1)
            probs = probs[:, : keys.shape[0]]
            output[b, s] = (probs @ keys[:, :head_dim_v].float()).to(output.dtype)
            lse[:, b * S + s] = torch.logsumexp(scores[:, : keys.shape[0]], dim=-1)
    return output, lse


# SOURCE: vllm/third_party/flashmla/flash_mla_interface.py flash_mla_sparse_fwd
#   —— HOST SEAM 参考数学（O(Lk)：只对选中条目真算）
def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length=None,
    out=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SOURCE: vllm/third_party/flashmla —— HOST SEAM 参考数学
    #   q [T, H, D]；kv [N, 1, D]；indices [T, 1, topk]
    T, H, D = q.shape
    head_dim_v = 512
    flat = kv.reshape(-1, kv.shape[-1])
    output = out if out is not None else torch.empty(
        T, H, head_dim_v, dtype=q.dtype)
    lse = torch.empty(H, T, dtype=torch.float32)
    for t in range(T):
        row = indices[t, 0]
        n = (int(topk_length[t].item()) if topk_length is not None
             else row.shape[0])
        ids = [int(i) for i in row.tolist()[:n] if i >= 0]
        if not ids:
            output[t] = 0
            lse[:, t] = float("-inf")
            continue
        keys = flat[ids].float()  # [K, D]
        scores = (q[t].float() @ keys.T) * sm_scale  # [H, K]
        probs = torch.softmax(scores, dim=-1)
        output[t] = (probs @ keys[:, :head_dim_v].float()).to(output.dtype)
        lse[:, t] = torch.logsumexp(scores, dim=-1)
    return output, lse
