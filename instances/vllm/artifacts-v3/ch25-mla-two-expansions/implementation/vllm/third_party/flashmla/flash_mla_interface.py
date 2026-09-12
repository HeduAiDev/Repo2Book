# SOURCE: vllm/third_party/flashmla/flash_mla_interface.py
# HOST SEAM：FlashMLA third-party 接口的 host 参考镜像。真实实现是
# DeepSeek FlashMLA CUDA 扩展（vllm/_flashmla_C / vllm/_flashmla_extension_C，
# Hopper/Blackwell tile-scheduler kernel）；host 无扩展——本镜像承载**同一
# 接口契约**与**文件头 Data-Movement Friendly 伪码的精确数学**（分页 gather
# → 吸收后的 576 维单头 MQA → LSE），使 decode 腿可数值追踪：
#   flash_mla_with_kvcache(q=[B,sq,N,L+R], k_cache=[blocks,page,1,L+R],
#     block_table, cache_seqlens, head_dim_v=L, tile_scheduler_metadata,
#     softmax_scale, causal, is_fp8_kvcache) -> (o=[B,sq,N,head_dim_v],
#     lse=[N,B,sq])
from __future__ import annotations

from dataclasses import dataclass, field

import torch


# SOURCE: vllm/third_party/flashmla/flash_mla_interface.py FlashMLASchedMeta
#   —— HOST SEAM：tile-scheduler 元数据容器（真实由 CUDA 扩展产出；host
#   参考数学不消费它，保契约字段位）
@dataclass
class FlashMLASchedMeta:
    # SOURCE: vllm/third_party/flashmla/flash_mla_interface.py —— 字段位
    tile_scheduler_metadata: torch.Tensor | None = field(default=None)
    num_splits: torch.Tensor | None = field(default=None)


# SOURCE: vllm/v1/attention/ops/flashmla.py:L178-L183 get_mla_metadata
#   —— HOST SEAM：tile-scheduler 计划（真实调 CUDA 扩展算每 SM 分片；
#   host 参考数学不消费——返回空计划容器，字段位同契约）
def get_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    is_fp8_kvcache: bool = False,
) -> tuple[FlashMLASchedMeta, torch.Tensor]:
    # SOURCE: vllm/v1/attention/ops/flashmla.py:L178-L183 —— HOST SEAM
    return FlashMLASchedMeta(), torch.zeros(
        cache_seqlens.shape[0] + 1, dtype=torch.int32
    )


# SOURCE: vllm/third_party/flashmla/flash_mla_interface.py flash_mla_with_
#   kvcache —— HOST SEAM：MQA kernel 的参考数学（文件头 L94-L118 伪码）：
#   q 已吸收（ql_nope ⊕ q_pe），cache 行 = (kv_c ⊕ k_pe)；分数 =
#   scale·(ql_nope·kv_c + q_pe·k_pe)，causal 按各请求序列位置；输出 =
#   softmax·kv_c（V 头 = kv_lora_rank 潜维本身），LSE=[N,B,sq]。
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata=None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SOURCE: vllm/third_party/flashmla/flash_mla_interface.py —— HOST SEAM
    assert not is_fp8_kvcache, "host mirror covers the bf16 path only"
    B, sq, N, D = q.shape
    L = head_dim_v
    R = D - L
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)
    pages = k_cache.view(k_cache.shape[0], k_cache.shape[1], k_cache.shape[-1])
    bs = pages.shape[1]
    o = torch.zeros(B, sq, N, L, dtype=q.dtype)
    # FA 家族契约：lse 为 float32（flashmla 返回 float32 lse）
    lse = torch.full((N, B, sq), -float("inf"), dtype=torch.float32)
    for b in range(B):
        s_len = int(cache_seqlens[b])
        rows = []
        for pos in range(s_len):
            blk = int(block_table[b, pos // bs])
            rows.append(blk * bs + pos % bs)
        keys = pages.reshape(-1, pages.shape[-1])[torch.tensor(rows, dtype=torch.long)]
        kv_c, k_pe = keys[:, :L], keys[:, L:]
        qb = q[b]  # [sq, N, D]（吸收后的 ql_nope ⊕ q_pe）
        # scores [N, sq, s_len]
        s = (
            torch.einsum("tnd,sd->nts", qb[..., :L], kv_c)
            + torch.einsum("tnd,sd->nts", qb[..., L:], k_pe)
        ) * softmax_scale
        if causal:
            pos_q = torch.arange(s_len - sq, s_len)
            mask = torch.arange(s_len).unsqueeze(0) > pos_q.unsqueeze(1)
            s.masked_fill_(mask.unsqueeze(0), -float("inf"))
        p = torch.softmax(s, dim=-1)
        # [N, sq, L] → [sq, N, L]（回填 o 的 [sq, N, head_dim_v] 槽位）
        o[b] = torch.einsum("nts,sl->ntl", p, kv_c).transpose(0, 1)
        lse[:, b, :] = torch.logsumexp(s, dim=-1)
    return o, lse
