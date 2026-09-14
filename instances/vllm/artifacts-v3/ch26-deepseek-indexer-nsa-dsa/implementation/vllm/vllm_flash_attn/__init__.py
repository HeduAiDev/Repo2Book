# SOURCE: vllm/vllm_flash_attn/__init__.py
# HOST SEAM（B1 kernel 镜像·精确数学）：FA varlen 入口的 host 参考镜像。
# 真实为 vllm_flash_attn 扩展（FA4 masked-MHA 消费面）；本镜像承载 varlen
# + topk 位掩码（mask_mod/aux_tensors 契约）的注意力数学：逐请求在
# cu_seqlens 段内做 softmax(q·kᵀ·scale)·v，dense_mask 位图门控（bit i =
# token (i*32+bit) 选中——与 _build_topk_mask 的打包契约同构）+ causal 语义。
from __future__ import annotations

import torch


def _decode_bitmask(dense_mask: torch.Tensor, req: int, qpos: int,
                    n_words: int) -> list[bool]:
    row = dense_mask[req, qpos]
    sel = []
    for w in range(n_words):
        word = int(row[w].item())
        for bit in range(32):
            sel.append(bool((word >> bit) & 1))
    return sel


# SOURCE: vllm/vllm_flash_attn flash_attn_varlen_func —— HOST SEAM 参考数学
def flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    softmax_scale,
    return_softmax_lse=False,
    causal=False,
    mask_mod=None,
    aux_tensors=None,
    aux_tensor_leading_dims=None,
    key_starts=None,
    fa_version=None,
    **kwargs,
):
    # SOURCE: vllm/vllm_flash_attn —— HOST SEAM（topk 位掩码 varlen 注意力）
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    dense_mask = aux_tensors[0] if aux_tensors else None
    n_words = dense_mask.shape[-1] if dense_mask is not None else 0
    B = len(cu_q) - 1
    H = q.shape[-2]
    Dv = v.shape[-1]
    out = torch.zeros(q.shape[0], H, Dv, dtype=q.dtype)
    lse = torch.empty(H, q.shape[0], dtype=torch.float32)
    key_off = key_starts.tolist() if key_starts is not None else None
    for b in range(B):
        qs, qe = cu_q[b], cu_q[b + 1]
        ks, ke = cu_k[b], cu_k[b + 1]
        for t in range(qe - qs):
            global_t = qs + t
            sel = None
            if dense_mask is not None:
                sel = _decode_bitmask(dense_mask, b, t, n_words)
            keys, vals, keep = [], [], []
            for j in range(ke - ks):
                pos = ks + j
                if key_off is not None:
                    pos_abs = key_off[b] + j
                else:
                    pos_abs = j
                if sel is not None and not (
                        pos_abs < len(sel) and sel[pos_abs]):
                    continue
                if causal and (ks + j) > global_t:
                    continue
                keys.append(k[pos])
                vals.append(v[pos])
                keep.append(True)
            if not keys:
                out[global_t] = 0
                lse[:, global_t] = float("-inf")
                continue
            K = torch.stack(keys).float()
            V = torch.stack(vals).float()
            scores = (q[global_t].float() @ K.T) * softmax_scale
            probs = torch.softmax(scores, dim=-1)
            out[global_t] = (probs @ V).to(q.dtype)
            lse[:, global_t] = torch.logsumexp(scores, dim=-1)
    if return_softmax_lse:
        return out, lse
    return out
