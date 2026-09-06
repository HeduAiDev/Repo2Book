# Subtract-only companion for v3 ch21 — vllm/v1/attention/backends/fa_utils.py
# (pin v0.27.1 / 6e448d0ea). 本章切面：FA 版本探测探针族（真身逐字——host
# 走 try/except ImportError 的真实回退）+ 平台 import 面（CUDA/XPU/ROCm 三支
# 中 CUDA 支的 vllm.vllm_flash_attn 名字面；host 无 CUDA 库——三 op 由包内
# HOST SEAM 镜像承载：flash_attn_varlen_func/get_scheduler_metadata 在
# flash_attn.py 尾部，reshape_and_cache_flash 在 _custom_ops.py）。
from __future__ import annotations

from typing import Any

import torch

from ...._host_seams import (
    current_platform,
    get_flash_attn_version,
    init_logger,
    is_flash_attn_varlen_func_available,
)

logger = init_logger(__name__)

# SUBTRACTED: ROCm/XPU 平台 import 支（L15-L70）——host 镜像承载面见下；
#   _ROCM_FLASH_ATTN_AVAILABLE 追踪位随 ROCm 支删。


# SOURCE: vllm/v1/attention/backends/fa_utils.py:L21-L27 平台 import 面
#   （CUDA 支原文：is_cuda → from vllm._custom_ops import reshape_and_cache_
#   flash; from vllm.vllm_flash_attn import flash_attn_varlen_func /
#   get_scheduler_metadata）。host 无 vllm 包——三名字由 HOST SEAM 承载：
#   * reshape_and_cache_flash → .._custom_ops（kernel 本体的逐 token 镜像）
#   * flash_attn_varlen_func / get_scheduler_metadata → flash_attn.py 尾部
#     HOST SEAM 镜像（精确 attention 数学 / FA2 无调度返 None）
if is_flash_attn_varlen_func_available():
    from ...._custom_ops import reshape_and_cache_flash  # noqa: F401

# SUBTRACTED: compile_flash_attn_varlen_func_from_specs / FlashAttention-
#   CuTeDSLCompileSpec（L73-L137）——FA4 编译预热域，本章零调用。

# SOURCE: vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version ——
#   HOST SEAM 装配位（真身 L140-L282 按算力代 + 配置覆写解析 3/4/2；host
#   恒 FA2——见 _host_seams.py 的装配注记）。此处 re-export 保住调用面。
__all__ = [
    "get_flash_attn_version",
    "is_fa_version_supported",
    "flash_attn_supports_kv_cache_dtype",
    "flash_attn_supports_quant_query_input",
    "flash_attn_supports_sinks",
    "is_flash_attn_varlen_func_available",
    "reshape_and_cache_flash",
    "current_platform",
]


# SOURCE: vllm/v1/attention/backends/fa_utils.py:L285-L293 is_fa_version_
#   supported ——（逐字；host 无 vllm_flash_attn → ImportError 回退 False）
def is_fa_version_supported(fa_version: int) -> bool:  # SOURCE: vllm/v1/attention/backends/fa_utils.py
    try:
        from vllm.vllm_flash_attn.flash_attn_interface import (  # type: ignore
            is_fa_version_supported as _is_fa_version_supported,
        )

        return _is_fa_version_supported(fa_version)
    except ImportError:
        return False


# SOURCE: vllm/v1/attention/backends/fa_utils.py:L296-L316 flash_attn_supports_
#   kv_cache_dtype ——（逐字）
def flash_attn_supports_kv_cache_dtype(  # SOURCE: vllm/v1/attention/backends/fa_utils.py
    kv_cache_dtype: str = "fp8_e4m3",
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> bool:
    if kv_cache_dtype == "fp8_e5m2":
        return False
    if current_platform.is_xpu():
        return True
    fa_version = get_flash_attn_version(
        requires_alibi=requires_alibi,
        head_size=head_size,
        head_size_v=head_size_v,
        has_sinks=has_sinks,
    )
    return (fa_version == 3 and current_platform.is_device_capability_family(90)) or (
        fa_version == 4 and current_platform.is_device_capability_family(100)
    )


# SOURCE: vllm/v1/attention/backends/fa_utils.py:L319-L320 flash_attn_supports_
#   quant_query_input ——（逐字）
def flash_attn_supports_quant_query_input() -> bool:  # SOURCE: vllm/v1/attention/backends/fa_utils.py
    return not current_platform.is_xpu()


# SOURCE: vllm/v1/attention/backends/fa_utils.py:L323-L326 flash_attn_supports_
#   sinks ——（逐字）
def flash_attn_supports_sinks() -> bool:  # SOURCE: vllm/v1/attention/backends/fa_utils.py
    if current_platform.is_xpu():
        return True
    return get_flash_attn_version() in (3, 4)


# SUBTRACTED: flash_attn_supports_mla（L329-L347）——MLA 域（ch24/25）。
# SUBTRACTED: is_flash_attn_varlen_func_available 真身（L350-L378——CUDA/XPU
#   恒 True、ROCm 看 _ROCM_FLASH_ATTN_AVAILABLE）——HOST SEAM 装配位在
#   _host_seams.py（host 镜像承载三 op，恒 True 保住真实 import 面）。


# ── HOST SEAM：vllm.vllm_flash_attn 两个 CUDA op 的 CPU 镜像 ────────────────
# 真实导入位：vllm/v1/attention/backends/fa_utils.py:L23-L27
#   `from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_
#   metadata`（host 无该库——名字面与调用面逐字对齐，见 impl-notes §Seam）。

# SOURCE: vllm/v1/attention/backends/fa_utils.py:L63-L65 ROCm 的 get_
#   scheduler_metadata stub（真身同型：FA3 特性在无 FA3 平台返 None）——
#   HOST SEAM 装配位：host 恒 FA2 → 无 AOT 调度。
def get_scheduler_metadata(*args: Any, **kwargs: Any) -> None:  # SOURCE: vllm/v1/attention/backends/fa_utils.py
    return None


# SOURCE: vllm/vllm_flash_attn/flash_attn_varlen.py flash_attn_varlen_func
#   （调用位 vllm/v1/attention/backends/flash_attn.py:L1041）——HOST SEAM
#   精确数学镜像：每请求穿 block_table 逐逻辑块 gather K/V（间接寻址——
#   paged 读侧语义）、GQA 广播、causal（bool 或 int32 per-seq 张量——FA4
#   dynamic_causal 的消费面）、逐 Q 头 softmax(QK^T·scale)V（ch20 已立数学；
#   fp64 softmax 稳定数值）。只承载本章用面（(-1,-1) 无窗口、无 alibi/
#   softcap/descale）——滑窗/级联数学 → ch20。
def flash_attn_varlen_func(  # SOURCE: vllm/v1/attention/backends/flash_attn.py
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool | torch.Tensor,
    alibi_slopes=None,
    window_size=None,
    block_table: torch.Tensor | None = None,
    softcap: float = 0.0,
    scheduler_metadata=None,
    fa_version: int = 2,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    dynamic_causal=None,
    num_splits: int = 0,
    s_aux=None,
    mask_mod=None,
    aux_tensors=None,
):
    window_ok = window_size is None or list(window_size) == [-1, -1]
    assert (causal is True or isinstance(causal, torch.Tensor)) and window_ok and (
        alibi_slopes is None
    ), (
        "HOST SEAM 镜像只承载本章用面（causal=True 或 per-seq 张量、无窗口/"
        "alibi——(-1,-1) 即 FA 的「无窗口」约定）——滑窗/级联数学 → ch20"
    )
    causal_per_req = causal.tolist() if isinstance(causal, torch.Tensor) else None
    cu = cu_seqlens_q.tolist()
    num_kv_heads = k.shape[2]
    num_heads = q.shape[1]
    rep = num_heads // num_kv_heads
    for r in range(len(cu) - 1):
        start, end = cu[r], cu[r + 1]
        q_len = end - start
        if q_len == 0:
            continue  # HOST SEAM：padded 空请求无活干（真 kernel 天然跳过）
        seq_len = int(seqused_k[r])
        q_r = q[start:end]  # [q_len, H, D]
        # 穿表间接寻址：每个逻辑块号现场查物理块再取行（读侧内景）。
        k_rows, v_rows = [], []
        num_blocks_needed = (seq_len + k.shape[1] - 1) // k.shape[1]
        for b in range(num_blocks_needed):
            phys = int(block_table[r, b])
            rows = min(k.shape[1], seq_len - b * k.shape[1])
            k_rows.append(k[phys, :rows])
            v_rows.append(v[phys, :rows])
        k_hist = torch.cat(k_rows, dim=0)  # [seq_len, Hk, D]
        v_hist = torch.cat(v_rows, dim=0)
        if rep > 1:  # GQA：KV 头广播到 Q 头
            k_hist = k_hist.repeat_interleave(rep, dim=1)
            v_hist = v_hist.repeat_interleave(rep, dim=1)
        # causal：query i 的绝对位置 = seq_len - q_len + i，只看 ≤ 该位置的键
        # （per-seq 张量口径：causal_per_req[r] 为 0 的请求放宽为双向）。
        context_offset = seq_len - q_len
        head_dim = q_r.shape[-1]
        bidirectional = causal_per_req is not None and not int(causal_per_req[r])
        for i in range(q_len):
            key_len = seq_len if bidirectional else context_offset + i + 1
            for h in range(num_heads):
                scores = (k_hist[:key_len, h] @ q_r[i, h]) * softmax_scale
                p = torch.softmax(scores.to(torch.float64), dim=-1).to(q.dtype)
                out[start + i, h * head_dim:(h + 1) * head_dim] = (
                    p @ v_hist[:key_len, h]
                )
    return out
