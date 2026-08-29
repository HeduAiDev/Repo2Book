# SOURCE: vllm/v1/attention/backends/flash_attn.py
# ch22 切面（站13/站14 + m5/m6/m7）：代表性消费端——
#   * FlashAttentionBackend 两个语义位：get_supported_kernel_block_sizes=
#     [MultipleOf(16)]（m9 约束源）与 forward_includes_kv_cache_update=False
#     （m7 双口径裁决的由来）；
#   * FlashAttentionImpl.do_kv_cache_update → reshape_and_cache_flash（写腿
#     落地：slot<0 跳过、slot//bs 与 slot%bs 逆分解）；
#   * FlashAttentionImpl.forward → flash_attn_varlen_func(block_table=)（读腿：
#     kernel 穿表间接寻址读历史 KV——F7 回收点）。
# HOST SEAM：vllm_flash_attn 的两个 CUDA op 在 CPU host 无库——
#   reshape_and_cache_flash 以 cache_kernels.cu:L315-L344 kernel 本体的逐行
#   torch 镜像承载（同一 PAD 跳过 + 同一逆分解）；flash_attn_varlen_func 以
#   精确 attention 数学承载（每请求穿 block_table 逐块 gather K/V——正是
#   flash varlen kernel 的读侧语义，ch20 已立数学）。builder/AOT/cascade/
#   量化/FA3-4 面归 ch21。
from __future__ import annotations

from dataclasses import dataclass

import torch

from ._host_seams import (
    get_dcp_group,
    get_flash_attn_version,
    init_logger,
    is_quantized_kv_cache,
)
from .backend import AttentionBackend, AttentionType

logger = init_logger(__name__)

# SUBTRACTED: fa_utils 版本探测/平台 import 归一（get_flash_attn_version /
#   is_flash_attn_varlen_func_available 等 → HOST SEAM 装配位）；cascade/
#   merge_attn_states/dcp ops import → ch21/分布式 Part。


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L242 FlashAttentionMetadata
#   —— FA 后端的 per-layer 元数据载体（block_table 字段即读腿的表——m6）
@dataclass
# SOURCE: vllm/v1/attention/backends/flash_attn.py:L243 FlashAttentionMetadata
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Split counts for FA2 DCP context attention. num_prefill_* tracks
    # context-bearing extend rows; pure prefills do not attend to DCP context.
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool | torch.Tensor = True

    sliding_window: tuple[int, int] | None = None

    # PrefixLM bidirectional ranges for multimodal tokens.
    # Shape: (num_seqs, max_ranges, 2) int32, [start, end] per range.
    mm_prefix_range_tensor: torch.Tensor | None = None

    # Reference Sliding Window Attention (R-SWA) fields.
    # rswa_prefix_lens:  per-request prompt lengths [num_reqs], int32, CUDA.
    # rswa_window:       sliding window size (scalar int, for logic checks).
    # rswa_window_tensor: [1] int32 CUDA tensor — pre-allocated in build() so
    #   that no CPU→CUDA copy is needed inside forward() during CUDA graph capture.
    # Only populated when the model uses R-SWA (Unlimited-OCR).
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None
    rswa_window_tensor: torch.Tensor | None = None


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L72 FlashAttentionBackend
#   —— 本章只保留两个语义位 + 名字位；选择/validate 全景 → ch21
class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: list[str] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L82-L84 —— FA kernel
    #   块尺寸约束 [MultipleOf(16)]（m9：16 对齐要求驱动 hybrid 细分）
    @staticmethod
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L83 get_supported_kernel_block_sizes
    def get_supported_kernel_block_sizes() -> list[int | object]:
        from .backend import MultipleOf

        return [MultipleOf(16)]

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L86 —— KV 写不含在
    #   forward() 里（两个 op：do_kv_cache_update + forward——m7 裁决源）
    forward_includes_kv_cache_update: bool = False

    # SUBTRACTED: get_preferred_block_size/get_name/supports_* 面（L88-L118）
    #   与 kv_cache_shape/builder 面 → ch21。


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L743 FlashAttentionImpl ——
#   代表性后端实现（写腿 do_kv_cache_update + 读腿 forward）
class FlashAttentionImpl:
    can_return_lse_for_decode: bool = True

    # SOURCE: vllm/v1/worker/…/backend.py AttentionImplBase.__new__ 的组探测
    #   （vllm/v1/attention/backend.py:L862-L883：__new__ 里 try/except 取
    #   dcp/pcp 组，未初始化时退化 world=1/rank=0——HOST SEAM 同型）
    # SOURCE: vllm/v1/attention/backend.py:L862 AttentionImplBase.__new__ 的组探测半边
    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        return self

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L746-L759 __init__ 签名
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L760-L782（逐字）
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L783-L788（版本探测
        #   ——HOST SEAM：get_flash_attn_version 装配位，host 恒 FA2）
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            requires_local_attention=sliding_window is not None,
            head_size=head_size,
            has_sinks=sinks is not None,
        )
        # SUBTRACTED: logger.info_once 版本日志（L789-L792）——观测面；
        #   VLLM_BATCH_INVARIANT 缓存（L793-L794）——ch20 批不变域；
        #   量化 kv dtype 支持检查（L796-L808）——量化域 → ch14/ch21（本章
        #   kv_cache_dtype="auto" 恒不触发）；
        #   sinks 断言块（L810-L818）——FA3 专属（本章 sinks=None）；
        #   dcp_a2a combine 装配（L822-L836）——分布式 Part（CP 多 rank 部署
        #   形态不进本章，单卡 dcp_world_size=1）。
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L820（量化 query
        #   输入支持位——HOST SEAM 装配：host 无 FA3，恒 False）
        self.supports_quant_query_input = False
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L810 sinks 字段位
        self.sinks = sinks

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L838 forward —— 读腿
    #   （block_table 进 flash_attn_varlen_func 穿表间接寻址，站14/F7）
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L865-L867
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L869-L872
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L874-L876 profiling
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L889 真 token 数
        num_actual_tokens = attn_metadata.num_actual_tokens

        # SUBTRACTED: encoder attention 支（L891-L902——_forward_encoder_
        #   attention，encoder-only 模型无 paged KV，delete[7] 同域；本章
        #   attn_type 恒 DECODER）。

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L904-L905 K/V 页张量
        #   视图（(B,H,N,2D) → ((B,N,H,D),(B,N,H,D))——HND 页布局 K/V 打在
        #   内容维）
        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        # SUBTRACTED: stride 规范化块（L906-L922——FA3/4 TMA 的 16 字节对齐
        #   修正，ch21 域；host 镜像无 TMA）。

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L924-L927（量化
        #   dtype 的 fp8 视图——"auto" 恒不触发，判别位保留）
        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L929-L935 非级联主
        #   路的元数据解包——block_table 从这里进 kernel（站14 原文行）
        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            # SOURCE: vllm/v1/attention/backends/flash_attn.py:L937-L945 descale
            #   展开
            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            # SUBTRACTED: DCP 前向支（L947-L960——_forward_with_dcp，CP 多
            #   rank 部署形态 → 分布式 Part；单卡 dcp_world_size=1 不进）。
            # SOURCE: vllm/v1/attention/backends/flash_attn.py:L961-L973 窗口
            #   与 causal 位
            window = (
                attn_metadata.sliding_window
                if attn_metadata.sliding_window is not None
                else self.sliding_window
            )
            sliding_window_size: list[int] | None = (
                list(window) if window is not None else None
            )

            causal = attn_metadata.causal
            is_dynamic_causal = isinstance(causal, torch.Tensor)

            # SUBTRACTED: mm_prefix mask_mod 块（L974-L1007——多模态域）；
            #   R-SWA mask_mod 块（L1009-L1026——R-SWA 域 → ch21）。

            # SOURCE: vllm/v1/attention/backends/flash_attn.py:L1028-L1039
            #   per-sequence causal（FA4 动态 causal 张量）
            dynamic_causal = None
            if isinstance(causal, torch.Tensor):
                if self.vllm_flash_attn_version != 4:
                    raise NotImplementedError(
                        "Per-sequence causal requires FA4. Current version: "
                        f"FA{self.vllm_flash_attn_version}"
                    )
                dynamic_causal = causal
                has_window = (
                    sliding_window_size is not None and sliding_window_size[1] >= 0
                )
                causal = not has_window

            # SOURCE: vllm/v1/attention/backends/flash_attn.py:L1041-L1067 ——
            #   读腿本体：flash_attn_varlen_func(block_table=…) 穿表间接寻址
            #   读历史 KV（F7 回收点）
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                dynamic_causal=dynamic_causal,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
                # SUBTRACTED: mask_mod/aux_tensors 两参（rswa_mask_mod_fn or
                #   mm_mask_mod / rswa_aux or mm_aux——R-SWA/多模态分支的
                #   产物，其生产块已删）。
            )
            return output

        # SUBTRACTED: cascade attention 支（L1069-L1096——cascade 域 → ch21；
        #   本章 use_cascade 恒 False）。
        raise NotImplementedError("cascade attention → ch21 域")

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L1098 do_kv_cache_update
    #   —— 写腿落地（站13：do_kv_cache_update → reshape_and_cache_flash）
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L1106-L1109（encoder
        #   注意力不落池——判别位保留）
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L1111-L1114 K/V 页
        #   张量视图 + 散写
        # Scatter write into the KV cache using slot_mapping indices.
        # No TMA kernel is invoked here, so stride canonicalization is not needed.
        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L1116-L1132 ——
        #   woosuk NOTE 原文：key/value 可 padded 而 slot_mapping 不必 pad，
        #   op 以 slot_mapping 的形状决定 token 数
        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    # SUBTRACTED: _forward_with_dcp / _forward_encoder_attention（L1134 起）
    #   ——分布式 Part / encoder 域。


# ── HOST SEAM：vllm_flash_attn 两个 CUDA op 的 CPU 镜像 ────────────────────
# 真实导入位：vllm/v1/attention/backends/flash_attn.py:L37-L42
#   `from ...flash_attn_varlen import flash_attn_varlen_func, reshape_and_cache_flash`
# host 无 CUDA 库——两个 op 各以 kernel 本体的逐行 torch 镜像承载，签名与
# 语义逐字对齐（见 impl-notes §Seam 清单）。

# HOST SEAM —— reshape_and_cache_flash：csrc/libtorch_stable/cache_kernels.cu
#   L315-L342 kernel 本体的逐 token 镜像。
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    # SOURCE: csrc/libtorch_stable/cache_kernels.cu:L326-L331 PAD 消费端
    #   （'slot_idx can be -1 if the token is padded' → 该 token 线程直接
    #   return；循环镜像里等价于 continue）
    for token_idx in range(slot_mapping.shape[0]):
        slot_idx = int(slot_mapping[token_idx])
        # NOTE: slot_idx can be -1 if the token is padded
        if slot_idx < 0:
            continue  # HOST SEAM：kernel 的 per-thread return 之循环镜像
        # SOURCE: csrc/libtorch_stable/cache_kernels.cu:L332-L333 slot 逆分解
        #   （block_idx = slot//block_size、block_offset = slot%block_size）
        block_idx = slot_idx // key_cache.shape[1]
        block_offset = slot_idx % key_cache.shape[1]
        # SOURCE: csrc/libtorch_stable/cache_kernels.cu:L336-L344 源/目的行拷贝
        #   （host 镜像：fp8 缩放 "auto" 恒恒等；向量化拷贝退化为行赋值）
        key_cache[block_idx, block_offset] = key[token_idx]
        value_cache[block_idx, block_offset] = value[token_idx]


# HOST SEAM —— flash_attn_varlen_func：读腿 kernel 的精确数学镜像——每请求
#   穿 block_table 逐逻辑块 gather K/V（间接寻址），做 softmax(QK^T·scale)V
#   的 causal attention（ch20 已立的数学；flash varlen 的读侧语义）。
# SOURCE: vllm/v1/attention/backends/flash_attn.py:L1041 flash_attn_varlen_func 调用位（HOST SEAM 承载）
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
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
) -> torch.Tensor:
    window_ok = window_size is None or list(window_size) == [-1, -1]
    assert causal is True and window_ok and alibi_slopes is None, (
        "HOST SEAM 镜像只承载本章用面（causal=True、无窗口/alibi——"
        "(-1,-1) 即 FA 的「无窗口」约定）——滑窗/alibi/softcap 数学 → ch20/ch21"
    )
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
        # 穿表间接寻址：每个逻辑块号现场查物理块再取行（F7 的读侧内景）。
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
        # causal：query i 的绝对位置 = seq_len - q_len + i，只看 ≤ 该位置的键。
        # 逐 Q 头算（GQA 的 KV 头已在上方 repeat 到 Q 头维）。
        context_offset = seq_len - q_len
        head_dim = q_r.shape[-1]
        for i in range(q_len):
            key_len_i = context_offset + i + 1
            for h in range(num_heads):
                scores = (k_hist[:key_len_i, h] @ q_r[i, h]) * softmax_scale
                p = torch.softmax(scores.to(torch.float64), dim=-1).to(q.dtype)
                out[start + i, h * head_dim:(h + 1) * head_dim] = (
                    p @ v_hist[:key_len_i, h]
                )
    return out
