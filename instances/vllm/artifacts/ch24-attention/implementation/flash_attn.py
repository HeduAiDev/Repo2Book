"""ch24 精简版 — FlashAttention 后端四件套 + PagedAttention 读写（只做减法）。

对应 vllm/v1/attention/backends/flash_attn.py：
  - FlashAttentionBackend：声明 KV cache 逻辑 shape=(2,num_blocks,block_size,
    num_kv_heads,head_size) 与物理 stride_order(NHD/HND)，外加能力探针。
  - FlashAttentionMetadata：后端专属 dataclass（block_table/slot_mapping/scheduler_metadata/
    cascade 字段）。
  - FlashAttentionMetadataBuilder.build：从 CommonAttentionMetadata 解构共享字段、装配专属
    metadata（block_table=block_table_tensor、slot_mapping=slot_mapping）。
  - FlashAttentionImpl.forward：kv_cache.unbind(0) 拆 K/V，flash_attn_varlen_func 照
    block_table 读 paged KV（PagedAttention 读，f14）；do_kv_cache_update 用
    reshape_and_cache_flash 照 slot_mapping 写（PagedAttention 写，f14）。

两个真实 CUDA 算子（reshape_and_cache_flash / flash_attn_varlen_func）在 host 无法运行；本章
据 dossier 记录的可观察语义给出 CPU 等价实现（标 # SUBTRACTED 说明真实是 CUDA 内核），让精简版
能在 host 跑出与 vLLM 一致的 paged 读写数值。命名/控制流/签名与真实一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)


# ---- KV cache layout 全局（对应 vllm.v1.attention.backends.utils 的 set/get） ----
# SUBTRACTED: 真实 set_kv_cache_layout/get_kv_cache_layout 在
# vllm/v1/attention/backends/utils.py，带环境变量 VLLM_KV_CACHE_LAYOUT 默认值；本章用模块级
# 全局变量复现其「设一次、各后端读」的语义。
_KV_CACHE_LAYOUT = "NHD"


# SOURCE: vllm/v1/attention/backends/utils.py (set_kv_cache_layout)
def set_kv_cache_layout(layout: str) -> None:
    global _KV_CACHE_LAYOUT
    _KV_CACHE_LAYOUT = layout


# SOURCE: vllm/v1/attention/backends/utils.py (get_kv_cache_layout)
def get_kv_cache_layout() -> str:
    return _KV_CACHE_LAYOUT


# ============================================================================
# PagedAttention 的两个真实 CUDA 算子（host 用 CPU 等价实现复现可观察语义）
# ============================================================================

# SOURCE: vllm/_custom_ops.py:L2713 (reshape_and_cache_flash)
def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    # SUBTRACTED: 真实 wrapper 仅 `torch.ops._C_cache_ops.reshape_and_cache_flash(...)` 转发到
    # CUDA 内核（_custom_ops.py:L2723）。host 无该内核，本章据 dossier 记录的可观察语义给出等价
    # CPU 实现（PagedAttention『写』）：对每个 token i，slot=slot_mapping[i]，
    # block_idx=slot//block_size，block_offset=slot%block_size，把 key[i]/value[i]
    # （[num_kv_heads, head_size]）写进 key_cache[block_idx, block_offset]/
    # value_cache[...]。slot==-1 表示该 token 跳过（padding）。量化 k_scale/v_scale 在
    # auto/bf16 路径下为 no-op，故此处不做缩放（与非量化主路径一致）。
    block_size = key_cache.shape[1]
    num_tokens = slot_mapping.shape[0]
    for i in range(num_tokens):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        key_cache[block_idx, block_offset] = key[i]
        value_cache[block_idx, block_offset] = value[i]


# SOURCE: vllm/vllm_flash_attn (flash_attn_varlen_func) — backend 经此读 paged KV
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
    block_table: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # SUBTRACTED: 真实 flash_attn_varlen_func 是 vllm-flash-attn 的 CUDA kernel，含
    # alibi/sliding_window/sinks/scheduler_metadata/fa_version/q,k,v_descale/num_splits 等参数
    # （flash_attn.py:L795-L818）。host 无 kernel，本章据可观察语义给出 CPU 等价实现
    # （PagedAttention『读』）：对每请求 r，取该请求 query 段 [cu_seqlens_q[r]:
    # cu_seqlens_q[r+1]]，按 block_table[r] 列出的逻辑块号、seqused_k[r] 个 KV，从 paged
    # key_cache/value_cache 拼出连续的 K/V，做带 causal mask 的 GQA softmax 注意力，写回 out。
    # 这复现 vLLM『照 block_table 读历史 KV、照 seq_lens 决定读多少』的行为。
    num_blocks, block_size, num_kv_heads, head_size = k.shape
    num_query_heads = q.shape[1]
    num_reqs = cu_seqlens_q.shape[0] - 1
    queries_per_kv = num_query_heads // num_kv_heads

    for r in range(num_reqs):
        q_start = int(cu_seqlens_q[r].item())
        q_end = int(cu_seqlens_q[r + 1].item())
        q_len = q_end - q_start
        if q_len == 0:
            continue
        kv_len = int(seqused_k[r].item())

        # 照 block_table[r] 顺着逻辑块号把该请求的历史 KV 从 paged cache 读成连续张量。
        req_blocks = block_table[r]
        k_blocks = []
        v_blocks = []
        gathered = 0
        for blk in req_blocks:
            if gathered >= kv_len:
                break
            k_blocks.append(k[int(blk.item())])  # [block_size, num_kv_heads, head_size]
            v_blocks.append(v[int(blk.item())])
            gathered += block_size
        k_cont = torch.cat(k_blocks, dim=0)[:kv_len]   # [kv_len, num_kv_heads, head_size]
        v_cont = torch.cat(v_blocks, dim=0)[:kv_len]

        q_req = q[q_start:q_end]  # [q_len, num_query_heads, head_size]
        # GQA: 把 KV head 扩展到 query head 数。
        k_exp = k_cont.repeat_interleave(queries_per_kv, dim=1)  # [kv_len, H, d]
        v_exp = v_cont.repeat_interleave(queries_per_kv, dim=1)

        # scores: [H, q_len, kv_len]
        scores = torch.einsum("qhd,khd->hqk", q_req, k_exp) * softmax_scale
        if causal:
            # 第 i 个 query token 对应绝对位置 (kv_len - q_len + i)，只能看到 <= 该位置的 KV。
            ctx_len = kv_len - q_len
            qi = torch.arange(q_len).unsqueeze(1)
            ki = torch.arange(kv_len).unsqueeze(0)
            mask = ki > (ctx_len + qi)  # True = 屏蔽
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.einsum("hqk,khd->qhd", probs, v_exp)  # [q_len, H, d]
        out[q_start:q_end] = attn.to(out.dtype)

    return out


# ============================================================================
# FlashAttention 四件套
# ============================================================================

# SOURCE: vllm/v1/attention/backends/flash_attn.py:L66 (FlashAttentionBackend)
class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L74 (get_supported_kernel_block_sizes)
        # SUBTRACTED: 真实返回 [MultipleOf(16)]（混合 mamba float32 cache 时 [16,32,64]，
        # flash_attn.py:L74-L92）；本章用 16 表达「块大小须为 16 的倍数」这一 FA 约束。
        return [16]

    @staticmethod
    def get_name() -> str:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L102 (get_name)
        return "FLASH_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L106 (supports_batch_invariance)
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L110 (supports_non_causal)
        return True

    @classmethod
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L112 (supports_attn_type)
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @staticmethod
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L129 (get_impl_cls)
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L133 (get_builder_cls)
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L137 (get_kv_cache_shape)
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L148 (get_kv_cache_stride_order)
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        # SUBTRACTED: include_num_layers_dimension 的两个 6 维分支（把 num_layers 维算进物理
        # 布局，flash_attn.py:L153-L162）——本章不展开多层物理打包，删之不改 NHD/HND 主结论。
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @classmethod
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L181 (supports_head_size)
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        # SUBTRACTED: head_size>256 时真实问 is_fa_version_supported(4)（FA4 支持到 512，
        # flash_attn.py:L188-L190）；host 无 FA 版本探测，保守返回 False。
        return False

    @classmethod
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L208 (supports_compute_capability)
    def supports_compute_capability(cls, capability) -> bool:
        from platform_cuda import DeviceCapability

        return capability >= DeviceCapability(8, 0)


@dataclass
# SOURCE: vllm/v1/attention/backends/flash_attn.py:L222 (FlashAttentionMetadata)
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
    # 注意『翻译』：CommonAttentionMetadata.block_table_tensor 在这里改名 block_table、
    # slot_mapping 直接搬过来——共享字段直接搬入 + 后端特有字段新增。
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True

    # SUBTRACTED: DCP 字段 max_dcp_context_kv_len/dcp_context_kv_lens 与 aot 的
    # prefix_scheduler_metadata（flash_attn.py:L243-L252）——解码上下文并行/AOT 调度旁支，删之
    # 不影响标准 decoder 主路径。


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L308 (FlashAttentionMetadataBuilder)
class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    supports_update_block_table: bool = True

    def __init__(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L597 (FlashAttentionImpl.__init__)
        self,
        kv_cache_spec=None,
        layer_names=None,
        vllm_config=None,
        device=None,
    ):
        # SUBTRACTED: 真实 __init__ 从 vllm_config 解出 model_config/parallel_config/
        # compilation_config、算 num_heads_q/kv、决定 aot_schedule(FA3) 与
        # full-cuda-graph 的 scheduler_metadata 预分配、DCP 缓冲（flash_attn.py:L310-L386）。
        # host 无完整 VllmConfig，这些都是性能/并行旁支；精简版令 build 走默认调度
        # （scheduler_metadata=None），主链数值不变。
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.dcp_world_size = 1
        self.aot_schedule = False

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L388 (build)
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        # 开场：从 CommonAttentionMetadata 解构出跨层共享字段。
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # SUBTRACTED: build 中段的 AOT scheduler 闭包（schedule()→get_scheduler_metadata）、
        # DCP 与 cascade 三条分支、full-cudagraph 的 scheduler_metadata 拷贝
        # （flash_attn.py:L408-L555）。标准 decoder 走 use_cascade=False、dcp_world_size==1、
        # scheduler_metadata=None 的主分支即可端到端跑通。
        use_cascade = False
        scheduler_metadata = None
        max_num_splits = 0

        # 落点：把共享字段原样搬入（block_table=block_table_tensor、slot_mapping=slot_mapping），
        # 其余为 FA 特有字段。这就是『Common → 后端专属 metadata』的翻译。
        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            scheduler_metadata=scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L594 (FlashAttentionImpl)
class FlashAttentionImpl(AttentionImpl[FlashAttentionMetadata]):
    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L597 (__init__)
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        # 真实 AttentionImplBase 用 __new__ 设 dcp/pcp 通信组大小（见 backend.py）；这里直接
        # 调基类 __init__ 设 dcp_world_size=1（host 非分布式），不经抽象 AttentionImpl.__init__。
        from backend import AttentionImplBase

        AttentionImplBase.__init__(self)
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        # SUBTRACTED: alibi/sliding_window 转 tuple、FA 版本探测(get_flash_attn_version)、
        # FP8 quant cache 校验、sinks 校验、DCP a2a 通信 dtype（flash_attn.py:L611-L678）——
        # 都是可选特性/并行/版本路径，本章非量化、非 sliding-window、单卡主路径不依赖它们。
        self.sliding_window = (-1, -1)
        self.alibi_slopes = None
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap if logits_soft_cap is not None else 0
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        self.sinks = None
        self.supports_quant_query_input = False

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L682 (forward)
    def forward(
        self,
        layer,
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
                [2, num_blocks, block_size, num_kv_heads, head_size]
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # SUBTRACTED: encoder-only 分支(_forward_encoder_attention)、FP8 量化 cache 的 view、
        # descale 构造、DCP 分支(_forward_with_dcp)（flash_attn.py:L735-L789）——
        # 都是特例/量化/并行路径，标准 decoder 不走，删之不改主路径数值。

        # ① kv_cache.unbind(0) 把头维 2 拆成 key_cache 与 value_cache。
        key_cache, value_cache = kv_cache.unbind(0)

        # SUBTRACTED: cascade 分支（cascade_attention，flash_attn.py:L822-L848）——共享前缀加速
        # 特例；标准路径恒 use_cascade=False。
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        scheduler_metadata = attn_metadata.scheduler_metadata

        # ② flash_attn_varlen_func 拿 block_table（每请求逻辑块号表）去 paged KV cache 按块读
        # K/V——这就是 PagedAttention 的『读』那一半（f14 回收）。
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
            causal=attn_metadata.causal,
            block_table=block_table,
            scheduler_metadata=scheduler_metadata,
        )
        return output

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L851 (do_kv_cache_update)
    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention, we use direct Q, K, V without caching.
            return

        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. The reshape_and_cache_flash op uses the slot_mapping's
        # shape to determine the number of actual tokens.
        # 这是 PagedAttention 的『写』那一半（f14 回收）。
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            getattr(layer, "_k_scale", None),
            getattr(layer, "_v_scale", None),
        )
