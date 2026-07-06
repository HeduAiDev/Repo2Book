"""ch25 companion — DeepSeek-V4 MLA 执行层（只做减法）。

对应 vllm/model_executor/layers/deepseek_v4_attention.py。本章只交付 MLA 在「模型侧」可读的
结构与算子边界：
  - DeepseekV4MultiHeadLatentAttentionWrapper.forward：调 deepseek_v4_attention 自定义算子，
    输出端走 inverse-RoPE/FP8 einsum(wo_a 低秩) + wo_b 两段投影。
  - attn_gemm_parallel_execute：多 CUDA stream 并行输入 GEMM——fused_wqa_wkv(最重)走默认流，
    compressor/indexer 三个轻 GEMM 走 aux stream，用 execute_in_parallel + event 重叠
    （对照 ch22 Llama 单 qkv_proj 一把过）。
  - attention_impl：split([q_lora_rank, head_dim]) → fused_q_kv_rmsnorm（对 q/kv 低秩潜变量
    分别归一，对应标准 MLA 的 q_a_layernorm/kv_a_layernorm）。

注意力内核本身（deepseek_v4_attention 算子 / 稀疏 SWA / FlashMLA / kv_insert）下放 ch24；
本文件保留它们的调用边界。GPU-only 算子（torch.ops.vllm.* / fp8 量化 / fused rmsnorm）保留
调用点，内核实现见对应专章。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from ._runtime import (
    QuantizationConfig,
    get_tensor_model_parallel_world_size,
)

# SUBTRACTED: SparseAttnIndexer / DeepseekCompressor / DeepseekV4MLAModules 等稀疏注意力组件
#             与 v1 attention backend 的导入——稀疏 SWA/indexer/compressor/FlashMLA 内核下放
#             ch24（dossier 批准删）。本文件保留它们在 attn_gemm_parallel_execute 里作为
#             aux-stream 输入 GEMM 的「存在性分支」，不展开实现。


# SOURCE: vllm/utils/multi_stream_utils.py:61 (execute_in_parallel)
def execute_in_parallel(default_fn, aux_fns, start_event, done_events, aux_streams):
    # SUBTRACTED: 真实在多条 CUDA stream 上用 event fan-out/fan-in 重叠执行（GPU-only）；
    #             精简版顺序执行以保留「默认流 + N 路 aux」的接口契约与返回结构。
    main_out = default_fn()
    aux_out = [fn() if fn is not None else None for fn in aux_fns]
    return main_out, tuple(aux_out)


# SOURCE: vllm/v1/attention/ops/deepseek_v4_ops.py (fused_q_kv_rmsnorm)
def fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
    # SUBTRACTED: 真实是把 q 段(q_lora_rank)与 kv 段(head_dim)的 RMSNorm 融成一个 CUDA 算子；
    #             精简版用两次 native RMSNorm，语义等价（对应标准 MLA q_a_layernorm/kv_a_layernorm）。
    def _rmsnorm(x, w):
        # SOURCE: vllm/model_executor/models/deepseek_v4.py (_rmsnorm)
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (x.to(dt)) * w
    return _rmsnorm(qr, q_weight), _rmsnorm(kv, kv_weight)


# SOURCE: vllm/v1/attention/ops/deepseek_v4_ops.py (fused_inv_rope_fp8_quant)
def fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, *, n_groups, heads_per_group,
                             nope_dim, rope_dim, tma_aligned_scales):
    # SUBTRACTED: 真实是输出端 inverse-RoPE + FP8 量化的 fused 算子（GPU-only，ch24）；
    #             精简版保留调用边界并返回 (o_fp8, o_scale) 占位形状。
    raise NotImplementedError(
        "fused_inv_rope_fp8_quant 内核下放 ch24；本章仅保留输出端 O 投影的调用边界。"
    )


# SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:113 (class DeepseekV4MultiHeadLatentAttentionWrapper)
class DeepseekV4MultiHeadLatentAttentionWrapper(nn.Module):
    """MLA 执行层（V4 专属）：持有 fused_wqa_wkv/q_norm/wq_b/kv_norm/wo_a/wo_b/attn_sink/rope，
    forward 调 deepseek_v4_attention 自定义算子 + 输出端低秩两段投影；attn_gemm_parallel_execute
    做多 stream 并行输入 GEMM。是「MLA 低秩 + 多流 GEMM」性能 delta 的载体。

    # SUBTRACTED: 真实继承 PluggableLayer（OOT 后端可替换整层）；精简版用 nn.Module。
    # SUBTRACTED: 真实经 DeepseekV4MLAModules dataclass 传投影模块；精简版直接收平铺 kwargs，
    #             去掉容器样板但保留每个投影/归一/sink 字段（must_keep 符号全部在）。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        fused_wqa_wkv: nn.Module,
        q_norm: nn.Module,
        wq_b: nn.Module,
        kv_norm: nn.Module,
        wo_a: nn.Module,
        wo_b: nn.Module,
        attn_sink: torch.Tensor,
        rotary_emb,
        indexer,
        aux_stream_list: list | None = None,
        compress_ratio: int | None = None,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:132 (__init__)
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # FlashMLA 稀疏内核只支持 64 或 128 头；pad 到下一个支持值。
        if num_heads <= 64:
            self.padded_heads = 64
        elif num_heads <= 128:
            self.padded_heads = 128
        else:
            raise ValueError(
                f"DeepseekV4 attention does not support {num_heads} heads (must be <= 128)."
            )

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix
        self.layer_name = prefix

        tp_size = get_tensor_model_parallel_world_size()
        self.eps = None  # 由下方从 q_norm/kv_norm 同源（精简版未持 config，用注释表意）
        self.rope_head_dim = qk_rope_head_dim
        self.nope_head_dim = head_dim - qk_rope_head_dim
        self.o_lora_rank = o_lora_rank

        # 存投影/归一/sink 模块（must_keep：fused_wqa_wkv/q_norm/wq_b/kv_norm/wo_a/wo_b/attn_sink）。
        self.fused_wqa_wkv = fused_wqa_wkv
        self.q_norm = q_norm
        self.wq_b = wq_b
        self.kv_norm = kv_norm
        self.wo_a = wo_a
        self.wo_b = wo_b
        self.attn_sink = attn_sink
        self.rotary_emb = rotary_emb
        self.indexer = indexer
        self.aux_stream_list = aux_stream_list

        # SUBTRACTED: compressor（稀疏 SWA 压缩器）下放 ch24；dense 路径恒为 None。
        self.compressor = None
        # SUBTRACTED: QuantFP8 wo_a 激活量化器、fp8_einsum recipe/TMA 对齐的 GPU 能力分支
        #             （SM90/SM100 不同 sfb_gran）按 dossier 下放 ch24；保留输出端调用边界。
        self.n_local_groups = None  # 真实从 config.o_groups // tp_size 得，输出端 einsum 用

        # SUBTRACTED: ln_events（多 stream 重叠用的 CUDA event 列表，GPU-only）实例化下放 ch24。
        self.ln_events = None

    def forward(self, positions, hidden_states, llama_4_scaling=None):
        # SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:287 (forward)
        # 预分配 FlashMLA-padded 头数的输出；算子写入 o_padded，再切回 n_local_heads。
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        # 注意力主体在自定义算子里（torch.compile 边界）。内核下放 ch24。
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states, positions, o_padded, self.layer_name,
        )
        o = o_padded[:, : self.n_local_heads, :]

        # O 投影：inverse-RoPE + FP8 量化 → deepseek_v4_fp8_einsum(wo_a 低秩) → wo_b 回 hidden。
        # 这是「连输出投影都低秩」的 V4 特征（区别于标准 MLA 直接 o_proj）。
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o, positions, self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // (self.n_local_groups or 1),
            nope_dim=self.nope_head_dim, rope_dim=self.rope_head_dim,
            tma_aligned_scales=False,
        )
        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = getattr(self.wo_a, "weight_scale_inv", None)
        z = torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device, dtype=torch.bfloat16,
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8, o_scale, wo_a_fp8, wo_a_scale, z, "bhr,hdr->bhd", [],
        )
        return self.wo_b(z.flatten(1))

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        # SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:356 (attn_gemm_parallel_execute)
        # 多 stream GEMM 并行：fused_wqa_wkv(最重)走默认流；compressor/indexer 三个轻 GEMM 走
        # aux stream 0..2（仅当其 owning module 存在）。这是 V4 模型侧的工程亮点 delta。
        assert self.aux_stream_list is not None
        assert len(self.aux_stream_list) >= 3

        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.compressor is not None:
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                # SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:113 (compressor_kv_score)
                return torch.mm(
                    hidden_states, compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:113 (indexer_weights_proj)
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                # SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:113 (indexer_compressor_kv_score)
                return torch.mm(
                    hidden_states, indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )
            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            # SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:113 (fused_wqa_wkv)
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv, aux_fns,
            self.ln_events[0] if self.ln_events else None,
            self.ln_events[1:4] if self.ln_events else None,
            self.aux_stream_list[:3],
        )
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def attention_impl(self, hidden_states, positions, out) -> None:
        # SOURCE: vllm/model_executor/layers/deepseek_v4_attention.py:416 (attention_impl)
        # MLA 前处理 delta：低秩 latent split 成 q 段(q_lora_rank)与 kv 段(head_dim)，分别 fused
        # RMSNorm（对应标准 MLA 的 q_a_layernorm/kv_a_layernorm，V4 融成一个算子）。
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr, kv, self.q_norm.weight.data, self.kv_norm.weight.data, self.eps,
        )
        # SUBTRACTED: 此后 wq_b 升 q + 解耦 RoPE(只旋 rope 段) + kv_insert + FlashMLA 内核
        #             下放 ch24（dossier 批准删）。本章保留低秩 split + fused RMSNorm 的前处理骨架。
        raise NotImplementedError(
            "wq_b 升维 / 解耦 RoPE / kv_insert / FlashMLA 内核下放 ch24。"
        )
