# SOURCE: vllm/model_executor/layers/mla.py —— ch25 主文件 1（站 3/9）
# 全文减法：MLAModules（L14-L31 逐字）+ MultiHeadLatentAttentionWrapper
#（@PluggableLayer.register 位 + __init__ L55-L148 / forward L150-L226）。
# SUBTRACTED（挂 dossier.subtraction_plan.delete 批准项）：
#   delete[0]：dcp_q_replicate 分支（L103-L104/L196-L197/L211-L213）
#   delete[3]：indexer 调用位（L205-L206）与 topk 布线（L105-L108）、
#     indexer_op 的 dense_mha 绑定块（L128-L147）——稀疏 MLA 归 ch26
# 章节 elide：llama_4_scaling 分支（L208-L209）——章界外（Llama4 面）
from dataclasses import dataclass

import torch

from vllm.config import CacheConfig
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.platforms import current_platform  # noqa: F401  (indexer 绑定块的消费位已随 delete[3] 删)


# SOURCE: vllm/model_executor/layers/mla.py:L14-L31 MLAModules —— 逐字
#   （must_keep：装配契约——模型侧打包投影交 Wrapper 的数据结构）
@dataclass
class MLAModules:
    # SOURCE: vllm/model_executor/layers/mla.py:L14-L31（锚点双置：声明上方注释同文）
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None
    g_proj: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
# SOURCE: vllm/model_executor/layers/mla.py:L35 @PluggableLayer.register 位
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    # SOURCE: vllm/model_executor/layers/mla.py:L55-L148 __init__ —— 减法子集
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config=None,
        prefix: str = "",
        skip_topk: bool = False,
        non_causal_multi_token_decode: bool = False,
        allow_short_prefill_indexer_scoring_skip: bool = False,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/mla.py:L73-L94 属性面 —— 逐字
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.g_proj = mla_modules.g_proj

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        # SUBTRACTED: dcp_q_replicate 探测（mla.py:L101-L104
        #   q_proj_layer.qrep_active）——delete[0]（单进程 DCP=1 恒 False）
        self.dcp_q_replicate = False
        # SUBTRACTED: topk 缓冲布线（mla.py:L105-L108）——delete[3]
        #   （topk 相关字段；稀疏 MLA 归 ch26）

        # SOURCE: vllm/model_executor/layers/mla.py:L110-L127 MLAAttention
        #   插座构造 —— 逐字（kv_b_proj 引用一并交接——吸收重排要靠它）
        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            dcp_q_replicate=self.dcp_q_replicate,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=mla_modules.topk_indices_buffer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )
        # SUBTRACTED: indexer_op 的 dense_mha_metadata_layer_name 绑定块
        #   （mla.py:L128-L147）——delete[3]（稀疏 MLA 归 ch26）
        self.prefix = prefix

    # SOURCE: vllm/model_executor/layers/mla.py:L150-L226 forward —— 减法
    #   子集（must_keep：外层主角——低秩链 + 解耦 RoPE + o_proj 全程）
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/mla.py:L156-L187 低秩链双支 —— 逐字
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q_proj_layer = self.q_b_proj
            q_proj_input = q_c
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q_proj_layer = self.q_proj
            q_proj_input = hidden_states

        # SOURCE: vllm/model_executor/layers/mla.py:L189-L192 KV 压缩与解耦
        #   —— 逐字（split [kv_lora_rank, rope]；kv_a_layernorm 只打 kv_c）
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        # SOURCE: vllm/model_executor/layers/mla.py:L194-L198 q 上投影 —— 逐字
        q = q_proj_layer(q_proj_input)[0]
        heads = self.num_heads
        # SUBTRACTED: dcp_q_replicate 的 heads *= group_size（L196-L197）
        #   ——delete[0]
        q = q.view(-1, heads, self.qk_head_dim)

        # SOURCE: vllm/model_executor/layers/mla.py:L200-L203 RoPE 只作用
        #   rope 段 —— 逐字（q[..., nope:] 与 k_pe；nope 段不旋转——低秩
        #   可吸收的数学前提，m02 的代码化身）
        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        # SUBTRACTED: indexer 调用位（mla.py:L205-L206）——delete[3]
        #   （if self.indexer and self.is_sparse and not self.skip_topk:
        #     self.indexer(hidden_states, q_c, positions, ...)——ch26）
        # SUBTRACTED: llama_4_scaling 分支（mla.py:L208-L209）——章界外
        #   （Llama4 面）
        # SUBTRACTED: dcp_q_replicate 的 q 本地视图段（mla.py:L211-L213）
        #   ——delete[0]
        q_dcp_replicated = None

        # SOURCE: vllm/model_executor/layers/mla.py:L215-L221 交内层插座
        #   —— 逐字
        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
            q_dcp_replicated=q_dcp_replicated,
        )

        # SOURCE: vllm/model_executor/layers/mla.py:L223-L226 o_proj 收尾
        #   —— 逐字（g_proj 门控分支保留位：None 时直通）
        if self.g_proj is not None:
            attn_out = attn_out * self.g_proj(hidden_states)[0].sigmoid()

        return self.o_proj(attn_out)[0]
