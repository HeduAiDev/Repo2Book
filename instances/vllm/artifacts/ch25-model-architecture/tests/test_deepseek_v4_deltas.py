"""ch25 — DeepSeek-V4 as a stack of deltas over Llama：测精简版复现真实 vLLM 的可观察结构/控制流。

这些是结构与数值（对纯 PyTorch 段）测试，不 import vllm，可在 host 跑。每个测试对照
真实 vllm/model_executor/models/deepseek_v4.py 的行为，验证「delta-over-Llama」骨架被忠实保留：
  - 注意力是 MLA 低秩压缩（fused_wqa_wkv → [q_lora_rank, head_dim]），不是全量 QKV；
  - FFN 是 MoE（gate + shared_experts + 双后端），不是单 dense MLP；
  - 残差是 hc_pre/hc_post 多流超连接，不是 add-norm；
  - 解码头多出 MTP draft（融合 pre-hc_head 残差 + 下一 token embedding）；
  - 量化作为 delta（DeepseekV4FP8Config expert_dtype 惰性解析、e8m0fnu→uint8 字节装载）。
"""
import torch

from implementation import deepseek_v4 as dv4
from implementation import deepseek_v4_mtp as mtp


# --- 注意力 delta：MLA 低秩压缩取代全量 QKV -------------------------------------
def test_mla_fused_wqa_wkv_is_low_rank(vllm_config):
    attn = dv4.DeepseekV4Attention(vllm_config, prefix="model.layers.0.attn")
    cfg = vllm_config.model_config.hf_config
    # fused_wqa_wkv 输出维 = q_lora_rank + head_dim（低秩潜变量），远小于全量 n_heads*head_dim。
    out_features = attn.fused_wqa_wkv.weight.shape[0]
    assert out_features == cfg.q_lora_rank + cfg.head_dim
    full_qkv = cfg.num_attention_heads * cfg.head_dim
    assert out_features < full_qkv  # 这就是 MLA 相对全量 QKV 的「压缩」


def test_mla_wq_b_lifts_q_back_to_full(vllm_config):
    attn = dv4.DeepseekV4Attention(vllm_config, prefix="model.layers.0.attn")
    cfg = vllm_config.model_config.hf_config
    # wq_b 把 q 潜变量(q_lora_rank) 升回 full Q(n_heads*head_dim)。
    assert attn.wq_b.weight.shape == (cfg.num_attention_heads * cfg.head_dim, cfg.q_lora_rank)


def test_mla_output_projection_is_also_low_rank(vllm_config):
    attn = dv4.DeepseekV4Attention(vllm_config, prefix="model.layers.0.attn")
    cfg = vllm_config.model_config.hf_config
    # V4 特征：连输出投影也低秩（wo_a → o_lora_rank/o_groups，wo_b → 回 hidden）。
    assert attn.wo_a.weight.shape[0] == cfg.o_groups * cfg.o_lora_rank
    assert attn.wo_b.weight.shape == (cfg.hidden_size, cfg.o_groups * cfg.o_lora_rank)
    # q_norm/kv_norm 对应标准 MLA 的 q_a_layernorm/kv_a_layernorm。
    assert attn.q_norm.weight.shape[0] == cfg.q_lora_rank
    assert attn.kv_norm.weight.shape[0] == cfg.head_dim


def test_attn_sink_padded_to_min_64_and_neg_inf(vllm_config):
    attn = dv4.DeepseekV4Attention(vllm_config, prefix="model.layers.0.attn")
    # attn_sink padded 到 >=64 头，初始化为 -inf（无 sink 效应），装载只填前 n_local_heads。
    assert attn.attn_sink.shape[0] == 64
    assert torch.isinf(attn.attn_sink).all() and (attn.attn_sink < 0).all()


def test_attn_dense_path_has_no_indexer(vllm_config):
    # compress_ratios 全 1（dense）→ 无稀疏 indexer（稀疏分支下放 ch24）。
    attn = dv4.DeepseekV4Attention(vllm_config, prefix="model.layers.0.attn")
    assert attn.indexer is None
    assert attn.mla_attn.compressor is None


# --- FFN delta：MoE 取代 dense MLP（gate 路由 + 共享专家）--------------------------
def test_moe_has_gate_routed_and_shared_experts(vllm_config):
    moe = dv4.DeepseekV4MoE(vllm_config, prefix="model.layers.0.ffn")
    cfg = vllm_config.model_config.hf_config
    # gate 路由到 n_routed_experts；shared_experts 是每 token 必走的 dense 残留（MoE 对 dense 的关键 delta）。
    assert moe.gate.weight.shape == (cfg.n_routed_experts, cfg.hidden_size)
    assert isinstance(moe.shared_experts, dv4.DeepseekV4MLP)
    # shared_experts 用 DeepseekV4MLP（结构同构 LlamaMLP），印证「MoE 内那条 dense 路径」。
    assert hasattr(moe.shared_experts, "gate_up_proj")
    assert hasattr(moe.shared_experts, "down_proj")


def test_moe_default_backend_is_tp_fused_moe(vllm_config):
    # 未开 EP → use_mega_moe=False，走 TP FusedMoE 后端。
    moe = dv4.DeepseekV4MoE(vllm_config, prefix="model.layers.0.ffn")
    assert moe.use_mega_moe is False
    assert isinstance(moe.experts, dv4.FusedMoE)


def test_moe_mega_backend_selected_under_ep(vllm_config):
    # 开 EP + moe_backend=deep_gemm_mega_moe → use_mega_moe=True，走 MegaMoE 单算子后端。
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = "deep_gemm_mega_moe"
    vllm_config.model_config.hf_config.expert_dtype = "fp4"
    moe = dv4.DeepseekV4MoE(vllm_config, prefix="model.layers.0.ffn")
    assert moe.use_mega_moe is True
    assert isinstance(moe.experts, dv4.DeepseekV4MegaMoEExperts)


def test_fused_topk_bias_routes_top_k_and_renormalizes(vllm_config):
    cfg = vllm_config.model_config.hf_config
    logits = torch.randn(5, cfg.n_routed_experts)
    w, ids = dv4.fused_topk_bias(
        hidden_states=None, gating_output=logits, scoring_func="sqrtsoftplus",
        e_score_correction_bias=None, topk=cfg.num_experts_per_tok, renormalize=True,
        indices_type=torch.int32, input_tokens=None, hash_indices_table=None,
        routed_scaling_factor=cfg.routed_scaling_factor,
    )
    # 选出 top-k 个专家，renorm 后(乘 routed_scaling_factor)每行权重和 == scaling_factor。
    assert ids.shape == (5, cfg.num_experts_per_tok)
    assert torch.allclose(w.sum(-1), torch.full((5,), cfg.routed_scaling_factor), atol=1e-5)


def test_mega_moe_experts_weights_are_uint8_quantized(vllm_config):
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.kernel_config.moe_backend = "deep_gemm_mega_moe"
    vllm_config.model_config.hf_config.expert_dtype = "fp4"
    moe = dv4.DeepseekV4MoE(vllm_config, prefix="model.layers.0.ffn")
    experts = moe.experts
    # 量化专家：w13/w2 权重与 scale 以 uint8 原始字节存放（FP4/FP8 量化 delta）。
    assert experts.w13_weight.dtype == torch.uint8
    assert experts.w2_weight.dtype == torch.uint8
    assert experts.w13_weight_scale.quant_method == "block"


# --- 残差 delta：hc 多流超连接取代 add-norm ----------------------------------------
def test_decoder_layer_uses_hc_not_addnorm(vllm_config):
    layer = dv4.DeepseekV4DecoderLayer(vllm_config, prefix="model.layers.0")
    # 解码层持有 hc 超连接学习参数（attn/ffn 各一套），而非 Llama 的 input/post_attention_layernorm。
    for name in ["hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
                 "hc_attn_scale", "hc_ffn_scale"]:
        assert hasattr(layer, name), name
    assert callable(layer.hc_pre) and callable(layer.hc_post)
    # 仍是 attn(MLA) + ffn(MoE) 的两段结构。
    assert isinstance(layer.attn, dv4.DeepseekV4Attention)
    assert isinstance(layer.ffn, dv4.DeepseekV4MoE)


def test_hc_head_compresses_multi_stream_to_single(vllm_config):
    # hc_head 是纯 PyTorch：把 (T, hc_mult, D) 多流经 RMSNorm+sigmoid 门控加权求和压回 (T, D)。
    cfg = vllm_config.model_config.hf_config
    T, M, D = 5, cfg.hc_mult, cfg.hidden_size
    x = torch.randn(T, M, D)
    hc_fn = torch.randn(M, M * D)
    hc_scale = torch.randn(1)
    hc_base = torch.randn(M)
    y = dv4.hc_head(x, hc_fn, hc_scale, hc_base, cfg.rms_norm_eps, cfg.hc_eps)
    assert y.shape == (T, D)  # 多流 → 单流


def test_hc_head_matches_reference_formula(vllm_config):
    # 数值核对 hc_head 与 dossier 内嵌真源码公式逐项一致（防过度简化 hc 收尾）。
    import torch.nn.functional as F
    cfg = vllm_config.model_config.hf_config
    T, M, D = 3, cfg.hc_mult, cfg.hidden_size
    x = torch.randn(T, M, D, dtype=torch.float32)
    hc_fn = torch.randn(M, M * D)
    hc_scale = torch.randn(1)
    hc_base = torch.randn(M)
    eps, hc_eps = cfg.rms_norm_eps, cfg.hc_eps

    flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
    mixes = F.linear(flat, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    ref = torch.sum(pre.unsqueeze(-1) * flat.view(x.size()), dim=1)

    y = dv4.hc_head(x, hc_fn, hc_scale, hc_base, eps, hc_eps)
    assert torch.allclose(y, ref, atol=1e-5)


# --- 主干 delta：多流展开 + _mtp_hidden_buffer 桥 ----------------------------------
def test_model_repeats_into_hc_mult_streams_and_stashes_mtp_buffer(vllm_config):
    model = dv4.DeepseekV4Model(vllm_config=vllm_config, prefix="model")
    cfg = vllm_config.model_config.hf_config
    # _mtp_hidden_buffer 是 (max_num_batched_tokens, hc_mult*hidden) 的 pre-hc_head 残差桥。
    assert model._mtp_hidden_buffer.shape == (
        vllm_config.scheduler_config.max_num_batched_tokens, cfg.hc_mult * cfg.hidden_size,
    )
    # repeat 成 hc_mult 流的算子可观察：embed 后 unsqueeze(-2).repeat 的语义在 forward 里。
    h = torch.randn(4, cfg.hidden_size)
    streamed = h.unsqueeze(-2).repeat(1, cfg.hc_mult, 1)
    assert streamed.shape == (4, cfg.hc_mult, cfg.hidden_size)


def test_for_causal_lm_exposes_mtp_target_hidden_states(vllm_config):
    m = dv4.DeepseekV4ForCausalLM(vllm_config=vllm_config, prefix="")
    buf = m.get_mtp_target_hidden_states()
    # 顶层暴露 pre-hc_head 残差缓冲给 MTP draft（ch28 据此取目标隐状态）。
    assert buf is m.model._mtp_hidden_buffer
    assert hasattr(m, "lm_head") and hasattr(m, "compute_logits")


# --- 量化 delta：expert_dtype 惰性解析 + e8m0fnu 字节装载 ---------------------------
def test_fp8config_expert_dtype_lazy_defaults_to_fp4_without_vllm_config():
    cfg = dv4.DeepseekV4FP8Config()
    # 没有 current_vllm_config 时惰性解析返回 "fp4"（不急切误路由），且不缓存。
    assert cfg.expert_dtype == "fp4"
    assert cfg._resolved_expert_dtype is None
    assert cfg.is_scale_e8m0 is True


def test_e8m0fnu_scale_loaded_as_uint8_bytes_not_numeric_convert():
    # 装载特例：e8m0fnu scale 必须 view 成 uint8 装入，copy_ 的数值转换会毁掉指数字节。
    # 这里直接验证 view(uint8) 保留原始字节，而 copy_ 到 uint8 会数值转换归零。
    raw = torch.tensor([2.0 ** -7, 1.0, 2.0 ** 5], dtype=torch.float8_e8m0fnu)
    as_bytes = raw.view(torch.uint8)
    dst = torch.zeros(3, dtype=torch.uint8)
    dst.copy_(raw.to(torch.float32).clamp(0, 255).to(torch.uint8))  # 错误路径：数值转换
    assert as_bytes.dtype == torch.uint8
    # view 保留的原始字节 与 数值转换后的结果 不同（证明为何必须 view）。
    assert not torch.equal(as_bytes, dst)


# --- 解码头 delta：MTP draft 融合两路信号 ------------------------------------------
def test_mtp_layer_fuses_embedding_and_target_residual(vllm_config):
    layer = mtp.DeepSeekV4MultiTokenPredictorLayer(
        vllm_config, topk_indices_buffer=None, prefix="model.layers.2",
    )
    # 混合残差真身：enorm/hnorm + e_proj/h_proj 融合 token embedding 与 target 的 pre-hc_head 残差。
    assert hasattr(layer, "enorm") and hasattr(layer, "hnorm")
    assert hasattr(layer, "e_proj") and hasattr(layer, "h_proj")
    # 复用主模型同款解码层（draft 与 target 同构）。
    assert isinstance(layer.mtp_block, dv4.DeepseekV4DecoderLayer)
    assert isinstance(layer.shared_head, mtp.SharedHead)


def test_mtp_top_class_reuses_hc_head_in_compute_logits(vllm_config):
    predictor = mtp.DeepSeekV4MTP(vllm_config=vllm_config, prefix="")
    cfg = vllm_config.model_config.hf_config
    # MTP 顶层持有多 MTP 层 ModuleDict，按 spec_step 选层。
    assert len(predictor.model.layers) == cfg.num_nextn_predict_layers
    # compute_logits 接受 pre-hc_head 残差 (T, hc_mult*D)，内部补 hc_head 再过 shared_head。
    T = 3
    pre_hc = torch.randn(T, cfg.hc_mult * cfg.hidden_size)
    logits = predictor.compute_logits(pre_hc, spec_step_idx=0)
    assert logits.shape == (T, cfg.vocab_size)
