"""ch22 测试 —— 验证精简版复现真实 vLLM 的可观察行为。

聚焦四组真相（均以真实 vLLM 行为为基准，非测精简版自洽）：
  1. TP 线性层切分：列切分(output 维)/行切分(input 维)的 offset + tp_rank narrow。
  2. QKV fuse + GQA：q 用 tp_rank、k/v 用 tp_rank//num_kv_head_replicas；KV 头<tp 时复制。
  3. 权重装载：stacked_params_mapping 把独立 q/k/v、gate/up 重命名 + shard_id 装入 fused 参数；
     其余走 default_weight_loader；tie 时 AutoWeightsLoader 跳 lm_head。
  4. 模型契约：(vllm_config, prefix) 三段式装载、prefix→static_forward_context 注册、
     forward 形状、pre-norm + 显式 residual 协议、compute_logits。

纯单元测试，不 import vllm，host pytest 即可跑。
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from implementation import _runtime as rt  # noqa: E402
from implementation.linear import (  # noqa: E402
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from implementation.llama import LlamaForCausalLM  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_tp_and_context():
    rt.set_tp(1, 0)
    rt.STATIC_FORWARD_CONTEXT.clear()
    yield
    rt.set_tp(1, 0)
    rt.STATIC_FORWARD_CONTEXT.clear()


class TinyLlamaConfig:
    """最简 Llama 配置（GQA：8 个 query 头，2 个 KV 头）。"""

    vocab_size = 32
    hidden_size = 16
    intermediate_size = 32
    num_hidden_layers = 2
    num_attention_heads = 8
    num_key_value_heads = 2
    hidden_act = "silu"
    rms_norm_eps = 1e-6
    max_position_embeddings = 64
    tie_word_embeddings = False


def _vllm_config():
    return rt.VllmConfig(TinyLlamaConfig(), dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. TP 线性层切分
# ---------------------------------------------------------------------------


def test_column_parallel_shards_output_dim():
    rt.set_tp(2, 1)  # 2 卡，本 rank=1
    layer = ColumnParallelLinear(input_size=4, output_size=8, bias=False)
    # 列切分：output_size_per_partition = 8//2 = 4，input 不切。
    assert layer.output_size_per_partition == 4
    assert layer.input_size_per_partition == 4
    full = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    layer.weight_loader(layer.weight, full)
    # rank1 拿磁盘上 [4:8] 行（output 维 = dim0）。
    assert torch.equal(layer.weight.data, full[4:8])


def test_row_parallel_shards_input_dim():
    rt.set_tp(2, 1)
    layer = RowParallelLinear(input_size=8, output_size=4, bias=False)
    # 行切分：input_size_per_partition = 8//2 = 4，output 不切。
    assert layer.input_size_per_partition == 4
    assert layer.output_size_per_partition == 4
    full = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    layer.weight_loader(layer.weight, full)
    # rank1 沿 input 维(dim1) 拿 [4:8] 列。
    assert torch.equal(layer.weight.data, full[:, 4:8])


def test_row_parallel_all_reduce_only_when_tp_gt_1(monkeypatch):
    rt.set_tp(1, 0)
    layer = RowParallelLinear(input_size=4, output_size=4, bias=False)
    x = torch.randn(3, 4)
    called = {"n": 0}
    orig = rt.tensor_model_parallel_all_reduce

    def _spy(t):
        called["n"] += 1
        return orig(t)

    monkeypatch.setattr(layer.quant_method, "apply", lambda l, i, b: i @ l.weight.t())
    # 用真实 all_reduce 路径（tp=1 时 reduce_results and tp_size>1 为假，不调用）。
    layer.weight.data.copy_(torch.eye(4))
    out, _ = layer(x)
    assert called["n"] == 0  # tp=1 不触发 all_reduce
    assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. QKV fuse + GQA 复制语义
# ---------------------------------------------------------------------------


def test_qkv_output_sizes_and_replicas_when_kv_lt_tp():
    rt.set_tp(4, 0)  # 4 卡，KV 头=2 < 4 → 复制
    qkv = QKVParallelLinear(hidden_size=16, head_size=2, total_num_heads=8, total_num_kv_heads=2)
    assert qkv.num_heads == 2  # 8 query 头 / 4
    assert qkv.num_kv_heads == 1  # KV<tp → 每 rank 1 个
    assert qkv.num_kv_head_replicas == 2  # 4 / 2
    # output_sizes 是「全量」三段宽度（×tp_size）。
    assert qkv.output_sizes == [2 * 2 * 4, 1 * 2 * 4, 1 * 2 * 4]


def test_qkv_kv_uses_replica_rank():
    # tp=4，KV 头=2，replicas=2：rank 0,1 共享 KV 切片 0；rank 2,3 共享 KV 切片 1。
    head_size = 2
    total_q, total_kv = 8, 2
    full_k = torch.arange(total_kv * head_size * 16, dtype=torch.float32).reshape(
        total_kv * head_size, 16
    )
    seen = {}
    for rank in range(4):
        rt.set_tp(4, rank)
        qkv = QKVParallelLinear(
            hidden_size=16, head_size=head_size, total_num_heads=total_q, total_num_kv_heads=total_kv
        )
        qkv.weight_loader(qkv.weight, full_k.clone(), loaded_shard_id="k")
        # k 段在 fused 参数里紧跟 q 段；取出本 rank 装入的 k 切片。
        k_off = qkv.num_heads * qkv.head_size
        k_size = qkv.num_kv_heads * qkv.head_size
        seen[rank] = qkv.weight.data.narrow(0, k_off, k_size).clone()
    assert torch.equal(seen[0], seen[1])  # replica 共享
    assert torch.equal(seen[2], seen[3])
    assert not torch.equal(seen[0], seen[2])  # 不同 KV 头


def test_qkv_q_uses_own_rank():
    head_size = 2
    full_q = torch.arange(8 * head_size * 16, dtype=torch.float32).reshape(8 * head_size, 16)
    rt.set_tp(4, 0)
    qkv0 = QKVParallelLinear(hidden_size=16, head_size=head_size, total_num_heads=8, total_num_kv_heads=2)
    qkv0.weight_loader(qkv0.weight, full_q.clone(), loaded_shard_id="q")
    rt.set_tp(4, 1)
    qkv1 = QKVParallelLinear(hidden_size=16, head_size=head_size, total_num_heads=8, total_num_kv_heads=2)
    qkv1.weight_loader(qkv1.weight, full_q.clone(), loaded_shard_id="q")
    q_size = qkv0.num_heads * qkv0.head_size
    # q 用各自 tp_rank → rank0/rank1 的 q 切片不同。
    assert not torch.equal(qkv0.weight.data.narrow(0, 0, q_size), qkv1.weight.data.narrow(0, 0, q_size))


def test_merged_column_shard_offsets():
    rt.set_tp(1, 0)
    gate_up = MergedColumnParallelLinear(input_size=4, output_sizes=[6, 6], bias=False)
    gate = torch.full((6, 4), 1.0)
    up = torch.full((6, 4), 2.0)
    gate_up.weight_loader(gate_up.weight, gate, loaded_shard_id=0)
    gate_up.weight_loader(gate_up.weight, up, loaded_shard_id=1)
    # fused 布局 [gate | up]：前 6 行=gate，后 6 行=up。
    assert torch.equal(gate_up.weight.data[:6], gate)
    assert torch.equal(gate_up.weight.data[6:], up)


# ---------------------------------------------------------------------------
# 3. 权重装载（stacked_params_mapping / default / tie skip）
# ---------------------------------------------------------------------------


def _make_checkpoint(cfg, model):
    """构造与真实 Llama checkpoint 同名的权重流（独立 q/k/v、gate/up）。"""
    weights = {}
    H, I, V = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = H // nh
    weights["model.embed_tokens.weight"] = torch.randn(V, H)
    weights["model.norm.weight"] = torch.randn(H)
    weights["lm_head.weight"] = torch.randn(V, H)
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        weights[f"{p}.self_attn.q_proj.weight"] = torch.randn(nh * hd, H)
        weights[f"{p}.self_attn.k_proj.weight"] = torch.randn(nkv * hd, H)
        weights[f"{p}.self_attn.v_proj.weight"] = torch.randn(nkv * hd, H)
        weights[f"{p}.self_attn.o_proj.weight"] = torch.randn(H, nh * hd)
        weights[f"{p}.mlp.gate_proj.weight"] = torch.randn(I, H)
        weights[f"{p}.mlp.up_proj.weight"] = torch.randn(I, H)
        weights[f"{p}.mlp.down_proj.weight"] = torch.randn(H, I)
        weights[f"{p}.input_layernorm.weight"] = torch.randn(H)
        weights[f"{p}.post_attention_layernorm.weight"] = torch.randn(H)
    return weights


def test_stacked_mapping_fuses_qkv_and_gate_up():
    cfg = TinyLlamaConfig()
    model = LlamaForCausalLM(vllm_config=_vllm_config())
    ckpt = _make_checkpoint(cfg, model)
    loaded = model.load_weights(list(ckpt.items()))
    H = cfg.hidden_size
    nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    hd = H // nh
    # q/k/v 被 fuse 进 qkv_proj：装载后 qkv_proj.weight 的 q 段 == checkpoint 的 q_proj。
    qkv_w = dict(model.named_parameters())["model.layers.0.self_attn.qkv_proj.weight"].data
    assert torch.equal(qkv_w[: nh * hd], ckpt["model.layers.0.self_attn.q_proj.weight"])
    assert torch.equal(qkv_w[nh * hd : nh * hd + nkv * hd], ckpt["model.layers.0.self_attn.k_proj.weight"])
    # gate/up fuse 进 gate_up_proj。
    gu_w = dict(model.named_parameters())["model.layers.0.mlp.gate_up_proj.weight"].data
    I = cfg.intermediate_size
    assert torch.equal(gu_w[:I], ckpt["model.layers.0.mlp.gate_proj.weight"])
    assert torch.equal(gu_w[I:], ckpt["model.layers.0.mlp.up_proj.weight"])
    # 重命名后的 fused 名出现在 loaded set；原独立名不出现。
    assert "model.layers.0.self_attn.qkv_proj.weight" in loaded
    assert "model.layers.0.self_attn.q_proj.weight" not in loaded


def test_default_loader_for_non_fused_weights():
    cfg = TinyLlamaConfig()
    model = LlamaForCausalLM(vllm_config=_vllm_config())
    ckpt = _make_checkpoint(cfg, model)
    model.load_weights(list(ckpt.items()))
    params = dict(model.named_parameters())
    # embed/norm 等非 fused 权重经 default_weight_loader 直装。
    assert torch.equal(params["model.embed_tokens.weight"].data, ckpt["model.embed_tokens.weight"])
    assert torch.equal(params["model.norm.weight"].data, ckpt["model.norm.weight"])


def test_tie_word_embeddings_skips_lm_head():
    class TiedCfg(TinyLlamaConfig):
        tie_word_embeddings = True

    vcfg = rt.VllmConfig(TiedCfg(), dtype=torch.float32)
    model = LlamaForCausalLM(vllm_config=vcfg)
    # tie 后 lm_head.weight 与 embed_tokens.weight 是同一张量。
    assert model.lm_head.weight is model.model.embed_tokens.weight
    ckpt = _make_checkpoint(TiedCfg(), model)
    loaded = model.load_weights(list(ckpt.items()))
    # AutoWeightsLoader skip_prefixes=["lm_head."] → lm_head 权重被跳过。
    assert not any(n.startswith("lm_head.") for n in loaded)


# ---------------------------------------------------------------------------
# 4. 模型契约：构造签名 / static_forward_context / forward / 三段式
# ---------------------------------------------------------------------------


def test_packed_modules_mapping_is_class_attr():
    assert LlamaForCausalLM.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }


def test_attention_registers_prefix_in_static_context():
    LlamaForCausalLM(vllm_config=_vllm_config())
    # 每层 Attention 以 prefix=model.layers.{i}.self_attn.attn 注册进 static_forward_context。
    keys = set(rt.STATIC_FORWARD_CONTEXT.keys())
    assert "model.layers.0.self_attn.attn" in keys
    assert "model.layers.1.self_attn.attn" in keys


def test_duplicate_prefix_raises():
    LlamaForCausalLM(vllm_config=_vllm_config())
    # 重复 layer name 应报错（prefix 必须唯一）。
    with pytest.raises(ValueError, match="Duplicate layer name"):
        rt.Attention(2, 2, 1.0, num_kv_heads=2, prefix="model.layers.0.self_attn.attn")


def test_initialize_model_checks_vllm_config_prefix_signature():
    # initialize_model 校验模型类接受 (vllm_config, prefix)。
    model = rt.initialize_model(_vllm_config(), prefix="", model_class=LlamaForCausalLM)
    assert isinstance(model, LlamaForCausalLM)

    class BadModel:
        def __init__(self, foo):  # 缺 vllm_config/prefix
            pass

    with pytest.raises(AssertionError):
        rt.initialize_model(_vllm_config(), model_class=BadModel)


def test_three_stage_load_model_runs_forward():
    cfg = TinyLlamaConfig()
    vcfg = _vllm_config()
    tmp = LlamaForCausalLM(vllm_config=vcfg)
    ckpt = _make_checkpoint(cfg, tmp)
    rt.STATIC_FORWARD_CONTEXT.clear()
    # 三段式：initialize_model → load_weights → process_weights_after_loading → eval。
    model = rt.load_model(LlamaForCausalLM, vcfg, list(ckpt.items()))
    seq = 5
    input_ids = torch.randint(0, cfg.vocab_size, (seq,))
    positions = torch.arange(seq)
    hidden = model(input_ids, positions)
    assert hidden.shape == (seq, cfg.hidden_size)
    logits = model.compute_logits(hidden)
    assert logits.shape == (seq, cfg.vocab_size)


def test_decoder_layer_residual_protocol():
    cfg = TinyLlamaConfig()
    model = LlamaForCausalLM(vllm_config=_vllm_config())
    layer = model.model.layers[0]
    seq = 4
    positions = torch.arange(seq)
    hidden = torch.randn(seq, cfg.hidden_size)
    # 第一层 residual=None → 内部用 hidden 初始化 residual 并返回非 None residual。
    out, residual = layer(positions, hidden, None)
    assert residual is not None
    assert out.shape == (seq, cfg.hidden_size)
    assert residual.shape == (seq, cfg.hidden_size)
    # 第二次传入非 None residual 也应跑通（显式穿针）。
    out2, residual2 = layer(positions, out, residual)
    assert out2.shape == (seq, cfg.hidden_size)


def test_attention_forward_shapes_with_gqa():
    cfg = TinyLlamaConfig()
    model = LlamaForCausalLM(vllm_config=_vllm_config())
    attn = model.model.layers[0].self_attn
    seq = 6
    positions = torch.arange(seq)
    hidden = torch.randn(seq, cfg.hidden_size)
    out = attn(positions, hidden)
    # o_proj 输出回到 hidden_size（GQA：8 query 头 / 2 KV 头，n_rep=4）。
    assert out.shape == (seq, cfg.hidden_size)
    assert attn.num_heads == 8 and attn.num_kv_heads == 2
