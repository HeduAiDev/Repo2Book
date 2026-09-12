# ch25 explainer 驱动公共件 —— 复用测试电池（tests/test_mla_two_expansions.py）
# 的真实注入件与装配工具：参考 decode/prefill 后端经真实注册流注入
# （register_backend(CUSTOM) / MLAPrefillBackendEnum.CUSTOM），CUDA kernel 面
# （FlashMLA MQA kernel / merge_attn_states / concat_and_cache_mla / gather_cache）
# 由 impl-notes §Seam B1 的 HOST SEAM 镜像承载精确数学（= 真实源码文件头
# mla_attention.py:L44-L118 两路伪码的逐式实现），等价性测试已对其数值闭环
# （38 passed）。本文件只做装配/跑数/落盘，不引入新的数学。
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.dirname(os.path.dirname(HERE))  # 本章目录

sys.path.insert(0, os.path.join(CH, "tests"))
sys.path.insert(0, os.path.join(CH, "implementation"))

import torch  # noqa: E402

import test_mla_two_expansions as T  # noqa: E402  真实注入件 + 注册流（模块级注册）

# host 平台的算力代占位：SM90（与测试 _sm90_platform fixture 同款）——真实
# 选择器的显式 mla_prefill_backend 支在 device_capability 为 None 时不会走到，
# 以 SM90 走真实显式选择分支（validate_configuration → get_class → CUSTOM 覆盖表）。
from vllm.platforms import current_platform  # noqa: E402
from vllm.platforms.interface import DeviceCapability  # noqa: E402

type(current_platform).get_device_capability = classmethod(
    lambda cls: DeviceCapability(9, 0)
)

from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.v1.kv_cache_interface import MLAAttentionSpec  # noqa: E402
from vllm.v1.attention.backends.mla.flashmla import (  # noqa: E402
    FlashMLAMetadataBuilder,
)

BLOCK = T.BLOCK
MINI = T.MINI
# DSV3 实尺维度（deepseek_v3 config：hidden 7168 / q_lora 1536 / 128 头 /
# nope 128 / rope 64 / kv_lora 512 / v 128）——用于形状账，不跑前向
DSV3 = dict(
    hidden_size=7168,
    q_lora_rank=1536,
    num_heads=128,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    v_head_dim=128,
)
LAYER_NAME = "m.l0.self_attn.attn"


def setup(dims=MINI, max_model_len=64, max_num_seqs=8, block=BLOCK, seed=2525,
          num_blocks=32):
    """装配 MINI 数值档的一层（选型三岔 → 投影积木 → Wrapper → 插座 → 吸收重排）
    + FlashMLA builder（真实注入位）。返回 (vllm_config, layer, inner, builder)。"""
    torch.manual_seed(seed)
    vllm_config = T.enable_ref_backends(
        T.make_vllm_config(dims=dims, max_model_len=max_model_len,
                           max_num_seqs=max_num_seqs, block_size=block))
    layer, inner = T.build_mla_layer(vllm_config, prefix="m.l0.self_attn",
                                     dims=dims)
    spec = MLAAttentionSpec(block_size=block, num_kv_heads=1, head_size=576,
                            dtype=torch.float32)
    with T.set_current_vllm_config(vllm_config):
        builder = FlashMLAMetadataBuilder(spec, [LAYER_NAME], vllm_config,
                                          torch.device("cpu"))
    inner.kv_cache = torch.zeros(num_blocks, block, 576)
    return vllm_config, layer, inner, builder


def run(layer, builder, vllm_config, query_lens, seq_lens, h, pos, block=BLOCK):
    """一拍前向：构造 common metadata → builder.build（split 分流）→ set_forward_context → 层前向。"""
    common, slots = T.make_common(query_lens, seq_lens, block=block)
    md = builder.build(0, common)
    with set_forward_context({LAYER_NAME: md}, vllm_config,
                             slot_mapping={LAYER_NAME: slots}):
        out = layer(pos, h, None)
    return out, md, common


def oracle_mha(layer, inner, common, h_new, pos_new, query_lens, seq_lens,
               dims=MINI, block=BLOCK):
    """上投影 MHA 参照（真实源码文件头 Compute Friendly 伪码 mla_attention.py
    L44-L118 的逐式实现——与测试电池同款 oracle）：重建 q（含 rope），逐请求
    从分页 cache 取潜行、kv_b_proj 上投影、标准注意力、o_proj 收尾。
    返回 (ref_out, q_full)；q_full 供吸收腿对照。"""
    N, P, R, Lkv, V = (dims["num_heads"], dims["qk_nope_head_dim"],
                       dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                       dims["v_head_dim"])
    Lq = dims["q_lora_rank"]
    scale = (P + R) ** -0.5
    cache = inner.kv_cache.view(-1, 576)
    bt = common.block_table_tensor
    q_c = layer.q_a_layernorm((h_new @ layer.fused_qkv_a_proj.weight.T)[:, :Lq])
    q = (q_c @ layer.q_b_proj.weight.T).view(-1, N, P + R)
    q_rot, _ = layer.mla_attn.rotary_emb(pos_new, q[..., P:].clone(),
                                         torch.zeros(q.shape[0], 1, R))
    q_full = torch.cat([q[..., :P], q_rot], dim=-1)
    ref = torch.zeros(q.shape[0], N, V)
    off = 0
    for r, (qlen, s) in enumerate(zip(query_lens, seq_lens)):
        rows = [int(bt[r, p // block]) * block + p % block for p in range(s)]
        lat = cache[torch.tensor(rows)]
        w = inner.kv_b_proj.weight
        kv_nope = (lat[:, :Lkv] @ w.T).view(s, N, P + V)
        k_nope, v = kv_nope.split([P, V], dim=-1)
        k = torch.cat([k_nope, lat[:, Lkv:].unsqueeze(1).expand(-1, N, -1)],
                      dim=-1)
        s_mat = torch.einsum("tnd,snd->nts", q_full[off:off + qlen], k) * scale
        pos_q = pos_new[off:off + qlen]
        mask = torch.arange(s).unsqueeze(0) > pos_q.unsqueeze(1)
        s_mat.masked_fill_(mask.unsqueeze(0), -float("inf"))
        p_attn = torch.softmax(s_mat, dim=-1)
        ref[off:off + qlen] = torch.einsum("nts,snv->tnv", p_attn, v)
        off += qlen
    return layer.o_proj(ref.reshape(-1, N * V))[0], q_full


def fmt(x, nd=4):
    """表用数值的定长十进制串——trace 与 explainer 表格共用同一字面值。"""
    return f"{float(x):.{nd}f}"


def dump(name, obj):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print("written", path)
