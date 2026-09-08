# ch23 m5 驱动脚本 —— 采样位切片与 compute_logits 出口契约
# （gpu_model_runner.py:L2232-L2240 + L4483-L4485 → llama.py:L528-L533 →
#   logits_processor.py:L137-L153）
# 跑精简版（implementation/）取真实数值轨迹。参考注入件 =
# 测试电池 tests/test_model_layer_assembly.py 的 RefAttentionImpl /
# inject_ref_backend / bind_kv_caches / gen_ckpt_weights（真实注入位 =
# attention.py L251 的 attn_backend 参数；数学 = ch20 已立的精确注意力）。
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import torch
from transformers import LlamaConfig as HFLlamaConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "implementation"))

import vllm.distributed as dist  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.model_executor.layers.attention import Attention  # noqa: E402
from vllm.model_executor.model_loader.utils import initialize_model  # noqa: E402
from vllm.v1.attention.backend import AttentionType  # noqa: E402
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402

dist._TP_STATE.update(rank=0, size=1, pp_rank=0, pp_size=1)

# ── 测试电池同款注入件（tests/test_model_layer_assembly.py L137-L302）──


@dataclass
class RefAttnMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor


class RefAttentionImpl:
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, **extra):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(self, attn_layer, key, value, kv_cache, slot_mapping):
        for t, slot in enumerate(slot_mapping.tolist()):
            if slot >= 0:
                kv_cache[slot, 0] = key[t]
                kv_cache[slot, 1] = value[t]

    def forward(self, attn_layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        qsl = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        _, hq, hd = query.shape
        hk = self.num_kv_heads
        rep = hq // hk
        out = torch.zeros_like(query)
        for r in range(len(qsl) - 1):
            qs, qe = qsl[r], qsl[r + 1]
            total = seq_lens[r]
            K = kv_cache[:total, 0].reshape(total, hk, hd)
            V = kv_cache[:total, 1].reshape(total, hk, hd)
            for h in range(hq):
                kv_h = h // rep
                q = query[qs:qe, h].double()
                k = K[:, kv_h].double()
                v = V[:, kv_h].double()
                scores = (q @ k.T) * self.scale
                causal = torch.arange(total).unsqueeze(0) <= torch.arange(qs, qe).unsqueeze(1)
                scores = scores.masked_fill(~causal, float("-inf"))
                p = torch.softmax(scores, dim=-1)
                out[qs:qe, h] = (p @ v).to(query.dtype)
        if output is not None:
            output.copy_(out)
            return None
        return out


class RefAttentionBackend:
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name():
        return "REFERENCE"

    @staticmethod
    def get_impl_cls():
        return RefAttentionImpl


def make_hf_config(**over):
    kwargs = dict(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    kwargs.update(over)
    cfg = HFLlamaConfig(**kwargs)
    cfg.architectures = ["LlamaForCausalLM"]
    return cfg


def make_vllm_config(hf_config):
    return VllmConfig(
        model_config=ModelConfig(hf_config=hf_config, dtype=torch.float32),
        cache_config=CacheConfig(),
        load_config=LoadConfig(load_format="auto"),
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
        parallel_config=ParallelConfig(),
    )


def inject_ref_backend(model):
    for _, m in model.named_modules():
        if isinstance(m, Attention):
            m.attn_backend = RefAttentionBackend
            m.impl = RefAttentionImpl(
                m.num_heads, m.head_size, m.head_size**-0.5,
                num_kv_heads=m.num_kv_heads, attn_type=m.attn_type,
            )


def gen_ckpt_weights(hf_config, seed=0):
    g = torch.Generator().manual_seed(seed)
    H, I, V = hf_config.hidden_size, hf_config.intermediate_size, hf_config.vocab_size
    L = hf_config.num_hidden_layers
    hd = H // hf_config.num_attention_heads
    nkvh = hf_config.num_key_value_heads
    weights = {
        "model.embed_tokens.weight": torch.randn(V, H, generator=g),
        "lm_head.weight": torch.randn(V, H, generator=g),
        "model.norm.weight": torch.randn(H, generator=g) * 0.1 + 1.0,
    }
    for i in range(L):
        p = f"model.layers.{i}"
        weights[f"{p}.self_attn.q_proj.weight"] = torch.randn(hf_config.num_attention_heads * hd, H, generator=g) * 0.05
        weights[f"{p}.self_attn.k_proj.weight"] = torch.randn(nkvh * hd, H, generator=g) * 0.05
        weights[f"{p}.self_attn.v_proj.weight"] = torch.randn(nkvh * hd, H, generator=g) * 0.05
        weights[f"{p}.self_attn.o_proj.weight"] = torch.randn(H, hf_config.num_attention_heads * hd, generator=g) * 0.05
        weights[f"{p}.mlp.gate_proj.weight"] = torch.randn(I, H, generator=g) * 0.05
        weights[f"{p}.mlp.up_proj.weight"] = torch.randn(I, H, generator=g) * 0.05
        weights[f"{p}.mlp.down_proj.weight"] = torch.randn(H, I, generator=g) * 0.05
        weights[f"{p}.input_layernorm.weight"] = torch.randn(H, generator=g) * 0.1 + 1.0
        weights[f"{p}.post_attention_layernorm.weight"] = torch.randn(H, generator=g) * 0.1 + 1.0
    return weights


# ── 场景：一拍 chunked prefill 批 = r0 全 prompt 5 token + r1 部分请求 3/5 token ──
hf = make_hf_config()
vllm_config = make_vllm_config(hf)
model = initialize_model(vllm_config)
inject_ref_backend(model)
ckpt = gen_ckpt_weights(hf, seed=0)
model.load_weights(iter(list(ckpt.items())))

T = 8
g = torch.Generator().manual_seed(3)
input_ids = torch.randint(0, hf.vocab_size, (T,), generator=g)
positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.long)
qsl = torch.tensor([0, 5, 8], dtype=torch.int32)          # r0 占 [0,5)、r1 占 [5,8)
seq_lens = torch.tensor([5, 3], dtype=torch.int32)
meta = {name: RefAttnMetadata(qsl, seq_lens) for name in vllm_config.compilation_config.static_forward_context}
sm = {name: torch.arange(T) for name in meta}
hd = hf.hidden_size // hf.num_attention_heads
kvs = [torch.zeros(T, 2, hf.num_key_value_heads, hd) for _ in range(hf.num_hidden_layers)]
for layer_name, attn in vllm_config.compilation_config.static_forward_context.items():
    attn.bind_kv_cache(kvs[int(layer_name.split(".")[2])])

runner = GPUModelRunner(
    vllm_config=vllm_config, model=model,
    query_start_loc=qsl, attn_metadata=meta, slot_mapping=sm,
)
logits = runner.execute_model(input_ids=input_ids, positions=positions)

# 对照 1：直接经模型契约两方法（forward 出 hidden → 切片 → compute_logits）
with set_forward_context(meta, vllm_config, slot_mapping=sm):
    hidden = model(input_ids=input_ids, positions=positions, intermediate_tensors=None)
logits_indices = qsl[1:] - 1
sample_hidden = hidden[logits_indices]
ref_logits = model.compute_logits(sample_hidden)

# 对照 2：手写 lm_head GEMM + 裁 padding（logits_processor 三步的算术本体）
padded_lm_head = model.lm_head.weight.double()          # [128, 64]（pad 到 64 倍数）
full_logits_padded = sample_hidden.double() @ padded_lm_head.T   # [2, 128]
hand_logits = full_logits_padded[..., : hf.vocab_size]  # 裁回 org_vocab 100

# 反事实：HF 式全位置物化（forward 里一口气全算）
full_hidden = hidden
full_all_positions = full_hidden.double() @ padded_lm_head.T    # [8, 128]

next_tokens = logits.argmax(dim=-1)

out = {
    "mechanism": "m5",
    "source": "run_m5.py @ implementation/ (vLLM v0.27.1 只做减法精简版, host CPU)",
    "params": {
        "vocab_size_org": hf.vocab_size,
        "vocab_size_padded": model.lm_head.weight.shape[0],
        "pad_to_multiple_of": 64,
        "hidden_size": hf.hidden_size,
        "num_layers": hf.num_hidden_layers,
        "req0_tokens_scheduled": 5,
        "req1_tokens_scheduled": 3,
        "req1_is_partial_chunked_prefill": True,
        "total_tokens": T,
        "num_requests": 2,
    },
    "runner_policy": {
        "query_start_loc": qsl.tolist(),
        "logits_indices_formula": "query_start_loc[1:] - 1",
        "logits_indices": logits_indices.tolist(),
        "sampled_row_of_req0": "批内第 4 行 = r0 本拍最后一个已算 token",
        "sampled_row_of_req1": "批内第 7 行 = r1 本拍最后一个已算 token（部分请求也采、结果被忽略——woosuk NOTE）",
    },
    "forward_contract": {
        "forward_returns": "hidden_states (不是 logits)",
        "hidden_states_shape": list(hidden.shape),
        "hidden_states_is_IntermediateTensors": False,
    },
    "export": {
        "sample_hidden_states_shape": list(sample_hidden.shape),
        "lm_head_weight_shape_padded": list(model.lm_head.weight.shape),
        "gemm_out_shape_padded": list(full_logits_padded.shape),
        "logits_shape_after_cut": list(logits.shape),
        "org_vocab_cut": f"logits[..., :{hf.vocab_size}]",
        "tp_size": 1,
        "gather_called": False,
        "gather_note": "lm_head.tp_size>1 才 _gather_logits；本例单卡退化不 gather（多 rank 部署归 ch34）",
    },
    "verification": {
        "runner_vs_compute_logits_max_abs_diff": float((logits - ref_logits).abs().max()),
        "compute_logits_vs_hand_gemm_max_abs_diff": float((ref_logits.double() - hand_logits).abs().max()),
    },
    "counterfactual_full_materialization": {
        "hf_style_all_positions_shape": list(full_all_positions.shape),
        "values_all_positions_padded": full_all_positions.numel(),
        "values_all_positions_org_vocab": T * hf.vocab_size,
        "values_sampled_org_vocab": logits.numel(),
        "reduction_factor_org_vocab": (T * hf.vocab_size) / logits.numel(),
    },
    "outcome": {
        "next_token_req0": int(next_tokens[0]),
        "next_token_req1_ignored_partial": int(next_tokens[1]),
        "logits_req0_top3": sorted(logits[0].tolist(), reverse=True)[:3],
        "logits_req1_top3": sorted(logits[1].tolist(), reverse=True)[:3],
        "logits_req0_argmax_value": float(logits[0].max()),
    },
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m5.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
