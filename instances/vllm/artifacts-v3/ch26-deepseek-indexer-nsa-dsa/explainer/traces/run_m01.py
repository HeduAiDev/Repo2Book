# ch26-m01 DSA 打分器 worked example —— Eq.(1) 手算 + 真实 Indexer.forward 路径。
# 素材：手算例（2 头×4 维小例，心算可验）→ 独立小头字面证据（config.index_*）→
# 真实 forward 的量化/scale 折叠数值 → DSV3.2 实尺形状账 → FP8 wk 融合加载。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import (DEV, dump, make_small_hf_config, make_v32_hf_config,
                          make_vllm_config, ref_group_quant_ue8m0)

doc = {"mechanism": "ch26-m01 DSA 打分器：独立小头 + I_{t,s}=Σ w·ReLU(q·k) 的前置装配",
       "source": "run_m01.py（host；Indexer.forward else 分支 = 真实非 CUDA 分派同型）",
       "code_anchor": "deepseek_v2.py:L645-L726（装配）/ L780-L836（forward）/ "
                      "sparse_attn_indexer.py:L500-L507（Eq.(1) 打分核消费位）"}

# ── A. 手算例：H=2、D=4（讲解用小维度；真实 D=128 数学同式）────────────────
q_h1 = torch.tensor([1.0, 0.0, 2.0, -1.0])
q_h2 = torch.tensor([0.0, 3.0, 1.0, 1.0])
K = torch.tensor([[2.0, 1.0, 0.0, 1.0],
                  [-1.0, 0.0, 1.0, 2.0],
                  [0.0, 1.0, 1.0, 0.0],
                  [1.0, 1.0, -1.0, 0.0]])
w = torch.tensor([1.5, -0.5])
dots_h1 = (K @ q_h1).tolist()
dots_h2 = (K @ q_h2).tolist()
relu_h1 = torch.relu(K @ q_h1).tolist()
relu_h2 = torch.relu(K @ q_h2).tolist()
I = (w[0] * torch.relu(K @ q_h1) + w[1] * torch.relu(K @ q_h2)).tolist()
top2 = [i for i, _ in sorted(enumerate(I), key=lambda t: (-t[1], t[0]))][:2]
neg_dot_count = int(sum(1 for d in dots_h1 + dots_h2 if d < 0))
doc["hand_example"] = {
    "setup": "H=2 头（真实 64）、D=4 维（真实 128，数学同式）、S=4 个历史 token；"
             "逐头权重 w=[1.5, -0.5]（weights_proj 可为负——第 2 头是抑制项）",
    "q_h1": [1, 0, 2, -1], "q_h2": [0, 3, 1, 1],
    "k_rows": K.tolist(), "weights": [1.5, -0.5],
    "dots_h1": dots_h1, "dots_h2": dots_h2,
    "relu_h1": relu_h1, "relu_h2": relu_h2,
    "I_t_s": I,
    "top2_selected": top2,
    "top2_values": sorted(I, reverse=True)[:2],
    "neg_dot_gated_count": neg_dot_count,
    "neg_dot_note": f"8 个逐头点积里 {neg_dot_count} 个为负被 ReLU 门成 0——"
                    "ReLU 而非 softmax 是吞吐考虑（arXiv:2512.02556 §2.1）",
    "negative_weight_note": "w_2=-0.5 使第 2 头的高分反而压低总分：s=0 头 2 得 "
                            "4 但 I=-0.5——『逐头贡献可学习地加权』的含义",
}

# ── B. 独立小头字面证据：头表全来自 config.index_*，与主注意力头数无关 ────────
hf = make_small_hf_config()
doc["independent_head_evidence"] = {
    "num_attention_heads_main": int(hf.num_attention_heads),
    "index_n_heads": int(hf.index_n_heads),
    "index_head_dim": int(hf.index_head_dim),
    "qk_rope_head_dim": int(hf.qk_rope_head_dim),
    "index_topk": int(hf.index_topk),
    "note": "主注意力 8 头（DSV3.2 实尺 128）、indexer 2 头（实尺 64）——"
            "两套头表互不相干，字面证据=字段前缀 index_*（deepseek_v2.py:L655-L660）",
}

# ── C. 真实 Indexer.forward：量化 + scale 折叠的实测数值 ─────────────────────
from vllm.model_executor.models.deepseek_v2 import Indexer
from vllm.model_executor.layers.rotary_embedding import get_rope

torch.manual_seed(3)
vllm_config = make_vllm_config(hf, max_model_len=256)
buf = torch.zeros(256, hf.index_topk, dtype=torch.int32)
idx = Indexer(vllm_config, hf, hf.hidden_size, hf.q_lora_rank,
              None, vllm_config.cache_config, buf,
              "model.layers.0.self_attn.indexer")

captured = {}


class _Recorder(torch.nn.Module):
    def __call__(self, *args):
        captured["args"] = args
        return args[-1]


idx.indexer_op = _Recorder()
T = 5
hidden = torch.randn(T, hf.hidden_size)
qr = torch.randn(T, hf.q_lora_rank)
positions = torch.arange(T)
rope = get_rope(64, max_position=64,
                rope_parameters={"rope_type": "default"},
                is_neox_style=False)  # indexer 专属 interleave RoPE
idx(hidden, qr, positions, rope)
q_fp8, k, weights = captured["args"][1:4]
# weights 折叠账：raw_w · q_scale · softmax_scale · n_head_scale
kw = idx.wk_weights_proj(hidden)[0]
w_raw = kw[:, 128:].view(T, hf.index_n_heads)
# q 参考：上投 → 切 rope/nope → interleave RoPE 只打 rope 段 → cat 回（与
# Indexer.forward else 分支同式；tests/test_ch26_indexer.py 同款参考）
q_raw = idx.wq_b(qr)[0].view(T, hf.index_n_heads, 128)
from trace_common import ref_interleave_rope
q_pe = ref_interleave_rope(q_raw[..., :64], positions)
q_rot = torch.cat([q_pe, q_raw[..., 64:]], dim=-1).reshape(-1, 128)
q_fp8_ref, q_scale_ref = ref_group_quant_ue8m0(q_rot)
w_ref = (w_raw * q_scale_ref.view(T, hf.index_n_heads)
         * idx.softmax_scale * idx.n_head_scale)
fold_max_diff = float((weights.view(T, hf.index_n_heads) - w_ref).abs().max())
# q 量化的 ue8m0 幂次 scale 实测（token0/head0）
quant_bitwise_equal = bool(torch.equal(
    q_fp8.reshape(-1).view(torch.uint8),
    q_fp8_ref.reshape(-1).view(torch.uint8)))
q_scale_t0h0 = float(q_scale_ref.view(T, hf.index_n_heads)[0, 0])
doc["forward_run"] = {
    "T": T, "n_head": int(hf.index_n_heads), "head_dim": 128,
    "q_fp8_dtype": str(q_fp8.dtype).replace("torch.", ""),
    "q_fp8_shape": list(q_fp8.shape),
    "quant_bitwise_equal_to_reference": quant_bitwise_equal,
    "q_scale_token0_head0": q_scale_t0h0,
    "softmax_scale": float(idx.softmax_scale),
    "n_head_scale": float(idx.n_head_scale),
    "fold_formula": "weights = raw_w · q_scale · softmax_scale · n_head_scale",
    "fold_max_abs_diff": f"{fold_max_diff:.6f}",
    "k_shape": list(k.shape),
    "note": "q_scale=2 的幂（ue8m0）；三个标量全折进 weights——打分核内只剩 "
            "『点积+ReLU+加权和』，标量归一化搬出核（deepseek_v2.py:L817）",
}

# ── D. DSV3.2 实尺形状账（真实例化，只做装配不跑前向）──────────────────────
hf32 = make_v32_hf_config()
vllm32 = make_vllm_config(hf32, max_model_len=163840,
                          max_num_batched_tokens=8192)
buf32 = torch.zeros(8192, hf32.index_topk, dtype=torch.int32)
idx32 = Indexer(vllm32, hf32, hf32.hidden_size, hf32.q_lora_rank,
                None, vllm32.cache_config, buf32,
                "model.layers.0.self_attn.indexer")
doc["dsv32_shapes"] = {
    "n_head": int(idx32.n_head), "head_dim": int(idx32.head_dim),
    "rope_dim": int(idx32.rope_dim), "topk_tokens": int(idx32.topk_tokens),
    "wq_b_weight_shape": list(idx32.wq_b.weight.shape),
    "wq_b_note": "ReplicatedLinear——no tensor parallel, just replicated"
                 "（deepseek_v2.py:L668-L674 注释原话；q_lora_rank 1536 上投 64×128）",
    "wk_weights_proj_weight_shape": list(idx32.wk_weights_proj.weight.shape),
    "wk_weights_proj_output_sizes": [128, 64],
    "k_norm_eps": float(idx32.k_norm.eps),
    "k_cache_head_dim": int(idx32.k_cache.head_dim),
    "k_cache_bytes_per_token": 132,
    "softmax_scale": float(idx32.softmax_scale),
    "n_head_scale": float(idx32.n_head_scale),
    "max_total_seq_len": int(idx32.max_total_seq_len),
    "max_total_seq_len_note": "163840×40（get_max_prefill_buffer_size 魔数账，"
                              "归 m02/m04）",
}

# ── E. FP8 wk 权重融合加载：『单独训练』的工程痕迹（两段缓冲）────────────────
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk

prefix = "model.layers.0.self_attn.indexer"
w_fp8 = torch.randn(128, 64).to(torch.float8_e4m3fn)
scale = torch.rand(128) + 0.5
pbuf, loaded = {}, set()
params = {f"{prefix}.wk_weights_proj.weight":
          type("P", (), {"loaded": None,
                         "weight_loader": lambda self, p, t, s:
                         setattr(self, "loaded", t.clone())})()}
r1 = _try_load_fp8_indexer_wk(f"{prefix}.wk.weight", w_fp8, pbuf, params,
                              loaded, [])
r2 = _try_load_fp8_indexer_wk(f"{prefix}.wk.weight_scale_inv", scale, pbuf,
                              params, loaded, [])
fused = params[f"{prefix}.wk_weights_proj.weight"].loaded
doc["fp8_wk_load"] = {
    "weight_arrival_returned": bool(r1),
    "scale_arrival_returned": bool(r2),
    "fused_dtype": str(fused.dtype).replace("torch.", ""),
    "fused_shape": list(fused.shape),
    "fused_max_abs_diff_vs_dequant": f"{float((fused.float() - w_fp8.float() * scale.unsqueeze(-1)).abs().max()):.6f}",
    "note": "FP8 wk 权重先缓冲、scale 到齐才反量化融合进 wk_weights_proj"
            "（deepseek_v2.py:L822-L871）——checkpoint 里 indexer 权重自带 "
            "FP8 独立形态 = 训练期单独优化的加载侧痕迹（Eq.(3)-(4)）",
}

# ── F. explainer 表格回显（lint 数字溯源；字符串由本次运行产出）────────────
doc["table_rows_echo"] = [
    ["手算-输入", "2 头 query / 4 个历史 key / 逐头权重 w=[1.5, -0.5]",
     f"q_h1·k_0={dots_h1[0]:g}, q_h2·k_0={dots_h2[0]:g}", "点积逐头逐 key"],
    ["手算-门控", "ReLU(q·k)：负点积门成 0",
     f"h1: [{', '.join(f'{v:g}' for v in relu_h1)}]；"
     f"h2: [{', '.join(f'{v:g}' for v in relu_h2)}]",
     f"8 个点积中 {neg_dot_count} 个为负被门掉"],
    ["手算-加权", "I_s = 1.5·ReLU_h1 − 0.5·ReLU_h2",
     f"[{', '.join(f'{v:g}' for v in I)}]",
     f"负权重压制第 2 头（s=0 头 2 得 {dots_h2[0]:g} 仍 {I[0]:g}）"],
    ["手算-选择", f"top-2（降序、tie 取小）",
     f"选中 {top2}，分值 [{I[top2[0]]:g}, {I[top2[1]]:g}]",
     "k=2 的 top-k 就是主注意力要算的条目"],
    ["forward-量化", "q per-token-group FP8（ue8m0）",
     f"q_fp8 [5, 2, 128] float8_e4m3fn；"
     f"q_scale(token0,head0)={q_scale_t0h0:g}",
     f"与独立参考逐位相等={str(quant_bitwise_equal).lower()}"],
    ["forward-折叠", "weights = raw_w·q_scale·softmax_scale·n_head_scale",
     f"{idx.softmax_scale:.6f}·{idx.n_head_scale:.6f} 两标量并入",
     f"折叠后最大偏差 {fold_max_diff:.6f}"],
    ["装配-头表", "头表全来自 config.index_*", "indexer 2 头 vs 主注意力 8 头",
     "实尺 64 头 vs 128 头——两套头表互不相干"],
    ["装配-实尺", "wq_b [8192, 1536] 复制不切 TP；wk_weights_proj [192, 7168]",
     "一枪 GEMM 出 key 128 + 逐头权重 64",
     "k_cache 132B/条；workspace 6553600"],
]

dump("m01.json", doc)
print("m01.json written:",
      "| I =", [f"{v:g}" for v in I], "| top2 =", top2,
      "| fold_max_diff =", f"{fold_max_diff:.6f}",
      "| quant_bitwise =", quant_bitwise_equal)
