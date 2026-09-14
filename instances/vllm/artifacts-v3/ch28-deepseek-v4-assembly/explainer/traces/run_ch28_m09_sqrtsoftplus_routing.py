"""ch28-m09 驱动脚本 —— sqrtsoftplus 路由: 谁打分/谁选择/谁加权 三分离 + hash 查表对照。

跑法(host, 纯 CPU torch): python run_ch28_m09_sqrtsoftplus_routing.py
输出: ch28_m09_sqrtsoftplus_routing.json(与本脚本同目录)

素材对应 dossier 机制 ch28-m09(needs_worked_example)。

真源:
- _topk_softplus_sqrt_torch 逐字拷贝自
  vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:L75-L119
  (XPU/CPU 回退路径; CUDA 生产走 ops.topk_hash_softplus_sqrt,
   vllm_topk_softplus_sqrt L121-L149 分派)。该函数同时覆盖 hash 分支
   (expert_ids=tid2eid[input_tokens], 权重仍从无偏 scores gather)。
- 生产调用点: vllm/models/deepseek_v4/nvidia/model.py:L704-L743
  (fused_topk_bias(... indices_type=int64(mega)/int32, input_tokens, hash_indices_table,
  routed_scaling_factor) → experts(topk_weights, topk_ids, activation_clamp))。
- routed_scaling_factor=1.5 取自 tests/kernels/moe/test_topk_softplus_sqrt.py:L90
  参数化表 [1.0, 1.5](真实模型读 config.routed_scaling_factor, 默认 1.0,
  model.py:L543); DSv4-Flash 全部 bias≈8.08 的实证注释见
  fused_topk_bias_router.py:L101-L104(偏置当权重会把分布压平)。

参数(小而具体): E=4 专家 top-2, 2 个 token, bias=[0, 0.3, 0, 0](故意让 token0
的选择被 bias 翻转——非平凡分支; token1 边距大翻不动, 作对照)。
hash 层对照: tid2eid 查表结果 ≠ 打分 top-2 集合, 展示「ids 查表定死、只算权重」。
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- 以下函数逐字拷贝自 fused_topk_bias_router.py:L75-L119 (pin v0.27.1) ----


def _topk_softplus_sqrt_torch(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """Pure PyTorch fallback for topk_softplus_sqrt (XPU/CPU)."""
    # scores = sqrt(softplus(gating_output))
    scores = torch.sqrt(F.softplus(gating_output.float()))

    # Bias is used for expert SELECTION only, not for weight computation.
    # Using biased scores as weights flattens the distribution when the bias
    # is near-uniform (e.g., DSv4-Flash where all biases ≈ 8.08).
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.float()
    else:
        scores_for_choice = scores

    topk = topk_weights.shape[-1]

    if hash_indices_table is not None and input_tokens is not None:
        # Hash MoE: expert indices predetermined by lookup table
        # hash_indices_table: [vocab_size, topk] mapping token_id -> expert_ids
        expert_ids = hash_indices_table[input_tokens.long()]  # [M, topk]
        topk_indices.copy_(expert_ids)
        # Gather weights from unbiased scores
        weights = scores.gather(1, expert_ids.long())
    else:
        # Standard topk selection using biased scores
        _, indices = torch.topk(scores_for_choice, k=topk, dim=-1)
        topk_indices.copy_(indices)
        # Gather weights from unbiased scores
        weights = scores.gather(1, indices)

    if renormalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True).clamp(min=1e-20))

    topk_weights.copy_(weights * routed_scaling_factor)
    return topk_weights, topk_indices


# ---- 本例参数 ----
E, TOPK = 4, 2
RENorm, SCALE = True, 1.5
gating = torch.tensor([[1.2, 0.3, -0.5, 2.0],
                       [0.1, 1.9, 0.4, -1.0]], dtype=torch.float32)
bias = torch.tensor([0.0, 0.3, 0.0, 0.0], dtype=torch.float32)


def r(t, nd=4):
    if torch.is_tensor(t):
        return [r(v, nd) for v in t] if t.dim() else round(float(t), nd)
    return round(float(t), nd)


def run_case(tag, gating_output, bias_t, input_tokens=None, table=None):
    tw = torch.empty(gating_output.shape[0], TOPK, dtype=torch.float32)
    ti = torch.empty(gating_output.shape[0], TOPK, dtype=torch.int32)
    te = torch.empty(gating_output.shape[0], TOPK, dtype=torch.int32)
    _topk_softplus_sqrt_torch(tw, ti, te, gating_output,
                              renormalize=RENorm, e_score_correction_bias=bias_t,
                              input_tokens=input_tokens,
                              hash_indices_table=table,
                              routed_scaling_factor=SCALE)
    scores = torch.sqrt(F.softplus(gating_output.float()))
    choice = scores + bias_t if bias_t is not None else scores
    unbiased_top2 = torch.topk(scores, k=TOPK, dim=-1)[1].tolist()
    return {
        "gating": r(gating_output),
        "scores=sqrt(softplus(g))": r(scores),
        "scores_for_choice=scores+bias": r(choice),
        "unbiased_top2(若无bias会选)": unbiased_top2,
        "selected_topk_ids": ti.tolist(),
        "topk_weights(无偏gather→归一→×1.5)": r(tw),
        "weights_sum_after_scale": round(float(tw.sum(-1)[0]), 4),
    }


out = {
    "env": "host Miniconda python 3.11.11, torch 2.11.0+cu128 (纯 CPU), pin=vLLM v0.27.1 (6e448d0ea)",
    "refs_copied_verbatim": "vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:L75-L119 (_topk_softplus_sqrt_torch)",
    "params": {
        "E": E, "topk": TOPK, "renormalize": RENorm,
        "routed_scaling_factor": SCALE,
        "routed_scaling_factor_provenance": "tests/kernels/moe/test_topk_softplus_sqrt.py:L90 参数化 [1.0,1.5]; 模型默认 1.0 (model.py:L543)",
        "e_score_correction_bias": r(bias),
        "bias_note": "bias=[0,0.3,0,0] 为本例自选; DSv4-Flash 真机 bias≈8.08 近均匀 (fused_topk_bias_router.py:L101-L104 注释)",
    },
    "cases": {},
    "hand_check": {},
}

# 案例A: 标准层(noaux_tc 带 bias) —— token0 被 bias 翻转, token1 翻不动
out["cases"]["A_standard_layer_with_bias"] = run_case("A", gating, bias)

# 案例B: 同样 gating、无 bias、无 hash —— 对照「没有选择偏置」的基线
out["cases"]["B_standard_layer_no_bias"] = run_case("B", gating, None)

# 案例C: hash 层 —— tid2eid[7]=[3,2]/tid2eid[3]=[0,2](本例固定表, 真实模型从
# checkpoint 装载; dummy 模式才用 randint 初始化, model.py:L575-L581)。
# 故意让表与打分 top-2 集合不同(token7: 表含 e2 不含 e0; token3: 表含 e0 不含 e1)
# ——「ids 查表定死、分数只算权重」的非平凡分支。
tid2eid = torch.zeros(16, TOPK, dtype=torch.int64)
tid2eid[3] = torch.tensor([0, 2])
tid2eid[7] = torch.tensor([3, 2])
input_tokens = torch.tensor([7, 3], dtype=torch.int64)
out["cases"]["C_hash_layer_lookup"] = run_case("C", gating, None,
                                               input_tokens=input_tokens,
                                               table=tid2eid)
out["cases"]["C_hash_layer_lookup"]["tid2eid_rows"] = {"7": [3, 2], "3": [0, 2]}
out["cases"]["C_hash_layer_lookup"]["input_tokens"] = [7, 3]

# 手算复核(标准库 math, 与 torch 对拍): softplus(x)=log(1+e^x)
hc = {}
for j, g in enumerate(gating[0].tolist()):
    sp = math.log1p(math.exp(g))
    hc[f"e{j}: softplus({g})"] = round(sp, 6)
    hc[f"e{j}: sqrt"] = round(math.sqrt(sp), 6)
s = [math.sqrt(math.log1p(math.exp(g))) for g in gating[0].tolist()]
w_sel = [s[3], s[1]]  # biased top2 = {3,1}, 权重从无偏 scores 取
tot = sum(w_sel)
hc["token0_weights_before_scale"] = [round(w_sel[0] / tot * SCALE, 6), round(w_sel[1] / tot * SCALE, 6)]
out["hand_check"] = hc

dst = Path(__file__).parent / "ch28_m09_sqrtsoftplus_routing.json"
with open(dst, "w", newline="\n", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", dst)
print(json.dumps(out["cases"], ensure_ascii=True, indent=1))
