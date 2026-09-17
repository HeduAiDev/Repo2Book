# ch32 m12 驱动：为什么写 -inf——softmax 前概率精确归零、与温度/top-k 正交；
# 有限负值残留对照。真 xgrammar fill + 真 xgr.apply_token_bitmask_inplace（CUDA）。
import sys

import torch

from _ch32_common import (OFF_GRAMMAR, TOK_NO, TOK_YES, VOCAB, RealBackend,
                          allowed_ids, dump, get_tokenizer, make_vllm_config,
                          new_grammar)
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))
import xgrammar as xgr

EBNF = 'root ::= "yes" | "no"'
tok = get_tokenizer()
out = {"env": {"grammar": EBNF, "vocab": VOCAB,
               "cuda_device": torch.cuda.get_device_name(0)}}

# 位置 0 合法集（真 FSM）
g = new_grammar(EBNF)
bm = torch.zeros((1, -(-VOCAB // 32)), dtype=torch.int32)
g.fill_bitmask(bm, 0)
allowed0 = allowed_ids(bm[0], VOCAB)
out["mask"] = {"allowed_ids": allowed0,
               "allowed_tokens": [repr(tok.decode([t])) for t in allowed0]}

# 构造 logits：非法 token 4242('####') 得分最高
logits = torch.full((1, VOCAB), -10.0, device="cuda", dtype=torch.float32)
logits[0, OFF_GRAMMAR] = 5.0
logits[0, TOK_YES] = 2.0
logits[0, TOK_NO] = 1.0
probs_before = torch.softmax(logits, dim=-1)
argmax_before = int(logits[0].argmax())

# 真 xgr apply（CUDA kernel，bit=0 → -inf）
gpu_bm = bm.to("cuda", non_blocking=True)
xgr.apply_token_bitmask_inplace(logits, gpu_bm)
torch.cuda.synchronize()

probs_after = torch.softmax(logits / 1.0, dim=-1)
argmax_after = int(logits[0].argmax())
out["inf_apply"] = {
    "logit_4242_before": 5.0,
    "logit_8505_before": 2.0,
    "argmax_before": argmax_before,
    "argmax_before_token": repr(tok.decode([argmax_before])),
    "logit_4242_after": float(logits[0, OFF_GRAMMAR].item()),
    "logit_8505_after": float(logits[0, TOK_YES].item()),
    "prob_4242_before": float(probs_before[0, OFF_GRAMMAR].item()),
    "prob_4242_after_softmax": float(probs_after[0, OFF_GRAMMAR].item()),
    "argmax_after": argmax_after,
    "argmax_after_token": repr(tok.decode([argmax_after])),
}

# 温度正交性：softmax(logits/T) 在掩码位恒 0（T 变化不改 -inf）
temp_probs = {}
for T in (1.0, 0.5, 0.01, 10.0):
    p = torch.softmax(logits / T, dim=-1)
    temp_probs[f"T={T}"] = {
        "prob_4242": float(p[0, OFF_GRAMMAR].item()),
        "prob_8505": float(p[0, TOK_YES].item()),
    }
temp_probs["logit_4242_div_T10"] = float(logits[0, OFF_GRAMMAR].item())  # -inf/10 仍 -inf
out["temperature_orthogonality"] = {
    "masked_prob_at_temps": temp_probs,
    "note": "-inf 除以任何正温度仍为 -inf → exp 下溢精确 0.0（fp32 位级）——"
            "九步采样变换（ch30）对掩码无感知的全部理由",
}

# top-k 正交性：top-1 候选
topk = torch.topk(logits[0], 3)
out["topk_after_mask"] = {
    "top3_ids": topk.indices.tolist(),
    "top3_tokens": [repr(tok.decode([t])) for t in topk.indices.tolist()],
    "top3_values": [float(v) for v in topk.values],
}

# ── 对照：有限负值掩码的残留（-100 而非 -inf）─────────────────────────────────
finite = torch.tensor([5.0, 2.0, -100.0], dtype=torch.float32)
inf_row = torch.tensor([5.0, 2.0, float("-inf")], dtype=torch.float32)
p_fin = torch.softmax(finite / 1.0, dim=-1)
p_fin_hot = torch.softmax(finite / 10.0, dim=-1)   # 高温放大掩码不足
p_inf_hot = torch.softmax(inf_row / 10.0, dim=-1)
out["finite_mask_contrast"] = {
    "logits": [5.0, 2.0, -100.0],
    "prob_masked_T1": float(p_fin[2].item()),
    "prob_masked_is_zero_T1": float(p_fin[2].item()) == 0.0,
    "prob_masked_T10": float(p_fin_hot[2].item()),
    "prob_masked_is_zero_T10": float(p_fin_hot[2].item()) == 0.0,
    "prob_inf_masked_T10": float(p_inf_hot[2].item()),
    "prob_inf_masked_is_zero_T10": float(p_inf_hot[2].item()) == 0.0,
    "note": "有限负值 -100：T=1 下 exp(-105) 已低于 fp32 最小次正规数（~1.4e-45）"
            "→ 数值上恰好 0.0；但高温 T=10 下残留涨到 1.58e-05（非零——千 token/秒"
            "的采样必踩）；-inf 在任何温度下都是精确 0.0——『为什么必须是 -inf 而"
            "不是一个很大的负数』的数值答案：有限值的保护随温度变宽而失效",
}

dump("trace_m12_inf.json", out)
print("argmax flip:", out["inf_apply"]["argmax_before"], "->", out["inf_apply"]["argmax_after"])
print("temps:", {k: v["prob_4242"] for k, v in temp_probs.items() if isinstance(v, dict)})
print("finite contrast:", {k: v for k, v in out["finite_mask_contrast"].items() if k != "note"})
