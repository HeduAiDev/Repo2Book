"""ch32 explainer driver t06 - m17 为什么写 -inf 而不是一个「很大的负数」.

场景(GPU 真机 + 真 Triton kernel 做 -inf 那一路):
  |V|=96,语法此刻只允许 token 42;但模型此刻恰好不想要它——
  legal logit = -25.0,其余 95 个非法 token 的原始 logit = 5.0。
对照三种「屏蔽值」在 softmax 后的非法 token 总概率:
  -inf(kernel 实际写入)/ 有限常数 C=-20 / 有限常数 C=-10000,
再叠加温度与 top_k 这两种 ch30 的采样变换。
概率一律以字符串形式按 6 位小数记录(避免科学计数法的读数歧义)。
输出 JSON 存 t06_neg_inf_vs_constant.json。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CH, "implementation"))
sys.path.insert(0, os.path.join(CH, "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from input_batch import InputBatch  # noqa: E402
from structured_outputs import StructuredOutputsWorker  # noqa: E402

VOCAB = 96
LEGAL = 42
LEGAL_LOGIT = -25.0
OTHER_LOGIT = 5.0
device = torch.device("cuda")


def base_logits():
    x = torch.full((1, VOCAB), OTHER_LOGIT, dtype=torch.float32, device=device)
    x[0, LEGAL] = LEGAL_LOGIT
    return x


# ---- -inf 那一路:用真实 kernel 写入 -------------------------------------------
bm = np.zeros((1, -(-VOCAB // 32)), dtype=np.int32)
bm[0, LEGAL // 32] |= np.int32(1 << (LEGAL % 32))
cu = np.array([0, 1], dtype=np.int32)
ib = InputBatch(req_ids=["rA"],
                logits_indices=torch.arange(1, dtype=torch.int32, device=device),
                cu_num_logits=torch.from_numpy(cu).to(device),
                cu_num_logits_np=cu, has_structured_output_reqs=True)
worker = StructuredOutputsWorker(max_num_logits=4, vocab_size=VOCAB, device=device)
logits_inf = base_logits()
worker.apply_grammar_bitmask(logits_inf, ib, ["rA"], bm)
torch.cuda.synchronize()
kernel_wrote_neg_inf = bool(torch.isneginf(logits_inf[0, 0]).item())


def masked_with_constant(c):
    x = base_logits()
    mask = torch.ones(VOCAB, dtype=torch.bool, device=device)
    mask[LEGAL] = False
    x[0, mask] = c
    return x


def probs(x, temperature=1.0, top_k=None):
    y = x[0].double() / temperature
    if top_k is not None:
        kth = torch.topk(y, top_k).values[-1]
        y = torch.where(y >= kth, y, torch.tensor(float("-inf"), dtype=torch.float64,
                                                  device=y.device))
    p = torch.softmax(y, dim=-1)
    p_legal = float(p[LEGAL].item())
    p_illegal = float(1.0 - p_legal)
    legal_in_topk = True if top_k is None else bool(
        (torch.topk(x[0].double() / temperature, top_k).indices == LEGAL).any().item())
    return {"P_legal_token_42": "%.6f" % p_legal,
            "P_illegal_total": "%.6f" % p_illegal,
            "legal_token_in_top_k": legal_in_topk}


rows = []
configs = [
    ("-inf (kernel 实际写入)", logits_inf, 1.0, None),
    ("-inf (kernel 实际写入)", logits_inf, 0.1, None),
    ("-inf (kernel 实际写入)", logits_inf, 1.0, 4),
    ("有限常数 C=-20", masked_with_constant(-20.0), 1.0, None),
    ("有限常数 C=-20", masked_with_constant(-20.0), 0.1, None),
    ("有限常数 C=-20", masked_with_constant(-20.0), 1.0, 4),
    ("有限常数 C=-10000", masked_with_constant(-10000.0), 1.0, None),
    ("有限常数 C=-10000", masked_with_constant(-10000.0), 1.0, 4),
]
for name, x, t, k in configs:
    r = probs(x, temperature=t, top_k=k)
    r.update({"mask_value": name, "temperature": t,
              "top_k": k if k is not None else 0})
    rows.append(r)

out = {
    "params": {"vocab_size": VOCAB, "legal_token_id": LEGAL,
               "legal_token_logit": LEGAL_LOGIT,
               "illegal_token_logit_before_mask": OTHER_LOGIT,
               "num_illegal_tokens": VOCAB - 1},
    "kernel_wrote_neg_inf_on_illegal": kernel_wrote_neg_inf,
    "rows": rows,
    "note": "C=-10000 在本例「碰巧」有效,只因它低于每个合法 logit 且 exp 下溢到 0;"
            "把 legal logit 降到 -10001 它同样失效。-inf 是唯一与 logit 取值范围无关的选择。",
}

# 反例:C=-10000 也会失效——把合法 token 的 logit 压到 -10001
x = base_logits()
x[0, LEGAL] = -10001.0
mask = torch.ones(VOCAB, dtype=torch.bool, device=device)
mask[LEGAL] = False
x[0, mask] = -10000.0
r = probs(x, temperature=1.0, top_k=None)
r.update({"mask_value": "有限常数 C=-10000, 合法 logit=-10001",
          "temperature": 1.0, "top_k": 0})
out["counterexample_constant_below_legal_logit"] = r

path = os.path.join(HERE, "t06_neg_inf_vs_constant.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
