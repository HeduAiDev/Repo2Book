# ch29 m06 惩罚三件套张量化 — 驱动脚本（host）。
# 机制：step6 apply_penalties → apply_all_penalties（make_tensor_with_pad+H2D+−1
# 占位替换）→ get_token_bin_counts_and_mask（scatter_add_ 计数）→ repetition 自定义
# op（正除负乘）+ frequency×出现次数 + presence×是否出现（OpenAI 定义）。
# 行为基准：vllm/v1/sample/sampler.py:L419-L436、ops/penalties.py:L10-L56、
# model_executor/layers/utils.py:L34-L89、_custom_ops.py:L309-L323（真实 v0.27.1 行号；
# host seam：repetition 的 torch 参考算式与 CUDA 核逐元素等价）。
import json
import os
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from vllm.model_executor.layers.utils import (
    apply_penalties as apply_penalties_real,
    get_token_bin_counts_and_mask,
)
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.penalties import _convert_to_tensors
from vllm.v1.sample.sampler import Sampler

R = lambda x: round(float(x), 4)
out = {}

# ── 玩具词表 V=5：logits / prompt / output / 三惩罚系数 ──────────────
V = 5
logits0 = torch.tensor([[0.5, 1.0, 2.0, 0.0, -0.5]])
prompt = [[0, 1]]                      # 请求 0 的 prompt
generated = [[2, 2, 3]]                # 请求 0 已生成：token2 两次、token3 一次
freq = torch.tensor([0.5])
pres = torch.tensor([1.0])
rep = torch.tensor([1.2])
prompt_t = torch.tensor(prompt)
output_t = torch.tensor(generated)

# 逐 token 计数与出现 mask（真实 scatter_add_ 实现）
p_counts, p_mask = get_token_bin_counts_and_mask(prompt_t, V, 1)
o_counts, o_mask = get_token_bin_counts_and_mask(output_t, V, 1)
appearance = (p_mask | o_mask)[0].tolist()

# 分步算账：repetition 先（自定义 op），freq/pres 后（OpenAI 两式）
after_rep = logits0.clone()
from vllm._custom_ops import apply_repetition_penalties
apply_repetition_penalties(after_rep, p_mask, o_mask, rep)
after_freq = after_rep - freq.unsqueeze(1) * o_counts
after_pres = after_freq - pres.unsqueeze(1) * o_mask

# 真链一步到位（Sampler.apply_penalties → apply_all_penalties → apply_penalties）
md = dict(
    temperature=torch.tensor([1.0]), all_greedy=False, all_random=False,
    top_p=None, top_k=None, generators={}, max_num_logprobs=None,
    no_penalties=False, prompt_token_ids=torch.tensor(prompt),
    frequency_penalties=freq, presence_penalties=pres, repetition_penalties=rep,
    output_token_ids=generated, allowed_token_ids_mask=None,
    bad_words_token_ids={}, logitsprocs=LogitsProcessors(),
)
full_chain = Sampler.apply_penalties(logits0.clone(), SamplingMetadata(**md), generated)

out["penalties_trio"] = {
    "vocab_size": V,
    "logits": [0.5, 1.0, 2.0, 0.0, -0.5],
    "prompt_tokens": prompt[0],
    "output_tokens": generated[0],
    "coefficients": {"frequency": 0.5, "presence": 1.0, "repetition": 1.2},
    "output_bin_counts_scatter_add": o_counts[0].tolist(),
    "output_appearance_mask": o_mask[0].tolist(),
    "prompt_appearance_mask": p_mask[0].tolist(),
    "appearance_union_rep_scope": appearance,
    "logits_after_repetition": [R(v) for v in after_rep[0]],
    "repetition_rule": "已出现位: logit>0 → ÷1.2 / logit<0 → ×1.2（token3 logit=0.0 不变）；未出现位 no-op",
    "freq_delta": [R(v) for v in (-freq.unsqueeze(1) * o_counts)[0]],
    "pres_delta": [R(v) for v in (-pres.unsqueeze(1) * o_mask)[0]],
    "logits_after_all": [R(v) for v in after_pres[0]],
    "full_chain_logits": [R(v) for v in full_chain[0]],
    "argmax_before": 2,
    "argmax_after": 1,
    "claim": "repetition(÷1.2)+frequency(−0.5×2)+presence(−1.0×1) 三式联手把复读头名 token2 从 2.0 压到 -0.3333 → argmax 2 → 1",
}

# ── async 的 −1 占位替换：scatter 前把非法 -1 换成 vocab_size ─────────
out_raw = [[2, 2, 3], [-1, -1]]        # 行1 = async 下无惩罚行的占位
t_raw = _convert_to_tensors(out_raw, V, torch.device("cpu"))
t_fixed = t_raw.clone()
t_fixed.masked_fill_(t_fixed == -1, V)
c2, m2 = get_token_bin_counts_and_mask(t_fixed, V, 2)
row1_logits = torch.tensor([
    [0.5, 1.0, 2.0, 0.0, -0.5],      # 行0（同 A 部分，吃满三惩罚）
    [1.0, 2.0, 3.0, 0.0, -1.0],      # 行1（async 无惩罚行，-1 占位）
])
md2_fields = dict(
    temperature=torch.tensor([1.0, 1.0]), all_greedy=False, all_random=False,
    top_p=None, top_k=None, generators={}, max_num_logprobs=None,
    no_penalties=False, prompt_token_ids=torch.tensor([[0, 1], [0, 1]]),
    frequency_penalties=torch.tensor([0.5, 0.0]), presence_penalties=torch.tensor([1.0, 0.0]),
    repetition_penalties=torch.tensor([1.2, 1.0]),
    output_token_ids=out_raw, allowed_token_ids_mask=None,
    bad_words_token_ids={}, logitsprocs=LogitsProcessors(),
)
after2 = Sampler.apply_penalties(
    row1_logits.clone(), SamplingMetadata(**md2_fields), out_raw)
out["async_placeholder_minus_one"] = {
    "output_token_ids_raw": out_raw,
    "tensor_before_replace": t_raw.tolist(),
    "replace_rule": "masked_fill_(==-1, vocab_size=5)——pad 位在 bin_counts[:, :vocab_size] 切片时丢弃",
    "tensor_after_replace": t_fixed.tolist(),
    "row1_bin_counts": c2[1].tolist(),
    "row1_penalties_all_zero": True,
    "row1_logits_before": [1.0, 2.0, 3.0, 0.0, -1.0],
    "row1_logits_after": [R(v) for v in after2[1]],
    "claim": "−1 是非法 token id，scatter 前必须换成合法 pad 位 vocab_size；换完计数落 pad 列、切片丢弃 → 无惩罚行 logits 逐位不变",
}

out["table_rows_echo"] = [
    ["历史盘点(scatter_add_)", "output=[2, 2, 3]", "bin_counts=[0, 0, 2, 1, 0]", "出现 mask={2, 3}", "prompt mask={0, 1}——repetition 吃并集 {0, 1, 2, 3}"],
    ["step6a repetition(r=1.2)", "已出现位正 logit ÷1.2", "t0: 0.5→0.4167 / t1: 1.0→0.8333 / t2: 2.0→1.6667", "t3: 0.0 不变(零既不正也不负)", "t4 未出现: -0.5 原样"],
    ["step6b frequency(OpenAI)", "logit −= 0.5×出现次数", "t2: −0.5×2=−1.0 / t3: −0.5×1=−0.5", "未出现位 −0", "叠加后 t2=0.6667"],
    ["step6c presence(OpenAI)", "logit −= 1.0×是否出现", "t2: −1.0 / t3: −1.0", "出现≠次数、只看有没有", "叠加后 t2=-0.3333 / t3=-1.5"],
    ["结果", "final=[0.4167, 0.8333, -0.3333, -1.5, -0.5]", "argmax 2 → 1", "复读头名被三式压下", "与真链 full_chain 逐位一致"],
    ["async −1 占位", "行1 output=[-1, -1]", "替换 -1→5(pad)", "bin_counts=[0,0,0,0,0]→pad 列", "行1 penalties 全 0 → logits 逐位不变"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m06_penalties.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
