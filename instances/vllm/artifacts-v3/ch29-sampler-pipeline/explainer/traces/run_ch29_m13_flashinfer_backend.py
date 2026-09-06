# ch29 m13 FlashInfer 拒绝采样后端 — 驱动脚本（GPU 容器实跑，flashinfer 在场）。
# 机制：flashinfer_sample（topk_topp_sampler.py:L475-L512）按 k/p 是否为 None
# 分三条 FlashInfer API、全部 deterministic=True；与 random_sample 统计等价
# 而非逐位相同（docstring 原话 statistically equivalent）；forward_cuda 的
# 回退（k/p 全 None 或有逐请求 generator → forward_native）；构造期绑定的
# 判据 flashinfer_sampler_supported()（L28-L74：env+平台+算力）与
# PROCESSED_LOGPROBS_MODES 强制 native。
# 行为基准：vllm/v1/sample/ops/topk_topp_sampler.py:L77-L129/L155-L182/L475-L512
# （真实 v0.27.1 行号）。
import json
import os
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMPL = pathlib.Path(__file__).resolve().parent.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))
# 本驱动须在 VLLM_USE_FLASHINFER_SAMPLER 默认态（True）跑——构造期绑定正支路。
assert os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "1") != "0"

import torch

from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.platforms import current_platform
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p_pytorch,
    flashinfer_sample,
    flashinfer_sampler_supported,
    random_sample,
)

R = lambda x: round(float(x), 4)
dev = torch.device("cuda")
out = {
    "environment": {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "1(默认)"),
    }
}

# ── A. 构造期绑定：is_cuda + flashinfer 可用 → forward_cuda ─────────
cap = current_platform.get_device_capability()
sampler = TopKTopPSampler()  # 默认 raw_logprobs
sampler_processed = TopKTopPSampler(logprobs_mode="processed_logprobs")
out["constructor_binding"] = {
    "is_cuda": bool(current_platform.is_cuda()),
    "capability": f"sm_{cap.major}{cap.minor}",
    "flashinfer_sampler_supported": bool(flashinfer_sampler_supported()),
    "bound_forward_default_raw_mode": sampler.forward.__name__,
    "bound_forward_processed_logprobs_mode": sampler_processed.forward.__name__,
    "PROCESSED_LOGPROBS_MODES": sorted(PROCESSED_LOGPROBS_MODES),
    "claim": "构造期一次性绑定：默认 raw 模式绑 forward_cuda；processed 两态拿不到截断后中间量 → 强制 forward_native（L96-L102/L109-L115）",
}

# ── B. 三条 FlashInfer API 分发（间谍记录、透真实现）───────────────
import flashinfer.sampling as fis

api_calls = []


def make_spy(name, real):
    def spy(*args, **kwargs):
        api_calls.append({"api": name, "deterministic": kwargs.get("deterministic")})
        return real(*args, **kwargs)
    return spy


fis.top_p_sampling_from_probs = make_spy("top_p_sampling_from_probs", fis.top_p_sampling_from_probs)
fis.top_k_sampling_from_probs = make_spy("top_k_sampling_from_probs", fis.top_k_sampling_from_probs)
fis.top_k_top_p_sampling_from_logits = make_spy("top_k_top_p_sampling_from_logits", fis.top_k_top_p_sampling_from_logits)

V = 5
logits5 = torch.tensor([[3.0, 2.0, 1.0, 0.5, 0.1]], device=dev)
k1 = torch.tensor([2], device=dev)
p1 = torch.tensor([0.9], device=dev)

tok_k = flashinfer_sample(logits5.clone(), k1.clone(), None)
tok_p = flashinfer_sample(logits5.clone(), None, p1.clone())
tok_kp = flashinfer_sample(logits5.clone(), k1.clone(), p1.clone())
out["three_api_branches"] = {
    "vocab": V,
    "logits": [3.0, 2.0, 1.0, 0.5, 0.1],
    "k_only": {"k": 2, "api": api_calls[0]["api"], "token": int(tok_k[0])},
    "p_only": {"p": 0.9, "api": api_calls[1]["api"], "token": int(tok_p[0])},
    "both_kp": {"k": 2, "p": 0.9, "api": api_calls[2]["api"], "token": int(tok_kp[0])},
    "all_deterministic_True": all(c["deterministic"] is True for c in api_calls[:3]),
    "entry_note": "k 单独/p 单独 → 先 softmax 成 probs 再进 *_from_probs；双参 → top_k_top_p_sampling_from_logits 直接吃 logits、拒绝采样免排序（L487-L508）",
    "claim": "k/p 是否为 None 三分 API、全部 deterministic=True——docstring 自述与 random_sample 统计等价（statistically equivalent）而非逐位相同",
}

# ── C. 统计等价：flashinfer vs random_sample(Gumbel) vs 理论条件分布 ──
# 5-token 分布截 top-2：条件分布 = [p0/(p0+p1), p1/(p0+p1)] ≈ [0.7311, 0.2689]
probs5 = torch.softmax(logits5, dim=-1)[0]
cond0 = (probs5[0] / (probs5[0] + probs5[1])).item()
cond1 = (probs5[1] / (probs5[0] + probs5[1])).item()
N = 100000
torch.manual_seed(77)
fi_tokens = flashinfer_sample(logits5.repeat(N, 1), torch.full((N,), 2, dtype=torch.long, device=dev), None)
fi_freq = torch.bincount(fi_tokens, minlength=V).float() / N
trunc = apply_top_k_top_p_pytorch(logits5.clone(), k1.clone(), None)
torch.manual_seed(77)
gu_tokens = random_sample(trunc.repeat(N, 1).softmax(dim=-1), {})
gu_freq = torch.bincount(gu_tokens, minlength=V).float() / N
out["statistical_equivalence"] = {
    "probs": [R(p) for p in probs5.tolist()],
    "k": 2,
    "theoretical_conditional": [R(cond0), R(cond1)],
    "N": N,
    "flashinfer_freq_token0_token1": [R(fi_freq[0].item()), R(fi_freq[1].item())],
    "gumbel_freq_token0_token1": [R(gu_freq[0].item()), R(gu_freq[1].item())],
    "tail_tokens_freq_sum": R((fi_freq[2:].sum() + gu_freq[2:].sum()).item()),
    "claim": "k=2 截断后两家频率双双贴住条件分布——统计等价而非逐位相同（两家用各自的 RNG，逐位不比对）",
}

# ── D. forward_cuda 回退（走真 forward_cuda 代码，包实例 forward_native 记录）──
s = TopKTopPSampler()  # 构造期已绑 forward_cuda
native_paths = []
real_native = s.forward_native  # 已绑方法


def wrapped_native(logits, generators, k, p):
    native_paths.append({"k_none": k is None, "p_none": p is None, "has_generators": bool(generators)})
    return real_native(logits, generators, k, p)


s.forward_native = wrapped_native  # 实例属性遮蔽——真 forward_cuda 内 self.forward_native 会命中
_ = s(logits5.clone(), {}, None, None)  # k/p 全 None → 真分支递给 native
_ = s(logits5.clone(), {0: torch.Generator(device=dev).manual_seed(1)}, k1.clone(), None)  # 逐请求 generator → native
_ = s(logits5.clone(), {}, k1.clone(), None)  # 正常 k → 不经 native（直达 flashinfer）
out["forward_cuda_fallbacks"] = {
    "case_no_filter": native_paths[0],
    "case_per_request_generator": native_paths[1],
    "normal_k_call_hits_native": len(native_paths) > 2,
    "case_fp64": "use_fp64_gumbel=True → forward_native（L167-L168）——构造参数默认 False，本例未开",
    "claim": "FlashInfer 0.2.3+ 不支持逐请求 generator、对无截断的调用无事可做——两类都递回 forward_native（L159-L166）；正常 k 调用直达 flashinfer_sample",
}

out["table_rows_echo"] = [
    ["构造期绑定", "is_cuda=True + supported=True", f"capability=sm_{cap.major}{cap.minor}", f"默认模式绑 {sampler.forward.__name__}", "一次性、运行期零分支"],
    ["processed 模式", "logprobs_mode=processed_logprobs", f"绑 {sampler_processed.forward.__name__}", "FlashInfer 拿不到截断后中间量", "PROCESSED 两态强制 native"],
    ["k 单独", "k=2", api_calls[0]["api"], f"token={int(tok_k[0])}", "先 softmax 再进 API"],
    ["p 单独", "p=0.9", api_calls[1]["api"], f"token={int(tok_p[0])}", "先 softmax 再进 API"],
    ["k+p 双参", "k=2, p=0.9", api_calls[2]["api"], f"token={int(tok_kp[0])}", "直接吃 logits、免排序"],
    ["统计等价(N=10万)", f"理论条件分布=[{R(cond0)}, {R(cond1)}]", f"flashinfer 频率=[{R(fi_freq[0].item())}, {R(fi_freq[1].item())}]", f"Gumbel 频率=[{R(gu_freq[0].item())}, {R(gu_freq[1].item())}]", "等价不逐位"],
    ["回退·无过滤", "k=None, p=None", "→ forward_native", "FlashInfer 无事可做", "L159 条件 (k is None and p is None)"],
    ["回退·逐请求 seed", "generators 非空", "→ forward_native", "FlashInfer 0.2.3+ 不支持", "与 m10 的逐请求覆写路汇合"],
]

print(json.dumps(out, ensure_ascii=False, indent=1))
with open(pathlib.Path(__file__).parent / "ch29_m13_flashinfer_backend.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
