# ch32 m18 驱动：use_v2_model_runner 决策表（v0.27.1 两条 worker 落地路径的真实
# 选择器——真 VllmConfig property，精简版逐字）。13 路配置对照，产出 V1/V2 判定。
import os
import sys

from _ch32_common import dump

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))
from vllm.config import (DiffusionConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)

out = {"env": {"note": "真 use_v2_model_runner property（vllm/config/vllm.py:L577-L623 "
                       "逐字承载）；env 用 os.environ 注入；HAS_TRITON 用模块属性"
                       "覆盖（精简版 PEP 562 读取面）——与真实 env 语义一致"}}

import vllm.config.vllm as cv


def decide(name, *, env=None, parallel=None, spec=None, diffusion=None, model=None,
           has_triton=True, expect_note=""):
    if env is not None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = env
    else:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    saved = cv.HAS_TRITON
    cv.HAS_TRITON = has_triton
    try:
        cfg = VllmConfig(
            scheduler_config=SchedulerConfig(),
            model_config=model if model is not None else ModelConfig(
                architectures=("LlamaForCausalLM",)),
            speculative_config=spec,
            diffusion_config=diffusion,
            parallel_config=parallel if parallel is not None else ParallelConfig(),
        )
        result = cfg.use_v2_model_runner
    finally:
        cv.HAS_TRITON = saved
    return {"case": name, "use_v2": result, "why": expect_note}


cases = []
# 1-2 env 优先
cases.append(decide("env=1 强制 V2", env="1", expect_note="env 显式设置优先，直接短路"))
cases.append(decide("env=0 强制 V1", env="0",
                    expect_note="env=0：即便稠密默认 V2 也被压回 V1"))
# 3 PCP>1
_pcp_model = ModelConfig(architectures=("LlamaForCausalLM",))
_pcp_model.use_mla = True  # 真实 ModelConfig 字段（简化消费面经 getattr 读取）
cases.append(decide("PCP=2", parallel=ParallelConfig(prefill_context_parallel_size=2),
                    model=_pcp_model,
                    expect_note="PCP 运行时只在 V2 实现（无 MLA 也 force V2）"))
# 4 dspark
cases.append(decide("spec=dspark", spec=SpeculativeConfig(method="dspark"),
                    expect_note="DSpark 只在 V2 GPU runner 实现 → 强制 V2"))
# 5 diffusion
cases.append(decide("diffusion", diffusion=DiffusionConfig(canvas_length=64),
                    expect_note="diffusion 模型强制 V2"))
# 6 稠密默认（v0.27.1 的反转主角）
cases.append(decide("Llama 稠密生成（默认）", expect_note="非 MoE 生成模型 → V2"
                    "（v0.21『V2 opt-in』在 v0.27 反转为默认）——Llama 走 V2"))
# 7 MoE 不在名单
cases.append(decide("Mixtral MoE（不在默认名单）",
                    model=ModelConfig(architectures=("MixtralForCausalLM",), is_moe=True),
                    expect_note="MoE 且架构不在默认 V2 名单 → V1"))
# 8 MoE 在名单
cases.append(decide("Qwen2Moe（在默认名单）",
                    model=ModelConfig(architectures=("Qwen2MoeForCausalLM",), is_moe=True),
                    expect_note="名单内 MoE → V2"))
# 9 pooling
cases.append(decide("pooling 模型",
                    model=ModelConfig(architectures=("BertModel",), runner_type="pooling"),
                    expect_note="runner_type != generate → V1"))
# 10 attention-free
cases.append(decide("attention-free",
                    model=ModelConfig(architectures=("FalconH1ForCausalLM",),
                                      is_attention_free=True),
                    expect_note="is_attention_free → V1"))
# 11 无 Triton
cases.append(decide("稠密但无 Triton", has_triton=False,
                    expect_note="V2 requires Triton → 警告并回退 V1"))
# 12 ngram spec
cases.append(decide("spec=ngram（稠密）",
                    spec=SpeculativeConfig(method="ngram", num_speculative_tokens=3),
                    expect_note="ngram/ngram_gpu 未被 V2 支持 → 警告回退 V1"))
# 13 eagle spec
cases.append(decide("spec=eagle（稠密）",
                    spec=SpeculativeConfig(method="eagle", num_speculative_tokens=3),
                    expect_note="eagle 在 V2 支持名单 → V2"))

out["decision_table"] = cases
out["default_v2_architectures"] = sorted(cv.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES)

os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
dump("trace_m18_selector.json", out)
for c in cases:
    print(("V2" if c["use_v2"] else "V1"), "|", c["case"])
