# SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py
# ch26 消费面：get_mla_prefill_backend 的『无可用后端』真实失败面（host 无
# FA4/FlashInfer——与真实无加速器平台同型；MLAAttention 捕获 ValueError 落
# sparse top-k MQA-only）。选择面归 ch25。
from __future__ import annotations


# SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py
#   get_mla_prefill_backend —— 减法子集（无可用支逐字语义）
def get_mla_prefill_backend(vllm_config):
    # SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py —— HOST SEAM
    #   （显式 mla_prefill_backend 配置与可用性探测面归 ch25；host 无可选
    #   后端 → ValueError，与真实无 FA 平台同型）
    raise ValueError(
        "No MLA prefill backend supports this model on the current platform."
    )
