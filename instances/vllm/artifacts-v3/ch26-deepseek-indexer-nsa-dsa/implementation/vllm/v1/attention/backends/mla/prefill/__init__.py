# SOURCE: vllm/v1/attention/backends/mla/prefill/__init__.py
# ch26 消费面（ch25 同源域）：MLA prefill 家族的名字位——get_mla_prefill_
# backend 在 host（无 FA/FlashInfer）上的真实行为是『无可选后端 → ValueError』
#（MLAAttention.__init__ 的 try/except 消费此面：sparse 无 dense-MHA prefill
# 时落 top-k MQA-only）。完整注册表与优先级面归 ch25。
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend  # noqa: F401
from vllm.v1.attention.backends.mla.prefill.registry import (  # noqa: F401
    get_mla_prefill_backend,
)
