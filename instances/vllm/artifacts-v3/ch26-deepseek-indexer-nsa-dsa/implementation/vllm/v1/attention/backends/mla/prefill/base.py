# SOURCE: vllm/v1/attention/backends/mla/prefill/base.py
# ch26 消费面：MLAPrefillBackend 名字位（类型面；实现族归 ch25）。
from __future__ import annotations


# SOURCE: vllm/v1/attention/backends/mla/prefill/base.py MLAPrefillBackend
#   —— 章界标记类（ABC 面归 ch25）
class MLAPrefillBackend:
    """MLA dense-MHA prefill backend family (ch25 domain)."""

    def clone(self):
        # SOURCE: vllm/v1/attention/backends/mla/prefill/base.py clone 位
        raise NotImplementedError
