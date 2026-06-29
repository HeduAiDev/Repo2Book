# vllm/v1/attention/backends/flash_attn.py —— subtract-only 精简版（伪装样板）
#
# 本章点(3) 伪装 HACK 的「样板对象」：CUDA FlashAttention 后端的 get_name 返回 "FLASH_ATTN"。
# 昇腾在 V2 model-runner 下原样冒充这个名字，绕过 vLLM 按名字（而非类型）判定后端的断言。
#
# 整个 FlashAttentionBackend 有上千行（含 flash_attn 算子封装），本章只取 get_name 这一处样板。
from vllm.v1.attention.backend import AttentionBackend


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L69（类定义）
class FlashAttentionBackend(AttentionBackend):
    # SUBTRACTED: get_supported_dtypes / get_impl_cls / get_builder_cls / get_kv_cache_shape /
    #   supports_batch_invariance 等以及全部 flash_attn 算子实现（flash_attn.py:L70+，约上千行）——
    #   本章只需 get_name 这一处「伪装样板」，其余 CUDA FA 算子与昇腾路由立意正交，折叠。

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L105-L107
        return "FLASH_ATTN"
