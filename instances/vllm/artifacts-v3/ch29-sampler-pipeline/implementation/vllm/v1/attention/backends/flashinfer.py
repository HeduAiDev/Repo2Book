# SOURCE: vllm/v1/attention/backends/flashinfer.py
# HOST SEAM：本章消费面一件——flashinfer_sampler_supported 的能力裁决
# import（topk_topp_sampler.py:L51 `from vllm.v1.attention.backends.flashinfer
# import FlashInferBackend`，读 supports_compute_capability 判断 SM 版本
# 是否可跑 FlashInfer 采样）。真实 backend 类数百行（注意力域，归 ch21）；
# 此处只镜像被消费的类名与该方法（逐字）。flashinfer 本体仍由
# flashinfer_sample 里的 `import flashinfer` 直连真实安装。
from __future__ import annotations

from vllm.platforms import DeviceCapability


# SOURCE: vllm/v1/attention/backends/flashinfer.py:L343 FlashInferBackend
#   —— HOST SEAM：类位（真实继承 AttentionBackend，归 ch21 域）+ 被消费的
#   supports_compute_capability（逐字）
class FlashInferBackend:
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FlashInfer supports SM75+, but is currently broken on SM75 (Turing):
        # https://github.com/flashinfer-ai/flashinfer/issues/3620 (fix:
        # https://github.com/flashinfer-ai/flashinfer/pull/3621). Temporarily
        # raise the floor to SM80 so it is not auto-selected on SM75 until
        # that fix lands; revert to DeviceCapability(7, 5) once it does.
        # SOURCE: vllm/v1/attention/backends/flashinfer.py:L461-L470 FlashInferBackend.supports_compute_capability —— 逐字 （SM80–SM121：SM75 Turing 上 FlashInfer 采样已坏、临时抬门槛的 注释与边界原样保留）
        return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability(
            12, 1
        )
