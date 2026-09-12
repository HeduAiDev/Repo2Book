# SOURCE: vllm/utils/platform_utils.py
# HOST SEAM：num_compute_units 的 host 退化（真实经 pynvml/NVML 查 SM 数
# ——FlashMLA builder 的 CG 缓冲上界；host 无设备，返回名义 8）。
from __future__ import annotations


# SOURCE: vllm/utils/platform_utils.py num_compute_units —— HOST SEAM
def num_compute_units(device_index) -> int:
    # SOURCE: vllm/utils/platform_utils.py —— HOST SEAM 名义位
    return 8
