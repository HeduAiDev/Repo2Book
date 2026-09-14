# SOURCE: vllm/utils/platform_utils.py
# HOST SEAM：num_compute_units 的名义位（真实按当前设备返回 SM/CU 数——
# builder 的 DeepGEMM scheduler_metadata_buffer 尺寸消费；host 无设备，返回
# CPU 核数作名义值，不影响语义——schedule_metadata 在 host 面不参与打分数值）。
from __future__ import annotations

import os


# SOURCE: vllm/utils/platform_utils.py num_compute_units —— HOST SEAM
def num_compute_units(device_index) -> int:
    # SOURCE: vllm/utils/platform_utils.py num_compute_units —— HOST SEAM
    return os.cpu_count() or 8
