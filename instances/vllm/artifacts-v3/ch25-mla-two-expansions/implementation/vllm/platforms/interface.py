# SOURCE: vllm/platforms/interface.py
# HOST SEAM：DeviceCapability 载体（真实为 interface.py 的 @dataclass
# major/minor 元组序；prefill selector 的优先级分档消费它）。
from __future__ import annotations

from dataclasses import dataclass


# SOURCE: vllm/platforms/interface.py DeviceCapability —— HOST SEAM 逐字段
@dataclass
class DeviceCapability:
    # SOURCE: vllm/platforms/interface.py DeviceCapability.major
    major: int
    # SOURCE: vllm/platforms/interface.py DeviceCapability.minor
    minor: int = 0

    def __repr__(self):
        # SOURCE: vllm/platforms/interface.py DeviceCapability.__repr__
        return f"DeviceCapability(major={self.major}, minor={self.minor})"

    def as_version_str(self):
        # SOURCE: vllm/platforms/interface.py as_version_str
        return f"{self.major}.{self.minor}"
