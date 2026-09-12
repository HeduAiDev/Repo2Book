# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/platforms/__init__.py —— HOST SEAM（最小承载，ch29 同款）：
# 本章消费面只有 get_max_tokens（api_utils.py:L190）调
# current_platform.get_max_output_tokens(input_length)——真实平台层按
# 平台上限（如 TPU 地址空间约束）取输出预算上限，CPU/GPU 通用路径返回
# None（不设限）。host 精简环境统一走 None 支路。
# SUBTRACTED: Platform 接口全量（detect_platform/DeviceCapability/
# 各后端能力面）——平台层归 Part V 执行栈域。


# SOURCE: vllm/platforms/interface.py —— HOST SEAM：current_platform 的
# 最小替身（get_max_output_tokens → None，与真实 CPUPlatform 行为一致）
class _CpuLikePlatform:
    # SOURCE: vllm/platforms/interface.py —— HOST SEAM：平台上限不设限位
    def get_max_output_tokens(self, input_length: int) -> int | None:
        return None


current_platform = _CpuLikePlatform()
