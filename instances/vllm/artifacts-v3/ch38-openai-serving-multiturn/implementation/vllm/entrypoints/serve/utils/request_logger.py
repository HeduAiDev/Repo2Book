# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/serve/utils/request_logger.py —— HOST SEAM（最小
# 承载）：RequestLogger 是 serving 对象的可选观测件（enable_log_outputs
# 的日志块已按 delete[4] 删去，精简版恒 request_logger=None 路径）；
# 构造签名逐字，log_inputs/log_outputs 记录位保留。
from typing import Any


# SOURCE: vllm/entrypoints/serve/utils/request_logger.py:L17-L99 —— HOST SEAM：
# RequestLogger 消费面
class RequestLogger:
    # SOURCE: vllm/entrypoints/serve/utils/request_logger.py:L18-L33
    def __init__(self, *, max_log_len: int | None) -> None:
        self.max_log_len = max_log_len

    # SOURCE: vllm/entrypoints/serve/utils/request_logger.py:L35-L68 —— HOST SEAM：
    # 记录位（真实截断打印 prompt/params；精简环境 no-op 载体）
    def log_inputs(
        self,
        request_id: str,
        prompt: str | None,
        params: Any = None,
        lora_request: Any = None,
        **kwargs,
    ) -> None:
        return

    # SOURCE: vllm/entrypoints/serve/utils/request_logger.py:L70-L99 —— HOST SEAM：
    # 记录位（真实截断打印 outputs；delete[4] 删尽了调用点，保留方法面）
    def log_outputs(
        self,
        *,
        request_id: str,
        outputs: str,
        output_token_ids: Any = None,
        finish_reason: Any = None,
        is_streaming: bool = False,
        delta: bool = False,
    ) -> None:
        return
