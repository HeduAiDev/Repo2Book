# SOURCE: vllm/logger.py
# HOST SEAM：init_logger 的 no-op logger 载体（ch22/ch23 同款）——本章文件内
# 日志消费仅 debug/info/warning 级（插件加载与 FlashInfer 裁决的旁注），
# host 跑通不需要真实日志管道。
from __future__ import annotations


# SOURCE: vllm/logger.py —— HOST SEAM：no-op logger
class _NullLogger:
    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def debug(self, *a, **k):
        pass

    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def debug_once(self, *a, **k):
        pass

    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def info(self, *a, **k):
        pass

    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def info_once(self, *a, **k):
        pass

    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def warning(self, *a, **k):
        pass

    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def warning_once(self, *a, **k):
        pass

    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def error(self, *a, **k):
        pass

    # SOURCE: vllm/logger.py —— HOST SEAM no-op 位
    def exception(self, *a, **k):
        pass


# SOURCE: vllm/logger.py init_logger —— HOST SEAM
def init_logger(name):
    return _NullLogger()
