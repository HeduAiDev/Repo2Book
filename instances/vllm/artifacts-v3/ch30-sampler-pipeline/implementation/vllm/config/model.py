# SOURCE: vllm/config/model.py
# HOST SEAM：model 配置面里本章消费的 logprobs_mode 两常量（逐字）。
# 真实 ModelConfig 数千行（ch03/ch23 域），此处不重建。
from __future__ import annotations

from typing import Literal

# SOURCE: vllm/config/model.py:L99-L101 LogprobsMode —— 逐字（四态）
LogprobsMode = Literal[
    "raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"
]
# SOURCE: vllm/config/model.py:L102-L105 PROCESSED_LOGPROBS_MODES —— 逐字
#   （FlashInfer 拿不到 top-k/top-p 之后的 logits/logprobs → 强制 native
#   的判据，topk_topp_sampler.py:L96-L99/L176）
PROCESSED_LOGPROBS_MODES: tuple[LogprobsMode, ...] = (
    "processed_logits",
    "processed_logprobs",
)
