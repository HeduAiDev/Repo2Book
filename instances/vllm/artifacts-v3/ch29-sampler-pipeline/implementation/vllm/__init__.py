# ch29 精简版 vllm 包骨架（HOST SEAM 载体）。
# 真实 vllm/__init__.py 是数百行延迟导出门面；本章只消费
# `from vllm import SamplingParams`（logits_processor/interface.py:L11）——
# 按真实 re-export 位就地承载，不重建门面。
from vllm.sampling_params import SamplingParams  # SOURCE: vllm/__init__.py SamplingParams re-export 位
