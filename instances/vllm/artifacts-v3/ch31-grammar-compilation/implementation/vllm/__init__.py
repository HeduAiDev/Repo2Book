# ch30 精简版 vllm 包标记。
# SUBTRACTED: 真实 vllm/__init__.py 的公共 API 导出面（SamplingParams/LLM 等
#   顶层导出，vllm/__init__.py:L1-L60）——本章消费面全部走模块路径
#   （vllm.sampling_params / vllm.v1.*），顶层导出面不进精简版。
