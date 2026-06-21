# 只做减法的忠实精简版 —— 镜像 vllm/v1/outputs.py 的 logprobs 载荷类型（pin f3fef123）
# 本章只消费这两个 NamedTuple 的字段解包；保留字段定义即可。
#
# SUBTRACTED: slice_request / tolists / to_cpu_nonblocking / filter / empty_cpu 等
#             构造/搬运/切片工具方法（vllm/v1/outputs.py:L39-L48, L61-L105+）——
#             这些发生在 sampler 与 EngineCore（其它章节）；本章只解包字段。
# SUBTRACTED: numpy/torch 的 import —— 精简版用纯 Python 桩数组（numpy/torch 均支持
#             相同的 .shape/.flatten().tolist()/.tolist() 接口）以便 host 直接运行。
from typing import NamedTuple


class LogprobsLists(NamedTuple):
    # SOURCE: vllm/v1/outputs.py:L26-L37
    # EngineCore 传给 sample logprobs 的载荷：已是 list/numpy 形态（已 tolists）。
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: object
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: object
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: object
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    cu_num_generated_tokens: list[int] | None = None


class LogprobsTensors(NamedTuple):
    # SOURCE: vllm/v1/outputs.py:L51-L59
    # EngineCore 传给 prompt logprobs 的载荷：torch 张量，需在
    # _update_prompt_logprobs 内自行 Pythonize（.flatten().tolist()/.tolist()）。
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: object
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: object
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: object
    # [num_reqs]
    cu_num_generated_tokens: list[int] | None = None
