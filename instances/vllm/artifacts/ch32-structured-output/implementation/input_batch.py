# SOURCE: vllm/v1/worker/gpu/input_batch.py
# 只做减法的忠实精简版。真实 InputBatch 是 GPU model runner 侧一整批请求的输入
# 快照（block table/attention 元数据/采样参数等几十个字段），本章只保留掩码行
# 映射需要的四个字段。
#
# SUBTRACTED: SPDX 版权头、block table/attention 元数据/采样参数等其余字段
# （input_batch.py 其余部分，与本章无关）。
import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class InputBatch:
    # SOURCE: vllm/v1/worker/gpu/input_batch.py:L78-85（精简到本章相关字段）
    # [num_reqs] 本步 batch 里的请求 id，按 GPU runner 自己的顺序排列——不保证与
    # 调度侧 structured_output_request_ids 的顺序一致（本章第一个不变式的另一半）。
    req_ids: "list[str]"

    # [total_num_logits]
    logits_indices: "torch.Tensor"
    # [num_reqs + 1] 请求 -> logits 行区间的前缀和：请求 i 占据
    # [cu_num_logits[i], cu_num_logits[i+1]) 这一段行。
    cu_num_logits: "torch.Tensor"
    cu_num_logits_np: "np.ndarray"

    # Whether any requests in batch use structured output.
    has_structured_output_reqs: bool = False
