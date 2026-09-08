# SOURCE: vllm/sequence.py:L12-L47 IntermediateTensors（逐字——PP 分段载体，
# m4/m14 消费：LlamaModel.forward 非末 rank 打包、make_empty_intermediate_
# tensors_factory 造空壳）
from __future__ import annotations

from dataclasses import dataclass

import torch


# cannot use msgspec.Struct here because Dynamo does not support it
# SOURCE: vllm/sequence.py:L11-L12 IntermediateTensors
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    # SOURCE: vllm/sequence.py:L18 tensors 字段声明
    tensors: dict[str, torch.Tensor]

    # SOURCE: vllm/sequence.py:L20-L28 __init__（Dynamo 保源文件信息的手工定义）
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    # SOURCE: vllm/sequence.py:L30-L34 __getitem__（str 取项 / slice 整体切片）
    def __getitem__(self, key: str | slice):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    # SOURCE: vllm/sequence.py:L36-L37 __setitem__
    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    # SOURCE: vllm/sequence.py:L39-L40 items
    def items(self):
        return self.tensors.items()

    # SOURCE: vllm/sequence.py:L42-L43 __len__
    def __len__(self):
        return len(self.tensors)

    # SOURCE: vllm/sequence.py:L45-L50 __eq__（逐张量 torch.equal）
    def __eq__(self, other: object):
        if not isinstance(other, self.__class__):
            return False
        if self.tensors.keys() != other.tensors.keys():
            return False
        return all(torch.equal(self.tensors[k], other.tensors[k]) for k in self.tensors)

    # SOURCE: vllm/sequence.py:L52-L53 __repr__
    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"
