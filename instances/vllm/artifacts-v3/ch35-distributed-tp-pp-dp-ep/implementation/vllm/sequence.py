# SOURCE: vllm/sequence.py
# ch34 切面：IntermediateTensors（PP 段间载荷本体，L10-L49 逐字；AsyncIntermediate
# Tensors 的懒同步包装在 v1/worker/gpu_worker.py 切面）。真实文件的其余面
# （Request/SequenceData 族）属 ch04 域。

from dataclasses import dataclass

import torch


# cannot use msgspec.Struct here because Dynamo does not support it
# SOURCE: vllm/sequence.py:L12-L62 IntermediateTensors —— 逐字（手写 __init__ 保
#   Dynamo 溯源的注释原话；__eq__/__repr__/empty_like 按 SUBTRACTED 裁除）
@dataclass
class IntermediateTensors:
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    tensors: dict[str, torch.Tensor]

    # SOURCE: vllm/sequence.py:L20-L28 __init__ —— 逐字（锚点双置）
    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        # manually define this function, so that
        # Dynamo knows `IntermediateTensors()` comes from this file.
        # Otherwise, dataclass will generate this function by evaluating
        # a string, and we will lose the information about the source file.
        self.tensors = tensors

    # SOURCE: vllm/sequence.py:L30-L34 __getitem__ —— 逐字
    def __getitem__(self, key: str | slice):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    # SOURCE: vllm/sequence.py:L36-L37 __setitem__ —— 逐字
    def __setitem__(self, key: str, value: torch.Tensor):
        self.tensors[key] = value

    # SOURCE: vllm/sequence.py:L39-L40 items —— 逐字
    def items(self):
        return self.tensors.items()

    # SOURCE: vllm/sequence.py:L42-L43 __len__ —— 逐字
    def __len__(self):
        return len(self.tensors)

    # SUBTRACTED: __eq__/__repr__（L45-L53）——测试比较面，不影响载荷语义。
    # SUBTRACTED: empty_like（L56-L62）——torch.compile 图替代构造面（ch19 域）。
