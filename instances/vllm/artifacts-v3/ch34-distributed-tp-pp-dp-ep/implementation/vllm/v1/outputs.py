# SOURCE: vllm/v1/outputs.py
# ch34 切面：ModelRunnerOutput / AsyncModelRunnerOutput 的载体面（Worker.
# execute_model 的 isinstance 判据；logprobs/采样细节归 ch08/ch29）。

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# SOURCE: vllm/v1/outputs.py:L261-L321 ModelRunnerOutput —— 字段子集载体
@dataclass
class ModelRunnerOutput:  # HOST/ch08 SEAM
    # SOURCE: vllm/v1/outputs.py:L261-L321（锚点双置）
    req_ids: list[str] = field(default_factory=list)
    req_id_to_index: dict[str, int] = field(default_factory=dict)
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    # SUBTRACTED: logprobs/kv_connector_output 族——ch08/ch16 域。


# SOURCE: vllm/v1/outputs.py:L325-L334 AsyncModelRunnerOutput —— 逐字 ABC
class AsyncModelRunnerOutput(ABC):
    @abstractmethod
    def get_output(self) -> ModelRunnerOutput:
        """Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are available,
        which might involve copying device tensors to the host.
        This method should only be called once per AsyncModelRunnerOutput.
        """
        # SOURCE: vllm/v1/outputs.py:L327-L334（锚点双置）
        pass
