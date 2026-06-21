"""InputProcessor 中本章唯一相关的一步：assign_request_id。

它是 child id 派生的前提——把外部 id 存进 external_req_id、再给父 request
生成带 8 hex 后缀的内部 request_id。
"""

from __future__ import annotations

from .types import EngineCoreRequest, random_uuid


# SOURCE: vllm/v1/engine/input_processor.py — class InputProcessor（本章仅保留 assign_request_id）
class InputProcessor:
    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        # SOURCE: vllm/v1/engine/input_processor.py:L214 @staticmethod def assign_request_id
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        # SUBTRACTED: VLLM_DISABLE_REQUEST_ID_RANDOMIZATION 分支（关闭随机化、保留原 id 并告警，
        #            vllm/v1/engine/input_processor.py:L225-L231）——默认走随机化路径即可。
        request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    # SUBTRACTED: process_inputs（tokenize/多模态/校验/clone）与全部其余方法
    #            （vllm/v1/engine/input_processor.py）——属 ch04/ch05 主题，本章不触及。
