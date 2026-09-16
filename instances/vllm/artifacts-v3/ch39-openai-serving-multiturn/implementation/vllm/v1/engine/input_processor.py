# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/v1/engine/input_processor.py —— HOST SEAM（最小承载）：
# assign_request_id 逐字（外部 id→内部 id 的 8 位随机后缀双轨，ch7 m22
# 已立）；process_inputs 按 tokens-dict 已渲染路径退化承载（真实含校验/
# 原始 prompt 预处理/多模态装配/max_tokens 回填全链——ch6 边界，本章黑盒
# 回指）。
import time
from typing import Any, Mapping

import vllm.envs as envs
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/input_processor.py:L38 —— InputProcessor 类位
# （HOST SEAM：只承载消费面两方法）
class InputProcessor:
    # SOURCE: vllm/v1/engine/input_processor.py —— 构造位（真实持有
    # vllm_config/model_config/renderer/tokenizer/input_preprocessor）
    def __init__(self, vllm_config=None, model_config=None, renderer=None):
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.renderer = renderer

    # SOURCE: vllm/v1/engine/input_processor.py:L231-L249 —— assign_request_id 逐字
    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            logger.warning_once(
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    # SOURCE: vllm/v1/engine/input_processor.py:L251-L345 —— HOST SEAM：
    # process_inputs 退化（已渲染 EngineInput dict → EngineCoreRequest；
    # 真实的参数校验/DP rank 校验/eos 合并/max_tokens 回填/多模态装配在
    # ch6/ch15 边界内，精简环境不逐项复刻；tokens 主线的字段映射一致）
    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple = (),
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest:
        if isinstance(prompt, dict) and "type" in prompt:
            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())
            processed_inputs: EngineInput = prompt
        else:
            # SUBTRACTED: vllm/v1/engine/input_processor.py:L296-L301 原始
            # prompt 的 input_preprocessor.preprocess 预处理（ch6 边界）。
            raise ValueError(
                "raw prompts are out of scope of the ch38 reduced build; "
                "pass a rendered EngineInput (tokens_input(...))"
            )

        prompt_token_ids = processed_inputs.get("prompt_token_ids")
        sampling_params = params if isinstance(params, SamplingParams) else None
        pooling_params = params if isinstance(params, PoolingParams) else None

        # SUBTRACTED: vllm/v1/engine/input_processor.py:L318-L345
        # sampling_params 的 max_tokens None 回填（max_model_len−seq_len）、
        # update_from_generation_config/update_from_tokenizer 与多模态
        # mm_features 装配——ch6/ch15 边界。

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            mm_features=None,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time if arrival_time is not None else time.time(),
            lora_request=lora_request,
            cache_salt=processed_inputs.get("cache_salt"),
            data_parallel_rank=data_parallel_rank,
            priority=priority,
            trace_headers=trace_headers,
        )

    # SUBTRACTED: vllm/v1/engine/input_processor.py 其余（_validate_params/
    # _validate_model_inputs/process_inputs_async 同步镜像）——ch6 边界内
    # 的校验族；本章消费面走同步退化位。
