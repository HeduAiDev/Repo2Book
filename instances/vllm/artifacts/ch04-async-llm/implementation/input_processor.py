"""Stage1 前处理（精简版，黑盒）。

本章只把 InputProcessor 当作 add_request 的第一步：把原始 prompt + 采样参数转成已 tokenize
的 EngineCoreRequest。内部 tokenize/校验/多模态留 ch05。
"""

from __future__ import annotations

import itertools

from messages import EngineCoreRequest, SamplingParams


class InputProcessor:
    # SOURCE: vllm/v1/engine/input_processor.py:L36 (class InputProcessor)
    def __init__(self, vllm_config=None, renderer=None):
        # SUBTRACTED: vllm_config/renderer/tokenizer 的真实持有与使用 —— Stage1 内部留 ch05。
        self._counter = itertools.count()

    # SOURCE: vllm/v1/engine/input_processor.py:L234 (process_inputs)
    def process_inputs(
        self,
        request_id: str,
        prompt,
        params: SamplingParams,
        supported_tasks=None,
        arrival_time: float = 0.0,
        **kwargs,
    ) -> EngineCoreRequest:
        # SUBTRACTED: 真实的 tokenize/多模态/校验/lora/data_parallel 等全部处理 —— 留 ch05。
        #   精简版把 prompt 当作「已是 token id 列表或可忽略」，只产出结构正确的 EngineCoreRequest，
        #   保住三段式入口转换语义（prompt -> EngineCoreRequest）。原 vllm/v1/engine/input_processor.py:L234+
        prompt_token_ids = list(prompt) if isinstance(prompt, (list, tuple)) else []
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=params,
            arrival_time=arrival_time,
        )

    # SOURCE: vllm/v1/engine/input_processor.py:L215 (assign_request_id)
    def assign_request_id(self, request: EngineCoreRequest) -> None:
        # SUBTRACTED: 真实版把 external_req_id 复制到内部 id 并做去重计数 —— 留 ch05。
        #   精简版 no-op：request_id 已由调用方给定。
        pass
