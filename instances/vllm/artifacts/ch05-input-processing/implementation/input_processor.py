"""Stage 1 输入处理：InputProcessor.process_inputs() —— 只做减法的精简版。

把用户的 PromptType/EngineInput + SamplingParams/PoolingParams 转成一个
EngineCoreRequest：参数校验、LoRA 校验、（兜底）tokenize、模型输入校验、
SamplingParams 克隆补全、多模态 placeholder 排序展平、组装 EngineCoreRequest，
并经 assign_request_id() 注入 8 字符随机后缀保证请求 id 唯一。

与真实 vllm/v1/engine/input_processor.py **同名、同结构、同控制流**；每个被删除的
分支都标 # SUBTRACTED，删除项均为 dossier subtraction_plan.delete 批准项。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from config import VllmConfig, current_platform
from messages import (
    EngineCoreRequest,
    GENERATION_TASKS,
    MultiModalFeatureSpec,
    POOLING_TASKS,
    PoolingParams,
    SamplingParams,
    argsort_mm_positions,
    json_iter_leaves,
    length_from_prompt_token_ids_or_embeds,
    random_uuid,
    split_enc_dec_input,
)
from preprocess import InputPreprocessor

# VLLM_DISABLE_REQUEST_ID_RANDOMIZATION 环境开关的替身（默认 False）。
# SOURCE: vllm/envs.py (VLLM_DISABLE_REQUEST_ID_RANDOMIZATION)
VLLM_DISABLE_REQUEST_ID_RANDOMIZATION = False


class InputProcessor:
    # SOURCE: vllm/v1/engine/input_processor.py:L36 (class InputProcessor)
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer=None,
        *,
        mm_registry=None,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = model_config = vllm_config.model_config
        self.lora_config = vllm_config.lora_config
        self.speculative_config = vllm_config.speculative_config
        self.structured_outputs_config = vllm_config.structured_outputs_config
        # SUBTRACTED: cache_config/scheduler_config/observability_config 持有
        #   — 本章未触及，原 vllm/v1/engine/input_processor.py:L46-L51。

        self.generation_config_fields = model_config.try_get_generation_config()

        self.renderer = renderer

        # SUBTRACTED: supports_mm_inputs 探测 + MultiModalBudget 分配
        #   （mm_encoder_cache_size / skip_prompt_length_check） — 多模态预算细节，
        #   原 vllm/v1/engine/input_processor.py:L57-L66。精简版用可注入默认值。
        self.supports_mm_inputs = False
        self.mm_encoder_cache_size = 1 << 30
        self.skip_prompt_length_check = False

        self.input_preprocessor = InputPreprocessor(
            vllm_config,
            renderer=renderer,
            mm_registry=mm_registry,
        )

    @property
    def tokenizer(self):
        # SOURCE: vllm/v1/engine/input_processor.py:L74 (tokenizer)
        return None if self.renderer is None else self.renderer.tokenizer

    def get_tokenizer(self):
        # SOURCE: vllm/v1/engine/input_processor.py:L78 (get_tokenizer)
        return self.renderer.get_tokenizer()

    # SOURCE: vllm/v1/engine/input_processor.py:L81 (_validate_params)
    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[str, ...],
    ) -> None:
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid."""
        if isinstance(params, SamplingParams):
            supported_generation_tasks = [
                task for task in supported_tasks if task in GENERATION_TASKS
            ]
            if not supported_generation_tasks:
                raise ValueError("This model does not support generation")

            params.verify(
                self.model_config,
                self.speculative_config,
                self.structured_outputs_config,
                self.tokenizer,
            )

            # SUBTRACTED: thinking_token_budget 与 reasoning_config 的一致性校验
            #   （reasoning 特性次要分支） — dossier 批准删，
            #   原 vllm/v1/engine/input_processor.py:L101-L109。
        elif isinstance(params, PoolingParams):
            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]
            if not supported_pooling_tasks:
                raise ValueError("This model does not support pooling")

            # SUBTRACTED: pooling task 默认补全（token_embed/token_classify/plugin）
            #   与 task 合法性细查 — dossier 批准删，
            #   原 vllm/v1/engine/input_processor.py:L117-L129。

            params.verify(self.model_config)
        else:
            raise TypeError(
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

    # SOURCE: vllm/v1/engine/input_processor.py:L138 (_validate_lora)
    def _validate_lora(self, lora_request) -> None:
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise ValueError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            # SUBTRACTED: per-LoRA tokenizer deprecation 警告文案 — 纯日志，
            #   原 vllm/v1/engine/input_processor.py:L149-L155。
            pass

    # SOURCE: vllm/v1/engine/input_processor.py:L157 (_get_mm_identifier)
    def _get_mm_identifier(self, mm_hash: str, lora_request) -> str:
        """When enable_tower_connector_lora is True, multi-modal embeddings
        vary depending on the LoRA request. Therefore the mm_hash must be
        generated based on the LoRA request to prevent incorrect cache hits."""
        if (
            lora_request is None
            or self.lora_config is None
            or not self.lora_config.enable_tower_connector_lora
        ):
            return mm_hash
        return f"{lora_request.lora_name}:{mm_hash}"

    # SUBTRACTED: inject_into_mm_cache()（L175-L212）整个方法 —— 前端把已处理 mm_kwargs
    #   注入 processor cache 的旁路优化，不在 process_inputs 主控制流上。dossier 批准删。

    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        # SOURCE: vllm/v1/engine/input_processor.py:L214 (assign_request_id)
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            # SUBTRACTED: deprecation 警告文案 — 纯日志，原 input_processor.py:L226-L230。
            pass
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    # SOURCE: vllm/v1/engine/input_processor.py:L234 (process_inputs)
    def process_inputs(
        self,
        request_id: str,
        prompt,
        params: SamplingParams | PoolingParams,
        supported_tasks: tuple[str, ...],
        arrival_time: float | None = None,
        lora_request=None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers=None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest:
        self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise ValueError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            # SUBTRACTED: tokenization_kwargs deprecation 警告分支（应改传 Renderer）
            #   — 纯日志，原 input_processor.py:L262-L267。
            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())

            processed_inputs = prompt
        else:
            # raw prompt（deprecated，v0.18 移除）：现场走 InputPreprocessor 兜底 tokenize。
            # SUBTRACTED: raw-prompt deprecation 警告文案 — 纯日志，原 input_processor.py:L274-L278。
            if arrival_time is None:
                arrival_time = time.time()

            processed_inputs = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
            )

        current_platform.validate_request(processed_inputs, params)

        encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_inputs["type"] == "embeds":
            prompt_embeds = decoder_inputs["prompt_embeds"]
            prompt_token_ids = decoder_inputs.get("prompt_token_ids")
            prompt_is_token_ids = decoder_inputs.get("is_token_ids")
        else:
            prompt_token_ids = decoder_inputs["prompt_token_ids"]
            prompt_embeds = None
            prompt_is_token_ids = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id() if self.renderer else None,
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                # SUBTRACTED: 错误文案细节 — dossier 批准简写，原 input_processor.py:L335-L339。
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(
                            base_mm_hash,
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
        )

    # SOURCE: vllm/v1/engine/input_processor.py:L379 (_validate_prompt_len)
    def _validate_prompt_len(
        self,
        prompt_len: int,
        prompt_type: Literal["encoder", "decoder"],
    ):
        if self.skip_prompt_length_check:
            return

        if prompt_len == 0 and prompt_type == "decoder":
            raise ValueError(f"The {prompt_type} prompt cannot be empty")

        model_config = self.model_config
        max_prompt_len = (
            model_config.max_model_len
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size
        )
        if prompt_len > max_prompt_len:
            # SUBTRACTED: suggestion 文案构造分支（mm/纯文本两版措辞）
            #   — dossier 批准删，原 input_processor.py:L397-L409。
            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}."
            )
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
            # SUBTRACTED: suggestion 文案 — dossier 批准删，原 input_processor.py:L416-L419。
            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}."
            )

    # SOURCE: vllm/v1/engine/input_processor.py:L426 (_validate_model_input)
    def _validate_model_input(
        self,
        prompt_input,
        prompt_type: Literal["encoder", "decoder"],
    ) -> None:
        model_config = self.model_config
        tokenizer = self.tokenizer

        prompt_ids = (
            None
            if prompt_input["type"] == "embeds"
            else prompt_input["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_input["prompt_embeds"] if prompt_input["type"] == "embeds" else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        if prompt_input["type"] == "multimodal":
            decoder_mm_positions = prompt_input["mm_placeholders"]
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_position in mm_positions:
                    num_embeds = mm_position.get_num_embeds()
                    if num_embeds > self.mm_encoder_cache_size:
                        # SUBTRACTED: 编码器缓存超限错误文案 — dossier 批准简写,
                        #   原 input_processor.py:L451-L459。
                        raise ValueError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with {num_embeds} embedding tokens, which exceeds the "
                            f"encoder cache size {self.mm_encoder_cache_size}."
                        )

        if prompt_ids and tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)
            # NOTE: 取 tokenizer.max_token_id 与 model_vocab_size-1 的较大者判越界。
            # SUBTRACTED: 关于 Qwen3 tokenizer/model vocab 不一致的长注释
            #   — dossier 批准压成一行，原 input_processor.py:L464-L473。
            model_vocab_size = model_config.get_vocab_size()
            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
                raise ValueError(f"Token id {max_input_id} is out of vocabulary")

    # SOURCE: vllm/v1/engine/input_processor.py:L478 (_validate_model_inputs)
    def _validate_model_inputs(self, encoder_input, decoder_input):
        if encoder_input is not None:
            self._validate_model_input(encoder_input, prompt_type="encoder")

        self._validate_model_input(decoder_input, prompt_type="decoder")
