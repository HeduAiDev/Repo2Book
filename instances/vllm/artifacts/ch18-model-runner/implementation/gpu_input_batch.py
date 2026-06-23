# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Subtract-only reduced companion for ch18.
# SOURCE: vllm/v1/worker/gpu_input_batch.py (pin f3fef123)
# CachedRequestState + InputBatch: the cross-iteration persistent batch. Kept
# same-name / same-structure / same-control-flow; only approved subtraction_plan
# items (M-RoPE/XD-RoPE, prompt_embeds, async spec decode, pooling, LoRA,
# PP/KV-connector, thinking-budget) are removed.

from dataclasses import dataclass

import numpy as np
import torch

from _support import (
    SamplingParams,
    SamplingType,
    length_from_prompt_token_ids_or_embeds,
)
from block_table import MultiGroupBlockTable
from logits_processor_state import BatchUpdateBuilder, MoveDirectionality


# SOURCE: vllm/v1/worker/gpu_input_batch.py:L33  CachedRequestState
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    generator: torch.Generator | None

    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int]

    # SUBTRACTED: mm_features / mrope_positions / xdrope_positions /
    #   lora_request / prompt_embeds / prompt_is_token_ids / prev_num_draft_len /
    #   pooling_params / pooling_states fields. Approved deletions (multimodal,
    #   M-RoPE/XD-RoPE, LoRA, prompt_embeds, async spec decode, pooling).
    #   Orig: vllm/v1/worker/gpu_input_batch.py:L37-L62

    def __post_init__(self):
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L64
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, None
        )

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L72  num_tokens
    @property
    def num_tokens(self) -> int:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L72
        return self.num_prompt_tokens + len(self.output_token_ids)

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L76  get_token_id
    def get_token_id(self, idx: int) -> int:
        if idx < self.num_prompt_tokens:
            if self.prompt_token_ids is None:
                raise ValueError(
                    f"Tried to access token index {idx}, but that token was "
                    "provided via prompt_embeds, and its ID is unknown."
                )
            return self.prompt_token_ids[idx]
        if idx - self.num_prompt_tokens < len(self.output_token_ids):
            return self.output_token_ids[idx - self.num_prompt_tokens]
        return -1


# SOURCE: vllm/v1/worker/gpu_input_batch.py:L89  InputBatch
class InputBatch:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks_per_req: list[int] | None = None,
    ):
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L90
        # SUBTRACTED: thinking_budget_state_holder / is_pooling_model / num_spec_tokens
        #   / reasoning_config / logitsprocs init args. Approved deletions.
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L101-L116
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        # SUBTRACTED: is_token_ids / req_prompt_embeds (prompt_embeds mixed-mode).
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L138-L145
        self.num_tokens_no_spec_cpu_tensor = torch.zeros(
            (max_num_reqs,), device="cpu", dtype=torch.int32, pin_memory=pin_memory
        )
        self.num_tokens_no_spec = self.num_tokens_no_spec_cpu_tensor.numpy()
        self.num_prompt_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,), device="cpu", dtype=torch.int32, pin_memory=pin_memory
        )
        self.num_prompt_tokens = self.num_prompt_tokens_cpu_tensor.numpy()
        self.num_computed_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,), device="cpu", dtype=torch.int32, pin_memory=pin_memory
        )
        self.num_computed_tokens_cpu = self.num_computed_tokens_cpu_tensor.numpy()

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=pin_memory,
            device=device,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            max_num_blocks=max_num_blocks_per_req,
        )

        # Sampling-related.
        self.temperature_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.temperature_cpu = self.temperature_cpu_tensor.numpy()
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=pin_memory
        )
        self.top_p_cpu = self.top_p_cpu_tensor.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.top_k_cpu = self.top_k_cpu_tensor.numpy()
        self.top_k_reqs: set[str] = set()

        self.frequency_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.frequency_penalties_cpu = self.frequency_penalties_cpu_tensor.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        self.presence_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.presence_penalties_cpu = self.presence_penalties_cpu_tensor.numpy()
        self.presence_penalties_reqs: set[str] = set()

        self.repetition_penalties_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.float, device="cpu", pin_memory=pin_memory
        )
        self.repetition_penalties_cpu = self.repetition_penalties_cpu_tensor.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # Speculative decoding: by default 1 token is accepted per request.
        self.num_accepted_tokens_cpu_tensor = torch.ones(
            (max_num_reqs,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
        )
        self.num_accepted_tokens_cpu = self.num_accepted_tokens_cpu_tensor.numpy()

        # SUBTRACTED: LoRA mapping dicts (request_lora_mapping / lora_id_to_*).
        #   Approved LoRA deletion. Orig: vllm/v1/worker/gpu_input_batch.py:L242-L245

        # req_index -> generator
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}
        self.logprob_token_ids: dict[str, list[int]] = {}
        self.in_progress_prompt_logprobs_cpu: dict = {}

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        self.has_allowed_token_ids: set[str] = set()
        self.allowed_token_ids_mask: torch.Tensor | None = None
        self.allowed_token_ids_mask_cpu_tensor: torch.Tensor | None = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.req_output_token_ids: list[list[int] | None] = []

        # Last speculative tokens for the sampler.
        self.spec_token_ids: list[list[int]] = [[] for _ in range(max_num_reqs)]

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

        # Cached reference to the previous step's sampled tokens (async sched).
        self.prev_sampled_token_ids: torch.Tensor | None = None
        self.prev_req_id_to_index: dict[str, int] | None = None

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L304  req_ids
    @property
    def req_ids(self) -> list[str]:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L304
        return self._req_ids  # type: ignore[return-value]

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L310  _register_add_request
    def _register_add_request(self, request: "CachedRequestState") -> int:
        """Track add-request operations for logits processors."""
        # Fill the next empty index if there is one.
        if (new_req_index := self.batch_update_builder.pop_removed()) is None:
            # Append to end otherwise.
            new_req_index = self.num_reqs

        assert new_req_index < self.max_num_reqs
        self.batch_update_builder.batch_changed = True
        if request.sampling_params:
            self.batch_update_builder.added.append(
                (
                    new_req_index,
                    request.sampling_params,
                    request.prompt_token_ids,
                    request.output_token_ids,
                )
            )

        return new_req_index

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L336  add_request
    def add_request(self, request: "CachedRequestState") -> int:
        req_index = self._register_add_request(request)

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
            self.spec_token_ids.append([])
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids
            self.spec_token_ids[req_index].clear()

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, None
        )
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        if request.prompt_token_ids is not None:
            self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
        # SUBTRACTED: is_token_ids / req_prompt_embeds writes (prompt_embeds).
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L363-L372
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
        # Number of tokens without spec decode tokens.
        self.num_tokens_no_spec[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        # Sampling parameter packing into this slot's columns.
        sampling_params = request.sampling_params
        assert sampling_params is not None  # reduced companion: sampling only
        if sampling_params.sampling_type == SamplingType.GREEDY:
            # Should avoid division by zero later when apply_temperature.
            self.temperature_cpu[req_index] = 0.0
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu[req_index] = sampling_params.temperature
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        top_k = sampling_params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
        else:
            top_k = self.vocab_size
        self.top_k_cpu[req_index] = top_k
        self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties_cpu[req_index] = sampling_params.presence_penalty
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties_cpu[req_index] = sampling_params.repetition_penalty
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)

        # NOTE(woosuk): self.generators should not include requests that do not
        # have their own generator.
        if request.generator is not None:
            self.generators[req_index] = request.generator

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = (
                self.vocab_size
                if sampling_params.logprobs == -1
                else sampling_params.logprobs
            )
        if sampling_params.logprob_token_ids is not None:
            self.logprob_token_ids[req_id] = sampling_params.logprob_token_ids

        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask_cpu_tensor is None:
                # Lazy allocation for this tensor, which can be large.
                self.allowed_token_ids_mask = torch.zeros(
                    self.max_num_reqs, self.vocab_size, dtype=torch.bool, device=self.device
                )
                self.allowed_token_ids_mask_cpu_tensor = torch.zeros(
                    self.max_num_reqs, self.vocab_size, dtype=torch.bool, device="cpu"
                )
            self.allowed_token_ids_mask_cpu_tensor[req_index] = True
            self.allowed_token_ids_mask_cpu_tensor[req_index][
                sampling_params.allowed_token_ids
            ] = False

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[req_index] = sampling_params.bad_words_token_ids

        # SUBTRACTED: pooling_params branch + lora_request mapping. Approved
        #   pooling/LoRA deletions. Orig: vllm/v1/worker/gpu_input_batch.py:L454-L481

        # Speculative decoding: by default 1 token is generated.
        self.num_accepted_tokens_cpu[req_index] = 1

        return req_index

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L484  update_req_spec_token_ids
    def update_req_spec_token_ids(
        self, request: CachedRequestState, scheduled_spec_tokens: dict[str, list[int]]
    ) -> None:
        req_id = request.req_id
        req_index = self.req_id_to_index[req_id]
        cur_spec_token_ids = self.spec_token_ids[req_index]
        cur_spec_token_ids.clear()
        spec_token_ids = scheduled_spec_tokens.get(req_id, ())
        num_spec_tokens = len(spec_token_ids)
        if not spec_token_ids:
            return
        start_index = self.num_tokens_no_spec[req_index]
        end_token_index = start_index + num_spec_tokens
        self.token_ids_cpu[req_index, start_index:end_token_index] = spec_token_ids
        cur_spec_token_ids.extend(spec_token_ids)

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L510  remove_request
    def remove_request(self, req_id: str) -> int | None:
        """This method must always be followed by a call to condense()."""
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None

        self.batch_update_builder.removed_append(req_index)
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None
        self.spec_token_ids[req_index].clear()
        self.block_table.clear_row(req_index)

        # SUBTRACTED: LoRA unbind + pooling early-return. Approved deletions.
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L530-L543

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.logprob_token_ids.pop(req_id, None)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)
        if self.prev_req_id_to_index is not None:
            self.prev_req_id_to_index.pop(req_id, None)

        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask_cpu_tensor is not None:
            self.allowed_token_ids_mask_cpu_tensor[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)
        # SUBTRACTED: thinking_token_budget_reqs.discard (thinking budget feature).
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L564
        return req_index

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L567  swap_states
    def swap_states(self, i1: int, i2: int) -> None:
        old_id_i1 = self._req_ids[i1]
        old_id_i2 = self._req_ids[i2]
        # Only swap the active token prefix for each request.
        i1_active_token_count = self._get_active_token_count(i1)
        i2_active_token_count = self._get_active_token_count(i2)
        max_active_token_count = max(i1_active_token_count, i2_active_token_count)

        self._req_ids[i1], self._req_ids[i2] = self._req_ids[i2], self._req_ids[i1]
        self.req_output_token_ids[i1], self.req_output_token_ids[i2] = (
            self.req_output_token_ids[i2],
            self.req_output_token_ids[i1],
        )
        self.spec_token_ids[i1], self.spec_token_ids[i2] = (
            self.spec_token_ids[i2],
            self.spec_token_ids[i1],
        )
        assert old_id_i1 is not None and old_id_i2 is not None
        self.req_id_to_index[old_id_i1], self.req_id_to_index[old_id_i2] = (
            self.req_id_to_index[old_id_i2],
            self.req_id_to_index[old_id_i1],
        )
        self.num_tokens_no_spec[i1], self.num_tokens_no_spec[i2] = (
            self.num_tokens_no_spec[i2],
            self.num_tokens_no_spec[i1],
        )
        self.num_prompt_tokens[i1], self.num_prompt_tokens[i2] = (
            self.num_prompt_tokens[i2],
            self.num_prompt_tokens[i1],
        )
        self.num_computed_tokens_cpu[i1], self.num_computed_tokens_cpu[i2] = (
            self.num_computed_tokens_cpu[i2],
            self.num_computed_tokens_cpu[i1],
        )

        # NOTE: the following is unsafe; temporarily copy one index's data.
        tmp_token_ids = self.token_ids_cpu[i1, :max_active_token_count].copy()
        self.token_ids_cpu[i1, :max_active_token_count] = self.token_ids_cpu[
            i2, :max_active_token_count
        ]
        self.token_ids_cpu[i2, :max_active_token_count] = tmp_token_ids

        # SUBTRACTED: is_token_ids / req_prompt_embeds swap (prompt_embeds).
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L613-L627

        self.block_table.swap_row(i1, i2)

        # SUBTRACTED: request_lora_mapping swap + pooling early-return. Approved.
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L631-L638

        self.batch_update_builder.moved.append((i1, i2, MoveDirectionality.SWAP))

        self.temperature_cpu[i1], self.temperature_cpu[i2] = (
            self.temperature_cpu[i2],
            self.temperature_cpu[i1],
        )
        self.top_p_cpu[i1], self.top_p_cpu[i2] = self.top_p_cpu[i2], self.top_p_cpu[i1]
        self.top_k_cpu[i1], self.top_k_cpu[i2] = self.top_k_cpu[i2], self.top_k_cpu[i1]
        self.frequency_penalties_cpu[i1], self.frequency_penalties_cpu[i2] = (
            self.frequency_penalties_cpu[i2],
            self.frequency_penalties_cpu[i1],
        )
        self.presence_penalties_cpu[i1], self.presence_penalties_cpu[i2] = (
            self.presence_penalties_cpu[i2],
            self.presence_penalties_cpu[i1],
        )
        self.repetition_penalties_cpu[i1], self.repetition_penalties_cpu[i2] = (
            self.repetition_penalties_cpu[i2],
            self.repetition_penalties_cpu[i1],
        )
        self.num_accepted_tokens_cpu[i1], self.num_accepted_tokens_cpu[i2] = (
            self.num_accepted_tokens_cpu[i2],
            self.num_accepted_tokens_cpu[i1],
        )

        # SUBTRACTED: generators / bad_words / allowed_token_ids_mask swap and
        #   LoRA mapping swap — symmetric sampler-state moves off the ch18 spine.
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L667-L677

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L679  _get_active_token_count
    def _get_active_token_count(self, req_index: int) -> int:
        return int(self.num_tokens_no_spec[req_index]) + len(
            self.spec_token_ids[req_index]
        )

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L684  condense
    def condense(self) -> None:
        """Slide non-empty requests down into lower, empty indices.

        Any consecutive empty indices at the very end of the list are not
        filled.
        """
        num_reqs = self.num_reqs

        if not (empty_req_indices := self.batch_update_builder.removed):
            # All removed requests were replaced by added requests, or else no
            # requests were removed at all. No condense() needed
            return
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            self.spec_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = self.batch_update_builder.peek_removed()
            assert empty_index is not None
            if empty_index >= last_req_index:
                break

            # Move active request down into empty request index.
            self.batch_update_builder.pop_removed()
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens = self._get_active_token_count(last_req_index)

            (self.spec_token_ids[last_req_index], self.spec_token_ids[empty_index]) = (
                self.spec_token_ids[empty_index],
                self.spec_token_ids[last_req_index],
            )
            self.spec_token_ids[last_req_index].clear()

            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens
            ]
            # SUBTRACTED: is_token_ids / req_prompt_embeds move (prompt_embeds).
            #   Orig: vllm/v1/worker/gpu_input_batch.py:L744-L750
            self.num_tokens_no_spec[empty_index] = self.num_tokens_no_spec[
                last_req_index
            ]
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_req_index]
            self.num_computed_tokens_cpu[empty_index] = self.num_computed_tokens_cpu[
                last_req_index
            ]
            self.block_table.move_row(last_req_index, empty_index)

            # SUBTRACTED: request_lora_mapping move + pooling early-continue.
            #   Orig: vllm/v1/worker/gpu_input_batch.py:L760-L767

            self.batch_update_builder.moved.append(
                (last_req_index, empty_index, MoveDirectionality.UNIDIRECTIONAL)
            )

            self.temperature_cpu[empty_index] = self.temperature_cpu[last_req_index]
            self.top_p_cpu[empty_index] = self.top_p_cpu[last_req_index]
            self.top_k_cpu[empty_index] = self.top_k_cpu[last_req_index]
            self.frequency_penalties_cpu[empty_index] = self.frequency_penalties_cpu[
                last_req_index
            ]
            self.presence_penalties_cpu[empty_index] = self.presence_penalties_cpu[
                last_req_index
            ]
            self.repetition_penalties_cpu[empty_index] = self.repetition_penalties_cpu[
                last_req_index
            ]
            self.num_accepted_tokens_cpu[empty_index] = self.num_accepted_tokens_cpu[
                last_req_index
            ]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator

            if self.allowed_token_ids_mask_cpu_tensor is not None:
                self.allowed_token_ids_mask_cpu_tensor[empty_index] = (
                    self.allowed_token_ids_mask_cpu_tensor[last_req_index]
                )

            bad_words_token_ids = self.bad_words_token_ids.pop(last_req_index, None)
            if bad_words_token_ids is not None:
                self.bad_words_token_ids[empty_index] = bad_words_token_ids

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]
        del self.spec_token_ids[num_reqs:]

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L812  refresh_metadata
    def refresh_metadata(self):
        """Apply any batch updates to sampling metadata."""
        # SUBTRACTED: is_pooling_model branch + logitsprocs.update_state loop +
        #   thinking_budget sync. Approved pooling / logits-processor deletions.
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L815-L828
        batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)
        # Update sampling metadata if batch state is changed.
        if batch_update:
            self.sampling_metadata = self._make_sampling_metadata()

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L832  _make_sampling_metadata
    def _make_sampling_metadata(self):
        # SUBTRACTED: full SamplingMetadata assembly (temperature/top_p/top_k/
        #   penalties GPU copies, prompt/output token id tensors, logitsprocs,
        #   allowed_token_ids mask, thinking budget). The sampler & its metadata
        #   belong to the sampling chapter; ch18 only needs the snapshot to be
        #   rebuilt when the batch changes.
        #   Orig: vllm/v1/worker/gpu_input_batch.py:L833-L935
        return {
            "num_reqs": self.num_reqs,
            "all_greedy": self.all_greedy,
            "req_ids": list(self.req_ids),
        }

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1082  num_reqs
    @property
    def num_reqs(self) -> int:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1082
        return len(self.req_id_to_index)

    # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1086  all_greedy
    @property
    def all_greedy(self) -> bool:
        # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1086
        return len(self.random_reqs) == 0
