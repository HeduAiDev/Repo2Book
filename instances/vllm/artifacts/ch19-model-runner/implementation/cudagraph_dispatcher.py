# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Subtract-only reduced companion for ch19.
# SOURCE: vllm/v1/cudagraph_dispatcher.py (pin f3fef123)
# CudagraphDispatcher.dispatch() picks the CUDA-graph runtime mode for a batch:
# FULL (whole forward replayed, needs an exact num_reqs match) -> PIECEWISE
# (non-attention op segments replayed, num_reqs=None relaxed key) -> NONE (eager).
# Reduced to the single-rank, no-LoRA, no-DP path per the approved
# subtraction_plan; CUDAGraphMode / BatchDescriptor are faithful local stand-ins
# for the vLLM enum / dataclass (same fields, same values).

import enum
from dataclasses import dataclass, replace


# SOURCE: vllm/config/compilation.py:L53  CUDAGraphMode
class CUDAGraphMode(enum.Enum):
    """Constants for the cudagraph mode. The subset NONE / PIECEWISE / FULL are
    also the concrete runtime modes used for cudagraph runtime dispatching."""

    # SOURCE: vllm/config/compilation.py:L59
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    # SUBTRACTED: composite separate-routine modes FULL_DECODE_ONLY /
    #   FULL_AND_PIECEWISE (tuple-valued) and their decode_mode/mixed_mode/
    #   has_mode/max_cudagraph_mode helpers — capture-time wiring lives in the
    #   compilation chapter; dispatch only consumes the three runtime modes.
    #   Orig: vllm/config/compilation.py:L62-L94

    # SOURCE: vllm/config/compilation.py:L89  separate_routine
    def separate_routine(self) -> bool:
        # SUBTRACTED: tuple-valued composite modes — reduced companion only has
        #   the three scalar runtime modes, none of which is a separate routine.
        #   Orig: vllm/config/compilation.py:L90
        return False

    # SOURCE: vllm/config/compilation.py:L92  valid_runtime_modes
    @classmethod
    def valid_runtime_modes(cls) -> frozenset["CUDAGraphMode"]:
        # SOURCE: vllm/config/compilation.py:L93
        return frozenset({cls.NONE, cls.PIECEWISE, cls.FULL})


# SOURCE: vllm/forward_context.py:L31  BatchDescriptor
@dataclass(frozen=True)
class BatchDescriptor:  # SOURCE: vllm/forward_context.py:L31
    """Uniquely describes the padded batch for cudagraph dispatching."""

    num_tokens: int
    # num_reqs can be None for PIECEWISE cudagraphs, which can handle any
    # number of requests.
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0


# SOURCE: vllm/v1/cudagraph_dispatcher.py:L15  CudagraphDispatcher
class CudagraphDispatcher:
    """Runtime cudagraph dispatcher.

    Stores two sets of dispatch keys, one for PIECEWISE and one for FULL. The
    keys are the only source of truth for which cudagraphs can be dispatched at
    runtime. dispatch() returns the runtime mode (FULL / PIECEWISE / NONE) plus
    a valid batch descriptor; the forward context then trusts that key to
    replay (mode matches) or pass through eager (NONE / no match).
    """

    def __init__(
        self,
        cudagraph_mode: CUDAGraphMode,
        max_cudagraph_capture_size: int | None,
        max_num_seqs: int,
        capture_sizes: list[int] | None = None,
    ):
        # SOURCE: vllm/v1/cudagraph_dispatcher.py:L34
        # SUBTRACTED: vllm_config / compilation_config plumbing, speculative
        #   uniform_decode_query_len (>1 only with spec decode), and the
        #   piecewise-compilation assertion. Approved (spec decode / compilation
        #   wiring). Orig: vllm/v1/cudagraph_dispatcher.py:L35-L60
        self.uniform_decode_query_len = 1
        self.max_cudagraph_capture_size = max_cudagraph_capture_size
        self.max_num_seqs = max_num_seqs

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        self.keys_initialized = False
        # SUBTRACTED: specialize_lora_count (LoRA capture). Approved.
        #   Orig: vllm/v1/cudagraph_dispatcher.py:L63-L67
        # Default cudagraph_mode to NONE until initialize_cudagraph_keys is called.
        self.cudagraph_mode = CUDAGraphMode.NONE

        # capture_sizes drives the bs->padded-size mapping; provided directly
        # here instead of read off compilation_config.
        self._capture_sizes = sorted(capture_sizes or [])
        self._init_cudagraph_mode = cudagraph_mode

    # SOURCE: vllm/v1/cudagraph_dispatcher.py:L71  _compute_bs_to_padded_graph_size
    def _compute_bs_to_padded_graph_size(self) -> None:
        """Pre-compute the mapping from batch size to padded graph size."""
        max_size = self.max_cudagraph_capture_size
        capture_sizes = self._capture_sizes
        assert max_size is not None
        assert capture_sizes is not None
        self._bs_to_padded_graph_size: list[int] = [0] * (max_size + 1)
        for end, start in zip(
            capture_sizes + [max_size + 1],
            [0] + capture_sizes,
        ):
            for bs in range(start, end):
                if bs == start:
                    self._bs_to_padded_graph_size[bs] = start
                else:
                    self._bs_to_padded_graph_size[bs] = end
        # SUBTRACTED: compile_sizes padding validation (compile-time guard).
        #   Approved. Orig: vllm/v1/cudagraph_dispatcher.py:L92-L108

    # SUBTRACTED: _get_lora_cases (LoRA capture cases). Approved (no LoRA).
    #   Orig: vllm/v1/cudagraph_dispatcher.py:L110-L129

    # SOURCE: vllm/v1/cudagraph_dispatcher.py:L131  _create_padded_batch_descriptor
    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
    ) -> BatchDescriptor:
        max_num_seqs = self.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        num_tokens_padded = self._bs_to_padded_graph_size[num_tokens]

        # SUBTRACTED: cudagraph_mode.has_mode(FULL) gate — for the scalar FULL
        #   mode in the reduced companion this is just `cudagraph_mode == FULL`.
        #   Orig: vllm/v1/cudagraph_dispatcher.py:L142
        if uniform_decode and self.cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    # SOURCE: vllm/v1/cudagraph_dispatcher.py:L157  add_cudagraph_key
    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    # SOURCE: vllm/v1/cudagraph_dispatcher.py:L165  initialize_cudagraph_keys
    def initialize_cudagraph_keys(self):
        # This should be called only after the attention backend is initialized,
        # so we know the resolved cudagraph mode.
        cudagraph_mode = self._init_cudagraph_mode
        self.cudagraph_mode = cudagraph_mode

        # Early exit if cudagraphs are disabled.
        if cudagraph_mode == CUDAGraphMode.NONE:
            self.keys_initialized = True
            return

        self._compute_bs_to_padded_graph_size()

        # SUBTRACTED: LoRA capture-case product + the FULL-decode separate-routine
        #   key block (only fires for composite modes like FULL_AND_PIECEWISE,
        #   which the reduced companion does not have). What remains is the
        #   mixed_mode() registration loop: for a scalar mode, mixed_mode() == the
        #   mode itself. PIECEWISE relaxes the key to num_reqs=None (any req
        #   count); FULL keeps the exact num_reqs (uniform=False, non-relaxed).
        #   Approved. Orig: vllm/v1/cudagraph_dispatcher.py:L179-L230
        for bs in self._capture_sizes:
            batch_desc = self._create_padded_batch_descriptor(bs, False, False, 0)
            # Only relax for PIECEWISE; FULL needs the exact num_reqs because the
            # full graph captures attention, whose metadata depends on it.
            if cudagraph_mode == CUDAGraphMode.PIECEWISE:
                batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
            self.add_cudagraph_key(cudagraph_mode, batch_desc)

        self.keys_initialized = True

    # SOURCE: vllm/v1/cudagraph_dispatcher.py:L234  dispatch
    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes=None,
        invalid_modes=None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """Dispatch to a cudagraph runtime mode and a valid batch descriptor.

        A new batch descriptor may be returned, as a uniform batch can be
        dispatched to a graph that supports a more general (non-uniform) batch.
        """
        allowed_modes = valid_modes or CUDAGraphMode.valid_runtime_modes()

        if invalid_modes:
            allowed_modes -= invalid_modes

        assert len(allowed_modes) >= 1, (
            f"No allowed cudagraph modes: valid_modes={valid_modes}, "
            f"invalid_modes={invalid_modes}"
        )
        max_size = self.max_cudagraph_capture_size

        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or max_size is None
            or num_tokens > max_size
            or allowed_modes <= {CUDAGraphMode.NONE}
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        # SUBTRACTED: LoRA effective_num_active_loras normalization
        #   (specialize / bisect / max_loras+1). With no LoRA, has_lora is False
        #   and this whole block is skipped. Approved.
        #   Orig: vllm/v1/cudagraph_dispatcher.py:L282-L299
        effective_num_active_loras = num_active_loras

        # SUBTRACTED: separate_routine() gate — scalar modes are never separate
        #   routines, so normalized_uniform == uniform_decode.
        #   Orig: vllm/v1/cudagraph_dispatcher.py:L301
        normalized_uniform = uniform_decode and self.cudagraph_mode.separate_routine()
        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, normalized_uniform, has_lora, effective_num_active_loras
        )

        if CUDAGraphMode.FULL in allowed_modes:
            # check if key exists for full cudagraph (exact num_reqs)
            batch_desc_to_check = batch_desc
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc_to_check

        if CUDAGraphMode.PIECEWISE in allowed_modes:
            # also check the relaxed key for the more "general" piecewise
            # cudagraph (num_reqs=None)
            batch_desc_to_check = replace(batch_desc, num_reqs=None, uniform=False)
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
                return CUDAGraphMode.PIECEWISE, batch_desc_to_check

        assert CUDAGraphMode.NONE in allowed_modes, (
            f"No matching cudagraph found and NONE is not in "
            f"allowed_modes={allowed_modes}"
        )
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
