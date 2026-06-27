"""Lazy pipeline-parallel synchronization — subtract-only companion.

Faithful subset of vLLM's PP receive/send path, preserving the exact control
flow that makes the PP `wait` *lazy*:

  * `AsyncIntermediateTensors` (vllm/v1/worker/gpu_worker.py) holds the irecv
    handles and defers `.wait()` until `.tensors` is first read.
  * `irecv_tensor_dict` / `isend_tensor_dict` (vllm/distributed/parallel_state.py)
    return non-blocking handles instead of synchronizing.
  * `Worker.execute_model` (vllm/v1/worker/gpu_worker.py) stitches it together:
    wait previous isend → irecv into AsyncIntermediateTensors → forward → isend,
    deferring this send's completion check to the *next* call.

Subtractions are marked `# SUBTRACTED:` and are all approved delete-items.
torch.distributed handles are modeled by `_bridge.Handle` (see _bridge.py); no
vLLM concept is invented.
"""

from __future__ import annotations

from typing import Any, Callable

from ._bridge import Handle


# SOURCE: vllm/sequence.py:L18
class IntermediateTensors:
    """Hidden states + residuals passed from one PP stage to the next."""

    # SOURCE: vllm/sequence.py:L29
    def __init__(self, tensors: dict[str, Any]) -> None:
        # SUBTRACTED: kv_connector_output param — KV-transfer plumbing,
        #   orthogonal to PP lazy sync · vllm/sequence.py:L32
        self.tensors = tensors

    # SOURCE: vllm/sequence.py:L41
    def __getitem__(self, key: str):
        return self.tensors[key]


# SOURCE: vllm/v1/worker/gpu_worker.py:L74
class AsyncIntermediateTensors(IntermediateTensors):
    """IntermediateTensors with lazy comm synchronization"""

    # SOURCE: vllm/v1/worker/gpu_worker.py:L77
    def __init__(
        self,
        tensors: dict[str, Any],
        comm_handles: list[Handle] | None = None,
        comm_postprocess: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__(tensors)
        self._comm_handles = comm_handles
        self._comm_postprocess = comm_postprocess
        self._comm_waited = False

    # SOURCE: vllm/v1/worker/gpu_worker.py:L88
    def wait_for_comm(self) -> None:
        if self._comm_waited:
            return
        if self._comm_handles:
            for handle in self._comm_handles:
                handle.wait()
        if self._comm_postprocess:
            for fn in self._comm_postprocess:
                fn()
        self._comm_waited = True

    # SOURCE: vllm/v1/worker/gpu_worker.py:L99
    def __getattribute__(self, name: str):
        # ensure `.tensors` is ready before use
        if name == "tensors" and not object.__getattribute__(self, "_comm_waited"):
            object.__getattribute__(self, "wait_for_comm")()
        return object.__getattribute__(self, name)


# SOURCE: vllm/distributed/parallel_state.py (GroupCoordinator PP P2P subset)
class PPGroupCoordinator:
    """Pipeline-parallel group: non-blocking tensor-dict send/recv.

    Subtracts everything except the irecv/isend non-blocking core that the
    lazy-sync story rides on.
    """

    # SOURCE: vllm/distributed/parallel_state.py:L309 (GroupCoordinator.__init__ subset)
    def __init__(self, rank_in_group: int, world_size: int) -> None:
        self.rank_in_group = rank_in_group
        self.world_size = world_size

    @property
    def is_first_rank(self) -> bool:  # SOURCE: parallel_state.py:L418 (is_first_rank)
        return self.rank_in_group == 0

    @property
    def is_last_rank(self) -> bool:  # SOURCE: parallel_state.py:L422 (is_last_rank)
        return self.rank_in_group == self.world_size - 1

    # SOURCE: vllm/distributed/parallel_state.py:L954
    def irecv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: "PPGroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[dict[str, Any] | None, list[Handle], list[Callable[[], None]]]:
        # SOURCE: vllm/distributed/parallel_state.py:L964
        if self.world_size == 1:
            return None, [], []

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        # SUBTRACTED: use_cpu_custom_send_recv synchronous fast-path — custom CPU
        #   communicator, not the GPU NCCL main line · parallel_state.py:L971-978

        # First receive metadata (shape/dtype) so we can pre-allocate the recv
        # buffer — this is *why* the irecv can be async.
        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        handles: list[Handle] = []
        postprocess: list[Callable[[], None]] = []

        # SOURCE: vllm/distributed/parallel_state.py:L993
        for key, value in recv_metadata_list:
            if _is_tensor_metadata(value):
                full_tensor = _empty_like_metadata(value)
                # SUBTRACTED: numel()==0 empty-tensor short-circuit — boundary
                #   case · parallel_state.py:L998-1000
                # SUBTRACTED: _should_use_all_gather / sequence-parallel slice +
                #   all_gather postprocess closure — SP is a layered optional
                #   optimization; without SP this reduces to a plain irecv ·
                #   parallel_state.py:L1002-1027
                handle = _dist_irecv(full_tensor, src=self.ranks(src))
                handles.append(handle)
                tensor_dict[key] = full_tensor
            else:
                tensor_dict[key] = value

        return tensor_dict, handles, postprocess

    # SOURCE: vllm/distributed/parallel_state.py:L859
    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, Any],
        dst: int | None = None,
        all_gather_group: "PPGroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Handle]:
        # SOURCE: vllm/distributed/parallel_state.py:L866
        if self.world_size <= 1:
            return []

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        # SUBTRACTED: use_cpu_custom_send_recv synchronous path ·
        #   parallel_state.py:L873-880

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst)

        tensor_keys = [k for k, v in tensor_dict.items() if _is_tensor(v)]
        assert len(tensor_keys) == len(tensor_list)

        # SOURCE: vllm/distributed/parallel_state.py:L896
        handles: list[Handle] = []
        for key, tensor in zip(tensor_keys, tensor_list):
            # SUBTRACTED: numel()==0 skip + _should_use_all_gather SP reshape +
            #   cuda record_stream — SP/CUDA-stream details, not the non-blocking
            #   send main line · parallel_state.py:L898-911
            handle = _dist_isend(tensor, dst=self.ranks(dst))
            handles.append(handle)

        return handles

    # --- metadata side-channel (bridge stand-ins; see _bridge rationale) ---
    # Real vLLM uses recv_object/send_object over the CPU group to ship the
    # tensor metadata list. We keep the call sites (so the control flow is
    # intact) and back them with an in-process channel the test wires up.
    # SOURCE: vllm/distributed/parallel_state.py:L988 (recv_object — metadata recv)
    def recv_object(self, src: int) -> list[tuple[str, Any]]:
        return self._link.recv_metadata(src)

    # SOURCE: vllm/distributed/parallel_state.py:L891 (send_object — metadata send)
    def send_object(self, obj: Any, dst: int) -> None:
        self._link.send_metadata(obj, dst)

    # SOURCE: vllm/distributed/parallel_state.py:L1011 (self.ranks[src] global-rank map)
    def ranks(self, idx: int) -> int:
        return idx

    # bound by the test harness to an in-process PP link + handle factory
    _link: Any = None


# --- bridge helpers (not vLLM abstractions; see _bridge.py) ---------------
# These wrap the dependencies vLLM gets from torch.distributed / serial_utils.
_HANDLE_FACTORY: Callable[[], Handle] | None = None


# SOURCE: torch.distributed.irecv(full_tensor, src, group) (parallel_state.py:L1030)
def _dist_irecv(buf: Any, src: int) -> Handle:
    # mirrors torch.distributed.irecv → returns a wait()-able handle
    assert _HANDLE_FACTORY is not None
    return _HANDLE_FACTORY()


# SOURCE: torch.distributed.isend(tensor, dst, group) (parallel_state.py:L907)
def _dist_isend(tensor: Any, dst: int) -> Handle:
    # mirrors torch.distributed.isend → returns a wait()-able handle
    assert _HANDLE_FACTORY is not None
    return _HANDLE_FACTORY()


# SOURCE: isinstance(v, torch.Tensor) test (parallel_state.py:L893)
def _is_tensor(v: Any) -> bool:
    return getattr(v, "_is_tensor", False)


# SOURCE: isinstance(value, TensorMetadata) test (parallel_state.py:L994)
def _is_tensor_metadata(v: Any) -> bool:
    return getattr(v, "_is_metadata", False)


# SOURCE: torch.empty(value.size, dtype, device) recv buffer alloc (parallel_state.py:L995)
def _empty_like_metadata(meta: Any) -> Any:
    return meta.allocate()


# SOURCE: _split_tensor_dict(tensor_dict) (parallel_state.py:L890)
def _split_tensor_dict(d: dict[str, Any]) -> tuple[list[tuple[str, Any]], list[Any]]:
    metadata: list[tuple[str, Any]] = []
    tensors: list[Any] = []
    for k, v in d.items():
        if _is_tensor(v):
            metadata.append((k, v.metadata()))
            tensors.append(v)
        else:
            metadata.append((k, v))
    return metadata, tensors


# SOURCE: vllm/v1/worker/gpu_worker.py:L106 (Worker — PP-communication subset)
class Worker:
    """PP one-cell executor. Only the lazy PP send/recv control flow is kept."""

    # SOURCE: vllm/v1/worker/gpu_worker.py:L107 (Worker.__init__; _pp_send_work init)
    def __init__(self, pp_group: PPGroupCoordinator, model_runner: Any) -> None:
        self._pp_group = pp_group
        self.model_runner = model_runner
        self._pp_send_work: list[Handle] = []

    # SOURCE: get_pp_group() accessor (vllm/distributed/parallel_state.py:L1700)
    def get_pp_group(self) -> PPGroupCoordinator:
        return self._pp_group

    # SOURCE: vllm/v1/worker/gpu_worker.py:L782
    def execute_model(self, scheduler_output: Any) -> Any:
        # ensure any previous non-blocking PP sends are complete
        if self._pp_send_work:
            for handle in self._pp_send_work:
                handle.wait()
            self._pp_send_work = []

        intermediate_tensors = None
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        all_gather_tensors: dict[str, bool] = {}
        # SUBTRACTED: sequence-parallel all_gather_tensors computation (PP+SP) —
        #   only affects whether residual needs all-gather; lazy-sync main line
        #   is unaffected with SP off · vllm/v1/worker/gpu_worker.py:L799-L826

        if forward_pass and not self.get_pp_group().is_first_rank:
            tensor_dict, comm_handles, comm_postprocess = (
                self.get_pp_group().irecv_tensor_dict(
                    all_gather_group=None,
                    all_gather_tensors=all_gather_tensors,
                )
            )
            assert tensor_dict is not None
            intermediate_tensors = AsyncIntermediateTensors(
                tensor_dict,
                comm_handles=comm_handles,
                comm_postprocess=comm_postprocess,
            )

        output = self.model_runner.execute_model(
            scheduler_output, intermediate_tensors
        )
        # SUBTRACTED: pooling-model output is None → pool() side-branch ·
        #   vllm/v1/worker/gpu_worker.py:L846-L851
        if isinstance(output, _ModelRunnerOutputMarker) or output is None:
            return output

        assert isinstance(output, IntermediateTensors)
        # SUBTRACTED: external_launcher / is_last_rank asserts — never fire for a
        #   normal PP middle rank · vllm/v1/worker/gpu_worker.py:L858-L862

        # launch non-blocking send of intermediate tensors
        self._pp_send_work = self.get_pp_group().isend_tensor_dict(
            output.tensors,
            all_gather_group=None,
            all_gather_tensors=all_gather_tensors,
        )

        return None


# SOURCE: ModelRunnerOutput | AsyncModelRunnerOutput type test (gpu_worker.py:L842)
class _ModelRunnerOutputMarker:
    """Marker for a real ModelRunnerOutput / AsyncModelRunnerOutput.

    In vLLM the early-return checks `isinstance(output, ModelRunnerOutput |
    AsyncModelRunnerOutput | NoneType)`. We keep that branch with a marker base
    so the *last-rank returns output, middle-rank returns None* control flow is
    preserved without importing the full runner output types.
    """
