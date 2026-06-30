# 精简版（只做减法）— 对照真实源码 vllm_ascend/compilation/acl_graph.py
#
# ACLGraphWrapper 对位 vLLM 的 CUDAGraphWrapper（platform.get_static_graph_wrapper_cls() 返回它，
# 整体顶替 CUDAGraphWrapper）：用 torch.npu.NPUGraph 做 capture/replay，按 forward_context 的
# BatchDescriptor 分桶——不同 batch 形状各捕一张图（NPUGraph 绑定固定地址/形状）。差异点相对
# vLLM 原型：torch.npu.NPUGraph 替 torch.cuda.CUDAGraph、torch.npu.graph 替 torch.cuda.graph、
# 新增 stream-resource 耗尽错误码 207008 的兜底、replay 前按 FULL/eagle/enpu 条件 synchronize。
#
# 减法：full-graph 模式的 GraphParams/workspace 簿记（update_full_graph_params、三套 set/get/
# update_*_graph_params* 全局函数、weak_ref_workspaces）按 dossier 批准删除——属图捕获周边内存
# 管理，不影响 capture/replay/分桶/207008 主干；__call__ 里对它们的调用一并 SUBTRACTED。
# ACLGraphWrapper 的生命周期/属性透传样板（clear_all_graphs/clear_graphs/unwrap/
# cudagraph_wrapper/__getattr__）亦按批准删除。
import dataclasses
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401  真实文件用它做 torch.npu.NPUGraph 等
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.ascend_forward_context import _EXTRA_CTX

from ..utils import weak_ref_tensors

_STREAM_RESOURCE_ERROR_CODE = "207008"
_STREAM_RESOURCE_ERROR_MARKERS = (
    "insufficient_stream_resources",
    "stream resources are insufficient",
)
_STREAM_RESOURCE_GUIDANCE = (
    "ACL graph capture failed with a known stream-resource exhaustion "
    "signature. Consider upgrading to a newer HDK/CANN stack, reducing "
    "cudagraph_capture_sizes, lowering max_cudagraph_capture_size, preferring "
    "FULL or FULL_DECODE_ONLY for mostly uniform decode workloads, or "
    "temporarily disabling graph mode to confirm the failure is capture-related."
)


# SOURCE: vllm_ascend/compilation/acl_graph.py:L41
def _is_stream_resource_capture_error(exc: RuntimeError) -> bool:
    message = str(exc)
    lowered_message = message.lower()
    has_error_code = _STREAM_RESOURCE_ERROR_CODE in message
    has_stream_resource_marker = any(marker in lowered_message for marker in _STREAM_RESOURCE_ERROR_MARKERS)
    return has_stream_resource_marker or (has_error_code and "stream resource" in lowered_message)


# SOURCE: vllm_ascend/compilation/acl_graph.py:L49
def _raise_stream_resource_capture_error(exc: RuntimeError) -> None:
    raise RuntimeError(f"{_STREAM_RESOURCE_GUIDANCE}\nOriginal error:\n{exc}") from exc


# SOURCE: vllm_ascend/compilation/acl_graph.py:L53
@dataclasses.dataclass
class ACLGraphEntry:
    # SOURCE: vllm_ascend/compilation/acl_graph.py:L53
    batch_descriptor: BatchDescriptor
    aclgraph: torch.npu.NPUGraph | None = None
    output: Any | None = None

    # for aclgraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None


# SOURCE: vllm_ascend/compilation/acl_graph.py:L64
class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    # SUBTRACTED: _all_instances WeakSet ClassVar + classmethod clear_all_graphs（原 :L89-L94）——
    #             跨实例批量清图的生命周期样板，与 capture/replay/分桶主线无关。

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
        *,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ):
        # SOURCE: vllm_ascend/compilation/acl_graph.py:L96
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}
        self.enable_enpu = enable_enpu
        self.use_eagle = use_eagle

        # SUBTRACTED: ACLGraphWrapper._all_instances.add(self)（原 :L129）——配合已删的
        #             clear_all_graphs 注册表。

    # SUBTRACTED: __getattr__ / unwrap / cudagraph_wrapper / clear_graphs（原 :L131-L150）——
    #             属性透传与生命周期样板，不影响 capture-replay-分桶控制流。

    def __call__(self, *args, **kwargs):
        # SOURCE: vllm_ascend/compilation/acl_graph.py:L152
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without aclgraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_aclgraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_aclgraph_entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = self.concrete_aclgraph_entries[batch_descriptor]

        if entry.aclgraph is None:
            if self.aclgraph_options.debug_log_enable:
                # Since we capture aclgraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor)
            # validate that aclgraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            entry.input_addresses = input_addresses
            aclgraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    # during every model forward for piecewise aclgraph
                    # mode, we will capture many pieces of aclgraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the aclgraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.

                # Sync offloader's copy stream before capture.
                # Ensure any pre-capture prefetches from offloader are complete.
                from vllm.model_executor.offloader.base import get_offloader

                get_offloader().sync_prev_onload()
                forward_context.capturing = True
                try:
                    with torch.npu.graph(aclgraph, pool=self.graph_pool):
                        # `output` is managed by pytorch's aclgraph pool
                        output = self.runnable(*args, **kwargs)
                        # Join offloader's copy stream after forward to avoid
                        # unjoined stream error. The last layer's start_prefetch
                        # forks copy_stream, but wait_prefetch only happens in
                        # the next forward pass.
                        get_offloader().join_after_forward()
                        if self.aclgraph_options.weak_ref_output:
                            # by converting it to weak ref,
                            # the original `output` will immediately be released
                            # to save memory. It is only safe to do this for
                            # the last graph in piecewise aclgraph mode, because
                            # the output of the last graph will not be used by
                            # any other acl graph.
                            output = weak_ref_tensors(output)
                except RuntimeError as exc:
                    if _is_stream_resource_capture_error(exc):
                        _raise_stream_resource_capture_error(exc)
                    raise

            # SUBTRACTED: weak_ref_workspaces(_graph_params / _draft_graph_params /
            #             _draft_graph_prefill_params)（原 :L227-L234）——full-graph 模式下把三套
            #             GraphParams 的 workspace 转弱引用省显存的内存簿记，与捕获主线正交。

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        logger.info_once("Replaying aclgraph")
        # In async scheduling or multi-threaded (MT) scenarios, it is possible that
        # the CPU's record event (from update_attn_params) for the iteration i completes
        # before the grph replay of iteration i-1.
        # To ensure proper ordering, we must call synchronize here before replaying,
        # so that update_attn_params only executes after the previous graph replay has fully completed.
        # If we do not in main model and in full-graph mode when using merge-eagle-graph,
        # we do not need to synchronize.
        # When enable_enpu is on, model_runner orders update vs replay; skip here.
        # When FULL + EAGLE draft (merge path), replay does not need this barrier.
        is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
        if not self.enable_enpu and need_sync:
            torch.npu.current_stream().synchronize()
        entry.aclgraph.replay()
        return entry.output


# SUBTRACTED: weak_ref_workspaces / update_full_graph_params / GraphParams dataclass /
#             reset_graph_params / 三套 set/get/update_*_graph_params(_workspaces) 全局函数
#             （原 :L275-L419）——full-graph 模式下 attn/conv1d 参数与 workspace 的更新簿记，
#             属图捕获的周边内存管理，capture/replay/分桶/207008 主干不依赖其细节。
