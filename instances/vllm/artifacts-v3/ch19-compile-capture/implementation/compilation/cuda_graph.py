# Subtract-only companion for v3 ch19 — vllm/compilation/cuda_graph.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`).
#
# Deletions here (dossier subtraction_plan.delete #5 + 章范围外收窄):
#   - CUDAGraphLogging 观测聚合类（L40-L124——统计表格，观测域）。
#     CUDAGraphStat（L32-L37）保留：gpu_model_runner._determine_batch_
#     execution_and_padding（station 10 / m13）直接消费这 6 行记录类，
#     删它会让 must_keep 方法残缺——delete[5] 的实质（表格聚合+offloader）
#     已删，见 impl-notes §对 delete[5] 的执行说明。
#   - offloader 同步调用（L308-L310/L320-L324/L357-L359——权重卸载扩展态）。
#   - _runnable_str 调试字段（L192 定义 + __getattr__ 的 raise 分支连带删，
#     剩裸 raise AttributeError；is_debugging_mode 保留——回放断言在用）。
from __future__ import annotations

import dataclasses
import weakref
from contextlib import ExitStack
from typing import Any, ClassVar
from unittest.mock import patch

import torch

from .._host_seams import (
    current_platform,
    envs,
    init_logger,
    set_graph_pool_id,
)
from ..config import CUDAGraphMode
from ..forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from ..utils.torch_utils import current_stream, weak_ref_tensors
from .counter import compilation_counter
from .monitor import validate_cudagraph_capturing_enabled

logger = init_logger(__name__)


# SOURCE: vllm/compilation/cuda_graph.py:L32-L37 CUDAGraphStat —— 每拍
#   padding 观测记录（num_paddings = padded - unpadded 的账目载体）
@dataclasses.dataclass(frozen=True)
class CUDAGraphStat:  # SOURCE: vllm/compilation/cuda_graph.py:L32-L37
    num_unpadded_tokens: int
    num_padded_tokens: int
    num_paddings: int
    runtime_mode: str


# SUBTRACTED: CUDAGraphLogging（L40-L124——统计表格聚合/日志，观测域；
#   消费点在 gpu_model_runner 的 shutdown 观测面，不在本章切面）。


# SOURCE: vllm/compilation/cuda_graph.py:L127-L135 CUDAGraphEntry —— key→图
#   条目：捕获态 input_addresses/output/cudagraph 的载体
@dataclasses.dataclass
class CUDAGraphEntry:  # SOURCE: vllm/compilation/cuda_graph.py:L127-L135
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None
    output: Any | None = None

    # for cudagraph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None


# SOURCE: vllm/compilation/cuda_graph.py:L138-L142 CUDAGraphOptions —— 每片
#   捕获选项（首片 debug / 非首片禁 GC / 末片 weak_ref）
@dataclasses.dataclass
class CUDAGraphOptions:  # SOURCE: vllm/compilation/cuda_graph.py:L138-L142
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


# SOURCE: vllm/compilation/cuda_graph.py:L145-L361 CUDAGraphWrapper —— 捕获/
#   回放包装：盲信 forward context 的 mode/descriptor、按 key 首遇即捕、
#   回放前 DEBUG 逐 data_ptr 断言；不存持久缓冲不拷输入（固定地址是
#   runner 的职责）
class CUDAGraphWrapper:
    """Wraps a runnable to add CUDA graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the cudagraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for cudagraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform cudagraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: CUDAGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as
    a trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    _all_instances: ClassVar[weakref.WeakSet["CUDAGraphWrapper"]] = weakref.WeakSet()

    @classmethod
    def clear_all_graphs(cls) -> None:  # SOURCE: vllm/compilation/cuda_graph.py:L172-L176
        """Clear captured graphs from all CUDAGraphWrapper instances."""
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    # SOURCE: vllm/compilation/cuda_graph.py:L178-L209 __init__（_runnable_str
    #   字段随 delete[5] 删；is_debugging_mode 保留——回放断言在用）
    def __init__(  # SOURCE: vllm/compilation/cuda_graph.py:L178-L209
        self,
        runnable: Any,
        vllm_config: Any,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
    ) -> None:
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        # SUBTRACTED: _runnable_str 调试字段（L192——delete[5]，连同
        #   __getattr__ 的拼名 raise 分支）。

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry] = {}

        CUDAGraphWrapper._all_instances.add(self)

    # SOURCE: vllm/compilation/cuda_graph.py:L211-L220 __getattr__（delete[5]
    #   删 _runnable_str 后剩裸 raise AttributeError）
    def __getattr__(self, key: str) -> Any:  # SOURCE: vllm/compilation/cuda_graph.py:L211-L220
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError

    # SOURCE: vllm/compilation/cuda_graph.py:L222-L224 unwrap
    def unwrap(self) -> Any:
        # in case we need to access the original runnable.
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "CUDAGraphWrapper":  # SOURCE: vllm/compilation/cuda_graph.py:L226-L228
        return self

    # SOURCE: vllm/compilation/cuda_graph.py:L230-L231 clear_graphs
    def clear_graphs(self) -> None:
        self.concrete_cudagraph_entries.clear()

    # SOURCE: vllm/compilation/cuda_graph.py:L233-L261 __call__ 头 —— 无
    #   context 直通（视觉编码器等）；mode 不匹配直通（嵌套多 wrapper 各按
    #   mode 认领）；匹配则按 batch_descriptor 建表项
    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:
        if not is_forward_context_available():
            # No forward context means we are outside the normal
            # inference path (e.g. a vision encoder forward pass).
            # Just run the underlying function without cudagraphs.
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            cudagraph_runtime_mode == CUDAGraphMode.NONE
            or cudagraph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        if batch_descriptor not in self.concrete_cudagraph_entries:
            # create a new entry for this batch descriptor
            self.concrete_cudagraph_entries[batch_descriptor] = CUDAGraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_cudagraph_entries[batch_descriptor]

        # SOURCE: vllm/compilation/cuda_graph.py:L265-L344 捕获路径（offloader
        #   两处调用随 delete[5] 删；gc patch/共享图池/弱引用输出原文保留）
        if entry.cudagraph is None:
            if self.cudagraph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    # during every model forward for piecewise cudagraph
                    # mode, we will capture many pieces of cudagraphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(
                        patch("gc.collect", lambda *args, **kwargs: None)
                    )
                    stack.enter_context(
                        patch(
                            "torch.accelerator.empty_cache",
                            lambda *args, **kwargs: None,
                        )
                    )

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())

                # SUBTRACTED: offloader 捕获前同步（L308-L310——权重卸载
                #   扩展态，delete[5]）。

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(
                    cudagraph,
                    pool=self.graph_pool,
                    stream=current_stream(),
                ):
                    # `output` is managed by pytorch's cudagraph pool
                    output = self.runnable(*args, **kwargs)
                    # SUBTRACTED: offloader join_after_forward（L320-L324
                    #   ——delete[5]）。
                    if self.cudagraph_options.weak_ref_output:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph in piecewise cuadgraph mode, because
                        # the output of the last graph will not be used by
                        # any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        # SOURCE: vllm/compilation/cuda_graph.py:L346-L361 回放路径 —— DEBUG
        #   模式逐 data_ptr 断言与捕获时一致（『形状全等 AND 地址不变』的
        #   第二条运行期体检）
        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        # SUBTRACTED: offloader 回放前同步（L357-L359——delete[5]）。
        entry.cudagraph.replay()
        return entry.output
