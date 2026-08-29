# Subtract-only companion for v3 ch19 — vllm/compilation/piecewise_backend.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions, plus 章范围外域段 SUBTRACTED+归属注记。
#
# SUBTRACTED: to_bytes/load_all_ranges 预编译产物序列化/热启动路径
#   （L209-L243、L319-L341——standalone/AOT 缓存域，随 delete[3]）；
#   _log_compile_start 的 trace_structured 观测（L279-L317）。
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx

from .._host_seams import init_logger
from ..config import VllmConfig
from ..config.utils import Range

logger = init_logger(__name__)


# SOURCE: vllm/compilation/piecewise_backend.py:L26-L34 get_fake_args_from_graph
def get_fake_args_from_graph(graph: fx.GraphModule) -> list[Any]:
    """Get fake args directly from graph placeholder nodes."""
    fake_args = []
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            fake_args.append(node.meta["example_value"])
        else:
            break
    return fake_args


# SOURCE: vllm/compilation/piecewise_backend.py:L37-L76 create_concrete_args —
#   单尺寸编译：把 SymInt 形状落成具体 size 的 FakeTensor 输入
def create_concrete_args(graph: fx.GraphModule, size: int) -> list[Any]:
    """Create Fake example inputs with symbolic dims replaced by a concrete size.

    Used for single-size compilation where we need concrete-shaped inputs.
    The Dynamo-captured graph gives us example inputs with SymInts in them.
    """
    from torch._prims_common import compute_required_storage_length
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, is_symbolic

    def concretize(sym_val: Any) -> int:  # SOURCE: vllm/compilation/piecewise_backend.py:L47-L52
        """Replace all symbolic variables in a SymInt expression with size."""
        if not is_symbolic(sym_val):
            return int(sym_val)
        expr = sym_val.node.expr
        return int(expr.subs({s: size for s in expr.free_symbols}))

    fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    args: list[Any] = []
    with fake_mode:  # SOURCE: vllm/compilation/piecewise_backend.py:L57-L76
        for node in graph.graph.nodes:
            if node.op != "placeholder":
                break
            val = node.meta["example_value"]
            if isinstance(val, torch.SymInt):
                args.append(concretize(val))
            elif isinstance(val, torch.Tensor):
                new_shape = tuple(concretize(d) for d in val.shape)
                new_strides = tuple(concretize(s) for s in val.stride())
                new_storage_offset = concretize(val.storage_offset())
                needed_size = compute_required_storage_length(
                    new_shape, new_strides, new_storage_offset
                )
                t = torch.empty(needed_size, dtype=val.dtype, device=val.device)
                t = t.as_strided(new_shape, new_strides, new_storage_offset)
                args.append(t)
            else:
                args.append(val)
    return args


# SOURCE: vllm/compilation/piecewise_backend.py:L79-L83 RangeEntry —— 区间→
#   编译产物条目
@dataclasses.dataclass
class RangeEntry:  # SOURCE: vllm/compilation/piecewise_backend.py:L79-L83
    compile_range: Range
    compiled: bool = False
    runnable: Callable[..., Any] = None  # type: ignore


# SOURCE: vllm/compilation/piecewise_backend.py:L86-L190 PiecewiseBackend ——
#   片的编译管理器：按 compile_sizes/compile_ranges 预编译全部区间、运行期
#   按形状查条目（SUBTRACTED：to_bytes/load_all_ranges 预编译产物路径——
#   缓存域；_log_compile_start 观测）
class PiecewiseBackend:
    def __init__(
        self,
        graph: fx.GraphModule | None,
        vllm_config: VllmConfig,
        piecewise_compile_index: int,
        total_piecewise_compiles: int,
        sym_shape_indices: list[int],
        vllm_backend: Any,
        returns_tuple: bool,
        compiled_runnables: dict[str, Callable[..., Any]] | None = None,
        submod_name: str = "",
    ):
        # SOURCE: vllm/compilation/piecewise_backend.py:L99-L129
        """
        The backend for piecewise compilation.
        It mainly handles the compilation of static shapes and
        dispatching based on runtime shape.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.
        """
        assert bool(graph is not None) ^ bool(compiled_runnables is not None), (
            "exactly one of graph and compiled_runnables should be set."
        )

        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend
        self.compiled_runnables = compiled_runnables
        self.submod_name = submod_name

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = piecewise_compile_index == total_piecewise_compiles - 1

        self.is_full_graph = total_piecewise_compiles == 1
        self.is_encoder_compilation = vllm_backend.is_encoder

        self.compile_ranges = self.compilation_config.get_compile_ranges()
        # SUBTRACTED: encoder 编译区间的 int32 上界改写（L138-L149——多模态
        #   编译域）。

        log_string = f"PiecewiseBackend: compile_ranges: {self.compile_ranges}"
        logger.debug_once(log_string)

        self.compile_sizes = self.compilation_config.compile_sizes
        log_string = f"PiecewiseBackend: compile_sizes: {self.compile_sizes}"
        logger.debug_once(log_string)

        self.sym_shape_indices = sym_shape_indices
        self.returns_tuple = returns_tuple

        # the entries for ranges that we need to either
        self.range_entries: dict[Range, RangeEntry] = {}

        # We only keep compilation management inside this class directly.
        # SOURCE: vllm/compilation/piecewise_backend.py:L165-L184
        if self.compile_sizes is not None:
            for size in self.compile_sizes:
                if isinstance(size, str):
                    assert size == "cudagraph_capture_sizes"
                    raise NotImplementedError(
                        "cudagraph_capture_sizes not supported in compile_sizes."
                        "This should be handled in `post_init_cudagraph_sizes`."
                    )
                else:
                    assert isinstance(size, int)
                    range = Range(start=size, end=size)
                    if range not in self.compile_ranges:
                        self.range_entries[range] = RangeEntry(
                            compile_range=range,
                        )

        for range in self.compile_ranges:
            self.range_entries[range] = RangeEntry(
                compile_range=range,
            )

        # Track whether we've logged the graph for this subgraph (only log once)
        self._graph_logged = False

        if self.graph is not None:
            self.compile_all_ranges()
        else:
            # SUBTRACTED: load_all_ranges()（L192——预编译产物热启动路径，
            #   缓存域，随 delete[3]）。
            raise NotImplementedError(
                "compiled_runnables warm-start path is cache domain (deleted)"
            )

    # SOURCE: vllm/compilation/piecewise_backend.py:L194-L207 get_compiled_
    #   graph_wrapper —— tuple 解包外皮
    def get_compiled_graph_wrapper(
        self, compiled_graph: Callable[..., Any]
    ) -> Callable[..., Any]:
        def compiled_graph_wrapper(*args: Any) -> Any:  # SOURCE: vllm/compilation/piecewise_backend.py:L197-L205
            graph_output = compiled_graph(*args)
            # unpack the tuple if needed
            # TODO(rzou): the implication is that we're not
            # reading the python bytecode correctly in vLLM?
            if self.returns_tuple or not isinstance(graph_output, (tuple, list)):
                return graph_output
            else:
                return graph_output[0]

        return compiled_graph_wrapper

    # SUBTRACTED: to_bytes（L209-L243——standalone 编译产物序列化，缓存域）。

    # SOURCE: vllm/compilation/piecewise_backend.py:L245-L277 compile_all_
    #   ranges —— 启动期把全部区间一次编完（观测行 _log_compile_start 随
    #   观测域删）
    def compile_all_ranges(self) -> None:  # SOURCE: vllm/compilation/piecewise_backend.py:L245-L277
        """Compile all range entries for this piecewise subgraph up front."""
        assert self.graph is not None, (
            "Cannot compile without a graph. "
            "When loading from cache/AOT artifacts, "
            "compile_all_ranges should not be called."
        )

        for range_entry in self.range_entries.values():
            if range_entry.compiled:
                continue

            # SUBTRACTED: _log_compile_start 观测调用（L257——trace_structured
            #   观测域）。

            if range_entry.compile_range.is_single_size():
                args_list = create_concrete_args(
                    self.graph, range_entry.compile_range.start
                )
            else:
                args_list = get_fake_args_from_graph(self.graph)

            range_entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args_list,
                self.vllm_backend.inductor_config,
                self.compilation_config,
                compile_range=range_entry.compile_range,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                is_encoder=self.vllm_backend.is_encoder,
            )

            range_entry.compiled = True

    # SUBTRACTED: _log_compile_start（L279-L304——TORCH_TRACE 观测）与
    #   load_all_ranges（L319-L341——缓存热启动）。

    # SOURCE: vllm/compilation/piecewise_backend.py:L343-L356 _find_range_for_
    #   shape —— 运行期形状 → 区间条目
    def _find_range_for_shape(self, runtime_shape: int) -> RangeEntry | None:  # SOURCE: vllm/compilation/piecewise_backend.py:L343-L356
        # First we try to find the range entry for the concrete compile size
        # If not found, we search for the range entry
        # that contains the runtime shape.
        if self.compile_sizes is None:
            return None

        if runtime_shape in self.compile_sizes:
            return self.range_entries[Range(start=runtime_shape, end=runtime_shape)]
        else:
            for range in self.compile_ranges:
                if runtime_shape in range:
                    return self.range_entries[range]
        return None

    # SOURCE: vllm/compilation/piecewise_backend.py:L358-L380 __call__ ——
    #   运行期按形状查条目执行（全部区间已在 __init__ 编完）
    def __call__(self, *args: Any) -> Any:  # SOURCE: vllm/compilation/piecewise_backend.py:L358-L380
        if self.sym_shape_indices:
            runtime_shape = args[self.sym_shape_indices[0]]
            range_entry = self._find_range_for_shape(runtime_shape)
            assert range_entry is not None, (
                f"Shape: {runtime_shape} out of considered ranges: "
                f"{self.compile_ranges}"
            )
        else:
            # All inputs have static shapes; use the only compiled range_entry
            compiled_entries = [re for re in self.range_entries.values() if re.compiled]
            assert len(compiled_entries) == 1, (
                f"Expected exactly one compiled range_entry for static shape "
                f"compilation, but found {len(compiled_entries)}"
            )
            range_entry = compiled_entries[0]

        assert range_entry.compiled, (
            "All ranges should be compiled or loaded up front in "
            "PiecewiseBackend.__init__. "
            f"range_entry={range_entry.compile_range}"
        )
        return range_entry.runnable(*args)
