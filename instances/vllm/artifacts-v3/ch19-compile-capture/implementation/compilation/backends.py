# Subtract-only companion for v3 ch19 — vllm/compilation/backends.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`), plus
# 章范围外域段以 SUBTRACTED+归属注记收窄（impl-notes §范围裁剪）。
#
# Deletions here (dossier subtraction_plan.delete #3/#4 + 范围收窄):
#   #3  VllmBackend.__call__ 的缓存/落盘巨块（hash/cache_dir/元数据 json
#       L1021-L1101、计时与 instrument L1103-L1164、split_gm 副本与 depyf
#       钩子 L1181-L1199、codegen 序列化尾段 L1226-L1339）+
#       collect_standalone_compile_artifacts（L872-L932）+ configure_post_pass
#       的 pass 装配（L934-L971）；
#   #4  _decompose_size_nodes（L479-L550）与 _merge_empty_only_subgraphs
#       （L437-L476）两个边界优化 pass 及其调用行——删的是优化不是功能；
#       tests 里保留多切点样例验证切图仍正确。
#   范围收窄: codegen.py（generate_execution_code/compile_execution_fn）与
#       caching.py（VllmSerializableFunction）是序列化域——返回值改为
#       split_gm 本体（生成式 runtime_callable 逐个调用同一批 submod
#       callables、顺序相同；getattr 先查 __dict__ 再查 _modules）。
from __future__ import annotations

import dataclasses
import operator
import time
from contextlib import contextmanager
from collections.abc import Callable, Generator, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.fx as fx
from torch.fx._lazy_graph_module import _use_lazy_graph_module

from .._host_seams import (
    current_platform,
    envs,
    init_logger,
)
from ..config import CompilationConfig, CUDAGraphMode, VllmConfig
from ..config.utils import Range
from ..utils.import_utils import resolve_obj_by_qualname
from ..utils.torch_utils import is_torch_equal_or_newer
from .compiler_interface import (
    CompilerInterface,
    EagerAdaptor,
    InductorAdaptor,
)
from .counter import compilation_counter
from .partition_rules import (
    inductor_partition_rule_context,
    should_split,
)

logger = init_logger(__name__)

# SUBTRACTED: make_copy_and_call（L59-L93——cudagraph_copy_inputs 的输入拷贝
#   包装，消费点在 __call__ 序列化尾段，随 delete[3] 删）。


# SOURCE: vllm/compilation/backends.py:L96-L121 make_compiler —— backend →
#   编译器适配器（inductor/eager/custom）
def make_compiler(compilation_config: CompilationConfig) -> CompilerInterface:  # SOURCE: vllm/compilation/backends.py:L96-L121
    assert not envs.VLLM_USE_MEGA_AOT_ARTIFACT or envs.VLLM_USE_STANDALONE_COMPILE, (
        "VLLM_USE_MEGA_AOT_ARTIFACT=1 requires VLLM_USE_STANDALONE_COMPILE=1"
    )

    if compilation_config.backend == "inductor":
        # Use standalone compile only if requested, version is new enough,
        # and the symbol actually exists in this PyTorch build.
        # SUBTRACTED: InductorStandaloneAdaptor 分支（L104-L110——standalone
        #   实验态，delete[3] 族）。
        logger.debug("Using InductorAdaptor")
        return InductorAdaptor()
    elif compilation_config.backend == "eager":
        logger.debug("Using EagerAdaptor")
        return EagerAdaptor()
    else:
        logger.debug("Using custom backend: %s", compilation_config.backend)
        compiler = resolve_obj_by_qualname(current_platform.get_compile_backend())()
        assert isinstance(compiler, CompilerInterface)
        return compiler


# SOURCE: vllm/compilation/backends.py:L124-L161 CompilerManager（缓存目录
#   装配/落盘族 L163-L221 随 delete[3] 删；伴读版恒 disable_cache——
#   每次进程内重编译，行为与冷启动一致）
class CompilerManager:
    """
    A manager to manage the compilation process, including
    caching the compiled graph, loading the compiled graph,
    and compiling the graph.

    The cache is a dict mapping
    `(runtime_shape, graph_index, backend_name)`
    to `any_data` returned from the compiler.
    """

    def __init__(self, compilation_config: CompilationConfig) -> None:  # SOURCE: vllm/compilation/backends.py:L139-L144
        self.cache: dict[tuple[Range, int, str], Any] = dict()
        self.is_cache_updated = False
        self.compilation_config = compilation_config
        self.compiler = make_compiler(compilation_config)
        self.loaded_artifacts: dict[str, Any] = {}
        # HOST 注记: 伴读版缓存恒禁用（真源 initialize_cache 的目录装配/落盘
        #   属 delete[3] 缓存域）。
        self.disable_cache = True
        self.cache_dir = ""
        self.cache_file_path = ""

    def compute_hash(self, vllm_config: VllmConfig) -> str:  # SOURCE: vllm/compilation/backends.py:L146-L147
        return self.compiler.compute_hash(vllm_config)

    @contextmanager
    def compile_context(self, compile_range: Range) -> Generator[None, None, None]:  # SOURCE: vllm/compilation/backends.py:L149-L161
        """Provide compilation context for the duration of compilation to set
        any torch global properties we want to scope to a single Inductor
        compilation (e.g. partition rules, pass context)."""
        yield
        # SUBTRACTED: pass_context/inductor_partition_rule_context 装配
        #   （L154-L160——pass 域；use_inductor_graph_partition 默认 False 不达）。

    # SUBTRACTED: initialize_cache / save_to_file / load（L163-L261——缓存
    #   目录装配/落盘/读回，delete[3]）。

    # SOURCE: vllm/compilation/backends.py:L264-L399 compile —— 编译一个
    #   （子图，区间）：缓存读回与 in-memory artifact 复用随 delete[3] 删，
    #   直落适配器 compile；首图/末图计时保留骨架
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        additional_inductor_config: dict[str, Any],
        compilation_config: CompilationConfig,
        compile_range: Range,
        graph_index: int = 0,
        num_graphs: int = 1,
        is_encoder: bool = False,
    ) -> Any:
        if graph_index == 0:
            # before compiling the first graph, record the start time
            global compilation_start_time
            compilation_start_time = time.perf_counter()

        compilation_counter.num_backend_compilations += 1

        # SUBTRACTED: 缓存 load 读回（L282-L298——delete[3]）与 autograd_cache_key
        #   monkey-patch 的 in-memory artifact 复用块（L300-L364——缓存优化）。

        with self.compile_context(compile_range):
            # SUBTRACTED: maybe_key 装配（L302-L308——artifact 缓存键，缓存域）。
            compiled_graph, handle = self.compiler.compile(
                graph,
                example_inputs,
                additional_inductor_config,
                compile_range,
                None,
            )

        assert compiled_graph is not None, "Failed to compile the graph"

        # SUBTRACTED: 缓存落账（L368-L388——delete[3]）。

        # after compiling the last graph, record the end time
        # SOURCE: vllm/compilation/backends.py:L390-L397
        if graph_index == num_graphs - 1:
            elapsed = time.perf_counter() - compilation_start_time
            logger.info_once(
                "Compiling a graph for compile range %s takes %.2f s",
                str(compile_range),
                elapsed,
            )

        return compiled_graph


# SUBTRACTED: StopCompiling（L402-L403——in-memory artifact 复用的早退信号，
#   随复用块删）。


# SOURCE: vllm/compilation/backends.py:L406-L411 SplitItem —— 切片条目
@dataclasses.dataclass
class SplitItem:  # SOURCE: vllm/compilation/backends.py:L406-L411
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


# SUBTRACTED: _is_empty_allocation_node（L414-L433）与 _merge_empty_only_
#   subgraphs（L437-L476）——delete[4]：空子图回并的边界优化（删后
#   split_module 语义不变，只是可能多出平凡子图）；_decompose_size_nodes
#   （L479-L550）——delete[4]：torch.Size 跨切点的分解（仅多维 sym shape
#   模型需要）。


# SOURCE: vllm/compilation/backends.py:L553-L627 split_graph —— 切图算法：
#   遍历 FX 节点按 should_split 编 subgraph_id（连续切点合并）、
#   keep_original_order=True 保突变语义、split_module 切成子模块序列
def split_graph(
    graph: fx.GraphModule, splitting_ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    # SUBTRACTED: _decompose_size_nodes(graph)（L556——delete[4] 边界优化）。

    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id: dict[fx.Node, int] = {}
    split_op_graphs: list[int] = []
    # SOURCE: vllm/compilation/backends.py:L562-L591
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue

        # Check if this is a getitem operation on a node from an earlier subgraph.
        # If so, assign it to the same subgraph as its input to avoid passing entire
        # tuple as input to submodules, which is against standalone_compile and
        # AoTAutograd input requirement.
        if node.op == "call_function" and node.target == operator.getitem:
            # Assign this getitem to the same subgraph as its input
            input_node = node.args[0]
            if input_node.op != "placeholder":
                assert input_node in node_to_subgraph_id
                node_to_subgraph_id[node] = node_to_subgraph_id[input_node]
                continue

        if should_split(node, splitting_ops):
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)

            # keep consecutive splitting ops together
            # (we know node.next exists because node isn't the last (output) node)
            if should_split(node.next, splitting_ops):
                # this will get incremented by the next node
                subgraph_id -= 1
            else:
                subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # SUBTRACTED: _merge_empty_only_subgraphs 调用行（L593——delete[4]）。

    # `keep_original_order` is important!
    # otherwise pytorch might reorder the nodes and
    # the semantics of the graph will change when we
    # have mutations in the graph
    # SOURCE: vllm/compilation/backends.py:L595-L608
    with _use_lazy_graph_module(True):
        has_tuple_return = is_torch_equal_or_newer("2.12.0.dev")
        tuple_return_kwarg = {"tuple_return": True} if has_tuple_return else {}
        split_gm = torch.fx.passes.split_module.split_module(
            graph,
            None,
            lambda node: node_to_subgraph_id[node],
            keep_original_order=True,
            **tuple_return_kwarg,
        )

    # SOURCE: vllm/compilation/backends.py:L610-L627
    outputs = []

    names = [name for (name, module) in split_gm.named_modules()]

    for name in names:
        if "." in name or name == "":
            # recursive child module or the root module
            continue

        module = getattr(split_gm, name)

        graph_id = int(name.replace("submod_", ""))
        outputs.append(SplitItem(name, graph_id, (graph_id in split_op_graphs), module))

    # sort by integer graph_id, rather than string name
    outputs.sort(key=lambda x: x.graph_id)

    return split_gm, outputs


compilation_start_time = 0.0


# SOURCE: vllm/compilation/backends.py:L633-L684 wrap_with_cudagraph_if_needed
#   —— Dynamo 切片路线下每片包 CUDAGraphWrapper(PIECEWISE)（首片 debug/
#   非首片禁 GC/末片 weak_ref）；use_inductor_graph_partition 时整段跳过
def wrap_with_cudagraph_if_needed(  # SOURCE: vllm/compilation/backends.py:L633-L684
    piecewise_backend: Any,
    vllm_config: VllmConfig,
    compilation_config: CompilationConfig,
    is_first_graph: bool,
    is_last_graph: bool,
) -> Any:
    """
    Wrap a piecewise backend with CUDA graph wrapper if needed.
    This function is shared between VllmBackend and
    construct_serializable_fn_from_inductor_cache.

    Args:
        piecewise_backend: The backend to wrap
        vllm_config: The vLLM configuration
        compilation_config: The compilation configuration
        is_first_graph: Whether this is the first graph in the sequence
        is_last_graph: Whether this is the last graph in the sequence

    Returns:
        The wrapped backend if CUDA graphs are enabled, otherwise the original backend
    """
    if (
        not compilation_config.cudagraph_mode.has_piecewise_cudagraphs()
        or compilation_config.use_inductor_graph_partition
    ):
        return piecewise_backend

    # We're using Dynamo-based piecewise splitting, so we wrap
    # the whole subgraph with a static graph wrapper.
    from .cuda_graph import CUDAGraphOptions

    # resolve the static graph wrapper class (e.g. CUDAGraphWrapper
    # class) as platform dependent.
    static_graph_wrapper_class = resolve_obj_by_qualname(
        current_platform.get_static_graph_wrapper_cls()
    )

    # Always assign PIECEWISE runtime mode to the
    # CUDAGraphWrapper for piecewise_backend, to distinguish
    # it from the FULL cudagraph runtime mode, no matter it
    # is wrapped on a full or piecewise fx graph.
    return static_graph_wrapper_class(
        runnable=piecewise_backend,
        vllm_config=vllm_config,
        runtime_mode=CUDAGraphMode.PIECEWISE,
        cudagraph_options=CUDAGraphOptions(
            debug_log_enable=is_first_graph,
            gc_disable=not is_first_graph,
            weak_ref_output=is_last_graph,
        ),
    )


# SOURCE: vllm/compilation/backends.py:L687-L776 PiecewiseCompileInterpreter ——
#   片间拼跑：fx.Interpreter 按序过 split_gm，命中编译名单的子图建
#   PiecewiseBackend 并包 cudagraph wrapper，其余（切点算子）eager 直调
class PiecewiseCompileInterpreter(torch.fx.Interpreter):  # type: ignore[misc]
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given split graph interpreter, and for each submodule in
    `compile_submod_names`, creates a PiecewiseBackend and compiles all
    ranges up front.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        vllm_config: VllmConfig,
        vllm_backend: "VllmBackend",
    ) -> None:  # SOURCE: vllm/compilation/backends.py:L711-L724
        super().__init__(module)
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend
        # When True, it annoyingly dumps the torch.fx.Graph on errors.
        self.extra_traceback = False

    # SUBTRACTED: @instrument 装饰的 run 包装（L726-L728——tracing 观测域；
    #   fx.Interpreter.run 直达）。

    # SOURCE: vllm/compilation/backends.py:L730-L776 call_module —— 返回
    #   example_value 不执行子图；命中编译名单则建 PiecewiseBackend 挂回
    #   self.module.__dict__[target]（接缝 eager）
    def call_module(  # SOURCE: vllm/compilation/backends.py:L730-L776
        self,
        target: torch.fx.node.Target,
        args: tuple[torch.fx.node.Argument, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        assert isinstance(target, str)

        gm = getattr(self.module, target)
        outputs = gm.graph.output_node().args[0]
        output = fx.map_arg(outputs, lambda node: node.meta["example_value"])

        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)

            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]

            # Lazy import here to avoid circular import
            from torch._inductor.compile_fx import graph_returns_tuple

            from .piecewise_backend import PiecewiseBackend

            piecewise_backend = PiecewiseBackend(
                submod,
                self.vllm_config,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                self.vllm_backend,
                graph_returns_tuple(submod),
                submod_name=target,
            )

            self.module.__dict__[target] = wrap_with_cudagraph_if_needed(
                piecewise_backend,
                self.vllm_config,
                self.compilation_config,
                piecewise_backend.is_first_graph,
                piecewise_backend.is_last_graph,
            )

            compilation_counter.num_piecewise_capturable_graphs_seen += 1

        return output


# the tag for the part of model being compiled,
# e.g. backbone/eagle_head
# SOURCE: vllm/compilation/backends.py:L779-L782 model_tag/model_is_encoder
model_tag: str = "backbone"
model_is_encoder: bool = False


# SOURCE: vllm/compilation/backends.py:L785-L802 set_model_tag
@contextmanager
def set_model_tag(tag: str, is_encoder: bool = False) -> Generator[None, None, None]:  # SOURCE: vllm/compilation/backends.py:L785-L802
    """Context manager to set the model tag."""
    global model_tag
    global model_is_encoder
    assert tag != model_tag, (
        f"Model tag {tag} is the same as the current tag {model_tag}."
    )
    old_tag = model_tag
    old_is_encoder = model_is_encoder

    model_tag = tag
    model_is_encoder = is_encoder
    try:
        yield
    finally:
        model_tag = old_tag
        model_is_encoder = old_is_encoder


# SOURCE: vllm/compilation/backends.py:L805-L1339 VllmBackend —— torch.compile
#   的 vLLM 定制后端：Dynamo trace 完成后切图、逐片编译、拼成可调用体
class VllmBackend:
    """The compilation backend for `torch.compile` with vLLM.
    It is used for compilation mode of `CompilationMode.VLLM_COMPILE`,
    where we customize the compilation.

    The major work of this backend is to split the graph into
    piecewise graphs, and pass them to the piecewise backend.

    This backend also adds the PostGradPassManager to Inductor config,
    which handles the post-grad passes.
    """

    vllm_config: VllmConfig
    compilation_config: CompilationConfig
    _called: bool = False
    # the graph we compiled
    graph: fx.GraphModule
    # the stiching graph module for all the piecewise graphs
    split_gm: fx.GraphModule
    piecewise_graphs: list[SplitItem]
    returned_callable: Callable[..., Any]
    compiler_manager: CompilerManager
    # Copy of CompilationConfig.inductor_compile_config +
    # an entry for PostGradPassManager
    inductor_config: dict[str, Any]

    # SOURCE: vllm/compilation/backends.py:L833-L868 __init__（pass_manager/
    #   pass_key 两行随 delete[3] pass 装配删）
    def __init__(  # SOURCE: vllm/compilation/backends.py:L833-L868
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_encoder: bool = False,
    ) -> None:
        # if the model is initialized with a non-empty prefix,
        # then usually it's enough to use that prefix,
        # e.g. language_model, vision_model, etc.
        # when multiple parts are initialized as independent
        # models, we need to use the model_tag to distinguish
        # them, e.g. backbone (default), eagle_head, etc.
        self.prefix = prefix or model_tag

        # Mark compilation for encoder.
        self.is_encoder = is_encoder or model_is_encoder

        # SUBTRACTED: pass_manager 解析与 pass_key（L851-L854——delete[3]
        #   configure_post_pass 的 pass 装配域）。

        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config

        self.compiler_manager: CompilerManager = CompilerManager(
            self.compilation_config
        )

        # Deepcopy the inductor config to detach the post-grad custom pass
        # from CompilationConfig.
        # We want to avoid PostGradPassManager in CompilationConfig because
        # in future we need PostGradPassManager.uuid() to be executed
        # only at compile time.
        self.inductor_config = deepcopy(self.compilation_config.inductor_compile_config)
        # `torch.compile` is JIT compiled, so we don't need to
        # do anything here

    # SUBTRACTED: collect_standalone_compile_artifacts（L872-L932——delete[3]
    #   standalone 编译产物收集）。

    # SUBTRACTED: configure_post_pass（L934-L971——delete[3] pass 装配细节）。

    # SUBTRACTED: _log_compilation_config（L973-L1017——tlparse 观测）。

    # SOURCE: vllm/compilation/backends.py:L1019-L1339 __call__ —— Dynamo trace
    #   完成后的切图与逐片编译主链（缓存/计时/instrument/序列化巨块随
    #   delete[3] 删；@dynamo_timed 装饰随观测域删）
    def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any]) -> Any:
        # SUBTRACTED: VllmSerializableFunction 导入（L1021-L1023——caching
        #   序列化域）与 _log_compilation_config/env hash/cache_dir 巨块
        #   （L1027-L1101——delete[3]）。

        vllm_config = self.vllm_config

        # when dynamo calls the backend, it means the bytecode
        # transform and analysis are done
        compilation_counter.num_graphs_seen += 1
        # SUBTRACTED: Dynamo 计时与 instrument（L1150-L1164——观测域，
        #   delete[3]）。

        # we control the compilation process, each instance can only be
        # called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        # SUBTRACTED: configure_post_pass() 调用行（L1171——delete[3]）。

        if self.compilation_config.use_inductor_graph_partition:
            # Let Inductor decide partitioning; avoid FX-level pre-splitting.
            fx_split_ops: list[str] = []
        else:
            fx_split_ops = self.compilation_config.splitting_ops or []

        self.split_gm, self.piecewise_graphs = split_graph(graph, fx_split_ops)

        # SUBTRACTED: split_gm 副本与 depyf/trace_structured 钩子（L1181-
        #   L1199——delete[3] 缓存/观测）。

        compilation_counter.num_piecewise_graphs_seen += len(self.piecewise_graphs)
        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # Extract fake values from the graph to use them when needed.
        # SOURCE: vllm/compilation/backends.py:L1209-L1216
        all_fake_values = []
        for i in graph.graph.find_nodes(op="placeholder"):
            all_fake_values.append(i.meta["example_value"])

        fake_args = [
            all_fake_values[i] if isinstance(t, torch.Tensor) else t
            for i, t in enumerate(example_inputs)
        ]

        # propagate the split graph to the piecewise backend,
        # compile submodules with symbolic shapes, and compile all ranges
        # up front so that compilation is complete before the callable
        # is returned.
        # SOURCE: vllm/compilation/backends.py:L1222-L1224
        PiecewiseCompileInterpreter(
            self.split_gm, submod_names_to_compile, self.vllm_config, self
        ).run(*fake_args)

        # SUBTRACTED: 编译缓存落盘/值域改写/computation_graph 落盘/生成式
        #   runtime_callable 与 VllmSerializableFunction/cudagraph_copy_inputs
        #   尾段（L1226-L1339——delete[3] 序列化域）。返回 split_gm 本体：
        #   生成式 execution code 逐个调用 getattr(split_gm, submod) 得到的
        #   同一批 callables、同序（getattr 先查 __dict__ 的 PiecewiseBackend/
        #   wrapper，再落回 _modules 的原 FX 子图），拼跑语义等价。
        self._called = True
        return self.split_gm

