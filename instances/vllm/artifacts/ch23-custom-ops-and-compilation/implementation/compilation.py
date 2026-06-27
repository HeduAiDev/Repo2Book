"""ch23 第 2 级 dispatch：@support_torch_compile + piecewise 切图（只做减法）。

对应 vllm/compilation/decorators.py、vllm/compilation/backends.py、
vllm/compilation/partition_rules.py。

第 2 级 dispatch 在「整图」粒度工作：@support_torch_compile 把一个 nn.Module 包成可编译
（从 forward 注解推断动态维、注入 wrapper 基类、改写 __init__/__call__）。首次前向触发
VllmBackend：split_graph 按 splitting_ops 在 attention 等不可融合算子处切图，
PiecewiseCompileInterpreter 把非切分子图逐段送编译并按需包 CUDA graph，attention 子图保持 eager。
"""

from __future__ import annotations

import dataclasses
import inspect
import operator
from typing import Any

import torch
import torch.fx as fx

from ._runtime import CompilationMode, get_cached_compilation_config


# ===========================================================================
# partition_rules.py — 切点判定
# ===========================================================================
# SOURCE: vllm/compilation/partition_rules.py:L14 (should_split)
def should_split(node: torch.fx.Node, splitting_ops: list[str]) -> bool:
    """
    Check if a node should be split for dynamo graph partition.
    It operates on dynamo graph, so the node.target can be anything.
    We need to check and split only on OpOverload and OpOverloadPacket.
    """
    if node.op != "call_function":
        return False

    target = node.target

    if isinstance(target, torch._ops.OpOverloadPacket):
        # Example: "aten::add"
        return target._qualified_op_name in splitting_ops

    if isinstance(target, torch._ops.OpOverload):
        # Example: "aten::add"
        packet_name = target.name()
        # Example: "aten::add.default"
        op_overload_name = f"{packet_name}.{target._overloadname}"
        return op_overload_name in splitting_ops or packet_name in splitting_ops

    return False


# ===========================================================================
# backends.py — split_graph / PiecewiseCompileInterpreter / VllmBackend
# ===========================================================================
@dataclasses.dataclass
# SOURCE: vllm/compilation/backends.py:L406 (SplitItem)
class SplitItem:
    submod_name: str
    graph_id: int
    is_splitting_graph: bool
    graph: fx.GraphModule


# SOURCE: vllm/compilation/backends.py:L548 (split_graph)
def split_graph(
    graph: fx.GraphModule, splitting_ops: list[str]
) -> tuple[fx.GraphModule, list[SplitItem]]:
    # SUBTRACTED: _decompose_size_nodes(graph)（backends.py:L478-L544 的 FX 正确性预处理）
    # 与 _merge_empty_only_subgraphs（L436-L475）：不改变「在 splitting_ops 处切图」的主流程。

    # split graph by ops
    subgraph_id = 0
    node_to_subgraph_id: dict[fx.Node, int] = {}
    split_op_graphs: list[int] = []
    for node in graph.graph.nodes:
        if node.op in ("output", "placeholder"):
            continue

        # Check if this is a getitem operation on a node from an earlier subgraph.
        # Assign it to the same subgraph as its input.
        if node.op == "call_function" and node.target == operator.getitem:
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
            if should_split(node.next, splitting_ops):
                # this will get incremented by the next node
                subgraph_id -= 1
            else:
                subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id

    # `keep_original_order` is important! otherwise pytorch might reorder the
    # nodes and the semantics of the graph will change when we have mutations.
    # SUBTRACTED: _use_lazy_graph_module 上下文与 torch 2.12 的 tuple_return 兼容分支
    # (backends.py:L593-L595)：版本兼容细节，删之不改切分结果。
    split_gm = torch.fx.passes.split_module.split_module(
        graph,
        None,
        lambda node: node_to_subgraph_id[node],
        keep_original_order=True,
    )

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


# SOURCE: vllm/compilation/backends.py:L628 (wrap_with_cudagraph_if_needed)
def wrap_with_cudagraph_if_needed(
    piecewise_backend: Any,
    vllm_config: Any,
    compilation_config: Any,
    is_first_graph: bool,
    is_last_graph: bool,
) -> Any:
    """
    Wrap a piecewise backend with CUDA graph wrapper if needed.
    """
    # SUBTRACTED: 真实判定 cudagraph_mode.has_piecewise_cudagraphs() /
    # use_inductor_graph_partition，命中则用平台 CUDAGraphWrapper 包住 piecewise_backend
    # （backends.py:L649-L676）。host 无 CUDA graph，故记录「是否会被包 CUDA graph」的决定后
    # 直接返回原 backend——控制流（哪些子图会被包）不变，仅省去真实 CUDA graph 捕获。
    if not getattr(compilation_config, "cudagraph_enabled", True):
        return piecewise_backend
    # 标注本子图「会被包成 PIECEWISE CUDA graph」（真实由 CUDAGraphWrapper 承担）。
    piecewise_backend.wrapped_with_cudagraph = True
    return piecewise_backend


# SOURCE: vllm/compilation/backends.py:L682 (PiecewiseCompileInterpreter)
class PiecewiseCompileInterpreter(torch.fx.Interpreter):
    """It runs the given split graph interpreter, and for each submodule in
    `compile_submod_names`, creates a PiecewiseBackend and compiles all
    ranges up front."""

    # SOURCE: vllm/compilation/backends.py:L706 (__init__)
    def __init__(
        self,
        module: torch.fx.GraphModule,
        compile_submod_names: list[str],
        vllm_config: Any,
        vllm_backend: "VllmBackend",
    ) -> None:
        super().__init__(module)
        self.compile_submod_names = compile_submod_names
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.vllm_backend = vllm_backend

    # SOURCE: vllm/compilation/backends.py:L725 (call_module)
    def call_module(self, target, args, kwargs):
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
            # SUBTRACTED: 真实 PiecewiseBackend 持 Inductor 编译器、对每个 cudagraph 形状
            # 范围 up-front 编译（backends.py/piecewise_backend.py）。host 无 Inductor，故用一个
            # 记录元数据并直接转发到 submod 的轻量 PiecewiseBackend 替身，保留「非切分子图才建
            # backend + 按需包 CUDA graph」这一控制流（切分算子子图不建 backend、保持 eager）。
            piecewise_backend = PiecewiseBackend(
                submod,
                self.vllm_config,
                index,
                len(self.compile_submod_names),
                sym_shape_indices,
                self.vllm_backend,
                submod_name=target,
            )
            self.module.__dict__[target] = wrap_with_cudagraph_if_needed(
                piecewise_backend,
                self.vllm_config,
                self.compilation_config,
                piecewise_backend.is_first_graph,
                piecewise_backend.is_last_graph,
            )

        return output


# SOURCE: vllm/compilation/piecewise_backend.py (PiecewiseBackend — 轻量替身)
class PiecewiseBackend:
    # SUBTRACTED: 真实 PiecewiseBackend 持有 Inductor 编译产物、按 cudagraph 形状范围分桶
    # 编译并在运行期按形状选 runnable（vllm/compilation/piecewise_backend.py）。本章只需它
    # 标记「这是一个被编译/可 CUDA-graph 的子图」并能转发执行，故保留接口与 is_first/last_graph。
    # SOURCE: vllm/compilation/piecewise_backend.py:PiecewiseBackend.__init__
    def __init__(
        self,
        submod,
        vllm_config,
        index: int,
        total: int,
        sym_shape_indices,
        vllm_backend,
        submod_name: str,
    ) -> None:
        self.submod = submod
        self.index = index
        self.is_first_graph = index == 0
        self.is_last_graph = index == total - 1
        self.submod_name = submod_name
        self.wrapped_with_cudagraph = False

    # SOURCE: vllm/compilation/piecewise_backend.py:PiecewiseBackend.__call__
    def __call__(self, *args):
        return self.submod(*args)


# SOURCE: vllm/compilation/backends.py (VllmBackend — piecewise 编译后端主体)
class VllmBackend:
    # SOURCE: vllm/compilation/backends.py (__init__ — 精简)
    def __init__(self, vllm_config: Any) -> None:
        # SUBTRACTED: 真实 __init__ 建 CompilerManager（缓存/哈希）、post_grad pass、
        # 各类 inductor 配置。本章只需持 vllm_config + compilation_config 与单次调用守卫。
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self._called = False
        self.split_gm: fx.GraphModule | None = None
        self.piecewise_graphs: list[SplitItem] = []

    # SOURCE: vllm/compilation/backends.py:L1015 (__call__ — 切图→逐段编译)
    def __call__(self, graph: fx.GraphModule, example_inputs) -> Any:
        # SUBTRACTED: 真实 __call__ 开头是一大段缓存/哈希子系统（env/config/compiler/code
        # hash、cache_dir 计算、cache_key_factors.json 写出，backends.py:L997-L1122）与
        # configure_post_pass。它们不改变「切图→逐段编译→包 CUDA graph」主流程，删去。

        # we control the compilation process, each instance can only be called once
        assert not self._called, "VllmBackend can only be called once"

        self.graph = graph
        # SUBTRACTED: use_inductor_graph_partition 时 fx_split_ops=[]（让 Inductor 自己切）；
        # 本章走默认 FX 预切分主路径（backends.py:L1146-L1150）。
        fx_split_ops = self.compilation_config.splitting_ops or []

        self.split_gm, self.piecewise_graphs = split_graph(graph, fx_split_ops)

        # 非切分子图（is_splitting_graph=False）才送编译；切分算子(attention)子图保持 eager。
        submod_names_to_compile = [
            item.submod_name
            for item in self.piecewise_graphs
            if not item.is_splitting_graph
        ]

        # SUBTRACTED: 真实从 graph 的 placeholder meta 提取 fake_args 再 run（backends.py:
        # L1181-L1197）。本章直接以 example_inputs 跑解释器，达成「逐段建 PiecewiseBackend +
        # 包 CUDA graph」的相同效果，并返回可执行的 split_gm。
        PiecewiseCompileInterpreter(
            self.split_gm, submod_names_to_compile, self.vllm_config, self
        ).run(*example_inputs)

        self._called = True
        return self.split_gm


# ===========================================================================
# decorators.py — @support_torch_compile
# ===========================================================================
# SOURCE: vllm/compilation/decorators.py (TorchCompileWithNoGuardsWrapper — 轻量替身)
class TorchCompileWithNoGuardsWrapper:
    # SUBTRACTED: 真实 wrapper（vllm/compilation/wrapper.py）持 torch.compile 包装的 forward、
    # 维护 compiled code object、在首调时把 VllmBackend 作为 backend 触发 Dynamo。本章保留它在
    # 装饰器里的角色：被注入 __bases__、其 __init__ 在编译开启时被调、其 __call__ 在首/后续调用
    # 时分别触发首编/缓存。首编通过 torch.compile(self.forward, backend=VllmBackend(...)) 复现。
    # SOURCE: vllm/compilation/wrapper.py:L47 (TorchCompileWithNoGuardsWrapper.__init__)
    def __init__(self, compile_prefix: str = "", is_encoder: bool = False) -> None:
        self._compile_prefix = compile_prefix
        self._is_encoder = is_encoder
        self._compiled_callable = None

    # SOURCE: vllm/compilation/wrapper.py:L47 (TorchCompileWithNoGuardsWrapper.__call__)
    def __call__(self, *args, **kwargs):
        if self._compiled_callable is None:
            backend = VllmBackend(self.vllm_config)
            self._compiled_callable = torch.compile(
                self.forward, backend=backend, fullgraph=True
            )
        return self._compiled_callable(*args, **kwargs)


IGNORE_COMPILE_KEY = "_ignore_torch_compile"


# SOURCE: vllm/compilation/decorators.py:L118 (support_torch_compile — 入口装饰器)
def support_torch_compile(
    cls: type | None = None,
    *,
    dynamic_arg_dims: dict[str, int | list[int]] | None = None,
    enable_if=None,
):
    """A decorator to add support for compiling the forward method of a class."""

    # SOURCE: vllm/compilation/decorators.py:L201 (cls_decorator_helper — 推断动态维)
    def cls_decorator_helper(cls: type) -> type:
        if not hasattr(cls, "forward"):
            raise TypeError("decorated class should have a forward method.")
        sig = inspect.signature(cls.forward)
        inferred_dynamic_arg_dims = dynamic_arg_dims
        if inferred_dynamic_arg_dims is None:
            inferred_dynamic_arg_dims = {}
            # 从 forward 的 torch.Tensor 注解推断哪一维（默认 dim 0=token/batch）是动态的。
            for k, v in sig.parameters.items():
                if v.annotation in [
                    torch.Tensor,
                    torch.Tensor | None,
                ]:
                    inferred_dynamic_arg_dims[k] = 0
            # SUBTRACTED: FloatTensor / IntermediateTensors 注解分支（decorators.py:L214-L217）
            # 是多模态/中间张量的同类推断，本章以 torch.Tensor 为代表。

        if len(inferred_dynamic_arg_dims) == 0:
            raise ValueError(
                "No dynamic dimensions found in the forward method of "
                f"{cls}. Please provide dynamic_arg_dims explicitly."
            )
        # SUBTRACTED: 逐 key 校验是否在签名里（decorators.py:L233-L237）与 logger.debug。

        return _support_torch_compile(cls, inferred_dynamic_arg_dims, enable_if)

    if cls is not None:
        # use `support_torch_compile` as a decorator without arguments
        assert isinstance(cls, type)
        return cls_decorator_helper(cls)
    return cls_decorator_helper


def _should_ignore_torch_compile(cls: type) -> bool:
    # SOURCE: vllm/compilation/decorators.py:_should_ignore_torch_compile
    return getattr(cls, IGNORE_COMPILE_KEY, False)


# SOURCE: vllm/compilation/decorators.py:L331 (_support_torch_compile — 注入 wrapper + 改写 __init__/__call__)
def _support_torch_compile(
    cls: type,
    dynamic_arg_dims: dict[str, int | list[int]],
    enable_if=None,
) -> type:
    """Internal implementation of support_torch_compile decorator."""

    if TorchCompileWithNoGuardsWrapper in cls.__bases__:
        # support decorating multiple times
        return cls

    # take care of method resolution order: make sure super().__init__ is called
    # on the base class other than TorchCompileWithNoGuardsWrapper.
    cls.__bases__ = cls.__bases__ + (TorchCompileWithNoGuardsWrapper,)

    old_init = cls.__init__
    setattr(cls, IGNORE_COMPILE_KEY, False)

    # SOURCE: vllm/compilation/decorators.py:L353 (改写 __init__ — 定 do_not_compile)
    def __init__(self, *args, vllm_config=None, prefix: str = "", **kwargs):
        if vllm_config is None:
            # SUBTRACTED: 真实取 get_current_vllm_config()（decorators.py:L360-L361）。
            # 本章要求显式传 vllm_config（测试构造一个轻量 VllmConfig 替身）。
            raise ValueError("vllm_config is required in the companion build")
        # SUBTRACTED: 位置参数类型校验循环（decorators.py:L365-L383）：多版本兼容兜底，删之
        # 不改 do_not_compile 决策。
        old_init(self, *args, **kwargs)

        self.vllm_config = vllm_config
        self.compilation_config = self.vllm_config.compilation_config
        enable_compile = enable_if is None or enable_if(vllm_config)
        # for CompilationMode.STOCK_TORCH_COMPILE, the upper level model runner
        # will handle the compilation, so we don't need to do anything here.
        self.do_not_compile = (
            self.compilation_config.mode
            in [CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE]
            or _should_ignore_torch_compile(self.__class__)
            or not enable_compile
        )
        if self.do_not_compile:
            return

        self._dynamic_arg_dims = dynamic_arg_dims
        self.compiled = False
        # SUBTRACTED: was_aot_compile_fn_loaded_from_disk / compilation_counter 计数
        # （decorators.py:L401-L403）：AOT/统计旁路，删之不损主线。
        TorchCompileWithNoGuardsWrapper.__init__(self, compile_prefix="", is_encoder=False)

    cls.__init__ = __init__

    # SOURCE: vllm/compilation/decorators.py:L414 (_mark_dynamic_inputs — 标 token 维动态)
    def _mark_dynamic_inputs(mod, *args, **kwargs):
        sig = inspect.signature(mod.__class__.forward)
        bound_args = sig.bind(mod, *args, **kwargs)
        bound_args.apply_defaults()
        # SUBTRACTED: UNBACKED/shape_id 多分支与 torch 版本兼容（decorators.py:L417-L500）：
        # 细粒度动态形状策略。主路径即对每个动态维调 torch._dynamo.mark_dynamic，避免按 batch
        # size 重编译。
        for k, dims in dynamic_arg_dims.items():
            arg = bound_args.arguments.get(k)
            if arg is None or not isinstance(arg, torch.Tensor):
                continue
            dims_list = [dims] if isinstance(dims, int) else dims
            for d in dims_list:
                real_d = arg.ndim + d if d < 0 else d
                torch._dynamo.mark_dynamic(arg, real_d)

    # SOURCE: vllm/compilation/decorators.py:L502 (改写 __call__ — 首编触发 VllmBackend / 后续缓存)
    def __call__(self, *args, **kwargs):
        # torch.compiler.is_compiling() means we are inside the compilation
        if self.do_not_compile or torch.compiler.is_compiling():
            return self.forward(*args, **kwargs)

        # SUBTRACTED: forward_context.skip_compiled 旁路（decorators.py:L512-L513）与整段
        # AOT(VLLM_USE_AOT_COMPILE)缓存路径（L515-L575,L652-L670）：AOT 默认关闭，删之不损 JIT 主线。

        if self.compiled:
            return TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)

        # This is the path for the first compilation.
        # the first compilation needs to have dynamic shapes marked
        _mark_dynamic_inputs(self, *args, **kwargs)
        # SUBTRACTED: traced_files 收集 / patched_inline_call / dynamo+inductor config patch
        # （decorators.py:L592-L651）：缓存失效追踪与编译期配置，删之不改「首次走 wrapper.__call__
        # 触发 VllmBackend」这一主控制流。
        output = TorchCompileWithNoGuardsWrapper.__call__(self, *args, **kwargs)
        self.compiled = True
        return output

    cls.__call__ = __call__
    return cls
