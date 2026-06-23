"""ch23 测试 — 验证精简版复现真实 vLLM 的可观察行为（不 import vllm，host 纯 torch）。

测的是 dossier 记录的真实 vLLM 行为，不是精简版自洽：
  第 1 级 dispatch（CustomOp）:
    - 构造期一次性 dispatch：__init__ 后 _forward_method 已定死
    - enabled()/default_on()：custom_ops 含 'all'→默认开；'none'→默认关；+name/-name 覆盖
    - enabled & CUDA → forward_cuda；disabled → forward_native
    - 平台分支：rocm→hip、cpu/tpu/xpu/oot→native
    - RMSNorm 两路（native/cuda）数值一致
  第 2 级 dispatch（@support_torch_compile + 切图）:
    - 动态维从 forward 注解推断（torch.Tensor → dim 0）
    - do_not_compile 据 CompilationMode（NONE/STOCK→不编译；VLLM_COMPILE→编译）
    - wrapper 类被注入 __bases__
    - splitting_ops 默认取 _attention_ops（在 attention 处切）
    - split_graph 在 unified_attention_with_output 处切：attention 段 is_splitting_graph=True
      且不进编译；规整段 is_splitting_graph=False 进编译+包 CUDA graph
    - should_split：命中 splitting_ops 的 OpOverload 才切
  f17 回收:
    - unified_attention_with_output 注册为 torch.ops.vllm.* 不透明算子，可调用、原位写 output
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

IMPL = Path(__file__).resolve().parent.parent / "implementation"


def _load(modname):
    if str(IMPL.parent) not in sys.path:
        sys.path.insert(0, str(IMPL.parent))
    spec = importlib.util.spec_from_file_location(
        f"ch23_impl.{modname}", IMPL / f"{modname}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # 让相对 import (._runtime) 可解析：把包注册进 sys.modules
    pkgname = "implementation"
    if pkgname not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            pkgname, IMPL / "__init__.py", submodule_search_locations=[str(IMPL)]
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        sys.modules[pkgname] = pkg
        pkg_spec.loader.exec_module(pkg)
    full = f"{pkgname}.{modname}"
    spec = importlib.util.spec_from_file_location(full, IMPL / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


runtime = _load("_runtime")
custom_op = _load("custom_op")
attention_op = _load("attention_op")
compilation = _load("compilation")

CompilationConfig = runtime.CompilationConfig
CompilationMode = runtime.CompilationMode


@pytest.fixture(autouse=True)
def _reset_config():
    # 默认：custom_ops=['all']（非 Inductor 路径默认开），CUDA 平台。
    runtime.set_cached_compilation_config(
        CompilationConfig(custom_ops=["all"], mode=CompilationMode.NONE)
    )
    runtime.current_platform.kind = "cuda"
    yield


def _set_cfg(custom_ops=None, mode=CompilationMode.NONE):
    if custom_ops is None:
        custom_ops = ["all"]
    cfg = CompilationConfig(custom_ops=custom_ops, mode=mode)
    runtime.set_cached_compilation_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# 第 1 级 dispatch: default_on / enabled
# ---------------------------------------------------------------------------
def test_default_on_all_vs_none():
    _set_cfg(custom_ops=["all"])
    assert custom_op.CustomOp.default_on() is True
    _set_cfg(custom_ops=["none"])
    assert custom_op.CustomOp.default_on() is False


def test_default_on_requires_exactly_one_of_all_none():
    _set_cfg(custom_ops=[])  # 既无 all 也无 none → assert 失败
    with pytest.raises(AssertionError):
        custom_op.CustomOp.default_on()


def test_enabled_plus_minus_override():
    # Inductor 默认 'none' → 关；用 +rms_norm 单独打开
    _set_cfg(custom_ops=["none", "+rms_norm"])
    assert custom_op.RMSNorm.enabled() is True
    # 'all' 默认开；用 -rms_norm 单独关
    _set_cfg(custom_ops=["all", "-rms_norm"])
    assert custom_op.RMSNorm.enabled() is False


def test_enabled_cannot_both():
    _set_cfg(custom_ops=["all", "+rms_norm", "-rms_norm"])
    with pytest.raises(AssertionError):
        custom_op.RMSNorm.enabled()


def test_register_assigns_name_and_registry():
    assert custom_op.RMSNorm.name == "rms_norm"
    assert custom_op.op_registry["rms_norm"] is custom_op.RMSNorm


# ---------------------------------------------------------------------------
# 第 1 级 dispatch: 构造期一次性选 forward_method
# ---------------------------------------------------------------------------
def test_dispatch_at_construction_cuda_enabled_picks_forward_cuda():
    _set_cfg(custom_ops=["all"])  # 默认开
    runtime.current_platform.kind = "cuda"
    op = custom_op.RMSNorm(8)
    assert op._forward_method.__func__ is custom_op.RMSNorm.forward_cuda


def test_dispatch_disabled_picks_forward_native():
    _set_cfg(custom_ops=["none"], mode=CompilationMode.NONE)  # Inductor 默认关
    op = custom_op.RMSNorm(8)
    # enabled=False 且 compile_native 默认 False → maybe_compile 直接返回 forward_native
    assert op._forward_method.__func__ is custom_op.RMSNorm.forward_native


def test_dispatch_platform_branches():
    _set_cfg(custom_ops=["all"])
    runtime.current_platform.kind = "rocm"
    assert custom_op.RMSNorm(8)._forward_method.__func__ is custom_op.RMSNorm.forward_hip
    for kind, meth in [
        ("cpu", custom_op.RMSNorm.forward_cpu),
        ("tpu", custom_op.RMSNorm.forward_tpu),
        ("xpu", custom_op.RMSNorm.forward_xpu),
        ("oot", custom_op.RMSNorm.forward_oot),
    ]:
        runtime.current_platform.kind = kind
        assert custom_op.RMSNorm(8)._forward_method.__func__ is meth


def test_forward_dispatches_through_stored_method():
    _set_cfg(custom_ops=["none"])  # → forward_native
    op = custom_op.RMSNorm(8)
    x = torch.randn(3, 8)
    out = op(x)
    expected = op.forward_native(x)
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# 第 1 级 dispatch: RMSNorm 两路数值一致
# ---------------------------------------------------------------------------
def test_rmsnorm_native_vs_cuda_no_residual():
    op = custom_op.RMSNorm(16)
    with torch.no_grad():
        op.weight.copy_(torch.randn(16) * 0.1 + 1.0)
    x = torch.randn(5, 16)
    n = op.forward_native(x.clone())
    c = op.forward_cuda(x.clone())
    assert torch.allclose(n, c, atol=1e-5)


def test_rmsnorm_native_vs_cuda_with_residual():
    op = custom_op.RMSNorm(16)
    with torch.no_grad():
        op.weight.copy_(torch.randn(16) * 0.1 + 1.0)
    x = torch.randn(5, 16)
    r = torch.randn(5, 16)
    n_out, n_res = op.forward_native(x.clone(), r.clone())
    c_out, c_res = op.forward_cuda(x.clone(), r.clone())
    assert torch.allclose(n_out, c_out, atol=1e-5)
    assert torch.allclose(n_res, c_res, atol=1e-5)


def test_rmsnorm_matches_reference_formula():
    op = custom_op.RMSNorm(8, eps=1e-6)
    x = torch.randn(4, 8)
    out = op.forward_native(x)
    xf = x.to(torch.float32)
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(out, ref.to(x.dtype), atol=1e-5)


# ---------------------------------------------------------------------------
# f17 回收: attention 注册为不透明 torch custom op
# ---------------------------------------------------------------------------
def test_attention_registered_as_torch_op():
    assert hasattr(torch.ops.vllm, "unified_attention_with_output")


def test_attention_op_callable_writes_output_inplace():
    q = torch.randn(4, 8)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)
    out = torch.zeros(4, 8)
    ret = torch.ops.vllm.unified_attention_with_output(q, k, v, out, "layer.0")
    assert ret is None  # 原位写、返回 None
    assert torch.allclose(out, q)


# ---------------------------------------------------------------------------
# 第 2 级 dispatch: 动态维推断 / do_not_compile / wrapper 注入
# ---------------------------------------------------------------------------
def _make_vllm_config(mode):
    class _Cfg:
        def __init__(self):
            self.compilation_config = CompilationConfig(mode=mode)
            self.compilation_config.cudagraph_enabled = True

    return _Cfg()


def _decorate_simple_model():
    @compilation.support_torch_compile
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    return Model


def test_dynamic_arg_dims_inferred_from_annotation():
    Model = _decorate_simple_model()
    cfg = _make_vllm_config(CompilationMode.VLLM_COMPILE)
    m = Model(vllm_config=cfg)
    assert m._dynamic_arg_dims == {"x": 0}


def test_wrapper_injected_into_bases():
    Model = _decorate_simple_model()
    assert compilation.TorchCompileWithNoGuardsWrapper in Model.__bases__


def test_do_not_compile_for_none_and_stock():
    Model = _decorate_simple_model()
    for mode in (CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE):
        m = Model(vllm_config=_make_vllm_config(mode))
        assert m.do_not_compile is True


def test_compile_enabled_for_vllm_compile():
    Model = _decorate_simple_model()
    m = Model(vllm_config=_make_vllm_config(CompilationMode.VLLM_COMPILE))
    assert m.do_not_compile is False


def test_do_not_compile_call_falls_through_to_forward():
    Model = _decorate_simple_model()
    m = Model(vllm_config=_make_vllm_config(CompilationMode.NONE))
    x = torch.randn(3, 8)
    out = m(x)
    assert torch.allclose(out, m.forward(x))


def test_no_dynamic_dims_raises():
    with pytest.raises(ValueError):

        @compilation.support_torch_compile
        class Bad(torch.nn.Module):
            def forward(self, n):  # 无 torch.Tensor 注解
                return n


# ---------------------------------------------------------------------------
# 第 2 级 dispatch: splitting_ops 默认 / should_split
# ---------------------------------------------------------------------------
def test_set_splitting_ops_defaults_to_attention_ops():
    cfg = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    cfg.set_splitting_ops_for_v1()
    assert "vllm::unified_attention_with_output" in cfg.splitting_ops


def test_set_splitting_ops_empty_when_not_vllm_compile():
    cfg = CompilationConfig(mode=CompilationMode.NONE)
    cfg.set_splitting_ops_for_v1()
    assert cfg.splitting_ops == []


def test_should_split_matches_registered_attention_op():
    # 构造一个含 attention op 调用的 fx 图，验证 should_split 在该节点返回 True
    splitting_ops = ["vllm::unified_attention_with_output"]

    def fn(x):
        out = torch.empty_like(x)
        torch.ops.vllm.unified_attention_with_output(x, x, x, out, "l0")
        return out + 1.0

    gm = torch.fx.symbolic_trace(fn)
    hits = [
        n for n in gm.graph.nodes if compilation.should_split(n, splitting_ops)
    ]
    assert len(hits) == 1
    assert "unified_attention_with_output" in str(hits[0].target)


def test_should_split_false_for_non_matching():
    def fn(x):
        return x + 1.0

    gm = torch.fx.symbolic_trace(fn)
    assert not any(
        compilation.should_split(n, ["vllm::unified_attention_with_output"])
        for n in gm.graph.nodes
    )


# ---------------------------------------------------------------------------
# 第 2 级 dispatch: split_graph 在 attention 处切, attention 段不编译
# ---------------------------------------------------------------------------
def _build_two_attention_graph():
    """norm → attn0 → linear → attn1 → norm 的 fx 图（两层 attention）。"""

    def fn(x):
        x = x * 2.0  # 规整段 0（norm 替身）
        out0 = torch.empty_like(x)
        torch.ops.vllm.unified_attention_with_output(x, x, x, out0, "l0")
        y = out0 + 1.0  # 规整段 1（linear 替身）
        out1 = torch.empty_like(y)
        torch.ops.vllm.unified_attention_with_output(y, y, y, out1, "l1")
        return out1 * 3.0  # 规整段 2

    return torch.fx.symbolic_trace(fn)


def test_split_graph_splits_at_attention():
    gm = _build_two_attention_graph()
    split_gm, items = compilation.split_graph(
        gm, ["vllm::unified_attention_with_output"]
    )
    splitting = [it for it in items if it.is_splitting_graph]
    regular = [it for it in items if not it.is_splitting_graph]
    # 两层 attention → 两个切分子图；规整段在其间
    assert len(splitting) == 2
    assert len(regular) >= 2


def test_split_graph_preserves_semantics():
    gm = _build_two_attention_graph()
    split_gm, _ = compilation.split_graph(
        gm, ["vllm::unified_attention_with_output"]
    )
    x = torch.randn(4, 8)
    assert torch.allclose(gm(x.clone()), split_gm(x.clone()))


def test_no_split_when_splitting_ops_empty():
    gm = _build_two_attention_graph()
    _, items = compilation.split_graph(gm, [])
    assert all(not it.is_splitting_graph for it in items)


# ---------------------------------------------------------------------------
# 第 2 级 dispatch: VllmBackend 只对非切分子图建 PiecewiseBackend + 包 CUDA graph
#
# VllmBackend.call_module 读 node.meta["example_value"]——这是 Dynamo 在 trace 时塞进
# 每个节点的元数据。因此这里走真实编译入口：torch.compile(fn, backend=VllmBackend(...))，
# 与真实「Dynamo trace → 把整图交给 VllmBackend」完全一致，而非用 symbolic_trace 伪造图。
# ---------------------------------------------------------------------------
def _two_attention_fn(x):
    x = x * 2.0  # 规整段 0（norm 替身）
    out0 = torch.empty_like(x)
    torch.ops.vllm.unified_attention_with_output(x, x, x, out0, "l0")
    y = out0 + 1.0  # 规整段 1（linear 替身）
    out1 = torch.empty_like(y)
    torch.ops.vllm.unified_attention_with_output(y, y, y, out1, "l1")
    return out1 * 3.0  # 规整段 2


def _compile_through_backend(splitting_ops):
    cfg = _make_vllm_config(CompilationMode.VLLM_COMPILE)
    cfg.compilation_config.splitting_ops = splitting_ops
    backend = compilation.VllmBackend(cfg)
    torch._dynamo.reset()
    compiled = torch.compile(_two_attention_fn, backend=backend, fullgraph=True)
    out = compiled(torch.randn(4, 8))
    return backend, out


def test_vllm_backend_compiles_only_regular_subgraphs():
    backend, out = _compile_through_backend(["vllm::unified_attention_with_output"])
    assert out.shape == (4, 8)

    n_piecewise = 0
    n_attn_eager = 0
    for item in backend.piecewise_graphs:
        attr = backend.split_gm.__dict__.get(item.submod_name, None)
        if item.is_splitting_graph:
            # attention 子图保持 eager：不建 PiecewiseBackend
            n_attn_eager += 1
            assert not isinstance(attr, compilation.PiecewiseBackend)
        else:
            # 规整段被建成 PiecewiseBackend 且标记会被包 CUDA graph
            assert isinstance(attr, compilation.PiecewiseBackend)
            assert attr.wrapped_with_cudagraph is True
            n_piecewise += 1
    # 两层 attention → 两个 eager 切分子图，其间规整段进编译
    assert n_attn_eager == 2
    assert n_piecewise >= 2


def test_vllm_backend_called_once():
    cfg = _make_vllm_config(CompilationMode.VLLM_COMPILE)
    cfg.compilation_config.splitting_ops = ["vllm::unified_attention_with_output"]
    backend = compilation.VllmBackend(cfg)
    gm = _build_two_attention_graph()
    # 第一次手动喂入（带 example_value 缺失会失败，但这里只验证「单次调用」守卫：
    # 先标记已调用，再次调用必触发 assert）。
    backend._called = True
    x = torch.randn(4, 8)
    with pytest.raises(AssertionError):
        backend(gm, (x,))
