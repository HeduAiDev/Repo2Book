"""m7 —— register_custom_op 的真实内建样例 _index_select，及自定义算子参数的动态定型
(self.arg_type)。

对照真实源码 third_party/ascend/language/cann/extension/builtin_custom_ops.py:
L74-103：类字段 name/core/pipe/mode 声明 __builtin_index_select 的注册四要素；
__init__ 做 src/index 的指针/整数/rank 断言，并把 end_offset/start_offset/src_stride
三个参数的类型动态改成 index.dtype(self.arg_type[...] = index.dtype)——Python int
默认降为 device 端 int32，这是"运行时按张量类型重定参数类型"的逃生口。
"""
import pytest


def _tensor(env, dtype, is_ptr=False):
    tl = env.tl_core
    # 用独立的 dtype 实例(而非复用模块级单例 tl.float32)，这样"把这块内存标成指针"
    # 只影响这一个测试用张量，不会污染其它张量共用的 dtype 单例。
    dt = tl.dtype(dtype.name)
    if is_ptr:
        dt.is_ptr = lambda: True  # 实例属性遮蔽 dtype.is_ptr 这个 @staticmethod 默认值
        # 真实 triton 的指针类型(pointer_type)有 .element_ty 指向指向的元素类型；
        # 本章精简版的 tl.dtype 未保留 pointer_type(与 register_custom_op/libdevice
        # 三分野无关)，这里只给测试用的"指针"实例补一个够用的 element_ty，使
        # `out.dtype == src.dtype.element_ty` 这条真实断言能被驱动到。
        dt.element_ty = dt
    t = tl.tensor(handle=f"h-{dtype.name}", type=dt)
    return t


def test_index_select_is_registered_at_import_via_decorator(env):
    """@register_custom_op 在 builtin_custom_ops.py import 期就已执行——
    _index_select 是真实注册的类，不是 _get_op_class 的 __builtin_ 前缀哑类兜底。"""
    C = env.ext_core
    reg = env.custom_op._custom_op_registry
    assert "__builtin_index_select" in reg
    Op = reg["__builtin_index_select"]
    assert Op is env.builtin_custom_ops._index_select
    assert Op.core == C.CORE.VECTOR
    assert Op.pipe == C.PIPE.PIPE_V
    assert Op.mode == C.MODE.SIMT


def test_get_op_class_returns_real_registered_class_not_dummy(env):
    """_get_op_class 对已注册名字直接返回真实类(不是 L37-51 的 dummy 兜底)。"""
    Op = env.custom_op._get_op_class("__builtin_index_select")
    assert Op is env.builtin_custom_ops._index_select


def _valid_kwargs(env):
    tl = env.tl_core
    src = _tensor(env, tl.float32, is_ptr=True)
    index = _tensor(env, tl.int32)
    index.shape = [tl.constexpr(4)]  # 1D index -> idx_rank == 1
    out = _tensor(env, tl.float32)
    return dict(
        src=src, index=index, dim=0, bound=8,
        end_offset=(4, 4), start_offset=(0, 0), src_stride=(1, 1), out=out,
    )


def test_index_select_init_accepts_valid_2d_arguments_and_retypes_offsets(env):
    """__init__ 校验通过后，self.arg_type 把 end_offset/start_offset/src_stride
    动态改成 index.dtype(见 dossier design_decisions)——直接构造实例验证这一件事，
    不经完整 custom_semantic 数据流(那条路径见 test_custom_semantic_dataflow.py)。"""
    Op = env.builtin_custom_ops._index_select
    kwargs = _valid_kwargs(env)

    op = Op.__new__(Op)
    op.arg_type = {}
    Op.__init__(op, **kwargs)

    assert op.arg_type["end_offset"] is kwargs["index"].dtype
    assert op.arg_type["start_offset"] is kwargs["index"].dtype
    assert op.arg_type["src_stride"] is kwargs["index"].dtype
    assert op.extra_attr == "src_stride_len=2"


def test_index_select_rejects_src_rank_out_of_2_to_5(env):
    Op = env.builtin_custom_ops._index_select
    kwargs = _valid_kwargs(env)
    kwargs["src_stride"] = (1,)  # rank 1 -> 不在 [2,5]
    kwargs["start_offset"] = (0,)
    with pytest.raises(AssertionError, match=r"src rank should in \[2, 5\]"):
        Op(**kwargs)


def test_index_select_rejects_dim_out_of_range(env):
    Op = env.builtin_custom_ops._index_select
    kwargs = _valid_kwargs(env)
    kwargs["dim"] = 5  # src_rank=2 -> dim 必须在 [0,1]
    with pytest.raises(AssertionError, match="dim should in"):
        Op(**kwargs)


def test_index_select_requires_out(env):
    Op = env.builtin_custom_ops._index_select
    kwargs = _valid_kwargs(env)
    kwargs["out"] = None
    with pytest.raises(AssertionError, match="out is required"):
        Op(**kwargs)


def test_index_select_end_to_end_through_custom_semantic(env):
    """把 _index_select 走完整 custom_semantic 数据流(m2)一遍——验证真实内建样例
    与调用侧入口能对接：查表命中真实类(非 dummy)、走 __builtin_ 免 symbol/bitcode。"""
    builder = env.FakeBuilder()
    kwargs = _valid_kwargs(env)

    result = env.custom_op.custom_semantic(
        "__builtin_index_select", **kwargs, _builder=builder)

    assert result is not None
    create_calls = [c for c in builder.calls if c[0] == "create_custom_op"]
    assert len(create_calls) == 1
    _, name, attrs, inputs, outputs, arg_attrs = create_calls[0]
    assert name == "__builtin_index_select"
    assert "symbol" not in attrs and "bitcode" not in attrs
