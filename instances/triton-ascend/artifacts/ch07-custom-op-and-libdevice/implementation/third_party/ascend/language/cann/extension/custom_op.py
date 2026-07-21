# SOURCE: third_party/ascend/language/cann/extension/custom_op.py
# 本章主角——昇腾语言层相对基座 triton 多出的『注册自定义算子』能力：register_custom_op
# 把一个 Python 类注册进全局表，custom/custom_semantic 是调用侧入口，查表→实例化
# 校验→拆 operand→_make_attrs 造 IR 属性→create_custom_op emit hivm.CustomOp。

import inspect
import types
import typing
import itertools
import triton.language.core as tl
from . import core


# SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L33-34
# Registry for custom op, mapping name to its configuration.
_custom_op_registry = {}


def _get_op_class(name):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L37-51
    # Try to get op class in _custom_op_registry.
    op_class = _custom_op_registry.get(name)
    if op_class is None:
        # Allow bulitin custom ops used without registry.
        assert name.startswith('__builtin_'), f"Custom Op '{name}' not registered."
        # Return a dummy op class for builtin custom op.
        op_class = type("_builtin_custom_op", (object, ), {
            "name": name,
            "core": core.CORE.VECTOR,
            "pipe": core.PIPE.PIPE_V,
            "mode": core.MODE.SIMT,
            "signature": inspect.signature(object),
        })
    return op_class


def _unwrap_constexpr(arg):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L54-63
    if isinstance(arg, tl.constexpr):
        return arg.value
    if isinstance(arg, tuple):
        return tuple(_unwrap_constexpr(x) for x in arg)
    if isinstance(arg, list):
        return [_unwrap_constexpr(x) for x in arg]
    if isinstance(arg, dict):
        return {k: _unwrap_constexpr(v) for k, v in arg.items()}
    return arg


# SUBTRACTED: 真实 _to_value 按 dtype 逐个精确匹配 int64/uint64/int32/uint32/int16/
# uint16/int8/uint8(int 分支)与 fp64/fp32/fp16/bf16(float 分支)共 12 条同构的
# `if ty.is_xxx(): return builder.get_xxx(value)` 分支(third_party/ascend/language/
# cann/extension/custom_op.py:L78-108)。控制流是同一个"按 dtype 找 builder 工厂方法"
# 范式的平坦重复，这里只保留骨架 + 两条代表分支(int 默认 int32、float 默认 fp32)，
# 其余精确匹配分支删除；不影响 custom_semantic 的主数据流理解(operand 的具体位宽
# 由调用点的 type-hint/self.arg_type 决定，不是本章讲解重点)。
def _to_value(value, builder, ty=None):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L66-111(节选)
    ty = getattr(value, 'type', ty) if ty is None else ty
    if isinstance(value, tl.tensor):
        if not value.type.is_block() and isinstance(ty, tl.dtype) and value.type != ty:
            return tl.semantic.cast(value, ty, builder).handle
        return value.handle
    if isinstance(value, bool):
        return builder.get_int1(value)
    if isinstance(value, int):
        # default int32
        return builder.get_int32(value)
    if isinstance(value, float):
        # default float32
        return builder.get_fp32(value)
    if isinstance(value, tl.constexpr):
        return _to_value(value.value, builder)
    raise TypeError(f"Unsupported argument type {value} : {type(value)}")


def _to_operands(args, builder):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L114-124
    operands = []
    for value in args:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                operands.append(_to_value(item, builder))
        else:
            operands.append(_to_value(value, builder))
    return operands


def _get_element_type(ty):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L127-130
    if isinstance(ty, types.GenericAlias):
        return typing.get_args(ty)[0]
    return ty


def _args_to_operands(op, builder, args, kwargs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L133-152
    if not op.signature.parameters:
        # Without parameters in signature, use the actual parameter order.
        return _to_operands(itertools.chain(args, kwargs.values()), builder)

    # Convert arguments to operands according the signature.
    operands = []
    bind = op.signature.bind(*args, **kwargs)
    for param in op.signature.parameters.values():
        value = bind.arguments.get(param.name)
        if value is None:
            continue
        ty = op.arg_type.get(param.name, param.annotation)
        if isinstance(value, (list, tuple)):
            ty = _get_element_type(ty)
            for item in value:
                operands.append(_to_value(item, builder, ty))
        else:
            operands.append(_to_value(value, builder, ty))
    return operands


def _bind_op_arguments(op, args, kwargs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L155-158
    if not op.signature.parameters:
        return None
    return op.signature.bind(*args, **kwargs)


def _make_align_dim_attrs(op, builder, arg_attrs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L161-185
    # Find op argument by name using op.align_dim's key
    # We want to return a dict mapping for each align_dim key -> int attribute for the actual bound argument value.
    name = 'align_dim'
    if not hasattr(op, name):
        return

    # To find argument indices matching each align_dim key, check the op.signature parameters
    # and map align_dim key (argument name) to its index position.
    align_arg_indices = {}
    if hasattr(op, "signature"):
        param_names = list(op.signature.parameters.keys())
        for arg_name in op.align_dim.keys():
            if arg_name in param_names:
                align_arg_indices[arg_name] = param_names.index(arg_name)

    for arg, align_val in op.align_dim.items():
        if isinstance(arg, str) and arg in align_arg_indices:
            arg_attrs[align_arg_indices[arg]] = {name: builder.get_int_attr(align_val)}
        elif isinstance(arg, int):
            arg_attrs[arg] = {name: builder.get_int_attr(align_val)}
        else:
            assert False, f"{name}'s keys should be string or int"


def _make_arg_attrs(op, builder):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L188-193
    num_args = len(op.signature.parameters) if hasattr(op, "signature") else 0
    arg_attrs = [{} for _ in range(num_args)]

    _make_align_dim_attrs(op, builder, arg_attrs)
    return arg_attrs


def _add_optional_attr(op, name, builder, attrs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L196-198
    if hasattr(op, name):
        attrs[name] = builder.get_str_attr(getattr(op, name))


def _add_bitcode_attr(op, builder, attrs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L201-209
    name = 'bitcode'
    if not hasattr(op, name):
        return

    from pathlib import Path
    bitcode = Path(getattr(op, name))
    assert bitcode.exists(), f"Provided bitcode ({name}) not exist"
    attrs[name] = builder.get_str_attr(str(bitcode.absolute()))


def _add_optional_extra_buffer_attr(op, builder, attrs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L212-223
    name = 'extra_buffers'
    if not hasattr(op, name):
        return

    extra_buffers = getattr(op, name)
    if isinstance(extra_buffers, tuple):
        extra_buffers = [extra_buffers]

    extra_buffer_types, extra_buffer_sizes = zip(*extra_buffers)
    attrs[name + "_types"] = builder.get_type_array_attr([ty.to_ir(builder) for ty in extra_buffer_types])
    attrs[name + "_sizes"] = builder.get_i64_array_attr(list(extra_buffer_sizes))


def _add_optional_indexing_map_attr(op, builder, attrs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L226-234
    # Optional indexing map attribute:
    # `indexing_map` should be an iterable of al.affine_map (MLIR AffineMap) objects.
    # 语义(al.affine_map 具体怎么描述访问/迭代映射)归 P5——本章只讲"这个参数存在、
    # 会被原样挂成 IR 属性"这一件事。
    name = 'indexing_map'
    if not hasattr(op, name):
        return

    indexing_map = getattr(op, name)
    attrs[name] = builder.get_affine_map_array_attr(indexing_map)


def _add_optional_iterator_types_attr(op, builder, attrs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L237-242
    name = 'iterator_types'
    if not hasattr(op, name):
        return

    attrs[name] = builder.get_iterator_types_attr([iterator_type.value for iterator_type in getattr(op, name)])


def _make_attrs(op, builder):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L245-271
    attrs = {
        'hivm.tcore_type': builder.get_core_type_attr(op.core.value),
        'hivm.pipe': builder.get_pipe_attr(op.pipe.value),
        'hivm.vf_mode': builder.get_vf_mode_attr(op.mode.value),
    }

    if not op.name.startswith('__builtin_'):
        assert hasattr(op, 'symbol'), "Non builtin custom op, symbol is required."
        assert hasattr(op, 'bitcode'), "Non builtin custom op, bitcode path is required."

    # Add bit code path attribute, formalize to abosulte path.
    _add_bitcode_attr(op, builder, attrs)

    _add_optional_indexing_map_attr(op, builder, attrs)
    _add_optional_iterator_types_attr(op, builder, attrs)

    _add_optional_extra_buffer_attr(op, builder, attrs)

    _add_optional_attr(op, 'symbol', builder, attrs)
    _add_optional_attr(op, 'source', builder, attrs)
    _add_optional_attr(op, 'compile', builder, attrs)
    # Extra attributes can be added here, such as op.extra_attr="attr_a=xx"
    _add_optional_attr(op, 'extra_attr', builder, attrs)

    return attrs


def _to_result(res, res_types):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L274-281
    assert len(res) == len(res_types)
    n_res = len(res)
    if n_res == 0:
        return None
    if n_res == 1:
        return tl.tensor(res[0], res_types[0])
    return tuple(tl.tensor(res[i], res_types[i]) for i in range(n_res))


def _init_op(op_class, *args, **kwargs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L284-291
    op = op_class.__new__(op_class)
    # Add arg_type dict to support dynamic argument type specifying.
    setattr(op, 'arg_type', {})
    if op_class.signature.parameters:
        # Init with arguments validate.
        op_class.__init__(op, *args, **kwargs)
    return op


def custom_semantic(name, *args, _builder=None, **kwargs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L294-315
    name = _unwrap_constexpr(name)
    # Get op class according the name.
    op_class = _get_op_class(name)
    # Convert constexpr to value in arguments.
    args = _unwrap_constexpr(args)
    kwargs = _unwrap_constexpr(kwargs)
    # Create op instance from op class with the arguments.
    op = _init_op(op_class, *args, **kwargs)
    # Prepare inputs and outputs operands.
    out = kwargs.pop('out', [])
    outs = out if isinstance(out, (list, tuple)) else [out]
    outputs = _to_operands(outs, _builder)
    inputs = _args_to_operands(op, _builder, args, kwargs)
    # Setup attributes.
    attrs = _make_attrs(op, _builder)
    arg_attrs = _make_arg_attrs(op, _builder)
    # Build IR for the custom op.
    res = _builder.create_custom_op(name, attrs, inputs, outputs, arg_attrs)
    # Results with same types as outputs.
    res_types = [out.type for out in outs]
    return _to_result(res, res_types)


@core.builtin
def custom(name, *args, _builder=None, **kwargs):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L318-321
    """Invoke a custom operation with the given name and arguments."""
    return custom_semantic(name, *args, _builder=_builder, **kwargs)


def register_custom_op(op):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L324-345
    """Register a custom operation so that we can invoke it using al.custom()."""
    assert inspect.isclass(op), "@register_custom_op should decorate on a class."
    # Use class name if name not set.
    if not hasattr(op, 'name'):
        setattr(op, 'name', op.__name__)
    # The op name should not be used.
    assert op.name not in _custom_op_registry, f"Custom op name '{op.name}' already used."

    # Check required core, pipe, mode fields.
    assert hasattr(op, 'core'), "'core' field is required."
    assert hasattr(op, 'pipe'), "'pipe' field is required."
    assert hasattr(op, 'mode'), "'mode' field is required."
    assert isinstance(op.core, core.CORE), "Invalid 'core' field, CORE type is required."
    assert isinstance(op.pipe, core.PIPE), "Invalid 'pipe' field, PIPE type is required."
    assert isinstance(op.mode, core.MODE), "Invalid 'mode' field, MODE type is required."
    # Retrieve arguments signature from __init__ method and save it.
    signature = inspect.signature(op)
    setattr(op, 'signature', signature)
    # Register the custom op configuration.
    _custom_op_registry[op.name] = op
    return op


# 原样保留(dossier subtraction_plan 未批准删除)——服务外部 bitcode/C++ 头文件生成时
# 把 dtype 转成 C 类型名，与 register_custom_op/custom_semantic 的注册-调用主数据流
# 无直接关联，但不属于任何已批准的删除项，故不裁剪。
_dtype_cname_dict = {  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L348-366
    'int1': 'bool',
    'int8': 'int8_t',
    'int16': 'int16_t',
    'int32': 'int32_t',
    'int64': 'int64_t',
    'uint8': 'uint8_t',
    'uint16': 'uint16_t',
    'uint32': 'uint32_t',
    'uint64': 'uint64_t',
    'fp16': 'half',
    'bf16': 'bfloat16_t',
    'fp32': 'float',
    'fp64': 'double',
    'fp8e5': 'float8_e5m2_t',
    'fp8e4nv': 'float8_e4m3_t',
    # other float8 types are not supported yet,
    # such as 'fp8e4b8', 'fp8e4b15', 'fp8e5b16'.
}


def _cname(self):  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L369-371
    """Return the corresponding C name of the given tl.dtype"""
    return _dtype_cname_dict.get(self.name, self.name)


# Add 'cname' property to tl.dtype class.
tl.dtype.cname = property(_cname, None)  # SOURCE: third_party/ascend/language/cann/extension/custom_op.py:L374
