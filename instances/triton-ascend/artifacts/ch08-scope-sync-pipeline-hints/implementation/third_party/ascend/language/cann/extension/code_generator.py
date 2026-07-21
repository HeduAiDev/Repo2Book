# SOURCE: third_party/ascend/language/cann/extension/code_generator.py
"""
Ascend-specific code generation handlers for 'with' statement context managers.
"""

# SUBTRACTED(subtraction_plan.delete 批准): mangle_ty(原 L29-60)——类型名字修饰的
# override 钩子，与 with/scope 语义无关，归 ch04 题材；删掉后本文件只留
# handle_scope_with 及其六个私有助手。
__all__ = ["handle_scope_with"]
import ast


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L63-69
def _extract_scope_attributes(context_expr):
    """Extract attributes from scope(...) call."""
    scope_attrs = {}
    for keyword in context_expr.keywords:
        if isinstance(keyword.value, ast.Constant):
            scope_attrs[keyword.arg] = keyword.value.value
    return scope_attrs


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L72-81
def _py_value_to_mlir_attr(builder, value):
    """Convert Python value to MLIR attribute."""
    attr_creators = {
        str: lambda v: builder.get_str_attr(v),
        bool: lambda v: builder.get_bool_attr(v),
        int: lambda v: builder.get_int32_attr(v),
        list: lambda v: builder.get_i64_array_attr(v),
    }
    creator = attr_creators.get(type(value))
    return creator(value) if creator else value


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L84-93
def _handle_core_mode_attr(builder, core_mode):
    """Handle core_mode attribute conversion."""
    if core_mode not in ("cube", "vector"):
        return {}
    return {
        builder.get_t_core_type_attr_name(): (
            builder.get_t_core_type_cube_attr() if core_mode == "cube"
            else builder.get_t_core_type_vector_attr()
        )
    }


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L96-118
def _build_mlir_attrs_from_scope_attrs(builder, scope_attrs):
    """Convert Python scope attributes to MLIR attributes.

    Args:
        builder: The IR builder
        scope_attrs: Dict of scope attributes (e.g., {'core_mode': 'vector', 'noinline': True})

    Returns:
        Dict of MLIR attributes
    """
    mlir_attrs = {"noinline": builder.get_unit_attr()}
    for k, v in scope_attrs.items():
        if k == "core_mode":
            mlir_attrs.update(_handle_core_mode_attr(builder, v))
        elif k == "noinline":
            if not v:
                mlir_attrs.pop("noinline")
        elif k == "disable_auto_sync":
            if v:
                mlir_attrs["hivm.disable_auto_sync"] = _py_value_to_mlir_attr(builder, v)
        else:
            mlir_attrs[k] = _py_value_to_mlir_attr(builder, v)
    return mlir_attrs


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L121-129
def _verify_loop_carried_variable(_is_triton_value, _is_triton_tensor, name, loop_val, live_val):
    """Verify that loop-carried variable types are consistent."""
    assert _is_triton_value(loop_val), f'cannot reassign constxpr {name} in the loop'
    assert _is_triton_value(live_val), f'cannot reasign constexpr {name} in the loop'
    assert type(loop_val) == type(live_val), f'Loop carried variable {name} changed type'
    assert not _is_triton_tensor(loop_val) or loop_val.type == live_val.type, \
        f'Loop-carried variable {name} has initial type {live_val.type} '\
        f'but is re-assigned to {loop_val.type} in loop! '\
        f'Please make sure that the type stays consistent.'


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L132-134
def _reconstruct_value_from_ir(language, entry_block_arg, ret_type):
    """Reconstruct a tensor value from IR."""
    return language.core.tensor(entry_block_arg, ret_type)


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L137-208
def handle_scope_with(generator, node):
    """
    Handle 'with scope(...)' statements by creating a scope.scope operation.

    This creates a scope.scope operation with a region for the scope block.
    Uses SSA threading to properly handle variables modified inside the scope.

    Args:
        generator: The CodeGenerator instance
        node: AST node for the with statement
    """
    # Lazy imports to avoid circular dependency
    from triton import language
    from triton.compiler.code_generator import enter_sub_region, _is_triton_value, _is_triton_tensor

    context_expr = node.items[0].context_expr
    scope_attrs = _extract_scope_attributes(context_expr)

    with enter_sub_region(generator) as sr:
        liveins, _ = sr
        ip, last_loc = generator._get_insertion_point_and_loc()

        # This implementation is similar to visit_while
        dummy = generator.builder.create_block()
        generator.builder.set_insertion_point_to_start(dummy)
        generator.visit_compound_statement(node.body)
        scope_defs = generator.local_defs
        dummy.erase()

        # Verify and get return type of the scope.scope
        # (variables that exist in parent scope AND are modified in scope)
        names = []
        ret_types = []
        for name in scope_defs:
            scope_val = scope_defs[name]
            ret_types.append(scope_val.type)
            names.append(name)
            if name in liveins:
                live_val = liveins[name]
                _verify_loop_carried_variable(
                    _is_triton_value, _is_triton_tensor, name, scope_val, live_val)

        # Convert Python primitives to MLIR attributes
        mlir_attrs = _build_mlir_attrs_from_scope_attrs(generator.builder, scope_attrs)

        # Create scope operation with operands (values from outside)
        generator._set_insertion_point_and_loc(ip, last_loc)
        scope_op = generator.builder.create_scope_op(
            mlir_attrs,
            [ty.to_ir(generator.builder) for ty in ret_types]
        )

        # Create the entry block with arguments matching the operands
        entry_block = generator.builder.create_block_with_parent(scope_op.get_region(0), [])
        generator.builder.set_insertion_point_to_start(entry_block)

        # Initialize the scope's symbol table with liveins
        generator.lscope = liveins.copy()
        generator.visit_compound_statement(node.body)
        generator.builder.set_insertion_point_to_end(entry_block)
        reconstructed_values = []

        for i in range(len(names)):
            # generator.lscope[names[i]] is already a tensor, just get its IR handle
            reconstructed_values.append(generator.lscope[names[i]].handle)
        generator.builder.scope_return(reconstructed_values)

    # After exiting enter_sub_region, update symbol table with results
    # Convert IR values back to tensor objects
    for i, name in enumerate(names):
        generator.set_value(name, _reconstruct_value_from_ir(language, scope_op.get_result(i), ret_types[i]))
    return None
