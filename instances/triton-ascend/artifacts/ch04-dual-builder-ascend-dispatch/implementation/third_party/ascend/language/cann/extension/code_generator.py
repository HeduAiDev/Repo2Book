# fork 增量：Ascend 的 'with' 语句处理器 —— mangle_ty(类型名 mangling 的 override 钩子)
# 与 handle_scope_with(with al.scope(...) 的落地 handler)。
#
# SOURCE: third_party/ascend/language/cann/extension/code_generator.py
__all__ = ["handle_scope_with", "mangle_ty"]
import ast


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L29-60
def mangle_ty(ty):
    """
    Replacement implementation for triton.compiler.code_generator.mangle_ty.

    This is registered via ASCEND_WITH_DISPATCH["mangle_ty"] and picked up by
    triton.compiler.code_generator through its global WITH_DISPATCH table.
    """
    # SUBTRACTED: 真实实现最前面还有一支 `isinstance(ty, bl.buffer_type)` 分支，把
    # buffer 语言（另一套，经 setup_unified_builder_with_buffer_builder 挂接，本章
    # 已不涉及）的类型编码成 "B<elt>S<shape>S"。本章只演示『override 钩子怎么接上』，
    # 不覆盖 buffer 类型这条支线，故略去该分支，其余 ptr/int/float/block/void 分支
    # 与基座 python/triton/compiler/code_generator.py 的 mangle_ty 完全一致地保留。
    # 原：third_party/.../code_generator.py:L40-44。
    from triton.language.core import dtype

    if ty.is_ptr():
        return "P" + mangle_ty(ty.element_ty)
    if ty.is_int():
        SIGNED = dtype.SIGNEDNESS.SIGNED
        prefix = "i" if ty.int_signedness == SIGNED else "u"
        return prefix + str(ty.int_bitwidth)
    if ty.is_floating():
        return str(ty)
    if ty.is_block():
        elt = mangle_ty(ty.scalar)
        shape = "_".join(map(str, ty.shape))
        return f"{elt}S{shape}S"
    if ty.is_void():
        return "V"
    raise TypeError(f"Unsupported type {ty}")


# SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L137-208
# SUBTRACTED（subtraction_plan.delete 批准）：真实 handle_scope_with 的属性提取/MLIR 属性
# 构造/循环携带变量校验/IR 值重建分别经由 6 个私有助手完成——
#   _extract_scope_attributes（遍历 AST keyword 节点收集 scope 关键字）
#   _py_value_to_mlir_attr / _handle_core_mode_attr / _build_mlir_attrs_from_scope_attrs
#     （把 core_mode/noinline/disable_auto_sync/透传属性逐类转换成 MLIR attribute）
#   _verify_loop_carried_variable（校验 with 块内改写的变量类型与外部一致）
#   _reconstruct_value_from_ir（把 scope_op 的 IR 结果重建回 tl.tensor）
# 这些是 scope.scope op 的属性系统与 SSA 线程化细节，归后续 scope 专章的深水区，删除
# 不影响本章要讲的落地关系：「with al.scope(...) 经 WITH_DISPATCH 命中这个 handler，
# 最终调用挂在主 builder 上的 create_scope_op」。下面保留同一控制流骨架（哑 block 试跑
# 收集 scope_defs → 建 scope_op → 建 entry_block 重新 emit body → scope_return），
# 只把六个助手内联为其中最简单的等价写法（仅示范 core_mode 这一条属性路径）。
def handle_scope_with(generator, node):
    """
    Handle 'with scope(...)' statements by creating a scope.scope operation.

    Uses SSA threading to properly handle variables modified in the scope.
    """
    # SOURCE: third_party/ascend/language/cann/extension/code_generator.py:L137-208
    # Lazy imports to avoid circular dependency（与真实实现一致：code_generator.py
    # 模块加载时要拉 WITH_DISPATCH，而 WITH_DISPATCH 又要在模块加载时合入本文件的
    # handler，若在模块顶层 import triton.compiler.code_generator 会成环，故延后到
    # handler 真正被调用时才导入）。
    from triton.compiler.code_generator import enter_sub_region

    context_expr = node.items[0].context_expr
    scope_attrs = {
        kw.arg: kw.value.value
        for kw in context_expr.keywords
        if isinstance(kw.value, ast.Constant)
    }

    with enter_sub_region(generator) as sr:
        liveins, _ = sr
        ip, last_loc = generator._get_insertion_point_and_loc()

        # This implementation is similar to visit_while: 先用哑 block 试跑一遍 body，
        # 只为收集 scope 内被赋值的变量名/类型（不真正落地这段 IR）。
        dummy = generator.builder.create_block()
        generator.builder.set_insertion_point_to_start(dummy)
        generator.visit_compound_statement(node.body)
        scope_defs = generator.local_defs
        dummy.erase()

        names = list(scope_defs)
        ret_types = [scope_defs[name].type for name in names]

        # 只演示 core_mode 这一条属性路径（其余属性类型转换见上方 SUBTRACTED 说明）。
        mlir_attrs = {"noinline": generator.builder.get_unit_attr()}
        core_mode = scope_attrs.get("core_mode")
        if core_mode in ("cube", "vector"):
            mlir_attrs[generator.builder.get_t_core_type_attr_name()] = (
                generator.builder.get_t_core_type_cube_attr() if core_mode == "cube" else
                generator.builder.get_t_core_type_vector_attr())

        # Create scope operation with operands (values from outside)
        generator._set_insertion_point_and_loc(ip, last_loc)
        scope_op = generator.builder.create_scope_op(
            mlir_attrs, [ty.to_ir(generator.builder) for ty in ret_types])

        # Create the entry block with arguments matching the operands
        entry_block = generator.builder.create_block_with_parent(scope_op.get_region(0), [])
        generator.builder.set_insertion_point_to_start(entry_block)

        # Initialize the scope's symbol table with liveins, then really emit the body.
        generator.lscope = liveins.copy()
        generator.visit_compound_statement(node.body)
        generator.builder.set_insertion_point_to_end(entry_block)

        reconstructed_values = [generator.lscope[name].handle for name in names]
        generator.builder.scope_return(reconstructed_values)

    # After exiting enter_sub_region, update symbol table with results.
    from triton import language
    for i, name in enumerate(names):
        generator.set_value(name, language.core.tensor(scope_op.get_result(i), ret_types[i]))
    return None
