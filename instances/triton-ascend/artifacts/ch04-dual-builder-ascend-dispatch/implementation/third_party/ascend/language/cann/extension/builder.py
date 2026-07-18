# SOURCE: third_party/ascend/language/cann/extension/builder.py（全文件——已经很小，
# 是本章 m4 机制的完整实现，一行未删；只有 setup_unified_builder 里的 ascend_methods
# 清单按 subtraction_plan 精简为代表项，见下方注释）。
"""
Ascend-specific builder utilities for code generation.
"""

__all__ = [
    "create_builder_method_wrapper",
    "attach_builder_methods",
    "setup_unified_builder",
]


# SOURCE: third_party/ascend/language/cann/extension/builder.py:L32-53
def create_builder_method_wrapper(main_builder, delegate_builder, method_name):
    """
    Create a wrapper that delegates a method call to another builder while
    synchronizing insertion points and locations.
    """
    delegate_method = getattr(delegate_builder, method_name)

    def wrapper(*args, **kwargs):
        # SOURCE: third_party/ascend/language/cann/extension/builder.py:L39-49
        saved_ip = main_builder.get_insertion_point()
        saved_loc = main_builder.get_loc()
        delegate_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            delegate_builder.set_loc(saved_loc)
        result = delegate_method(*args, **kwargs)
        main_builder.restore_insertion_point(saved_ip)
        if saved_loc:
            main_builder.set_loc(saved_loc)
        return result

    wrapper.__name__ = method_name
    wrapper.__doc__ = getattr(delegate_method, '__doc__', None)
    return wrapper


# SOURCE: third_party/ascend/language/cann/extension/builder.py:L56-60
def attach_builder_methods(main_builder, delegate_builder, method_names):
    """Attach multiple methods from a delegate builder to the main builder."""
    for method_name in method_names:
        wrapper = create_builder_method_wrapper(main_builder, delegate_builder, method_name)
        setattr(main_builder, method_name, wrapper)


# SOURCE: third_party/ascend/language/cann/extension/builder.py:L63-86
def setup_unified_builder(main_builder, ascend_builder):
    """Set up a unified builder interface by attaching methods from specialized builders."""
    main_builder._ascend_builder = ascend_builder
    # SUBTRACTED: 真实清单还有 get_target_attribute/create_copy_tensor/
    # create_annotation_mark/create_bind_buffer/create_debug_barrier/is_910_95/
    # sync_block_set/sync_block_wait/sync_block_all/create_convert_layout（共 11 项）。
    # 机制是「逐个包 wrapper 后 setattr」，清单只是被挂方法的枚举；保留下面 6 项——
    # 前 5 个是本章 handle_scope_with 真正会调用到的（create_scope_op/scope_return
    # 落地 scope op 本身，get_t_core_type_* 三件套转换 core_mode 属性），第 6 个
    # create_copy_buffer 作为『其余 hivm 原语同理挂上』的代表——即可讲清，不需要
    # 全量枚举。原：third_party/.../builder.py:L66-84。
    ascend_methods = [
        'create_scope_op',
        'scope_return',
        'get_t_core_type_attr_name',
        'get_t_core_type_cube_attr',
        'get_t_core_type_vector_attr',
        'create_copy_buffer',
    ]
    attach_builder_methods(main_builder, ascend_builder, ascend_methods)
