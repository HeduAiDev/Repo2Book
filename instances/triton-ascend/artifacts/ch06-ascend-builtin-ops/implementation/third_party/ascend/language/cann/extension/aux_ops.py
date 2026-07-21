# SOURCE: third_party/ascend/language/cann/extension/aux_ops.py:L114-133
#
# SUBTRACTED: 真实 aux_ops.py 还有 sync_block_all/set/wait（核间同步，归 ch08）、
# parallel（bind_sub_block 的 range 子类，与 mem_ops/vec_ops 无关）、compile_hint
# （公开 builtin 入口，内部就是调 compile_hint_impl——本章只讲 cast/sort 怎么"挂
# 提示"这件事，公开入口本身不是 mem_ops/vec_ops 的必要依赖，按 dossier 批准的
# "未被引用符号"一并删除）、multibuffer。本章只保留 cast()/sort() 都要用的
# compile_hint_impl 这一个函数。

import triton.language.core as core


def compile_hint_impl(ptr, hint_name: str, hint_val, builder):  # SOURCE: third_party/ascend/language/cann/extension/aux_ops.py:L114-133(compile_hint_impl)
    # Check isinstance(hint_val, bool) first to handle False explicitly
    if isinstance(hint_val, bool):
        hint_val = builder.get_bool_attr(hint_val)
    elif not hint_val:
        hint_val = builder.get_unit_attr()
    elif isinstance(hint_val, int):
        hint_val = builder.get_int32_attr(hint_val)
    elif isinstance(hint_val, core.constexpr):
        hint_val = builder.get_str_attr(hint_val.value)
    elif isinstance(hint_val, list):
        # only support i64 array attr for now
        hint_val = builder.get_i64_array_attr(hint_val)
    else:
        raise ValueError(f"Unsupported hint value type: {type(hint_val)}")
    builder.create_annotation_mark(ptr.handle, hint_name, hint_val)
