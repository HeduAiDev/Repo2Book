# SOURCE: third_party/ascend/language/cann/extension/_utils.py
# SUBTRACTED(subtraction_plan.delete 批准): _is_int_like_elem/_assert_int_like_tuple/
# _convert_elem_to_ir_value(原 L18-54)——服务于 block pointer 参数转换，不在
# sync_block 调用链上。本文件只留 custom_op 这一个手写 dispatcher。随之删除只服务于
# 这些被删助手的 `import triton.language.core as tl`(custom_op 本身从不用 tl)；
# `from triton._C.libtriton import ir` 仍保留，custom_op 的类型注解逐字未改。
from triton._C.libtriton import ir


def custom_op(builder: ir.builder, op_name: str, **kwargs):  # SOURCE: L5-16
    if op_name == "sync_block_all":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["mode"], kwargs["event_id"])

    elif op_name == "sync_block_set":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])

    elif op_name == "sync_block_wait":
        return builder.create_custom_op_for_inter_core_sync(op_name, kwargs["sender"], kwargs["event_id"])

    raise ValueError(f"Unsupported custom op: {op_name}")
