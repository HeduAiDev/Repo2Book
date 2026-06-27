"""ch23 f17 回收：attention 被注册成不透明 torch 自定义算子（只做减法）。

对应 vllm/model_executor/layers/attention/attention.py（unified_attention_with_output +
注册）与 vllm/utils/torch_utils.py(direct_register_custom_op)。

ch22 埋下伏笔 f17：Attention.forward 吞掉的 self.attn 算子如何进 torch.compile 图。本章
闭环：attention 用 direct_register_custom_op 注册为带 fake_impl 的 torch.ops.vllm.* 不透明
算子。编译路径下 Dynamo trace 到它既不内联也不 graph break——它正好成为 split_graph 的默认
切点（splitting_ops=_attention_ops 含 vllm::unified_attention_with_output）。
"""

from __future__ import annotations

import torch

# 用一个专属 Library 承载 vllm:: 命名空间下的自定义算子（对应真实 vllm_lib）。
# SUBTRACTED: 真实 vllm_lib 在 vllm/utils/torch_utils.py 顶层创建；本章用同名空间 "vllm"
# 的 Library 复现 torch.ops.vllm.unified_attention_with_output 的注册与可调用性。
_vllm_lib = torch.library.Library("vllm", "FRAGMENT")


# SOURCE: vllm/utils/torch_utils.py:L931 (direct_register_custom_op)
def direct_register_custom_op(
    op_name: str,
    op_func,
    mutates_args: list[str] | None = None,
    fake_impl=None,
    target_lib: torch.library.Library | None = None,
    dispatch_key: str = "CPU",
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    """
    if mutates_args is None:
        mutates_args = []
    # SUBTRACTED: 真实 dispatch_key 默认取 current_platform.dispatch_key（CUDA），
    # schema 由 torch._library.infer_schema 从 op_func 注解+mutates_args 推断
    # (torch_utils.py:L926-L931)。host 无 CUDA，dispatch_key 用 "CPU"，并显式给出与真实算子
    # 等价的 schema 字符串（避免依赖 vLLM 的 infer_schema 工具）。注册流程 define+impl+
    # _register_fake 与真实一一对应。
    schema_str = (
        "(Tensor query, Tensor key, Tensor value, Tensor(a!) output, "
        "str layer_name, Tensor? kv_cache_dummy_dep=None) -> ()"
    )
    my_lib = target_lib or _vllm_lib
    my_lib.define(op_name + schema_str)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


# SOURCE: vllm/model_executor/layers/attention/attention.py:L706 (unified_attention_with_output)
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    # SUBTRACTED: 真实实现从 layer_name 经 get_attention_context 取回 attn_metadata/self/
    # kv_cache，再 self.impl.forward(...) 跑真实 attention kernel（attention.py:L719-L733）。
    # attention forward 已在前面章节讲过；本章重点是「它被注册成不透明 op + fake_impl」。精简版
    # 用一个占位计算（把 query 写进 output）复现「原位写 output、返回 None」的可观察契约，
    # 让算子可在 host 真实调用并出现在 fx 图里成为切点。
    output.copy_(query.reshape_as(output))


# SOURCE: vllm/model_executor/layers/attention/attention.py:L736 (unified_attention_with_output_fake)
def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # fake_impl 让 Dynamo 在 trace 时不需真执行就知道输出（这里原位写、无返回），
    # 从而把它当成一个稳定的、不可融合的图节点（既不 graph break 又天然成为 split 切点）。
    return


# SOURCE: vllm/model_executor/layers/attention/attention.py:L749 (注册)
direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output"],
    fake_impl=unified_attention_with_output_fake,
)
