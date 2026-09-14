# SOURCE: vllm/utils/torch_utils.py —— 本章只携带 direct_register_custom_op +
# vllm_lib（集合算子注册面，m4；ch19 同款携带）。其余（stream/LayerName/版本谓词）
# 不携带。

from __future__ import annotations

from collections.abc import Callable

import torch
from torch._library.infer_schema import infer_schema
from torch.library import Library


# create a library to hold the custom op
# SOURCE: vllm/utils/torch_utils.py:L897-L898 vllm_lib — 逐字
vllm_lib = Library("vllm", "FRAGMENT")  # noqa


# SOURCE: vllm/utils/torch_utils.py:L901-L939 direct_register_custom_op — 逐字
#   （dispatch_key 缺省改为经 seam 平台取——真实版 import current_platform 同义）
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of
    the library object. If you want to bind the operator to a different
    library, make sure the library object is alive when the operator
    is used.
    """
    # SOURCE: vllm/utils/torch_utils.py:L901-L939（锚点双置）
    if mutates_args is None:
        mutates_args = []

    if dispatch_key is None:
        from vllm.platforms import current_platform

        dispatch_key = current_platform.dispatch_key

    schema_str = infer_schema(op_func, mutates_args=mutates_args)

    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
