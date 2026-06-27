# 环境桥接（NOT a vLLM abstraction）—— 让精简版能在 host(CPU/gloo) 上运行。
#
# 真实 vLLM 从 vllm.utils.torch_utils 引入 direct_register_custom_op、从
# vllm.platforms 引入 current_platform、从 vllm.utils 引入 resolve_obj_by_qualname。
# host 上没有这些 vLLM 内部模块，本文件用语义等价的最小实现替换它们——
# 不杜撰任何 vLLM 没有的行为，只是把 vLLM 自己的工具在无 vLLM 环境下补齐。
from __future__ import annotations

import contextlib
import importlib
import os
from typing import Callable

import torch
from torch.library import Library, infer_schema


# 对应 vllm.utils.torch_utils.vllm_lib —— 注册自定义算子的目标 Library。
vllm_lib = Library("vllm", "FRAGMENT")


# direct_register_custom_op：按 CPU 主线裁剪 dispatch_key（原版从
# current_platform.dispatch_key 取，这里固定 CPU）。
# SOURCE: vllm/utils/torch_utils.py:L931
def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: "list[str] | None" = None,
    fake_impl: "Callable | None" = None,
    target_lib: "Library | None" = None,
    dispatch_key: "str | None" = None,
    tags: tuple = (),
):
    """
    Directly registers a custom op and dispatches it to the given backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    """
    if mutates_args is None:
        mutates_args = []
    if dispatch_key is None:
        # SUBTRACTED: 原版 `dispatch_key = current_platform.dispatch_key`
        # (vllm/utils/torch_utils.py:L958-L961) 在 CUDA 上取 "CUDA"；host 无 CUDA，
        # 固定 "CompositeExplicitAutograd" 使算子在 CPU 上可派发，语义等价。
        dispatch_key = "CompositeExplicitAutograd"
    schema_str = infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


# 对应 vllm.utils.resolve_obj_by_qualname —— 按全限定名解析类/对象。
# SOURCE: vllm/utils/import_utils.py:L104
def resolve_obj_by_qualname(qualname: str):
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# 对应 vllm.utils.suppress_stdout（GroupCoordinator 建 gloo 群组时压制 stdout）。
@contextlib.contextmanager
def suppress_stdout():
    # SOURCE: vllm/utils/system_utils.py:L62（host 上无需真正重定向，留空上下文）
    yield


# SOURCE: vllm/platforms/interface.py:L187（Platform：本章用到的判定方法子集）
class _Platform:
    """对应 vllm.platforms.current_platform 在本章用到的判定方法。
    host 无 CUDA：is_cuda_alike()=False → GroupCoordinator 走 CPU 设备分支；
    use_custom_op_collectives() 可被测试 monkeypatch 来切换两条派发路径。
    """

    # SOURCE: vllm/platforms/interface.py:L187
    def is_cuda_alike(self) -> bool:
        # 真实 vLLM 由平台插件确定（多 GPU 时 self.device=cuda:local_rank）。
        # 本章是 host-CPU companion：默认走 GroupCoordinator 的 CPU 设备分支
        # （device_group/cpu_group 都用 gloo），从而无需多块 GPU 即可在单机上
        # 跑通多 rank 集合原语与 P2P。设 VLLM_CH20_CUDA=1 可显式切回 cuda 分支
        # （需真实多 GPU，应在 vllm/vllm-openai 容器内）。
        if os.environ.get("VLLM_CH20_CUDA") != "1":
            return False
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception:
            return False

    # SOURCE: vllm/platforms/interface.py:L166
    def is_tpu(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py:L935
    def use_custom_op_collectives(self) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py:L770
    def get_device_communicator_cls(self) -> str:
        # 真实 CUDA 平台返回 CudaCommunicator 的全限定名；本章默认实现用
        # DeviceCommunicatorBase（torch.distributed/gloo 后端），语义一致。
        return "base_device_communicator.DeviceCommunicatorBase"


current_platform = _Platform()
