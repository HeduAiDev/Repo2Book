# 综合样本 —— vllm_ascend/patch/worker/patch_distributed.py（subtract-only）
#
# 一处文件同时演示三招：
#   技法① 整类替换         : GroupCoordinator = GroupCoordinatorPatch
#   技法④ 库函数 wrapper    : _wrap_destroy_distributed_environment（@wraps + 幂等标记）
#   技法⑤ from-import 缓存陷阱: 同一 destroy_fn 绑到 parallel_state 与再导出别名两个名字
#
# SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L16-L30
from __future__ import annotations

import logging
from functools import wraps
from typing import Any, cast

import torch
import vllm
from torch.distributed import Backend
from vllm.distributed.parallel_state import GroupCoordinator, _get_unique_name, _register_group

from vllm_ascend.patch.worker._hccl_pg_registry import HcclPgRegistry

# SUBTRACTED: NPUCommunicator / make_hccl_pg_key / create_hccl_pg_options import，及
#   _normalize_backend / _resolve_reuse_domain / _create_device_group / _acquire_hccl_group
#   等 HCCL helper —— 属被替换类的 HCCL 业务体，与三招无关 (patch_distributed.py:L28-L76)。

# SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L30-L31
_HCCL_PG_REGISTRY = HcclPgRegistry()
logger = logging.getLogger(__name__)


# 技法④：库函数 wrapper —— 闭包捕获原 destroy_fn，@wraps 保留元信息，
#         自定义幂等标记 _hccl_registry_clearing_wrapped 防止多入口触发导致重复包裹。
def _wrap_destroy_distributed_environment(destroy_fn):
    # SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L79-L91
    if getattr(cast(Any, destroy_fn), "_hccl_registry_clearing_wrapped", False) is True:
        return destroy_fn

    @wraps(destroy_fn)
    def wrapped(*args, **kwargs):
        # SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L83-L88
        try:
            return destroy_fn(*args, **kwargs)
        finally:
            _HCCL_PG_REGISTRY.clear()

    cast(Any, wrapped)._hccl_registry_clearing_wrapped = True
    return wrapped


# 技法⑤：from-import 缓存陷阱 —— 同一 destroy_fn 同时绑到顶层与再导出别名两个名字。
def _patch_destroy_distributed_environment():
    # SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L94-L97
    destroy_fn = _wrap_destroy_distributed_environment(vllm.distributed.parallel_state.destroy_distributed_environment)
    vllm.distributed.parallel_state.destroy_distributed_environment = destroy_fn
    vllm.distributed.destroy_distributed_environment = destroy_fn


# 技法①：整类替换 —— 子类整体重写构造，并新增 all_to_all 方法（原 GroupCoordinator 没有）。
class GroupCoordinatorPatch(GroupCoordinator):
    # SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L100-L229
    def __init__(
        self,
        group_ranks,
        local_rank: int,
        torch_distributed_backend,
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name=None,
    ):
        # SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L101-L179
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)
        # SUBTRACTED: HCCL pg 复用注册表 / NPUCommunicator / gloo cpu_group 的完整构造逻辑
        #   （约 100 行）—— 属被替换类的业务实现，与「整类替换」招式无关
        #   (patch_distributed.py:L114-L179)。

    def destroy(self):
        # SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L181-L210
        # SUBTRACTED: cpu_group / _acquired_hccl_keys / _unshared_hccl_groups / device_group /
        #   device_communicator 的逐项释放细节 (patch_distributed.py:L182-L210)。
        ...

    # all_to_all 是替换的核心动机：vLLM 原 GroupCoordinator 不支持 all_to_all。
    def all_to_all(
        self,
        input_,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes=None,
        gather_sizes=None,
    ):
        # SOURCE: vllm_ascend/patch/worker/patch_distributed.py:L212-L229
        if self.world_size == 1:
            return input_
        # SUBTRACTED: scatter_dim/gather_dim 合法性 assert (patch_distributed.py:L221-L227)。
        assert self.device_communicator is not None, "device_communicator should be initialized when world_size > 1"
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)


# 招式核心：整类替换 + 同名再导出双绑「同时」生效。
vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
_patch_destroy_distributed_environment()
