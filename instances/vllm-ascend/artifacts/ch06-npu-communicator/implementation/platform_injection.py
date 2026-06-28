"""换底座注入点①：platform 用一个字符串 qualname 回调把通信器类换成 NPUCommunicator。

只做减法的忠实精简版：把基座默认值 / NPU 覆写 / 进程组解析三处「最小骨架」并排，
其余类体一律 # SUBTRACTED。这三段都是纯字符串 / 解析逻辑，host 可读可跑（除被
SUBTRACTED 的 vllm 模块级符号外）。
"""
from __future__ import annotations

# SUBTRACTED: from vllm.utils import resolve_obj_by_qualname
# SUBTRACTED: from vllm.platforms import current_platform
#   —— 基座模块级符号；精简版不在 host import vllm，仅在 _resolve_device_communicator
#      的忠实摘录里按原样引用（该函数不在 host 调用，故不触发 NameError）。
#   原 vllm/distributed/parallel_state.py 顶部 import。


class _BasePlatform:
    # SOURCE: vllm/platforms/interface.py:L769-L774
    # SUBTRACTED: Platform 基类其余 ~700 行（设备能力 / 内存 / dtype 探测等），
    #   本章只看「换通信器」这一个 classmethod —— 原 vllm/platforms/interface.py:Lxxx。
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # SOURCE: vllm/platforms/interface.py:L769-L774
        """
        Get device specific communicator class for distributed communication.
        """
        return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"  # noqa


class NPUPlatform(_BasePlatform):
    # SOURCE: vllm_ascend/platform.py:L803-L805
    # SUBTRACTED: NPUPlatform 是 Platform 的 OOT 子类，ch01-ch02 已讲注册机制；
    #   这里只看「换通信器」这一个 classmethod 覆写 —— 原 vllm_ascend/platform.py:Lxxx。
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # SOURCE: vllm_ascend/platform.py:L803-L805
        return "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator"


# SOURCE: vllm/distributed/parallel_state.py:L370-L381
def _resolve_device_communicator(self):
    """GroupCoordinator.__init__ 里「字符串 qualname → 类 → 实例化」的忠实摘录。

    # SUBTRACTED: GroupCoordinator.__init__ 的其余初始化（rank/world_size/进程组建立、
    #   self.device 按 is_out_of_tree() 设为 npu:local_rank 等），原 parallel_state.py:L361-L369。
    # 注意 resolve_obj_by_qualname / current_platform 是 vllm 模块级符号（见顶部 SUBTRACTED），
    #   本函数仅作可读摘录，不在 host 调用。
    """
    self.use_device_communicator = use_device_communicator  # noqa: F821
    self.device_communicator = None
    if use_device_communicator and self.world_size > 1:  # noqa: F821
        device_comm_cls = resolve_obj_by_qualname(  # noqa: F821
            current_platform.get_device_communicator_cls()  # noqa: F821
        )
        self.device_communicator = device_comm_cls(
            cpu_group=self.cpu_group,
            device=self.device,
            device_group=self.device_group,
            unique_name=self.unique_name,
        )
