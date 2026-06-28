"""验证 PyHcclCommunicator 的 disabled 降级路径（must_keep: disabled / all_reduce）。

这条范式样本对位 pynccl：单卡（world_size==1）或缺库（非 NPU 环境）时安全降级，
all_reduce 返回 None、broadcast no-op。host 可跑（不触达真正的 libhccl.so 调用）。
"""
import types

import pyhccl  # noqa: E402  (conftest 已把 implementation/ 加进 sys.path)


def _stateless_group(rank, world_size):
    # pyhccl.StatelessProcessGroup 是基座类型的忠实占位；构造实例走 __init__ 的
    # stateless else 分支（self.rank=group.rank / self.world_size=group.world_size），
    # 无需初始化 torch.distributed。
    g = pyhccl.StatelessProcessGroup()
    g.rank = rank
    g.world_size = world_size
    return g


def test_world_size_one_disables():
    comm = pyhccl.PyHcclCommunicator(group=_stateless_group(0, 1), device=0)
    assert comm.disabled is True
    assert comm.available is False


def test_missing_library_disables():
    # world_size==2 → 越过单卡早退，去构造 HCCLLibrary；host 无 .so → except → disabled。
    comm = pyhccl.PyHcclCommunicator(group=_stateless_group(0, 2), device=0)
    assert comm.disabled is True
    assert comm.available is False


def test_all_reduce_and_broadcast_noop_when_disabled():
    comm = pyhccl.PyHcclCommunicator(group=_stateless_group(0, 1), device=0)
    # disabled 时 all_reduce 直接 return None（不碰 device assert / libhccl）。
    assert comm.all_reduce(in_tensor=types.SimpleNamespace()) is None
    # broadcast 同理 no-op（返回 None，不抛）。
    assert comm.broadcast(tensor=types.SimpleNamespace(), src=0) is None
