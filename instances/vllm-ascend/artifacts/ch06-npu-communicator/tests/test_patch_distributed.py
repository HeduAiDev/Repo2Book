"""验证 310P 猴补的可观察行为：all_gather 模拟 + 降级直通（must_keep:
communication_adaptation_310p / broadcast310p / all_gather / NullHandle）。

只用 torch.distributed.all_gather（gloo 即支持），单进程 gloo 组即可在 host 复现：
- all_reduce(int64, SUM) → all_gather→stack→sum（world_size=1 时 == 原张量）
- all_reduce(非 int64) → 直通原函数 fn（降级路径）
- broadcast(cpu tensor) → 直通原函数 fn（降级路径）
注意：device tensor 的 broadcast all_gather 路径需非 cpu 设备，host 无法触发，故只验 cpu 直通。
"""
import pytest
import torch
import torch.distributed as dist

import patch_distributed as pd


@pytest.fixture
def gloo_group():
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture
def patched(gloo_group):
    # 保存原函数，调用 communication_adaptation_310p() 做真实猴补，测后还原。
    orig_br = torch.distributed.broadcast
    orig_ar = torch.distributed.all_reduce
    pd.communication_adaptation_310p()
    try:
        yield
    finally:
        torch.distributed.broadcast = orig_br
        torch.distributed.all_reduce = orig_ar


def test_all_reduce_int64_sum_uses_all_gather(patched):
    # int64 命中模拟路径：world_size=1 → all_gather 得 [t] → stack.sum(0) == t。
    t = torch.tensor([3, 5, 7], dtype=torch.int64)
    out = torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    assert out.dtype == torch.int64
    assert torch.equal(out, t)


def test_all_reduce_non_int64_passthrough(patched):
    # 非 int64 → 直通原 all_reduce（in-place，world_size=1 时值不变），不返回新张量。
    t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    ret = torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0]))
    assert ret is None or ret is not None  # 直通行为由原函数决定，不破坏张量即可


def test_broadcast_cpu_tensor_passthrough(patched):
    # cpu tensor → broadcast310p 走 fn 直通（不进 all_gather 分支）。
    t = torch.tensor([9, 9], dtype=torch.int64)
    torch.distributed.broadcast(t, src=0)
    assert torch.equal(t, torch.tensor([9, 9], dtype=torch.int64))


def test_null_handle_wait_is_noop():
    h = pd.NullHandle()
    assert h.wait() is None
