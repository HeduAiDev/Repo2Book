"""多进程分布式测试（host gloo 后端，无 CUDA）：

用真实 torch.distributed（gloo）spawn 多个 rank，验证精简版 GroupCoordinator 的
集合原语与 P2P 复现真实 vLLM 的可观察数值行为：
  - all_reduce 求和、all_gather 拼接、reduce_scatter 求和后切片
  - broadcast / broadcast_tensor_dict 的双群组分流
  - send/recv 的 P2P 主线、barrier
  - initialize_model_parallel 的 5 维 rank 张量切分（TP/PP/DP group_ranks）

精简版用 CPU 设备分支（current_platform.is_cuda_alike()==False），device_group 与
cpu_group 都是 gloo，因此这些用例无需 GPU 即可跑通。
"""
import os
import sys
from pathlib import Path

# spawn 出的子进程会重新 import 本模块，但不会执行 conftest，因此在这里也把
# implementation/ 加入 sys.path，保证子进程能 import parallel_state。
_IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import parallel_state as ps


def _init(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29555")
    ps._reset_state_for_tests()
    ps.init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=rank,
        backend="gloo",
    )


def _run(rank, world_size, fn, port, out):
    os.environ["MASTER_PORT"] = str(port)
    try:
        _init(rank, world_size)
        fn(rank, world_size, out)
    finally:
        if dist.is_initialized():
            dist.barrier()
            ps._reset_state_for_tests()
            dist.destroy_process_group()


def _spawn(fn, world_size=4, port=29560):
    mgr = mp.Manager()
    out = mgr.dict()
    mp.spawn(_run, args=(world_size, fn, port, out), nprocs=world_size, join=True)
    return out


# ---- TP=4 单组，验证三大集合原语数值 ----

def _body_collectives(rank, world_size, out):
    ps.initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp = ps.get_tp_group()

    # all_reduce: 每 rank 贡献 (rank+1)，求和后所有 rank 应得到 sum(1..world).
    x = torch.full((3,), float(rank + 1))
    ar = tp.all_reduce(x.clone())
    expected_sum = sum(range(1, world_size + 1))
    assert torch.allclose(ar, torch.full((3,), float(expected_sum))), ar

    # all_gather(dim=0): 每 rank 贡献 [rank]，拼接后 == [0,1,...,world-1].
    g = tp.all_gather(torch.tensor([float(rank)]), dim=0)
    assert torch.allclose(g, torch.arange(world_size, dtype=torch.float)), g

    # reduce_scatter(dim=0): 输入 world_size 段，求和后每 rank 留 1 段。
    inp = torch.arange(world_size, dtype=torch.float) + rank  # 每 rank 偏移
    rs = tp.reduce_scatter(inp.clone(), dim=0)
    # 第 i 段的和 = sum_r (i + r) = world*i + sum(r)
    seg_sum = world_size * rank + sum(range(world_size))
    assert torch.allclose(rs, torch.tensor([float(seg_sum)])), rs

    if rank == 0:
        out["ok"] = True


def test_collectives_tp4():
    out = _spawn(_body_collectives, world_size=4, port=29561)
    assert out.get("ok") is True


# ---- broadcast / broadcast_tensor_dict 双群组分流 ----

def _body_broadcast(rank, world_size, out):
    ps.initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp = ps.get_tp_group()

    t = torch.zeros(2) if rank != 0 else torch.tensor([7.0, 8.0])
    bt = tp.broadcast(t.clone(), src=0)
    assert torch.allclose(bt, torch.tensor([7.0, 8.0])), bt

    # broadcast_object 走 cpu_group。
    obj = {"k": rank} if rank == 0 else None
    bo = tp.broadcast_object(obj, src=0)
    assert bo == {"k": 0}, bo

    # broadcast_tensor_dict: metadata 走 cpu_group、tensor 走 device_group。
    if rank == 0:
        td = {"meta": 123, "x": torch.tensor([1.0, 2.0, 3.0])}
    else:
        td = None
    rtd = tp.broadcast_tensor_dict(td, src=0)
    assert rtd["meta"] == 123
    assert torch.allclose(rtd["x"], torch.tensor([1.0, 2.0, 3.0])), rtd["x"]

    if rank == 0:
        out["ok"] = True


def test_broadcast_tensor_dict_tp4():
    out = _spawn(_body_broadcast, world_size=4, port=29562)
    assert out.get("ok") is True


# ---- PP send/recv 主线 + barrier ----

def _body_p2p(rank, world_size, out):
    # PP=world_size, TP=1: 每个 rank 自成一个 TP 组，PP 组 = 全部 rank.
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=world_size
    )
    pp = ps.get_pp_group()
    pp.barrier()  # 走 cpu_group

    # rank0 -> rank1 send/recv 单张量主线。
    if pp.rank_in_group == 0:
        pp.send(torch.tensor([42.0, 43.0]), dst=1)
    elif pp.rank_in_group == 1:
        r = pp.recv(torch.Size([2]), torch.float, src=0)
        assert torch.allclose(r, torch.tensor([42.0, 43.0])), r
        out["recv_ok"] = True

    if rank == 0:
        out["ok"] = True


def test_pp_send_recv():
    out = _spawn(_body_p2p, world_size=2, port=29563)
    assert out.get("ok") is True
    assert out.get("recv_ok") is True


# ---- initialize_model_parallel 的切分（TP=2,PP=2,DP=1 -> world=4） ----

def _body_slicing(rank, world_size, out):
    ps.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=2
    )
    if rank == 0:
        out["tp_ranks"] = ps.get_tp_group().ranks
        out["pp_ranks"] = ps.get_pp_group().ranks
        out["dp_ranks"] = ps.get_dp_group().ranks


def test_model_parallel_slicing():
    out = _spawn(_body_slicing, world_size=4, port=29564)
    # rank0 所在 TP 组相邻 [0,1]；PP 组跨 TP [0,2]；DP=1 单元素 [0].
    assert out["tp_ranks"] == [0, 1]
    assert out["pp_ranks"] == [0, 2]
    assert out["dp_ranks"] == [0]


# ---- world_size==1 短路 ----

def _body_singleton(rank, world_size, out):
    ps.initialize_model_parallel(tensor_model_parallel_size=1)
    tp = ps.get_tp_group()
    x = torch.tensor([3.0, 4.0])
    # world_size==1 时 all_reduce/all_gather/reduce_scatter 都原样返回。
    assert tp.all_reduce(x) is x
    assert tp.all_gather(x) is x
    assert tp.reduce_scatter(x) is x
    out["ok"] = True


def test_singleton_short_circuit():
    out = _spawn(_body_singleton, world_size=1, port=29565)
    assert out.get("ok") is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
