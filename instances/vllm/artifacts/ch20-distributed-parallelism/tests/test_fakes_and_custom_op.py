"""单元测试（纯 host，无 CUDA、无多进程）：

验证精简版复现真实 vLLM 的两类可观察行为：
  1. 集合算子的 fake(meta) 实现给出正确的输出 shape——这是 torch.compile 能在
     编译期对 all_gather/reduce_scatter 做形状推断而不真正通信的关键。
  2. 三个集合算子被注册为 torch.ops.vllm.*（custom-op），即『集合通信进 compiled
     graph 而不 graph break』的核心动作。
"""
import torch

import parallel_state as ps


def test_all_reduce_fake_keeps_shape():
    x = torch.randn(4, 8)
    out = ps.all_reduce_fake(x, group_name="tp:0")
    # all_reduce 输出 shape 不变（每 rank 得到全和）。
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_all_gather_fake_scales_dim_by_world_size():
    x = torch.randn(4, 8)
    out = ps.all_gather_fake(x, dim=0, world_size=4, group_name="tp:0")
    # all_gather 沿 dim 放大 world_size 倍。
    assert out.shape == (16, 8)


def test_reduce_scatter_fake_shrinks_dim_by_world_size():
    x = torch.randn(16, 8)
    out = ps.reduce_scatter_fake(x, dim=0, world_size=4, group_name="tp:0")
    # reduce_scatter 沿 dim 缩小 world_size 倍（与 all_gather 对偶）。
    assert out.shape == (4, 8)


def test_three_collectives_registered_as_vllm_custom_ops():
    # 注册发生在 import parallel_state 时（模块级 direct_register_custom_op）。
    assert hasattr(torch.ops.vllm, "all_reduce")
    assert hasattr(torch.ops.vllm, "all_gather")
    assert hasattr(torch.ops.vllm, "reduce_scatter")


def test_custom_op_fake_shape_via_meta_device():
    # 在 meta 设备上调用 custom op，应走 fake 实现做纯形状推断（不真正通信）。
    x = torch.randn(4, 8, device="meta")
    out = torch.ops.vllm.all_gather(x, 0, 4, group_name="nonexistent")
    assert out.shape == (16, 8)
    assert out.device.type == "meta"


def test_module_level_all_reduce_looks_up_group_by_name():
    # 模块级 all_reduce 按 group_name 查回 GroupCoordinator；未知名字应断言失败。
    try:
        ps.all_reduce(torch.randn(2, 2), group_name="does-not-exist")
    except AssertionError as e:
        assert "is not found" in str(e)
    else:
        raise AssertionError("expected AssertionError for unknown group name")
