"""ch24 — 第三条注册线：direct_register_custom_op 把 torch.ops.vllm.* 注册成
真实现(PrivateUse1) + fake(Python 版 meta)。测真实可观察行为：

  1. 10 个 op 全部注册成功，可经 torch.ops.vllm.<op> 取到。
  2. 每个 op 都登记了 fake（abstract impl）——缺它 torch.compile 无法推形状。
  3. 在 FakeTensorMode（= 图捕获『假跑只追形状』）下，dispatch 走 fake，输出 shape/dtype
     由 fake 推出，且真实现根本不运行（host 无 NPU 也能跑通）。
"""
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import register_custom_ops  # noqa: F401  导入即触发 10 处 direct_register_custom_op

EXPECTED_OPS = [
    "maybe_chunk_residual",
    "maybe_all_gather_and_maybe_unpad",
    "maybe_pad_and_reduce",
    "prefetch_preprocess",
    "prefetch_postprocess",
    "maybe_all_reduce_tensor_model_parallel",
    "matmul_and_reduce",
    "quantize",
    "npu_rotary_embedding",
    "muls_add",
]


def test_all_ten_ops_registered():
    for name in EXPECTED_OPS:
        assert hasattr(torch.ops.vllm, name), f"torch.ops.vllm.{name} 未注册"


def test_every_op_has_a_fake():
    # _register_fake 把 fake 登到 Meta/CompositeImplicitAutograd 抽象派发；查 Meta 注册表即可证明在场
    meta_regs = torch._C._dispatch_get_registrations_for_dispatch_key("Meta")
    for name in EXPECTED_OPS:
        assert f"vllm::{name}" in meta_regs, f"vllm::{name} 缺 fake/meta 注册 — 无法进图"


def test_fake_infers_shape_without_real_compute():
    with FakeTensorMode():
        x = torch.empty(8, 16)
        residual = torch.empty(8, 16)
        # maybe_chunk_residual 的 fake = empty_like(x) → 同形同 dtype
        out = torch.ops.vllm.maybe_chunk_residual(x, residual)
        assert out.shape == x.shape and out.dtype == x.dtype

        # npu_rotary_embedding 的 fake 直接 return query,key → 形状不变
        q = torch.empty(4, 8, 64)
        k = torch.empty(4, 8, 64)
        pos = torch.empty(4, dtype=torch.long)
        cache = torch.empty(2048, 64)
        oq, ok = torch.ops.vllm.npu_rotary_embedding(pos, q, k, cache, 64, 64, True)
        assert oq.shape == q.shape and ok.shape == k.shape

        # muls_add 的 fake = empty_like(x)
        y = torch.empty(8, 16)
        mout = torch.ops.vllm.muls_add(x, y, 2.0)
        assert mout.shape == x.shape

        # maybe_all_reduce_tensor_model_parallel 的 fake = lambda x: x（形状不变）
        ar = torch.ops.vllm.maybe_all_reduce_tensor_model_parallel(x)
        assert ar.shape == x.shape


def test_quantize_fake_changes_dtype_to_int8():
    # quantize 的 fake 复用 torch_npu.npu_quantize → 输出 int8（量化）
    with FakeTensorMode():
        t = torch.empty(4, 32, dtype=torch.float16)
        scale = torch.empty(32)
        out = torch.ops.vllm.quantize(t, scale, scale, scale)
        assert out.dtype == torch.int8
        assert out.shape == t.shape
