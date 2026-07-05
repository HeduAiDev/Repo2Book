"""ch24 — Python 侧 meta 兜底：当 C++ .so 没编进某算子的 meta 时，
meta_registration.py 用 Library("_C_ascend","IMPL") + register_meta_if_necessary 在
Python 层补 Meta 派发键实现，让算子能进 torch.compile/aclgraph。测真实可观察行为：

  conftest 已模拟 C++ .so DEF 了 _C_ascend::{get_masked_input_and_mask,bgmv_expand,sgmv_expand}
  （真实现存在、但无 C++ meta）。导入 meta_registration 后：
  1. 这 3 个算子获得了 Meta 派发键实现（之前没有）。
  2. meta 实现只推 shape/dtype、不真算（在 meta 设备张量上调用即得正确空壳）。
  3. register_meta_if_necessary 查注册表去重——对已有 Meta 的算子二次调用安全跳过（不重复注册报错）。
"""
import torch

import meta_registration  # noqa: F401  导入即触发 if not is_310p(): register_meta_if_necessary × 3

OPS = ["get_masked_input_and_mask", "bgmv_expand", "sgmv_expand"]


def test_three_python_metas_registered():
    meta_regs = torch._C._dispatch_get_registrations_for_dispatch_key("Meta")
    for op in OPS:
        assert f"_C_ascend::{op}" in meta_regs, f"_C_ascend::{op} 未补上 Python meta — 不能进图"


def test_get_masked_input_and_mask_meta_infers_shape_and_dtype():
    # 经 Meta 派发：在 meta 设备张量上跑 = 只推形状不真算
    inp = torch.empty(3, 4, device="meta", dtype=torch.float16)
    masked_input, mask = torch.ops._C_ascend.get_masked_input_and_mask(inp, 0, 4, 0, 0, 0)
    assert masked_input.shape == inp.shape and masked_input.dtype == inp.dtype
    assert mask.shape == inp.shape and mask.dtype == torch.bool
    assert masked_input.device.type == "meta"


def test_meta_fn_called_directly_returns_empty_shells():
    inp = torch.empty(2, 5)
    masked_input, mask = meta_registration.get_masked_input_and_mask_meta(inp, 0, 5, 0, 0, 0)
    assert masked_input.shape == inp.shape
    assert mask.dtype == torch.bool

    y = torch.empty(7, 9)
    out = meta_registration.bgmv_expand_meta(torch.empty(1), torch.empty(1), torch.empty(1), y, 0, 9)
    assert out.shape == y.shape  # bgmv_expand meta = empty_like(y)


def test_register_meta_if_necessary_is_idempotent():
    # get_masked_input_and_mask 已有 Meta 实现 → 二次调用应命中去重 return，不抛
    # （若没有去重，第二次 lib.impl 同 op+Meta 会报重复注册）
    def _other_meta(input, a, b, c, d, e):
        return torch.empty_like(input), torch.empty_like(input).to(torch.bool)

    meta_registration.register_meta_if_necessary("_C_ascend", "get_masked_input_and_mask", _other_meta)
    # 没有异常即证明走了 `if schema_to_find in meta_impl_list: return` 分支
