# ch24 精简版实现说明（只做减法）

本章可运行部分仅 Python 两条线（C++ `csrc/*.cpp` 按 INSTANCE 硬规则 host 不可编译，全程作只读真源码片段，不进精简版）。精简版与真源码同名/同结构/同控制流，只删 dossier `subtraction_plan.delete` 批准项。

## 文件
- `implementation/meta_registration.py` ← `vllm_ascend/meta_registration.py`（Python meta 兜底：Library(_C_ascend,IMPL) + register_meta_if_necessary 补 3 个 Meta 实现）
- `implementation/register_custom_ops.py` ← `vllm_ascend/ops/register_custom_ops.py`（direct_register_custom_op × 10：真实现 PrivateUse1 + _fake）
- `tests/conftest.py` ← host 脚手架（非源码）：注入 stub 模块到 sys.modules，并以 `Library("_C_ascend","DEF")` 模拟 C++ .so 已 DEF 算子；`direct_register_custom_op` stub 忠实复刻基座 `vllm/utils/torch_utils.py` 的 infer_schema→define→impl→_register_fake。
- `tests/test_register_custom_ops.py` / `tests/test_meta_registration.py`

## Source Map（精简版 ↔ 真源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真源码 :Lxxx | 改动 | 原因 |
|---|---|---|---|
| `register_meta_if_necessary` / `lib`(Library) | vllm_ascend/meta_registration.py:L44-L54 | 原样保留 | 章核心：去重(`_dispatch_get_registrations_for_dispatch_key("Meta")`)后 `lib.impl(op,fn,"Meta")` 补 meta |
| `get_masked_input_and_mask_meta` / `bgmv_expand_meta` / `sgmv_expand_meta` | meta_registration.py:L57/L71/L78 | 原样保留 | Python 版 meta：只 `empty_like` 推 shape/dtype，不真算 |
| 顶部 how-to docstring | meta_registration.py:L6-L42 | `# SUBTRACTED` 删中段，留一句点题 | 开发指引非运行逻辑，不影响注册行为 |
| `_maybe_chunk_residual_impl` 等 7 个 `_xxx_impl` | register_custom_ops.py:L23-L166 | `# SUBTRACTED` 把通信/预取/MoE 多分支裁到单一主路径 | 本章只讲注册范式；NPU 真算分支见 ch20~ch23，host 无 NPU 跑不了 |
| 全部 `_xxx_fake`（含 `_quantize_impl_fake`/`_rope_forward_oot_impl_fake`/`_muls_add_impl_fake`） | register_custom_ops.py:L108-L217 | 原样保留 | fake = Python 版 meta，是本章主角；缺它 torch.compile 推不了形状 |
| 10 处 `direct_register_custom_op(...)` | register_custom_ops.py:L220-L298 | 原样保留（op_func/fake_impl/dispatch_key="PrivateUse1"） | 注册产物 `torch.ops.vllm.*` + 真实现/fake 分离，须逐字保真 |
| 顶部 import（torch_npu / vllm.* / vllm_ascend.* 内部模块） | register_custom_ops.py:L1-L20 | 原样保留；host 由 conftest stub 提供 | dossier delete 批准 stub 化；保留 import 行使结构忠实、符号在场 |

## 可验证点（host `python3 -m pytest`，8 passed）
1. 10 个 `torch.ops.vllm.*` 全注册成功，且每个在 Meta 派发键有 fake（缺则不能进图）。
2. `FakeTensorMode`（= 图捕获『假跑只追形状』）下 dispatch 走 fake，shape/dtype 由 fake 推出、真实现不运行 → 印证「无 meta/fake 不能进图」。
3. meta_registration 给 `_C_ascend::{get_masked_input_and_mask,bgmv_expand,sgmv_expand}` 补 Meta 实现；meta 设备张量上调用只推空壳；`register_meta_if_necessary` 去重幂等（二次调用不重复注册报错）。

## 给 writer 的提示
- C++ 契约符号（TORCH_LIBRARY_EXPAND / TORCH_LIBRARY_IMPL_EXPAND / ops.def / ops.impl / torch::kPrivateUse1 / at::kMeta）只在只读 C++ 片段讲，不在本精简版；Python 同构是 `direct_register_custom_op` 的 define/impl(dispatch_key)/`_register_fake` 与 `lib.impl(...,"Meta")`。
- 贯穿样本 `get_masked_input_and_mask` 三处现身（C++ 真实现 / C++ meta / Python meta）。算子计数按主块绝对数：63 def / 57 C++ meta（缺口 6）/ 10 direct_register_custom_op / 3 Python meta。
