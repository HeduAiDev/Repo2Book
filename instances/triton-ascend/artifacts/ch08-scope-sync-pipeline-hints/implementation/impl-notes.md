# ch08 implementation notes — scope / sync_block / PIPE / compile_hint

只做减法的精简版，与目标代码仓 `third_party/ascend/language/cann/extension/` +
`python/triton/compiler/code_generator.py` 同名同结构同控制流，只删不增。

## Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `third_party/ascend/language/cann/extension/scope.py::scope` | `third_party/ascend/language/cann/extension/scope.py:L28-71` | 删文件头 license(L1-21) | 纯法务声明，不参与控制流 |
| `.../extension/dispatch.py::ASCEND_WITH_DISPATCH` | 同名文件:L31-34 | 删 `"mangle_ty": mangle_ty` 一项 | mangle_ty 与本章 with/scope 语义无关(subtraction_plan.delete 批准)，归 ch04 题材 |
| `.../extension/code_generator.py`(6 helper + handle_scope_with) | 同名文件:L29-208 | 删 `mangle_ty`(L29-60)，其余逐字保留 | mangle_ty 是类型名字修饰，非 with 特判/SSA 穿线主线 |
| `python/triton/compiler/code_generator.py::CodeGenerator` | 同名文件:L210-380 | `__init__` 只留 `builder/lscope/local_defs` 三字段，删双 builder 构造(`self.ascend_builder`/`setup_unified_builder`)、`gscope` 归一化、`attributes/constants/function_name/is_kernel/...`等前端通用状态 | 双 builder 构造是 ch04《双 builder 与 Ascend 内建的分发路由》的机制主线，本章不重复讲；只保留 `handle_scope_with` 实际读写的三个字段，`set_value`/`_get_insertion_point_and_loc`/`_set_insertion_point_and_loc`/`visit_compound_statement`/`enter_sub_region`(must_keep)原样保留 |
| `.../extension/core.py`(CORE/PIPE/MODE + builtin/is_builtin + create_sync_block + sync_block_*) | 同名文件:L1-244 | 删 `IteratorType`/4 个 `Fixpipe*` 枚举+`fixpipe`/`ascend_address_space_base`/`_group`/`copy`/`copy_from_ub_to_l1`/`sub_vec_id`/`sub_vec_num`/`int64`/`SYNC_IN_VF`/`debug_barrier`(原 L126-368) | 分属 ch05(地址空间/搬运)、ch06/ch07(算子)题材，与 scope/sync_block/PIPE/compile_hint 控制流不相交；随之删除只服务于这些成员的 import(`triton.language.core as tl`/`triton.extension.buffer.language as bl`/`NPUUtils`/`ir`)与 `__all__` 对应条目 |
| `.../extension/semantic.py`(PIPE + create_sync_block_set/wait) | 同名文件:L1-87 | 只留 `PIPE`/`create_sync_block_set`/`create_sync_block_wait`，删 `create_address_space`/`sub_vec_id`/`copy_from_ub_to_l1`/`copy`/`fixpipe`/`debug_barrier`等(原 L44-...) | subtraction_plan.delete 批准：本章数据流只经过这三项；随之删除只服务于被删部分的 `import triton.language.extra.cann.extension as al`/`triton.extension.buffer.language as bl` |
| `.../extension/aux_ops.py`(旧代 sync_block_*/parallel/compile_hint*/multibuffer) | 同名文件:L1-162 | 顶部 import 名单只留 `_constexpr_to_value/_tensor_member_fn/builtin/constexpr/tensor/core/range/ir/custom_op`(must_keep)，删 `_unwrap_iterable/dtype/check_bit_width`与整段 14 项 `from triton.language.semantic import (...)` | 本章保留的函数从不引用被删名字 |
| `.../extension/_utils.py::custom_op` | 同名文件:L5-16 | 删 `_is_int_like_elem`/`_assert_int_like_tuple`/`_convert_elem_to_ir_value`(原 L18-54) | 服务于 block pointer 参数转换，不在 sync_block 调用链上 |
| `python/triton/language/core.py`(支撑层) | 同名文件(节选，各符号行号见文件内注释) | 只留 `builtin/_tensor_member_fn/constexpr/_unwrap_if_constexpr/_constexpr_to_value/_value/tensor/range` | 基座 Triton、未被 fork 改动，只提供 import 依赖，非本章 dossier 机制主角 |

## 关键设计决策的落地位置

- **M1 with 特判**：`code_generator.py::WITH_DISPATCH` + `visit_With`（基座）与
  `dispatch.py::ASCEND_WITH_DISPATCH`（fork）。测试 `test_with_dispatch.py` 专门证明
  `scope(...)` 这个调用表达式在 with-分派路径下从未被求值（`__init__`/`__enter__`
  都不跑）。
- **M2 scope 空壳**：`scope.py`。测试 `test_scope_empty_shell.py` 覆盖 `__init__` 的
  core_mode 校验、`__enter__` 的 RuntimeError、`__exit__` 恒 `return False`，以及
  docstring `feature_a=True` 与必填 `core_mode` 的签名不一致（M18）。
- **M3/M4 两趟 visit + 属性翻译**：`extension/code_generator.py::handle_scope_with`
  与六个私有助手。测试 `test_handle_scope_with.py` 覆盖 dummy block 试跑与 erase、
  `noinline`/`core_mode`/`disable_auto_sync`/透传四条属性规则、`_verify_loop_carried_variable`
  的类型一致性校验、SSA 穿线回填（对照官方 UT `kernel_scope_escape`）。
- **M6-M11 sync_block 两代**：旧代 `aux_ops.py` 经 `_utils.custom_op` 落
  `create_custom_op_for_inter_core_sync`；新代 `core.py::create_sync_block` 经
  `semantic.py` 落 `builder.sync_block_set/wait/all`。测试
  `test_sync_block_old_vs_new.py` 覆盖两代的四条公共校验、pipe 缺省配对
  （cube→FIX/MTE2、vector→MTE3/MTE2）、`event_id` 三形态归一、以及新代独有的
  `all_sub_vector`。
- **M12/M13/M17 口径收窄与双绑定**：`test_utils_and_enums.py` 覆盖 PIPE/CORE/MODE
  枚举规模、`scope` 到不了 `CUBE_OR_VECTOR`/`CUBE_AND_VECTOR`、`core.py` 里 `PIPE`
  被绑定两次（`semantic.PIPE` 被同名 `class PIPE` 覆盖，非同一个类）。
- **M14/M15/M16 compile_hint**：`test_compile_hint_and_parallel.py` 覆盖五种 hint
  值类型分派、`bool` 判断必须先于 `not hint_val`（否则 `False` 被误吞成 unit_attr）、
  `multibuffer` 绕过外层 SIMT 早退（FIXME 现状如实保留）、`parallel(bind_sub_block=...)`。

## 测试环境说明

宿主无昇腾 NPU/CANN，`ir.builder`/`ascendnpu_ir_builder`/`ascend_ir.{CoreType,PIPE,MODE}`
均为编译期 C++ 绑定，`tests/conftest.py` 用 `FakeBuilder`（记录调用 + 返回可预测哨兵值）
与占位枚举组代替，不模拟 MLIR/硬件语义。`triton.language.semantic.to_tensor`
（基座、未被 fork 改动的真实函数）在真实仓库里要经过完整 dtype 系统与 `full()`，与本章
dossier 完全正交，按外部依赖处理，只保留其「value → 带 `.handle` 的容器」这一可观察
行为的测试替身，不在 `implementation/` 下重造 dtype/`full()`。
