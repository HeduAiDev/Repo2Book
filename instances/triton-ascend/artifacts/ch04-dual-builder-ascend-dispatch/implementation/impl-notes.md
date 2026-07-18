# ch04 精简版实现说明——双 builder 与 Ascend 内建的分发路由

## 范围

本章精简版只保留 dossier `mechanisms`(m1-m6)真正需要的六条接缝：模块级 `WITH_DISPATCH`
表 + `mangle_ty` override 钩子（m5）、`CodeGenerator.__init__` 的双 builder 构造块
（m1）、`visit_Call` 的第四岔分发（m2/m3）、`setup_unified_builder` 的方法挂载（m4）、
`visit_With` 的查表分发（m5）、两 builder 的插入点/loc 接力（m6）。

`third_party/ascend/language/cann/extension/` 下的算子语义目录（hivm 方言的具体
create_* 算子实现）不在本章范围——按 dossier `subtraction_plan.delete` 批准，只留
`sub_vec_id` 一个代表例演示 `@builtin` 双标记，其余 hivm 算子归 Part 5。

## Source Map

| 精简版位置 | 对应真实源码 | 改动 | 原因 |
|---|---|---|---|
| `implementation/python/triton/compiler/code_generator.py:24-30` `WITH_DISPATCH` + import | `python/triton/compiler/code_generator.py:L25-31` | 原样保留 | m5 全局分发表，fork 接缝之一，逐字保留 |
| 同上文件 `:33-49` `mangle_ty` 基座实现 | `python/triton/compiler/code_generator.py:L34-49` | 原样保留 | 与 ascend 版对照，示范同名函数被 override |
| 同上文件 `:55` `mangle_ty = WITH_DISPATCH.get(...)` | `python/triton/compiler/code_generator.py:L51` | 原样保留 | override 钩子本体，一行不可删 |
| 同上文件 `class CodeGenerator.__init__:104-142` | `python/triton/compiler/code_generator.py:L210-271` | 只删 SUBTRACTED 标注段（buffer_builder 支线、gscope/module_map/function_ret_types 等基座通用初始化），双 builder 构造块（`self.builder`/`self.ascend_builder`/`setup_unified_builder`）原样保留 | m1 主角，删除内容与双 builder 主线无关 |
| 同上文件 `builtin_namespace`(L149) | `python/triton/compiler/code_generator.py:L273-278` | 只删 print/min/max 三项条目 | 与本章分发路由无关，dossier 批准 |
| 同上文件 `statically_implemented_functions`(L161) | `python/triton/compiler/code_generator.py:L1332-1338` | 清空为空字典，保留同名符号使 `visit_Call` 首行结构不变、恒为 None | 常量折叠是 dossier theory 所称『第①岔』，与本章第③/④岔正交 |
| 同上文件 `_get/_set_insertion_point_and_loc:164-177` | `python/triton/compiler/code_generator.py:L353-365` | 原样保留 | m6 插入点接力的读写点，`visit_Call` 与 wrapper 都用它 |
| 同上文件 `visit_With:199-215` | `python/triton/compiler/code_generator.py:L801-814` | 原样保留 | m5 查表分发宿主方法，fork 新增（基座无此方法） |
| 同上文件 `visit_Call:219-264` | `python/triton/compiler/code_generator.py:L1168-1206` | 原样保留（含 JITFunction 分支的结构，但 `_check_fn_args`/`call_JitFunction` 不提供实现——测试从不构造 JITFunction 实例，从不走到该分支） | m2/m3 核心——第四岔 `_builder = self.ascend_builder if extension.is_builtin(fn) else self.builder` 逐字保留 |
| `implementation/python/triton/language/core.py` 全文件 | `python/triton/language/core.py`（本仓与上游共享，未被 fork 改动的模块） | 大幅精简：只留 `TRITON_BUILTIN`/`builtin`/`is_builtin`（双标记机制基座半边）+ `constexpr`/`_unwrap_if_constexpr`/`_constexpr_to_value`/`_value`/`tensor`/`dtype.SIGNEDNESS` 最小容器子集 | 供 `visit_Call`/`mangle_ty`/`handle_scope_with` 依赖，删掉的算子重载/类型目录与双 builder 分发无关 |
| `implementation/python/triton/runtime/jit.py` `class JITFunction: pass` | `python/triton/runtime/jit.py:L445` | 只留类型标识，无方法体 | 仅供 `isinstance(fn, JITFunction)` 判定；本章测试从不构造真实实例 |
| `implementation/python/triton/compiler/errors.py` | `python/triton/compiler/errors.py` | 只留 `__init__`/异常语义，删 `_format_message`/`__reduce__` | `visit_Call` 只把它当异常类型抛/接，不依赖消息格式化 |
| `implementation/third_party/.../extension/core.py` | `third_party/ascend/language/cann/extension/core.py:L66-90`（双标记）+ `:L166-171`（`sub_vec_id`） | 只留 `TRITON_BUILTIN`/`ASCEND_BUILTIN`/`builtin`/`is_builtin` + 一个代表算子 `sub_vec_id` | m3 双标记核心；其余 hivm 算子完整校验体归 Part 5，按 dossier 批准删除 |
| `implementation/third_party/.../extension/builder.py` | `third_party/ascend/language/cann/extension/builder.py`（全文件，逻辑未删） | 只精简 `setup_unified_builder` 内 `ascend_methods` 清单为 6 项代表（含 `create_scope_op`/`scope_return`/`get_t_core_type_*`/`create_copy_buffer`） | m4 主体，机制是「逐个包 wrapper 后 setattr」，清单只是枚举 |
| `implementation/third_party/.../extension/dispatch.py` | `third_party/ascend/language/cann/extension/dispatch.py` | 全文件未删 | m5 的 ascend 侧注册项，本身已经很小 |
| `implementation/third_party/.../extension/scope.py` | `third_party/ascend/language/cann/extension/scope.py` | 全文件未删 | `scope` 类是 `ASCEND_WITH_DISPATCH` 的键，本身很小 |
| `implementation/third_party/.../extension/code_generator.py` | `third_party/ascend/language/cann/extension/code_generator.py:L29-60`(`mangle_ty`) + `:L137-208`(`handle_scope_with`) | `mangle_ty` 删 buffer_type 分支；`handle_scope_with` 内联 6 个私有属性/SSA 助手为最简写法，只演示 `core_mode` 一条属性路径，控制流骨架（哑 block 试跑→建 scope_op→重新 emit body→scope_return）保留 | 属性系统与 SSA 线程化细节归后续 scope 专章 |
| `implementation/third_party/.../extension/__init__.py` | `third_party/ascend/language/cann/extension/__init__.py` | 只重导出 `builtin`/`is_builtin`/`sub_vec_id`/`scope` 四个名字 | 其余 custom_op/math_ops/aux_ops/mem_ops/vec_ops 等算子入口归 Part 5 |

## 已知的结构性留白（非缺陷）

- `visit_Call` 的 `isinstance(fn, JITFunction)` 分支引用了 `_check_fn_args`/
  `self.call_JitFunction`，本精简版未定义这两者——真实行为下它们递归展开
  `@triton.jit` 组合子，是 dossier theory 所称『第②岔』，与本章『第③/④岔』正交，
  归另一批章节。本章测试与叙事都只构造普通可调用对象（`@builtin` 装饰的函数），
  从不构造 `JITFunction` 实例，故该分支在本章语境下结构存在但从不被执行到。

## 验证

- `python3 -m pytest tests/` — 18 passed（纯 Python 单元测试，靠 `conftest.py` 的
  `FakeBuilder`/`FakeAscendBuilder` 站在真实 `ir.builder`/`ascendnpu_ir_builder`
  （C++ 绑定，host 无昇腾 NPU/CANN 工具链故无法拥有）位置上，验证的是『调用被路由
  到哪个对象』，不模拟 MLIR 语义——IR dump 级别验证需真机，按 INSTANCE.md 约束不在
  本章测试范围）。
- `python3 scripts/lint_fidelity.py {chapter_dir}` — 无 BLOCKING。
