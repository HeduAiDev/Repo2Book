# ch06 精简版实现说明——昇腾内建算子：索引搬运、向量算子与定制 cast

## 范围

本章精简版覆盖 dossier `mechanisms`(m1-m14) 需要的六个文件：

- `third_party/ascend/language/cann/extension/mem_ops.py`——`index_put` /
  `gather_out_to_ub` / `scatter_ub_to_out` / `index_select_simd` 四件套，全部
  函数体逐字保留（含各自的 `*_impl` 内层实现），只删长 docstring 的
  Constraints/Example 段落（dossier `subtraction_plan.delete` 第一项批准）与
  `index_select_simd` 里从未被调用的死代码 `process_param`（第二项批准）。
- `third_party/ascend/language/cann/extension/vec_ops.py`——`insert_slice` /
  `extract_slice` / `get_element` / `flip`(+`flip_simd`/`static_range`) /
  `sort` / `ascend_cast_impl` / `cast`。按 `subtraction_plan.delete` 批准删除：
  `ascend_cast_impl` 的 fp8e4b15/convert_custom_types 分支与指针相关分支；
  `flip_simd`/`sort_impl` 的 shape-未知防御式兜底分支。
- `third_party/ascend/language/cann/extension/aux_ops.py`——只留
  `compile_hint_impl`（cast 的 overflow_mode 与 sort 的自动饱和提示共用的挂载通道）。
- `third_party/ascend/language/cann/extension/_utils.py`——只留
  `_convert_elem_to_ir_value`（offsets/strides 的 i32/i64 折叠器）。
- `third_party/ascend/language/kernels/gather.py`——`gather_2d_simd`，m13 的
  第三条路（不用任何昇腾扩展算子，纯上游 `tl.gather`），逐字保留、未做任何删减。
- `third_party/ascend/language/cann/extension/__init__.py`——只保留
  `is_compile_on_910_95` 一行重导出。

以及供上述文件运行的最小 base-Triton 子集（`python/triton/language/{core,semantic,
standard}.py`、`python/triton/runtime/interpreter.py`、`python/triton/tools/
get_ascend_devices.py`）——这些不是 ascend fork 改动的文件，不在本章 code_spine
内，按 ch04/ch05 一贯做法只取本章实际调用到的最小子集（见各文件顶部注释里逐条列出
的裁剪理由）。

## 保真度要点（对应 brief 订正）

- **index_select_simd 没有 index_boundary，也没有三元组**：`index_select_simd`
  的参数是 `(src, dim, index, src_shape, src_offset, read_shape)`，与另外三个
  mem_ops 不同——docstring 明写 "Does not check if index contains
  out-of-bounds values"。测试 `test_index_select_simd_has_no_index_boundary_param`
  用 `inspect.signature` 断言签名里确实没有这些参数。
- **位宽契约不是统一契约**：`gather_out_to_ub`/`scatter_ub_to_out` 硬编码
  `src_stride`/`dst_stride` 恒 i64、`start_offset`/`end_offset` 恒 i32；
  `index_put` 用 `require_i64 = index.dtype.is_int64()` 一个开关同时决定三者。
  测试 `test_index_put_flattens_multi_rank_index_and_uses_single_i64_switch`/
  `test_index_put_uses_i64_when_index_is_int64` 直接验证同一个开关如何联动三个
  参数的位宽，不替它统一。
- **`is_compile_on_910_95` 是模块级硬件探测全局量，不是 builder 方法**——
  `ascend_cast_impl` 直接读这个 import 进来的名字。测试用
  `mods.vec_ops.is_compile_on_910_95 = True/False` 直接 monkeypatch 该模块全局，
  覆盖 910_95/非 910_95 两条分支（saturate 整型收窄 vs 绕道 float32）。
- **overflow_mode 的 docstring 拼写错误**：cast() 的 docstring 写的是
  `"sautrate"`，真实校验列表 `overflow_modes = ["trunc", "saturate"]` 只认
  `"saturate"`。测试 `test_overflow_mode_docstring_typo_vs_real_whitelist` 同时
  断言 docstring 里的拼写、以及照抄它会被 `ValueError` 拒绝。
- **flip 的双路径**：SIMD 模式（`is_simt_mode()==False`）一条 `create_flip`
  算子；SIMT 模式退回 `log2(n)` 步 xor-swap（`static_range` 迭代 + 每步
  `xor_sum` + 前后各一次 `.to(idtype, bitcast=True)`）。`standard.xor_sum` 的
  内部 combine-region/generator 机制不在本章 code_spine/must_keep 内（真实实现
  委托 `core.reduce` 走一整套上游通用规约的编译期展开，是 ch04 那类"codegen 内联
  @jit 组合子"的深层机制），精简版把它简化成"校验 + 占位返回"（见
  `standard.py` 顶部注释），flip_impl 本身（must_keep）逐字未改。

## Source Map（节选，完整逐符号 `# SOURCE:` 见各文件内联注释）

| 精简版位置 | 对应真实源码 | 改动 | 原因 |
|---|---|---|---|
| `mem_ops.py` 全部四个 builtin + 四个 `*_impl` | `third_party/ascend/language/cann/extension/mem_ops.py:L40-L636` | 只删长 docstring 的 Constraints/Example 段落 + 死代码 `process_param` | dossier `subtraction_plan.delete` 第 1/2 项批准；控制流、校验、位宽契约逐字保留 |
| `vec_ops.py` `insert_slice`/`extract_slice`/`get_element` | `vec_ops.py:L47-L177` | 只删长 docstring | 互逆切片对 + tile→标量，m7/m8 |
| `vec_ops.py` `flip`/`flip_impl`/`flip_simd`/`static_range` | `vec_ops.py:L179-L313` | `flip_simd` 删 shape 未知时的防御式兜底（第 5 项批准） | m9 双路径主角 |
| `vec_ops.py` `sort`/`sort_impl` | `vec_ops.py:L316-L397` | `sort_impl` 删 rank 未知时的兜底分支（第 6 项批准） | m10 |
| `vec_ops.py` `ascend_cast_impl` | `vec_ops.py:L400-L522` | 删 fp8e4b15/convert_custom_types 分支与指针分支（第 3 项批准） | m11 决策树主体 |
| `vec_ops.py` `cast` | `vec_ops.py:L524-L562` | 原样保留（含 docstring 里 "sautrate" 拼写错误） | m12 公开入口 |
| `aux_ops.py` `compile_hint_impl` | `aux_ops.py:L114-L133` | 原样保留 | overflow_mode/saturate 提示唯一挂载通道 |
| `_utils.py` `_convert_elem_to_ir_value` | `_utils.py:L36-L54` | 原样保留 | offsets/strides 的 i32/i64 折叠器 |
| `kernels/gather.py` `gather_2d_simd` | `kernels/gather.py:L32-L106` | 原样保留 | m13 第三条路：纯上游 `tl.gather` |
| `python/triton/language/core.py` | `python/triton/language/core.py`(节选) | 精简为 `constexpr`(全量，因算术重载被 flip 的 SIMT 分支依赖)+`dtype`/`pointer_type`/`block_type`/`tensor`(节选)+`builtin`/`is_builtin`/`_tensor_member_fn` | 供 mem_ops/vec_ops 依赖的最小值系统 + 双标记机制基座半边 |
| `python/triton/language/semantic.py` | `python/triton/language/semantic.py`(节选) | 只留 `wrap_tensor`/`full`/`splat`/`to_tensor`/`reshape`/`broadcast_impl_value`/`not_equal`/`xor_`/`bitcast`/`cast`/`_str_to_rounding_mode`，每个函数只留本章输入会触达的分支 | mem_ops 的 `real_semantic.*` 与 vec_ops 的 `semantic.*` 共同依赖 |
| `python/triton/language/standard.py` | `python/triton/language/standard.py`(节选) | `_is_power_of_two` 全量；`xor_sum` 简化为校验+占位返回（非 must_keep 符号，见上） | flip 的 SIMT 回退分支的唯一外部依赖 |

## 已知的结构性留白（非缺陷）

- `standard.xor_sum` 的简化版不复现真实的 `core.reduce`/combine-region/
  `_generator.call_JitFunction` 机制——真实实现把一个 `@jit` 组合子内联进 MLIR
  region，这一整套是上游 Triton 通用规约算子的编译期展开细节，既不在本章
  code_spine 内，也不在 dossier must_keep 里（must_keep 只列到
  `flip`/`flip_impl`/`static_range`）。`flip_impl` 本身（must_keep，逐字保留）
  在测试里能完整跑到底，验证的是"走了几步 xor-swap、每步前后 reshape/bitcast
  是否配对"这条控制流，不是规约算子的数值语义。
- `mem_ops.py`/`vec_ops.py` 里 `@builtin`/`@_tensor_member_fn` 装饰的函数，测试
  都用关键字参数 `_builder=` 直接调用，不模拟 `CodeGenerator.visit_Call` 按
  `is_builtin(fn)` 路由到某个 builder 的分发链路（该链路已在 ch04 讲清）；同理
  字符串/整数字面量在真实 `@triton.jit` 编译路径下会被 codegen 自动包成
  `tl.constexpr` 再传入——直接调用测试里对 `overflow_mode` 这类参数显式传
  `constexpr(...)`，是在模拟 codegen 已经做过的这一步（`test_cast.py` 里有
  对应的行内说明）。
- `gather_2d_simd`（m13 第三条路）不借助宿主 pip 安装的官方 triton（版本/fork
  漂移风险，同 ch04/ch05 的一贯原则）——`test_gather_2d_simd.py` 自建一个极简
  的 numpy 驱动"解释器"站在 `triton.jit`/`tl.arange`/`tl.load`/`tl.store`/
  `tl.gather` 的位置上，只覆盖这一个 kernel 用到的算子，不是通用 Triton
  解释器。

## 验证

- `python3 -m pytest tests/` — 37 passed（纯 Python 单元测试；靠 `conftest.py`
  的 `FakeBuilder` 站在真实 `ir.builder`（C++ 绑定，host 无昇腾 NPU/CANN 工具链
  故无法拥有）位置上，验证的是"校验/位宽折叠/返回 shape/compile_hint 挂载"这些
  Python 语言层可观察行为，不模拟 MLIR/hivm op 的真实语义——IR dump 级别验证需
  真机，按 INSTANCE.md 约束不在本章测试范围）。`gather_2d_simd` 另有独立的
  numpy 解释器验证实际数值语义（见上）。
- `python3 scripts/lint_fidelity.py {chapter_dir}` — 无 BLOCKING。
