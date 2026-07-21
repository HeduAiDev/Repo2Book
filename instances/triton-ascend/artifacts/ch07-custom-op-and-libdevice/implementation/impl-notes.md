# ch07 精简版实现说明——register_custom_op 与 libdevice 三分野

## 范围

本章精简版覆盖 dossier `mechanisms`(m1-m8)的两条主线：

1. **昇腾语言层多出的『注册自定义算子』能力**——`third_party/ascend/language/cann/
   extension/custom_op.py` 的 `register_custom_op`/`custom`/`custom_semantic`/
   `_get_op_class`/`_make_attrs` 全流程，以及 `builtin_custom_ops.py` 的真实内建
   样例 `_index_select`。
2. **libdevice 数学库的三条实现路径**——`libdevice.py` 的 `reciprocal`(最简 extern)/
   `tanh`(双分支 extern)/`acos`(纯 IR 多项式逼近)三个代表，加 `extension/
   math_ops.py` 的 `isfinited`(第三条路：`@jit` 组合原语)，以及 `cann/__init__.py`
   的 import 期命名空间覆盖挂载(m8)。

`bitcode`/`indexing_map` 的下降语义、hivm.CustomOp 在 MLIR pass 里的具体 lowering
不在本章范围——按 dossier `forward_defer_to_P5` 批准，本章只讲"这些参数存在、如何
挂成 IR 属性"这一层注册表面。

## Source Map

| 精简版位置 | 对应真实源码 | 改动 | 原因 |
|---|---|---|---|
| `extension/custom_op.py` 全文件(除 `_to_value`) | `third_party/ascend/language/cann/extension/custom_op.py` | 原样保留(含 `_bind_op_arguments`——真实源码里定义但从未被调用的方法，同样原样保留，不擅自判断"没用就删") | 本章主角，逐段都要讲 |
| `extension/custom_op.py::_to_value` | 同上 `:L66-111` | 按 subtraction_plan 批准：删掉 int64/uint64/int32/uint32/int16/uint16/int8/uint8(int 分支)与 fp64/fp32/fp16/bf16(float 分支)共 12 条同构精确匹配分支，只留骨架 + int→int32/float→fp32 两条默认路径 | 同一"按 dtype 找 builder 工厂方法"范式的平坦重复，删除不影响 custom_semantic 主数据流理解 |
| `extension/custom_op.py::_dtype_cname_dict`/`_cname` | 同上 `:L348-374` | 原样保留 | dossier 未批准删除，虽与注册-调用主数据流关联不大，但不擅自裁剪 |
| `extension/builtin_custom_ops.py::_index_select` | `third_party/ascend/language/cann/extension/builtin_custom_ops.py:L30-103` | 原样保留(docstring 里的 rank 逐项 Reference formula 按 elide 批注删除，正文摘 1~2 行即可) | 唯一保留的完整 register_custom_op 真实样例 |
| ~~`_index_put`/`_gather_load`/`_scatter_store`~~ | 同上 `:L106-220` | 整体删除 | 按 subtraction_plan 批准：四个内建算子结构同构，保留 `_index_select` 一个即展示用法 |
| `extension/core.py::CORE`/`PIPE`/`MODE`/`int64` | `third_party/ascend/language/cann/extension/core.py:L93-125` | 枚举成员用同名字符串(`"VECTOR"`/`"PIPE_V"`/`"SIMT"`/...)顶替真实 C++ 绑定值 `ascend_ir.CoreType.VECTOR` 等(host 无昇腾 NPU/CANN 工具链，编译产物不可复现)；`int64.__new__` 原样保留 | register_custom_op 的必填三要素 + arg_type 动态定型的逃生类型。**订正(2026-07-20 lint_fidelity 复核)**：此前用一个额外的 `_AscendIrEnumValue` 桩类包装字符串，该类/`__init__`/`__repr__` 都不在 pin 里，是纯粹的自造抽象——`_make_attrs`/测试全程只对 `.value` 做相等断言，从不依赖它的具体类型，故按"只做减法"原则删掉这层包装，直接用字符串做枚举值 |
| `extension/core.py::builtin`/`is_builtin` | 同上 `:L66-90` | 原样保留 | `custom_op.py` 的 `@core.builtin def custom(...)` 直接依赖，删了会连累主角报 AttributeError(实测踩到，见下方"已发现并修正的问题") |
| ~~`extension/core.py` 其余内容~~(copy/fixpipe/sync_block_*/IteratorType/Fixpipe 系列枚举/ascend_address_space) | 同上(约 368 行全文件) | 整体删除 | 归 ch04(双标记本身)/ch05(内存层级)/P5(hivm op 语义)，与本章 core/pipe/mode 三要素无关 |
| `extension/_utils.py::_is_int_like_elem`/`_assert_int_like_tuple` | `third_party/ascend/language/cann/extension/_utils.py:L18-33` | 原样保留 | `_index_select.__init__` 的形状/dtype 校验依赖 |
| ~~`_utils.py::custom_op`/`_convert_elem_to_ir_value`~~ | 同上 `:L5-15`,`:L36-54` | 删除 | 块间同步/块指针辅助函数，`_index_select` 不依赖 |
| `libdevice.py::reciprocal`/`tanh`/`acos` | `third_party/ascend/language/cann/libdevice.py:L28-34`,`L81-93`,`L215-273` | 原样保留 | dossier must_keep：三条实现路径的代表 |
| `libdevice.py::isnan`/`isinf`/`atan` | 同上 `:L127-134`,`L54-61`,`L73-79` | **额外保留**(subtraction_plan 未明确列出保留，但删除会破坏 must_keep) | `extension/math_ops.py`(must_keep 符号 `isfinited` 所在文件)顶部 `from ..libdevice import atan, isnan, isinf` 是模块级 import，字面删除这三个函数会让 `math_ops.py` 在加载期直接 `ImportError`，连带炸穿 `isfinited`。按"不能删到 must_keep 跑不起来"的原则修正，见下方说明 |
| ~~`libdevice.py` 其余约 27 个数学函数~~(log1p/relu/tan/ilogb/ldexp/pow/div_rz/fast_dividef/fast_expf/fmod/float_as_int/atan2/trunc/round/sinh/cosh/acosh/asinh/atanh/expm1/nextafter/hypot/cyl_bessel_i0/signbit/erfinv/gamma/lgamma/nearbyint/asin/log10/copysign/rint) | 同上(全文件约 1032 行) | 整体删除 | 按 subtraction_plan 批准：同一"extern 调 __hmf_ / 无符号退回纯 IR"范式的重复实例；erfinv/gamma/cyl_bessel_i0 还额外带数值逼近系数数组/迭代展开体，属特定数值分析细节 |
| `extension/math_ops.py` 全文件 | `third_party/ascend/language/cann/extension/math_ops.py` | 原样保留(全文件仅 3 个函数) | dossier 未批准任何删除 |
| `cann/extension/__init__.py` | `third_party/ascend/language/cann/extension/__init__.py`(约 156 行) | 大幅精简：只重导出 `CORE`/`PIPE`/`MODE`/`int64`/`custom`/`custom_semantic`/`register_custom_op`/`builtin_custom_ops`/`math_ops`/`atan2`/`isfinited`/`finitef` | 其余 scope/aux_ops/vec_ops/mem_ops/affine 绑定各归其它章节 |
| `cann/__init__.py` | `third_party/ascend/language/cann/__init__.py`(约 52 行) | 删掉 `extension.parallel = extension.aux_ops.parallel`、`libdevice.flip = extension.flip`(aux_ops/vec_ops 越界，本章 extension 包也未重导出它们)；`libdevice.X = math.X` 的 15 行同构复用只留 `sqrt`/`abs` 两行代表 | m8"覆盖 vs 复用"这条分支主线不受影响；保留的 `isfinited`/`finitef`/`atan2` 覆盖三行是 m8 真正的内容 |
| `python/triton/language/{core,semantic,math}.py`、`python/triton/runtime/jit.py` | 基座 triton 同名文件(节选) | 大幅精简为最小真实子集(同 ch04/ch05 对基座支撑层的处理方式) | 供 custom_op/libdevice 间接依赖的值系统(`tl.tensor`/`tl.dtype`/`tl.constexpr`)与标量算子(`add`/`sub`/`mul`/`truediv`/`less_than`/`where`/`abs`/`sqrt`)，本身不是本章机制 |

## 已发现并修正的问题(供复核)

1. **`extension/core.py` 最初漏留 `builtin`/`is_builtin`**：起草时误判它们"只是 ch04
   双标记机制的内容"整体删除，实测 `custom_op.py` 的 `@core.builtin def custom(...)`
   立即 `AttributeError`。已修正为保留(见上表)。
2. **`libdevice.py` 的 `isnan`/`isinf`/`atan` 三个函数**：subtraction_plan 的删除
   清单("除 reciprocal、tanh、acos 外的其余数学函数")字面上包含它们，但
   `extension/math_ops.py`(must_keep 符号 `isfinited` 所在文件)顶部有
   `from ..libdevice import atan, isnan, isinf` 模块级 import——删掉会导致
   `math_ops.py` 加载期 `ImportError`，连带 `isfinited` 无法工作。按"must_keep 优先
   于字面 delete 清单"的原则修正保留这三个函数(每个都是与 reciprocal 同构的
   ~7 行 extern 包装，不引入新范式)。**这是对 dossier subtraction_plan 的一处必要
   订正，非擅自扩大保留范围**——已在 `libdevice.py` 顶部注释与本文件 Source Map
   中说明，供 writer/reviewer 复核。

3. **`lint_fidelity` BLOCKING 复核(2026-07-20，10 处缺 `# SOURCE:`)**：逐个判定是
   "漏标注"还是"误加的抽象"，不为过 linter 而伪造行号：
   - **漏标注(补标即可)**：`python/triton/language/core.py` 的 `extern_elementwise`
     (pin L2691-L2730，之前只在段前写了大段 SUBTRACTED 注释，没在 `def` 自身行落
     `# SOURCE:`)；`builtin(fn)` 内层 `wrapper`(pin L30-35)；
     `third_party/ascend/language/cann/extension/core.py` 的 `builtin(fn)` 内层
     `wrapper`(pin L75-83)与 `int64.__new__`(pin L98-101)；
     `python/triton/language/math.py` 的 `_check_dtype` 内层 `wrapper`(pin L19-33)
     与 `check`(pin L22-31)——这 6 处都真实存在于 pin，只是 AST 意义上的"这个
     def/class 自己的行区间"里没有 `# SOURCE:` 字面量(外层函数上的标注不算数)，
     现已在每个内层函数自己的 `def` 行补上。
   - **误加的抽象(删掉，不补假 SOURCE)**：
     a) `python/triton/language/core.py::constexpr.__hash__`——已 grep 确认 pin 的
        `constexpr` 只有 `__eq__`/`__ne__`/`__bool__`，没有 `__hash__`；本章精简版
        也从未把 `constexpr` 实例当 dict key/放进 set。删除，原地留
        `# SUBTRACTED:` 说明"pin 没有，故不补"。
     b) `third_party/.../extension/core.py::_AscendIrEnumValue`(及其
        `__init__`/`__repr__`)——pin 里根本没有这个类，它是此前为了给
        `CORE`/`PIPE`/`MODE` 的枚举值提供一个"看起来像 C++ 绑定对象"的桩而多造的
        一层包装。复核 `_make_attrs`/全部测试后确认：下游只对
        `op.core.value`/`C.CORE.CUBE.value` 做相等比较，从不依赖该值的具体类型或
        `.name`/`repr()`。按"只做减法"原则删掉这层多余抽象，枚举值直接用同名
        字符串(`"VECTOR"`/`"PIPE_V"`/...)，`46 passed` 不受影响。

## 简化(非删减)说明

- `python/triton/language/core.py` 的 `static_assert`：真实实现挂 `@builtin` 装饰器
  (要求调用点传 `_builder`)，函数体本身是 `pass`(不做任何校验)。本章测试直接调
  `math_ops.isfinited.fn(x)` 复现 `@jit` 函数体的真实组合逻辑(不驱动完整 codegen
  trace，同 ch04 对 `JITFunction` 的处理方式)，这条路径下 `_builder` 不会被自动
  注入。去掉 `@builtin` 包装只是让它能在"不经 codegen trace"的测试环境下被直接
  调用，不改变它"什么都不做"的真实行为。
- `python/triton/language/core.py::extern_elementwise`：真实实现还做隐式广播
  (block 参数形状对齐)与"精确 dtype 不匹配时退化为算术类型提升再查表"的宽松匹配。
  本章 libdevice 样例(reciprocal/tanh)调用点的实参 dtype 总与 `arg_type_symbol_dict`
  某个 key 精确相等(标量、不分块)，故只保留"精确 dtype 元组查表 -> 调
  `create_extern_elementwise`"这条主干——这也正是本章要讲的机制本身:『extern 只能
  调预置符号』。

## 验证

- `python3 -m pytest tests/` —— 46 passed(纯 Python 单元测试，靠 `conftest.py` 的
  `FakeBuilder` 站在真实 `ir.builder`(C++ 绑定，host 无昇腾 NPU/CANN 工具链故无法
  拥有)位置上)。
  - `create_custom_op`/`create_extern_elementwise`/`get_core_type_attr`/... 等
    只有编译期 MLIR/C++ 才能回答"IR 属性长什么样"的方法做成"记录调用 + 返回可预测
    哨兵值"——验证的是"调用被路由到哪个符号/建了哪个属性"，不模拟 MLIR 语义(同
    ch04/ch05 的既有测试哲学)。
  - `create_fadd`/`create_fsub`/`create_fmul`/`create_fdiv`/`create_fcmpOLT`/
    `create_select`/`create_fabs`/`create_sqrt` 做成**真的浮点算术**(handle 直接是
    Python float)——这是本章唯一"数值可验证"的部分：`acos` 的纯 IR 多项式逼近本身
    是可在 CPU 上复现的数学，不依赖昇腾硬件。测试验证了这条精简版路径在
    `x ∈ {0, ±0.2, ±0.4, 0.55, ±0.7, ±0.85}`(跨 center/mid 两个子分支)上与
    `math.acos` 的差异 < 2e-3。
  - IR dump / MLIR 语义级别的验证需要真机(昇腾 NPU/CANN 工具链)，按 INSTANCE.md
    约束不在本章测试范围。
- `python3 scripts/lint_fidelity.py {chapter_dir}` —— 见执行记录。
