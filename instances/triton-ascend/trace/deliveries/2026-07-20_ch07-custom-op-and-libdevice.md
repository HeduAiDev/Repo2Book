# ch07 交付：自定义算子框架与 Ascend libdevice——register_custom_op 与数学库

- **Type**: delivery
- **Chapter**: ch07
- **Date**: 2026-07-20
- **Timestamp**: 2026-07-20
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, writer, reviewer, Lead, archivist
- **User present**: false
- **Tags**: triton-ascend, part-2, deep, language-layer, register-custom-op, custom-semantic, core-pipe-mode, hivm-customop, libdevice, hmf-symbols, extern-elementwise, open-set-vs-closed-set

## What happened

Part 2 语言层第四章（物理章号 ch07），kind=**deep**，deps=ch04。正文 570 行，10 项章级门禁全绿，精简版 **46 tests passed**，5 图（4 机制图 + chapter-map）blind_review 全 PASS。verdict=**APPROVED**（0 blocking / 7 non-blocking）。run_ledger：write↔review 3 轮、map 2 轮；测试站曾有一次 `test-exhausted` 逃生，以 `skip_dossier=true` 续跑（既有 dossier 已过 lint_dossier）。以 `skip_archive=true` 并行发车。

**主线（两条）**：

**一、注册**。`register_custom_op`（`third_party/ascend/language/cann/extension/custom_op.py:L324-L345`）是一道**类装饰器**闸门（`assert inspect.isclass(op)`——装在类上，不是函数上）。八条断言查「必须是类 / 名字不重 / `core`、`pipe`、`mode` 三要素齐且分别是 `CORE`/`PIPE`/`MODE` 枚举实例」；全过了才 `inspect.signature(op)` 抄一份签名存进类、往 `_custom_op_registry` 写一条。过不了闸门，注册表一个字都不写。调用侧 `al.custom`（同文件 `L294-L321`）六步：查表 → 实例化跑算子自己的 `__init__` 校验 → 摊操作数 → `_make_attrs`（`L245-L271`）把 `core`/`pipe`/`mode` 翻成 `hivm.tcore_type`/`hivm.pipe`/`hivm.vf_mode` 三条 IR 属性 → emit 一条 `hivm.CustomOp`（走 ch04 的双 builder）→ 结果转回 tensor。`__builtin_` 前缀是随包自带算子的通行证——免注册、免 `symbol` 与 `bitcode`；用户自己的算子没有这层豁免。真实样例 `_index_select`（`builtin_custom_ops.py:L74-L103`，VECTOR/PIPE_V/SIMT）。参数动态定型走 `self.arg_type` + `al.int64`（Python int 默认转 i32）。

**二、libdevice 比想象中厚**。`third_party/ascend/language/cann/libdevice.py`（1032 行、37 个顶层函数）不是一层转发壳，是**四类形态的拼装**：14 个只有一张符号菜单、2 个（`tanh`/`pow`）按 SIMT 开关与芯片型号换另一张菜单、18 个在符号与纯 IR 之间分流（最常见）、3 个全程纯 IR 从不点符号；另有 `extension/math_ops.py` 里 3 个用 `@jit` 组合已有原语。菜单上有的就零算术直调 `__hmf_` 符号（全库 **66 处引用、去重 60 个不同符号**），没有的就用几十条 IR 指令做多项式逼近——`acos` 的逼近在八个采样点上与标准库最大绝对误差 1e-05。最后由 `cann/__init__.py:L27-L52` 在 import 期把 17 行基座 `math` 复用与昇腾差异实现拼成同一个 `al.libdevice` 命名空间（`al.libdevice.exp` 这类 `libdevice.py` 里根本没定义的名字由此而来）。

**核心对照（本章论点）**：基座 `extern_elementwise`（`python/triton/language/core.py:L2690-L2730`，主干在 `dispatch` L2647-L2687）只能**点菜**——把实参 dtype 拼成元组去查写死在函数体里的静态菜单，查不到就 `raise`。不变量：**可引用符号集合恒等于定义处那张菜单的值域，调用期不可扩充**（字典是字面量、无插新键路径；取符号的唯一出口是那次下标，前置 `if not in: raise` 挡掉出界）。昇腾多出的 `register_custom_op` 则是**自带菜谱**——注册表可以一直长。一句话：**基座只能点菜，昇腾能自带菜谱；闭集 vs 开集**。

**取证口径**：host 无昇腾 NPU/CANN，精简版为纯 Python 单元测试，真实 `ir.builder` C++ 绑定由 FakeBuilder 站位；七轮 dtype/开关实测表（`reciprocal`/`tanh` × fp32/fp16/bf16 × simt/910_95）来自精简版，正文已就近说明：精简版按批准减法只留精确元组查表主干、查不到抛 `KeyError`，**真实源码抛 `ValueError` 且多参数场景先做隐式广播与类型提升再查表——异常类型以真实源码为准**。

## Why it matters

ch07 收束了 Part 2「语言层昇腾增量」的最后一块可扩展性问题：算子不够用时，用户能不能自己往语言里加一条。它给出的答案（能，代价是必须回答「跑在哪个核、占哪条流水线、什么执行模式」）把 ch02 的硬件模型直接顶到了语言表面——**硬件模型的差异最终会长成语言表面的差异**，这条全书线索在语言层的又一次现形。ch04 的双 builder 在此兑现为 `create_custom_op` 的 emit 路径。

本章明确**留给 P5**：`bitcode` 的加载/下降语义、`indexing_map`（`al.affine_map` / MLIR AffineMap）与 `iterator_types` 的下降语义、`hivm.CustomOp` 在 MLIR pass 中的 lowering。P5 相关章开工时应从此处接。

## What to remember

- **诚实边界**：host 无 NPU/CANN；数值表来自精简版 + FakeBuilder，标『需真机』的不越界解读；异常类型以真实源码为准。
- **本章无 arc-map 正式伏笔埋/回收**（`bible.py due ch07` 两清单皆空）。前向线索：bitcode / indexing_map / hivm.CustomOp lowering → P5；语言层下一站是 scope 与核间同步（ch08）。
- **事实校准点（勿再回退）**：①`register_custom_op` 是**类装饰器**，必填 core/pipe/mode 且必须是枚举实例；②`__builtin_` 前缀豁免的是 `symbol` 与 `bitcode`；③`__hmf_` = Huawei Math Function 前缀，66 处引用 / 60 个去重符号；④libdevice 是 **37 个函数、四类形态**（14/2/18/3 + math_ops 3），不是「一层壳」；⑤`dispatch` 在 L2647-L2687、`extern_elementwise` 本体在 L2690-L2730，两个行号都对、别互相顶替。
- Bible 回写：glossary +15 条、concepts +12 条、figures +5 条、interfaces 登记 ch07 精简版签名（规范路径前缀，按真实包树）。
