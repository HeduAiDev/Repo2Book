# ch05 图文口径同步 —— 结案(2026-07-20)。**本文前两版的结论都是错的,以本版为准。**

三轮才落定,过程本身值得留档(经验已记 exp-2026-07-20-04):

- 原正文:「7 级里暴露并校验 4 级(UB/GM/L1/L0C)」——错。
- 本文第一版提议:改成「3 级(UB/L1/L0C)」——**也错**(把「有契约」当成了「被校验」)。
- 本文第二版(Lead):「暴露 7 级 / 边校验 UB+L1 / L0C 仅契约」——**「7 级」仍错**,
  根因是我只读到 Python 侧的反射代码「遍历 __dict__」就当成 7 个全反射,没追到 pybind 那一行。

## 结案口径:**先收窄两层,再分四档**

**收窄链**(数错任何一环都会写错):
1. `.td` 定义 **7** 个:`third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td:L188-L194`
   (Zero=0 / GM=1 / L1=2 / L0A=3 / L0B=4 / L0C=5 / UB=6)
2. pybind 只导出 **5** 个:`third_party/ascend/ascend_ir.cc:L412-L417` 的 `py::enum_<hivm::AddressSpace>`
   只 `.value()` 了 L1/UB/L0A/L0B/L0C → **Zero 与 GM 不进 Python**
3. `ascend_address_space_group`(`extension/core.py:L152-L163`)遍历的是 **Python 侧** `__dict__`,
   故语言层只拿得到这 5 个;`bl.alloc` 对这 5 个原样透传、不做白名单。
   ⚠️ 「7」这个数**只存在于 .td**,把它记到 `ascend_ir.AddressSpace` 或 `core.py` 头上都是错的归属。

**四档**(按「语言层管得住的程度」排):
- 第一档 **UB / L1**:真被校验。全仓 `.space` 比较仅 5 处,全部只提这两个——
  `semantic.py:L104,L106`(copy_from_ub_to_l1)、`semantic.py:L123,L125`(copy)、`core.py:L300`(fixpipe)。
- 第二档 **L0C**:有契约、无校验。`fixpipe` docstring 要求 src 在 L0C,但代码只 `isinstance(src, tl.tensor)`,
  而 `tl.tensor` 类型上没有 `.space` 字段 → 前端无从读取。
- 第三档 **L0A / L0B**:进了 Python、可 alloc,但没有任何一条边比较它们。
- 第四档 **GM / Zero**:**根本没进语言层**。GM 在 MLIR 侧有(枚举值 1),却被 pybind 挡住 →
  kernel 里写不出 `space=GM` 的 buffer。这不是疏漏而是分工:GM↔UB 那一跳走 ch02 的显式搬运,
  以及 ch06 的索引搬运算子(`gather_out_to_ub(src=src_ptr, …)` 吃的是**基座 Triton 的裸指针**,
  不是带门牌号的 buffer)。

## 处置(已完成)

- **正文**:writer 定点改 5 处(L25 是「7 级」的真正出处 / L27 / L29 段 / L65 / L545;
  其中 L25、L65 是 writer 自己发现的,超出 Lead 派工)。九项章级门禁全绿。
  另修正一处连带错误:「后端新增一级内存 Python 自动跟上」→ 应为「**绑定层**多**导出**一级才跟得上」
  (.td 加一级而 pybind 不加,Python 什么也跟不上——GM 就是活例)。
- **dossier / explainer**:Lead 同步订正(含 explainer 的 figure_specs.claim、caption_draft、
  numbers「AddressSpace 成员数」→「7 / 5」并注明两处出处)。
- **精简版测试**:`conftest._make_address_space_enum()` 曾把 7 个名字全塞进假枚举,
  `test_address_space.py` 据此断言 `hasattr(ascend_address_space,"GM")` —— **自洽地通过、复现的却是错行为**
  (违反「只做减法」)。已派 tester 改为只造 pybind 真导出的 5 个,并**加反向断言**:Zero/GM 不存在。
- **图**:`fig-ch05-mem-hierarchy` 按四档重绘中(此前两版分别错成 4 级、3 级/7 级全反射)。
  另两张图(`fig-ch05-copy-checks` / `fig-ch05-fixpipe-pipeline`)Lead 已逐张 Read 核过,**本就正确**,不动——
  fixpipe 图的四道结构门里正确地**没有** L0C;copy 矩阵对未验证格子标「本例未覆盖(非臆造)」。
- **本章地图**:此前缺失(ch05 唯一 BLOCKING),已派 illustrator 补。

## 遗留(交 Lead/用户定夺,未动)

章标题 `显式内存层级——UB/GM/L1/L0C、…` 里的 `GM`,在新口径下略有误导(它恰是语言层**没有**的那一级)。
但该标题出自**用户已审批的** outline-final.json,且同串还出现在 roadmap.py、ARCHITECTURE.md、
ch01 的 book-map 图里——改动牵涉已审批产物与他章渲染图,**故保持原样**,由用户决定是否改成 `UB/L1/L0C`。

归档时可删本文件。
