# ch05 交付：显式内存层级——门牌号、订台与两条搬运边

- **Type**: delivery
- **Chapter**: ch05
- **Date**: 2026-07-20
- **Timestamp**: 2026-07-20
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, writer, reviewer, Lead, archivist
- **User present**: false
- **Tags**: triton-ascend, part-2, deep, language-layer, address-space, four-tiers, buffer-language, bl-alloc, al-copy, al-fixpipe, buffer-tensor-bridge, is-910-95, late-archive

## What happened

Part 2 语言层第二章，kind=**deep**，deps=ch04。正文 563 行，10 项章级门禁全绿，精简版 **41 tests passed**，4 图（`fig-ch05-mem-hierarchy` / `fig-ch05-copy-checks` / `fig-ch05-fixpipe-pipeline` + chapter-map）blind_review 全 PASS。

**补记归档（important）**：本章正文定稿后由 Lead 直接提交（commit `9e5a02bd`）、**当时未派归档**，故 Bible/trace 里一直没有它的条目；ch06/ch07 于 2026-07-20 归档时才发现缺口，本条为**事后补档**（ch06 归档记录里那条「地址空间四档口径（ch05 立、ch06 沿用）」的出处就是这里）。

**主线：昇腾把基座藏起来的片上内存层级，在语言层全部显式暴露。** 五站源码剖面：

1. **门牌号**（M1 `ascend_address_space`，`third_party/ascend/language/cann/extension/core.py:L143-L163`）——本章最硬也最容易数错的一条事实，见下「四档口径」。
2. **显式订台**（M2 `bl.alloc`，`python/triton/extension/buffer/language/core.py:L190-L208`，语义 `semantic.py:L35-L62`）：在指定楼层物化一块内存，返回门牌号**恒等于**所订层（恒等映射、5 个导出档原样透传、不做白名单），顺带挂 `effects` 注解——基座完全没有的一道手续。
3. **搬运边一**（M3 `al.copy`，`extension/semantic.py:L94-L129`）：`UB → {UB, L1}`。**六道校验**按序短路：`is_910_95` 芯片门禁（最先短路）∧ 两头皆 `bl.buffer` ∧ 同 shape ∧ 同 dtype ∧ `src=UB` ∧ `dst∈{UB,L1}`。不变量：`create_copy_buffer` 出现**当且仅当**六道全过（它写在全部校验之后，任一失败即 `raise`）。已弃用的旧接口 `copy_from_ub_to_l1` 是它**终点更严的真子集**（终点从 {UB,L1} 收紧为 {L1}）。
4. **搬运边二**（M4/M7 `al.fixpipe`，`extension/core.py:L273-L333`）：cube 算完堆在 L0C 的结果落回 UB，`NZ2ND` 顺带把 Fractal NZ 分形布局还原成 ND。四道结构门 + **按 dtype 分流的对齐算术**：32 位（fp32/int32）末维 `N%8`，非 NZ2ND 收紧到 `N%16`，列切分 dual 收紧到 `N%32`，NZ2DN 首维 `M%8`；16 位（fp16/int16/bf16）末维 `N%16`、NZ2DN 首维 `M%16`。落地建**六参** `create_fixpipe`（量化 / ReLU 前端写死不开）。
5. **两视角的桥**（M5 `to_tensor`/`to_buffer`/`subview`，`buffer/language/semantic.py:L87-L116`）：让吃 buffer 的搬运原语与吃 tensor 的计算算子在同一 kernel 接力。`check_subview`（`core.py:L249-L296`）**逐字保留了两处上游真实缺陷**——rank-1 分支引用从未定义的 `offset`（形参是复数 `offsets`）导致真实仓库里任何 rank-1 缓冲调 `subview()` 先炸 `NameError`；运行时 `tl.tensor` offset 直接 `return` 放弃静态对齐校验。**只解读、不顺手修好上游缺陷**。

**四档口径（三轮才落定，本章最重要的事实）**。收窄链：① `.td` 定义 **7** 个（`HIVMAttrs.td:L188-L194`，Zero=0/GM=1/L1=2/L0A=3/L0B=4/L0C=5/UB=6）；② pybind 只导出 **5** 个（`third_party/ascend/ascend_ir.cc:L412-L418` 的 `py::enum_` 只 `.value()` 了 L1/UB/L0A/L0B/L0C）——**Zero 与 GM 不进 Python**；③ `ascend_address_space_group` 遍历的是 **Python 侧** `__dict__`，故语言层只拿得到这 5 个。⚠️「7」这个数**只存在于 `.td`**，记到 `ascend_ir.AddressSpace` 或 `core.py` 头上都是错的归属。**四档**（按「语言层管得住的程度」排）：①**UB/L1** 真被校验（全仓 `.space` 比较仅 5 处：`semantic.py:L104,L106`、`L123,L125`、`core.py:L300`）；②**L0C** 有契约无校验（`fixpipe` docstring 要求 src 在 L0C，代码只 `isinstance(src, tl.tensor)`，而 `tl.tensor` 类型上**没有 `.space` 字段**）；③**L0A/L0B** 可 alloc 但无人比较；④**GM/Zero** 根本没进语言层。GM 缺席不是疏漏而是分工——GM↔UB 那一跳走 ch02 的显式搬运与 ch06 的索引搬运算子（`gather_out_to_ub` 吃的是**基座 Triton 的裸指针**，不是带门牌号的 buffer）。

**三轮走错的过程（经验已记 exp-2026-07-20-04）**：原正文写「7 级里暴露并校验 4 级（UB/GM/L1/L0C）」——错；第一版订正提议「3 级（UB/L1/L0C）」——也错（把「有契约」当成「被校验」）；第二版（Lead）「暴露 7 级 / 边校验 UB+L1 / L0C 仅契约」——「7 级」仍错，根因是只读到 Python 侧「遍历 `__dict__`」就当成 7 个全反射，**没追到 pybind 那一行**。第三版才落定成上面的收窄链 + 四档。**连带纠正一条**：「后端新增一级内存，Python 自动跟上」是错的 → 应为「**绑定层**多**导出**一级才跟得上」（`.td` 加一级而 pybind 不加，语言层什么也看不见——GM 就是活例）。

**精简版替身曾编码错模型**：`conftest._make_address_space_enum()` 一度照 `.td` 造出 7 个假枚举成员，`test_address_space.py` 据此断言 `hasattr(ascend_address_space, "GM")`——测试**自洽地通过、复现的却是错行为**（违反「只做减法」）。已改为只造 pybind 真导出的 5 个，并**新增反向断言**（Zero/GM 不存在），反射测试从子集断言 `<=` 收紧为**集合相等** `==`。

**评审**：`reviews/review-report.json` 首轮 verdict=REVISE——2 条 blocking 均在 M3/`al.copy` 段的**计数错误**（「九个调用」实为 7 行表格且两个宣称维度无实据；「五道校验」实为 6 项合取），另 3 条 non-blocking 格式建议。已修完（现正文为「七个调用」「六道校验」）。其余机制 M1/M2/M4/M5/M6/M7 逐项过账，M4/M7 计数逐行手算复核无误。chapter-map 此前缺失（本章唯一 BLOCKING），已补画。`reviews/LEAD-PENDING-FIX.md` 是这段口径同步的结案记录，可随本次归档删除。

**取证口径**：host 无 NPU/CANN，数值表来自精简版 + 只记调用/返回哨兵值的测试替身；`create_copy_buffer(handle#2, handle#5)` 这类行读作「前端校验全过、走到建 op 这一步」，不是真机 emit。hivm op 的 IR 语义与 `NZ2ND` 分形下降需真机，留 P5 的 HIVM 方言部分。

## Why it matters

ch05 是 Part 2 语言层「昇腾比基座多暴露了什么」的**内存侧地基**：门牌号（`buffer_type.space`）是后面一切搬运校验的读取对象，两条搬运边（`al.copy` / `al.fixpipe`）是 P4 PlanMemory / bind_buffer 与 P5 HIVM 方言的上游。**四档口径**已被 ch06 直接沿用（GM 不进 Python ⇒ 跨 GM 的带索引访问只能由 mem_ops 那批吃裸指针的内建承担），后续凡讲 buffer 归属或搬运方向都要引它——数错任何一环都会写错，故已完整写进 glossary。

## What to remember

- **诚实边界**：host 无 NPU/CANN；数值表来自精简版 + 记账替身；IR 语义与分形下降留 P5。
- **本章无 arc-map 正式伏笔埋/回收**（`bible.py due ch05` 两清单皆空；本书 arc-map.json 至今为空，已向 Lead 报为连贯性缺口）。前向线索：两条搬运边 → P4 PlanMemory/bind_buffer、P5 HIVM 方言；`is_910_95` 芯片门禁 → ch07 `tanh` 换菜单同一开关。
- **事实校准点（勿再回退）**：①7 只存在于 `.td`、pybind 导出 5、反射拿到 5；②四档=真校验 UB/L1 / 有契约无校验 L0C / 可 alloc 无人比较 L0A/L0B / 没进语言层 GM/Zero；③「绑定层多导出一级才跟得上」而非「Python 自动跟上」；④`al.copy` **六道**校验、`fixpipe` 建**六参** `create_fixpipe`；⑤旧接口 `copy_from_ub_to_l1` 是 `copy` 的真子集。
- **可复用经验（已进 glossary/interfaces 的 testing 约定）**：**精简版替身照「绑定层真正导出」的东西造，别照定义文件造**；能写等式断言就别写子集断言；反向断言（「什么不该存在」）与正向断言同等重要。
- **⚠️ 遗留，交用户定夺、未动**：章标题 `显式内存层级——UB/GM/L1/L0C、buffer 语言与 copy/fixpipe` 里的 `GM` 在新口径下略有误导（它恰是语言层**没有**的那一级）。但该标题出自**用户已审批的** `outline-final.json`，同一串还活在 `roadmap.py`、`ARCHITECTURE.md`、ch01 的 book-map 图里，改动牵涉已审批产物与他章渲染图，**故保持原样**。**这条尚未解决**，由用户决定是否改成 `UB/L1/L0C`。
- Bible 回写：glossary +11 条（并把 AddressSpace 条目重写为 ch05 出处版、补全四档明细）、concepts +16 条（并把「AddressSpace 七档/五档」概念的归属从 ch06 改正为 ch05）、figures +4 条、interfaces 登记 ch05 精简版签名与 testing 约定。
