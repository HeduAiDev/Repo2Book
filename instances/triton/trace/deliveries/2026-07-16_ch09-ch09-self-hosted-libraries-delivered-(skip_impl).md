# ch09-self-hosted-libraries-delivered-(skip_impl)

- **Type**: delivery
- **Chapter**: 09
- **Date**: 2026-07-16
- **Timestamp**: 2026-07-16T18:32:38Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch09, skip_impl, part-2-closer, standard.py, Philox, libdevice, bitonic-sort, inlining

## What happened

第九章《在 Triton 里写 Triton：自举标准库、数学/extern 与随机数》交付（Part II 收官章，kind=code/mode=skip_impl，pin triton==3.2.0 精确编译取证）。核心 stakes：证明 tl.* 的『上半部』不是黑盒——standard.py/random.py 的 softmax/sort/cumsum/Philox 大多是用 core 原语 @jit 自举出来的普通 Triton 代码，调用点会被内联进调用者 kernel 的 IR（不是免费函数调用，而是实打实的 IR 膨胀，用 n=16/64/1024 三档实测 TTIR 行数 452/779/1781、tt.call 恒为 0 量化展示）。四段：一（自举标准库）cdiv 最小自举例→softmax 减最大值数值稳定（回指 ch08 reduce 不重讲机理）→sort 完整 bitonic sort（constexpr 驱动 log2(n) 阶段定长展开，_compare_and_swap 用异或+where 实现无分支条件交换）→cumsum 借道 associative_scan（回指 ch08）。二（Philox 无状态 RNG）counter-based、umulhi+位运算，故意关 sanitize_overflow 做环绕算术（与 ch07 防溢出用法相反，对照回指）。三（数学两条路）内置数学桥到 builder.create_exp 直接建 IR vs extern_elementwise/dispatch 走外部 libdevice bitcode；extra/__init__.py 用 pkgutil 动态发现 cuda/hip 后端子模块（配对脊柱，triton-ascend 挂载点）。四（编译期诊断+优化提示）static_print/static_assert 追踪期诊断；tl.multiple_of/tl.max_contiguous 打 divisibility/contiguity 标记喂给后端 AxisInfo 静态分析（前瞻回指第 25 章）。13 机制（9 core+4 supporting/轻量），7 图+chapter-map+roadmap 全 blind PASS。禁区遵守：不重讲 ch04 的 @builtin/constexpr 两层结构、不重讲 ch07 的 load/访存/sanitize_overflow 机理、不重讲 ch08 的 reduce/associative_scan 机理（均回指不重述）。

## Why it matters

本章是 Part II（领域语言 tl.*）的收官章，把全书『constexpr 驱动追踪期特化』『@jit 内联』两条主线（ch01/ch04 立心智模型）落到一批读者天天调用却从未细看的标准库函数上，结构性证明库函数与用户 kernel 对追踪器是同一回事——解锁两个可直接落地的性能杠杆：①认清 tl.softmax/sort/cumsum 等内联进 kernel IR 会造成实打实的 IR 膨胀（块越大膨胀越猛）；②用 tl.multiple_of/tl.max_contiguous 给编译器喂对齐/连续性提示能让 load 被向量化（消费端留给第 25 章 AxisInfo）。review 提出 7 条 issue 均 negotiable/non-blocking：1 条是 m13 不变量陈述隐式而非显式（内容已在场，缺一句加粗收束）、2 条是公式规则④行内复合式重复三次的可读性打磨、3 条是 reader-comprehension 维度的术语/符号未显式绑定小卡点（bitonic sort 三层 i 语义未拆解、CAS 缩写先用后定义、AxisInfo/Coalesce 零角色提示、cdiv 公式变量 d 与源码 div 未绑定）。均判定不阻断归档。

## What to remember

ch09 done（kind=code/mode=skip_impl，Part II 收官章，pin triton==3.2.0，无精简版接口）。dossier foreshadow_due 为空（should_plant/should_recover 均空）——本章未新开 arc-map 伏笔条目、也未标记回收；AxisInfo/Coalesce 的前瞻回指（→ch25）是对已开放 f9（ch07 plant→ch25）的重复强化而非新伏笔，故 arc-map.json 未改动。glossary.json 新增 6 条术语（CAS/bitonic sort/Philox/libdevice/extern_elementwise/multiple_of·max_contiguous）。concepts.json 新增 11 条 → ch09。figures.json 新增 8 条（chapter-map + 7 个机制图）。write_review_rounds=2、blind_rounds=1（0 failures）、map_rounds=1（PASS）、无 escalation。
