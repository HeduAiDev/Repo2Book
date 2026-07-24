# ch33 能力边界：测试套件揭示的支持/未支持谱系（全书收官）

- **Type**: delivery
- **Chapter**: ch33
- **Date**: 2026-07-24
- **Timestamp**: 2026-07-24T00:00:00Z
- **Agents involved**: writer, reviewer, illustrator, archivist
- **User present**: False
- **Tags**: capability-boundary, testing, skip-xfail-skipif, honesty, capstone, 全书收官

## What happened

全书**最后一章**(kind=meta，deps=ch03；对位基座「度量与实战」一部)归档：不讲新机制，把 `third_party/ascend/unittest/` 下的测试当成一份「能力谱系文档」逐条核实。8 机制(m1-m4 core + m5-m8 supporting)全覆盖：①**对拍判据 validate_cmp**——按 dtype 三档容差(fp16/bf16 1e-3、fp32 1e-4、整型逐位相等)把「支持」钉成可执行定义；②**支持面正面清单**——`third_party/ascend/unittest/` 下共 323 个 `.py`，本章聚焦的三个子目录 `pytest_ut`(297)+`autotune_ut`(13)+`custom_op`(7) 合计 317 个，横跨 tutorials 01-18/逐算子/昇腾专属扩展/custom_op 四大分区；③**未支持面反面清单**——无条件 skip/xfail 的活跃文件 22 个(20+2+0)，`pytest_ut` 生效标记 40 处，按 reason 归堆五主类(等 TA 13、等 bishengir 9、NPUIR 回退 5、UB overflow 3、attn_cp 整批 3，合计 33)+ 零星半支持 6 + 唯一 xfail 哨兵 1；④**边界分层归因**——同为 skip，五主类分落上游软件层/闭源编译器层/版本回退层/硬件资源层/整块未纳管五层，卡在哪一层决定读者该等版本、等编译器、换 shape 还是这功能压根没纳 CI；⑤**skip vs xfail vs skipif 语义分野**——skip 是止血(报告符恒为 `s`)、xfail 是唯一能自发翻转成 `XPASS` 的回归哨兵、skipif 是硬件条件跳(边界随硬件浮动，如 `only support A5`/910_95 系列，不算「不能」)；⑥**skip 的粒度**——整测级/`pytest.param` 级/skipif 硬件级/docstring 说明级四种，粒度越细支持面守得越大；⑦**flaky 半支持**——`randomly failed`/`expm1 failed sometimes` 是「做得了但不够稳」，与 `test_atomic_cas.py` 那条确定性的 `full tensor has problem` 分开，且把被注释掉、不生效的 `test_3Dgrid.py` marker 排除在 40 之外；⑧**三面俱到收官**——能(对拍+正面清单)/不能(skip/xfail 标出并归因)/未证(参数矩阵未触及的路径，如 ch32 fused_attention 的 `causal=True` 分支源码在但未被真机对拍覆盖)，**回收伏笔 f8**(plant ch32→payoff ch33)。方法边界：`conftest.assign_npu` 是 module-scope autouse 夹具，host 上没有 CANN/NPU 整套测试根本跑不起来，本章全部数字是**静态读**标记与 reason 字符串得出、非真跑复现，独立复算(`census_skip_markers.py` 对准 `instances/triton-ascend/source/third_party/ascend/unittest`)与正文逐一吻合。4 张机制图(fig-m2-support-census/fig-m3-unsupported-census/fig-m4-boundary-layers/fig-m5-marker-pipeline)+ roadmap + chapter-map 全部独立盲审 PASS，write↔review 3 轮收敛，16 门禁全绿，verdict **APPROVED**，0 blocking + 6 non-blocking(1 处 algorithm-pedagogy 风格：m6/m7/m8 起段未统一「直觉:」标签；1 处 algorithm-pedagogy：attn_cp 锚点只转述未内嵌代码块；4 处 reader-comprehension：gather 整测级举例与正文「已跑通」陈述表面矛盾/fig-m3 图注「哨兵」术语先于正文定义/skipif 表格 A5 与 910_95 两个代号首次并列未展开/因果掩码 off-band·on-band 跨章回收未复述——均留存量回修批次，不影响本轮通过)。Bible 回写：glossary 新增 4 条(validate_cmp/skip-xfail-skipif 三分/attn_cp/三面俱到)+ 对既有 `A2_A3 / 910_95` 词条补 A5 别名说明防漂移；concepts.json 新增 7 条机制登记 ch33；figures.json 新增 4 条机制图登记；arc-map.json 的 f8 状态改为 resolved(resolved_in: ch33)。**全书 33 章至此全部归档完成。**

## Why it matters

一本源码解读书的收官章不该只讲「能做什么」，更要老实标出「还不能做什么、卡在哪一层」——本章把这份诚实做成了全书方法论的收尾示范：测试套件是维护者在提交那一刻对「此刻真过不了」的当场承认，比任何文档都新鲜。三面俱到(能/不能/未证)框架回应了全书从「只做减法、不杜撰」到「交叉验证靠什么、证明不了什么」的一贯纪律；f8 的回收也让 ch32 capstone 章留下的方法论悬念(测试覆盖到底证明了什么)有了系统性的收口。

## What to remember

ch33 是全书最后一章，APPROVED，0 blocking + 6 条 negotiable 建议留存量回修；伏笔 f8 已回收(arc-map 全部 8 条伏笔至此全部 resolved)；Bible(glossary/concepts/figures/arc-map)已回写；33/33 章全部归档，《triton-ascend 源码解读》主线写作完成。
