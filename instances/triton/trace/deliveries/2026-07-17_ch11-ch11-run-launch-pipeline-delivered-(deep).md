# ch11-run-launch-pipeline-delivered-(deep)

- **Type**: delivery
- **Chapter**: 11
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch11, deep, part-3, JITFunction.run, driver-boundary, launch-cache, CompiledKernel, lazy-handles, headless

## What happened

第十一章《run()：从缓存查询到编译再到内核发射的一次完整 launch》交付（Part III，kind=deep/mode=skip_impl，pin triton==3.2.0 精确编译取证；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 Lead 手工串行补归档）。全书发射编排主脊：`JITFunction.run` 六段——① `driver.active` 取 device/stream/target + `make_backend`（跨到 driver 子系统边界，host 无 GPU 断裂点）→ ② 惰性 `create_binder` 调 `self.binder` 得 5 元组 → ③ 拼内存 `cache[device][key]`（键构成见 ch10）查询，命中直达⑧ → ④ 未命中慢路径：`parse_options`→None→\*i8 签名修正→`get_attrs_descriptor`+`get_constants` 组 constants→`_call_hook`→`ASTSource`→`compile()`（内部留给 ch14）→回填缓存 → ⑤ `used_global_vals` 核对（机制属 ch10） → ⑥ 非 warmup 才规范化 grid → ⑦ `kernel.launch_metadata`+`kernel.run` 跨语言发射给 C++ launcher → 收口 `CompiledKernel.__getattribute__` 拦截 `.run` 触发 `_init_handles`（惰性把 cubin 真正装到 GPU，编译期与设备解耦）。13 机制(8 core+5 supporting)。性能落点：稳态命中的 Python 侧固定开销与 kernel 大小无关（headless warmup 路径实测约 4.398 µs 下界），是判断小算子被发射开销主导而非算力主导的判据。无精简版（run() 是编排胶水，牵动整条编译栈，skip_impl 处理）；6 图（chapter-map + 5 个机制图：launch-spine/driver-boundary/slowpath/emission-crosslang/lazy-handles）全 blind PASS（round1 抓出 fig-ch11-slowpath 的 claim「6 样编译输入」与图面/其余素材「5 样」矛盾，已修 explainer.json claim 字段，图未重画）；review APPROVED(7 negotiable/non-blocking issues：2 条 algorithm-pedagogy/格式类（逐机制勾选表存档 + 3 个 core 机制缺显式『不变量』收口句）+5 条 reader-comprehension 小卡点：特化位与特化描述子未打通/总览表 'D' 符号先用后解释/4.398µs 测量口径迟至第392行才补/98.379ms 跨小节复用数字未点明出自另一 constexpr 例子/headless 首现未即释)；write_review_rounds=3、blind_rounds=2(round1 1 failure已修/round2 0 failure)、map_rounds=1(PASS)、无 escalation。禁区遵守：不重讲 ch10 的 binder 代码生成/缓存键三桶/used_global_vals 机制本体（回指不重述）、不展开 driver 抽象/后端发现/autotune/磁盘缓存（留给 ch12）、不展开 compile() 内部五段驱动主循环（留给 ch14）。

## Why it matters

本章是全书发射路径的胶水枢纽——把 ch10 建好的 binder/缓存键、ch12 的 driver/后端、ch14 的 compile 主循环，用一个方法串成一次真实 launch，并首次把「无 GPU 断裂线」精确钉死到 `CompiledKernel._init_handles` 的 `load_binary` 这一具体方法（坐实 ch03 早先点名但未展开的同名概念）。新开两条正式伏笔 f11(driver 边界→ch12 展开)/f12(compile 五段主循环→ch14 展开)——不同于 ch09 对已开放 f9 的重复强化，这两条是 ch11 首次做出的、有明确下游依赖的具体承诺（outline 中 ch12/ch14 均 deps=["ch11"]）。

## What to remember

ch11 done（kind=deep/mode=skip_impl，Part III 承上启下章）。glossary.json 110→120（新增 10 条：JITFunction.run/driver.active/make_backend/create_binder/内存 launch 缓存(cache[device][key])/launch_metadata/kernel.run(跨语言发射一跳)/get_attrs_descriptor·AttrsDescriptor/headless/_init_handles；并enrich 既有 CompiledKernel 词条补 ch11 坐实标注）。concepts.json 新增 6 条→ch11（run 编排主脊、两层缓存正交、driver 边界跳转、跨语言发射一跳、CompiledKernel 惰性设备句柄、小 kernel 发射受限判据）。interfaces.json 新增 ch11 键（源码接口，非精简版）：`JITFunction.run` 签名、binder 5 元组契约、`kernel.run` 发射签名、`_init_handles`、driver 边界调用面——供 ch12/ch14 回指。arc-map.json 新增 f11(plant ch11→payoff ch12，driver 子系统边界)、f12(plant ch11→payoff ch14，compile 五段主循环)，`bible.py due ch12`/`due ch14` 已验证能正确捞出。reviews/review-report.json 与 run-ledger.json 由 Lead 预写，本次未改动。
