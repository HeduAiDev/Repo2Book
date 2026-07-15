# ch04 tl surface and constexpr delivered (skip_impl)

- **Type**: delivery
- **Chapter**: 04
- **Date**: 2026-07-15
- **Timestamp**: 2026-07-15T18:41:04Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch04, skip_impl, tl-namespace, constexpr, static-range, builtin-marker, tensor-member-fn

## What happened

第四章《Part II（领域语言 tl.*）开篇：tl.* 表面装配与 constexpr 编译期/运行期分野》交付。kind=skip_impl(取景框章,正文直接内嵌真源码，无独立精简版重实现)。与 ch01 划清边界：不复述 visit_Call 三岔分发/@builtin 与 @jit 两套实现策略/_builder 注入/constexpr 折叠等 ch01 已讲透的机制，只在需要处一句回指。本章增量：①tl.* 表面怎么铺——python/triton/language/__init__.py 把 core/standard/math/random 四段 re-export 汇聚成 tl.*(18/81/17/10=126 项)，__all__ 131 项门面账目对平(+3 陈旧死名+2 子模块挂载)；triton/__init__.py 顶层暴露 triton.jit/cdiv 等。②装饰器标记服务 AST 识别——_tensor_member_fn(把自由函数挂成 tensor 方法、返回 fn 本身)、is_builtin/@builtin 如何被 CodeGenerator 分岔(回指 ch01)，@builtin 脱离 @triton.jit 直接调用报错归位为 tl.* 调用契约。③constexpr 讲透(核心)——core.py 里重载全部 dunder 的包装类，__index__/__bool__ 是刻意留的出壳口(算术类 dunder 结果仍裹回 constexpr)，static_range(纯编译期全展开，0 scf.for/8 arith.addi)对照 range(保留 1 scf.for、携 num_stages/loop_unroll_factor 两个后端调度旋钮喂给流水线 pass，性能主线落点)。pin triton==3.2.0 headless 编译实测取证 IR 层面的展开/循环差异与 constexpr dunder 出壳行为。4 张图(fig-tl-namespace/fig-constexpr-forward/fig-static-vs-range/chapter-map)1 轮盲审全 PASS，本章地图 1 轮 PASS，写作-评审 2 轮收口，APPROVED（5 条 issue 全 negotiable/non-blocking：1 处代码块跳过未标注注释、§1 缺不变量标签、AOT str_to_ty 缺前向指针、§6 visit_For 与 ch01 重合缺桥接句、extra 子模块未先解释）。

## Why it matters

本章是 Part II 领域语言的取景框——把读者此后逐章要用到的 tl.* 表面与 constexpr 编译期/运行期分野一次性讲透，是 BLOCK_SIZE 等 tl.constexpr 标注为何是性能旋钮、range 的 num_stages/loop_unroll_factor 提示为何直接影响后端调度这两条读者收益主线的落点；同时确立与 ch01 的边界处理范式(增量优先、重合处一句回指)，供后续 tl.* 各章(ch05-ch09)沿用。

## What to remember

ch04 done(skip_impl,取景框章,无精简版接口可登记——narrative 直接内嵌真源码，bible.py due ch04 为空，无应埋/应回收伏笔正式登记；dossier.foreshadow_due.note 给了非正式建议：num_stages/loop_unroll_factor→pipeline pass 章回收、constexpr 折叠→autotune 章回收、str_to_ty→类型系统章回收，供 Lead 后续在大纲/依赖图层面补登)。4图入figures.json(tl-namespace-assembly/constexpr-dunder-class/static-range-vs-range/ch04-source-profile-map)。concepts.json 新增 8 项本章新建立术语(tl.* 命名空间装配/is_builtin 标记分派/_tensor_member_fn 双调用/constexpr 全dunder转发类/出壳口/_constexpr_to_value 兜底/static_range 全展开/range 后端提示)。review APPROVED,5条issue全negotiable/non-blocking(reviewer 报告与 run-ledger 已按上游对象原样落盘，不做改写)。
