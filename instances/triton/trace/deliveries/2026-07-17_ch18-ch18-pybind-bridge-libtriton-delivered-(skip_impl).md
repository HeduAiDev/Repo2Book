# ch18-pybind-bridge-libtriton-delivered-(skip_impl)

- **Type**: delivery
- **Chapter**: 18
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch18, part-4, pybind11, libtriton, TritonOpBuilder, PYBIND11_MODULE, backend-seam, skip_impl

## What happened

第十八章《双语桥：libtriton 的 pybind11 绑定层》交付（Part IV 收尾，全书首次拆开 `python/src/*.cc`，坐实前几章反复借用的 `_builder.create_*` 的 C++ 真身；并行发车 skip_archive 模式，Review+Map 已 APPROVED，本次由 archivist 串行补归档）。核心机制：①`create_*` 双语绑定链——`.def("create_make_range", lambda)` 挂法名，lambda 算返回类型后落到公共底座；表格逐层拆开 Python 前端→pybind11 派发→lambda 算类型→`create<OpTy>`→MLIR op 产出→返回 Python 六层，`ir.cc` 里数得 129 个 `create_*`；②`TritonOpBuilder`——`_builder` 的 C++ 真身，包一层 `mlir::OpBuilder` 并记住 `lastLoc`，模板方法 `create<OpTy>(args...)` 两行(取 loc→转调 builder->create)是全部 129 个 lambda 的公共底座，不变量『一次调用⇄一个 op，loc 恒等 lastLoc』由此代码级证明；③`PYBIND11_MODULE`——`main.cc` 一个入口宏逐行 `init_triton_xxx(m.def_submodule("xxx"))` 装配 ir/passes/interpreter/llvm 四个核心子模块，`FOR_EACH_P(INIT_BACKEND, TRITON_BACKENDS_TUPLE)` 展开 CMake 注入的后端元组(回指 ch01「最多 4 个后端」的硬边界)；④`passes.cc`——`add_*` 是挂载口不是启动键，`ADD_PASS_WRAPPER_0` 宏把 pass 名绑成『收 PassManager、调 pm.addPass(builder())』的 lambda，只挂不跑；⑤`interpreter.cc` 一瞥——`load`/`store` 按掩码逐元素 gather/scatter，回指 ch13。小结：`create_*`/`add_*`/`load` 三种面孔同一台机器(pybind11 `.def(name, callable)`)，贯穿分界线『Python 描述，C++/MLIR 执行』。本章无精简版(mode=meta/kind=skip_impl)——真相源即 `python/src/*.cc` 逐字 C++ 源码。8 机制；3 图(chapter-map+`fig-m1-binding-chain`+`fig-m3-module-tree`)全 blind PASS；review APPROVED。

## Why it matters

本章是 Part IV 的收尾与 Part V 的过渡：把前面几章反复写的 `_builder.create_xxx` 坐实到具体 C++ 落点，读者往后调栈回溯、开 `TRITON_INTERPRET`、或好奇某行 `tl.load` 建出什么 IR 时知道该往哪儿看。同时是全书「后端接缝」伏笔(f1: plant ch01→payoff ch36)的中途实证站——`DECLARE_BACKEND`/`INIT_BACKEND`/`TRITON_BACKENDS_TUPLE` 三件套的具体装配语法在本章首次逐字展开，为 ch36-38 后端章与姊妹篇《Triton-Ascend 源码解读》的绑定层回指提供了精确源码锚点(interfaces.json 新增 ch18 键)。判断：f1 本身是「点名」性质的既有伏笔，本章只是给出其消费机制的具体证据，不构成新的独立承诺，故未新开 arc-map 条目（`bible.py due ch18` 确认本章无应埋/应回收项）。

## What to remember

ch18 done（kind=skip_impl/mode=meta，Part IV 收尾）。glossary.json 184→194（新增 10 条：`pybind11`/`.def 绑定`/`PYBIND11_MODULE`/`_C 扩展`/`init_triton_ir 等四子模块装配函数`/`TritonOpBuilder`/`lastLoc`/`create<OpTy>()`/`return_value_policy`/`ADD_PASS_WRAPPER_0`；同时给已有词条 `TRITON_BACKENDS_TUPLE` 追加 ch18 落实的消费者机制说明）。concepts.json 143→147（新增 4 条→ch18：Python create_*→C++ builder→MLIR op 双语接缝、pybind11 .def 绑定机制、PYBIND11_MODULE 装配子模块树、TritonOpBuilder 记忆 lastLoc 免传 loc）。interfaces.json 新增 ch18 键（源码接口非精简版，供 ch36-38 后端章与姊妹篇 Triton-Ascend 回指：`py::class_<TritonOpBuilder>`、`create<OpTy>` 公共底座、`PYBIND11_MODULE` 入口、`DECLARE_BACKEND`/`INIT_BACKEND`+`FOR_EACH_P` 后端注册两件套、`init_triton_passes` 装配骨架、`ADD_PASS_WRAPPER_0` 宏、`init_triton_interpreter`）。arc-map.json 未新开伏笔——`bible.py due ch18` 为空且判断本章不构成独立于既有 f1(plant ch01→payoff ch36) 的新承诺；**未动其它伏笔状态**（resolved=f4/f5/f7/f11/f12 原样；open 的 f13→ch17/f14→ch20/f15→ch24 原样未碰，因 ch17/ch20 正在并行跑）。

一致性核验：全部 status=resolved 的伏笔（f4→ch16/f5→ch13/f7→ch06/f11→ch12/f12→ch14）均满足 payoff==resolved_in 且 payoff≤已交付章节，无异常；f13(payoff ch17)/f14(payoff ch20)/f15(payoff ch24) 仍为 open，ch17/ch20 尚未提交——未发现并行泄漏。

reviews/review-report.json 与 run-ledger.json 由 Lead 预写，本次未改动；narrative/chapter.md 由 writer 并行修，diagrams/ 由 illustrator 并行修一张图，dossier/dossier.json 未触碰——以上三者 archivist 均未动。state.json 已加 ch18 条目并刷新 updated 时间戳；trace/INDEX.md 已刷新（自检确认 ch18 已在列，保留最近 10 条）。
