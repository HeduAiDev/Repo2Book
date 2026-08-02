# ch39《度量:proton 钩子、roofline viewer 与 do_bench》定稿——Part IX 度量章

- **Type**: delivery
- **Chapter**: ch39
- **Date**: 2026-07-18
- **Timestamp**: 2026-07-18T06:16:39+08:00
- **Agents involved**: archivist
- **User present**: False
- **Tags**: triton, part-9, deep, skip_impl, proton, roofline, do_bench, launch-hook, lazydict, util, compute-bound, memory-bound, cuda-event, quantiles, review-exhausted

## What happened

Part IX 度量章(kind=deep/skip_impl)。回答『怎么量一个 kernel 快不快』的三件工具:(1)proton 零侵入钩子——CompiledKernel 类级属性 launch_enter_hook/launch_exit_hook(默认 None,写在 class 体故赋值一次对全体已编译 kernel 立即生效),发射器在 cuLaunchKernel 前后各回调一次(每次发射恰好 enter+exit 两次配对);register_triton_hook 全部『注入』= 把 TritonHook.enter/exit 赋进两个槽位、核代码零改动;两道零成本闸(launch_metadata 首行 launch_enter_hook is None→return None + enter_scope/exit_scope 首行 get_profiling_on 空转)使 profiling 关时用户 metadata_fn 零调用;LazyDict 惰性求值(add 只登记欠条、钩子 enter 里 get() 才逐个执行 metadata_fn 并 | 合并);会话状态机 start(set_profiling_on→register_triton_hook→libproton.start)→activate/deactivate→finalize(默认输出 hatchet);flops 按位宽分档 flops8/16/32/64。(2)roofline viewer(third_party/proton/proton/viewer.py)——hatchet 调用树原始计数经 database.pop(1) 拆调用树+device_info、update_inclusive_columns 上卷、derive_metrics 派生 flop/s·byte/s·util、filter_frames 过滤 __proton_launch_metadata 记账桶;get_min_time_flops/get_min_time_bytes 出算力/带宽两面屋顶(peak_flops 按 arch 硬编码、位宽越窄峰值越高;peak_bw=2×bus_width×mem_clock/8 DDR 双数据沿),util=max(两屋顶)/实测时间判 compute-bound vs memory-bound。(3)do_bench(python/triton/testing.py)——全书所有『实测 X µs』口径来源:五段协议 触发编译(不计时)→跑 5 次估 estimate_ms→按毫秒预算(默认 warmup=25/rep=100ms)换算 n_warmup/n_repeat(max(1,·)兜底)→预热→逐轮 cache.zero_() 冲冷 L2(后端 256MB int 缓冲 > A100 40MB/H100 50MB L2)+ 每轮专属 CUDA event 打点(异步 record、循环后一次 synchronize 批量 elapsed_time,优于 time.time()),收尾取 quantiles=[0.5,0.2,0.8](median 击穿点 50% 抗尖峰,mean 被线性拖偏)。诚实边界:proton/do_bench/viewer 真 Python 源逐字可核;host 无 GPU 处 do_bench 计时/proton 挂计数标『需真机』,viewer 喂仓库自带 example_cuda.json 无 GPU 照跑 roofline(纯读 json 派生),foo0(device 1/arch 90)手算链可溯源、foo1(device 0/arch 89)留读者仿算;CUPTI/roctracer/hatchet/libproton 第三方库点到即止(任务书定)。质量事件:本章 review-exhausted 逃逸(exp-0716-1 家族第 6 例、首个纯文字环:writer 末轮已落地全部修复但 reviewer 持早轮快照判 blocking→轮数耗尽,8 项 issue 有 6 项 writer 末轮已解),Lead 核实后改判 APPROVED;唯一 blocking 是 L386 foo0/foo1 计数消歧(1e11/1e8 vs 1e10/1e7,fidelity)已解;Lead 补 chapter-map(逃逸跳过 Map 站→派 illustrator 补)+ 补开篇图引。逐机制覆盖全绿、全 linter green。本章无伏笔埋/回收(bible.py due ch39 确为空,未 resolve 任何伏笔)。skip_impl 无精简版接口(按契约跳过 interfaces 登记)。bible 回写:glossary +12(478→490)、concepts +18(311→329)、figures +7(76→83,含新画 chapter-map)。

## Why it matters

本章是全书度量地基:do_bench 是前面各章所有『实测』性能数字的统一口径(autotune 挑 config 也用它计时,见 ch12),roofline 给出『优化往哪使劲』的判据(compute-bound 去喂满 Tensor Core、memory-bound 去优化访存),proton 给出零侵入拿到 flops/bytes 的通用挂载点。三件工具连成量多久→量瓶颈在哪的闭环,收束『量准了才谈得上优化』这条主线。过程教训:review-exhausted 逃逸后 Lead 必逐条核『issue 是否已在稿』、不可假设未解(本章 8 项有 6 项 writer 末轮已落地);若纯文字环 review-exhausted 复发≥2,考虑 reviewer 契约加『复核当前稿而非早轮快照』。

## What to remember

Part IX 度量章。三件工具量 kernel 快不快:proton 零侵入类级钩子(launch_enter/exit_hook + LazyDict 惰性求值 + register_triton_hook 一次赋值全体生效 + 会话状态机)挂 flops/bytes;roofline viewer 用 util=max(算力屋顶,带宽屋顶)/实测判 compute/memory-bound;do_bench 五段协议(触发编译→估时→毫秒预算定次数→预热→逐轮冲冷 L2+CUDA event 打点→取中位/分位)是全书『实测』数字口径。skip_impl 无接口;无伏笔;host 无 GPU 处标需真机、viewer 喂 example_cuda.json 照跑;第三方库点到即止。review-exhausted 逃逸→Lead 核实 writer 末轮已解 6/8→改判 APPROVED + 补 chapter-map。glossary+12/concepts+18/figures+7。
