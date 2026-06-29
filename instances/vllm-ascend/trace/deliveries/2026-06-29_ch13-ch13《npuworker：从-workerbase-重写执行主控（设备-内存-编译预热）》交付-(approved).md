# ch13《NPUWorker：从 WorkerBase 重写执行主控（设备/内存/编译预热）》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 13
- **Date**: 2026-06-29
- **Timestamp**: 2026-06-29T17:00:12Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch13, npuworker, worker-base, lifecycle, delivery

## What happened

ch13 通过多维评审 APPROVED 交付。讲清昇腾 Worker 为何重写而非继承 WorkerBase：四步生命周期 init_device(npu set_device+workspace num_ubatches=1+torch_npu._inductor) / determine_available_memory(memory_profiling+profile_cudagraph_memory δ 估算回退 gpu-memory-utilization) / compile_or_warm_up_model(warmup 降序+_warm_up_atb) / execute_model(派发 NPUModelRunner，唯一热路径)。对照基座 vllm/v1/worker/{gpu_worker,worker_base}.py。横切点名 WorkerProfiler 与 xlite/ 轻量路径不展开。

## Why it matters

确立"昇腾重写 WorkerBase 而非继承 GPU Worker"这一进程级执行主控的方法对位，为后续 NPUModelRunner 章铺垫 execute_model 派发细节。

## What to remember

ch13 无伏笔应埋/应回收（bible.py due 空）；新接口 6 条已登记 interfaces.json（NPUWorker 类+四步方法+对照 WorkerBase）。评审 APPROVED，11 条均 non-blocking 商榷（5 条算法维度松节奏/补论证 + 6 条 reader-comprehension 术语定义 ATB/ACL/HCCL/enforce_eager/workspace 等）。
