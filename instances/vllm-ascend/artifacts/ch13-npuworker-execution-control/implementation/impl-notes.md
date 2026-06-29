# ch13 精简版实现笔记 —— NPUWorker：从 WorkerBase 重写执行主控

只做减法的忠实精简版。主角 `worker.py` 与真实 `vllm_ascend/worker/worker.py` 同名同结构同
控制流，串起四步生命周期：`init_device → determine_available_memory →
compile_or_warm_up_model → execute_model`。另带两份对照基座的 subtract-only 摘录，支撑
本章 thesis「为何 Worker 选择重写而非继承」。

本章解读的真实源码（规范路径）：
- `vllm_ascend/worker/worker.py`（主角：NPUWorker 四步生命周期 + _warm_up_atb）
- `vllm/v1/worker/worker_base.py`（对照基座：WorkerBase 抽象 + 公共 __init__）
- `vllm/v1/worker/gpu_worker.py`（对照基座：GPU Worker.init_device 把设备钉死在 cuda）

## 验收判据
把真实 `worker.py` 删掉所有 `# SUBTRACTED` 标注的行（许可证头 + 不可在 host 导入的
torch_npu/vllm/vllm_ascend import + sleep mode / static_kernel / PP-KV transfer / profiler /
lora 等旁支分支），应当 ≈ 得到本精简版。四步生命周期的控制流一字未改。

## host 可读可跑边界
- **可跑（纯 Python 决策逻辑）**：`determine_available_memory` 的 KV 显存预算与
  gpu-memory-utilization 回退建议算术；`compile_or_warm_up_model` 的 warmup_sizes 计算 /
  capture_model 触发 / _warm_up_atb 接线；`execute_model` 派发给 model_runner；
  「NPUWorker(WorkerBase) 而非继承 GPU Worker」的方法对位。测试用 monkeypatch 把
  torch.npu / memory_profiling / CUDAGraphMode / envs_vllm / torch_npu 换成桩。
- **不跑（需 NPU/CANN）**：真实 `torch.npu.set_device` / `torch_npu._inductor` /
  `MemorySnapshot` / `torch_npu._npu_matmul_add_fp32` ATB 预热 / HCCL 分布式初始化。
  `_init_device` 等设备路径仅作可读控制流，host 不真跑。

## Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `worker.py` 文件头 + 模块级 import | `vllm_ascend/worker/worker.py:L1-L78` | 删除/占位 | Apache 许可证 + torch_npu/vllm/vllm_ascend 运行时 import（host 不可用）+ torch._dynamo trace-rule 注册（npu graph 追踪，非生命周期）；`from vllm.logger import logger` 用 stdlib logging 占位，logger.* 调用原样 |
| `worker.py` `NPUWorker.__init__` | `worker.py:L82-L130` | 逐字保留主线 + 删旁支 | 保留 adapt_patch/ATB 注册/ascend_config/super().__init__/cache_dtype/profiler 懒初始化；SUBTRACT COMPILE_CUSTOM_KERNELS warning、sleep buffers、WEIGHT_LOADER fixme、v2 回退、_pp_send_work、static_kernel 信号（L93-97,131-166） |
| `worker.py` `_init_device` | `worker.py:L260-L315` | 逐字保留主线 + 删特例 | 设备层重写第 1 步全保留（npu set_device / torch_npu._inductor / MemorySnapshot / hccl / triton 属性）；SUBTRACT A5 分支、DP visible_device_count 断言、报错文案细节 |
| `worker.py` `init_device` | `worker.py:L317-L332` | 删 v2 分支 | 拆层第二层全保留（存 device / init_workspace_manager / 构造 NPUModelRunner）；SUBTRACT use_v2_model_runner 下 V2 runner import（开发中，主线走 else） |
| `worker.py` `determine_available_memory` | `worker.py:L334-L462` | 逐字保留算法 + 删捷径/文案 | 第 2 步全保留（memory_profiling / pre-graph torch peak / profile_cudagraph_memory / 预算公式 / 回退建议算术）；SUBTRACT --kv-cache-memory fast path、DeepSeek-V4 特例、logger 完整文案 |
| `worker.py` `compile_or_warm_up_model` | `worker.py:L557-L659` | 逐字保留控制流 + 删诊断 log | 第 3 步全保留（warmup_sizes 计算 / _dummy_run / capture_model / _warm_up_atb 接线 / CompilationTimes）；SUBTRACT 实测vs估算 log、--kv-cache-memory 建议 log、enable_cpu_binding bind_cpus |
| `worker.py` `_warm_up_atb` | `worker.py:L661-L665` | 逐字一致 | 昇腾特有 ATB matmul 预热，区别于基座 kernel_warmup 的关键差异点 |
| `worker.py` `execute_model` | `worker.py:L474-L538` | 保留派发主线 + 删 PP/观测 | 单机首 rank 主线全保留（intermediate_tensors=None → model_runner.execute_model → 返回 ModelRunnerOutput）；SUBTRACT profile_memory/dp.step/profiler.step、非首/末 PP rank irecv/isend、kv_connector 透传 |
| `worker.py` `_init_worker_distributed_environment` | `worker.py:L836-L849` | 逐字一致 | hccl 后端分布式初始化，设备层重写一环（must_keep） |
| `worker_base.py` `WorkerBase` + `CompilationTimes` | `vllm/v1/worker/worker_base.py:L33-L143` | 摘录抽象骨架 | 对照基座：四步生命周期方法体全是 raise NotImplementedError + 公共 __init__ 摊开 vllm_config；SUBTRACT WorkerWrapperBase 及其余接口声明 |
| `gpu_worker.py` `Worker.init_device` | `vllm/v1/worker/gpu_worker.py:L106,L239-L309` | 摘录 if-cuda/else-raise 骨架 | 「不能继承只能重写」硬证据：非 cuda 直接 raise RuntimeError；SUBTRACT __init__、DP rank 调整、其余生命周期方法 |

## 「重写=搬结构、换设备层」核对（昇腾 vs 基座）
- 设备分配：`torch.accelerator.set_device_index(cuda:N)` → `torch.npu.set_device(npu:N)`
- 编译栈：CUDA inductor → `import torch_npu._inductor`（triton graph 模式）
- 分布式后端：`nccl` → `hccl`
- 算子预热：基座 `kernel_warmup(self)`（CUDA kernel 调优）→ 昇腾 `_warm_up_atb`（ATB matmul）
- graph 显存：CUDA graph → ACL/NPU graph（`profile_cudagraph_memory` / `capture_model`）
- **结构性差异（非换符号）**：快照与分布式初始化的先后顺序——基座先初始化分布式后拍快照
  （NCCL buffer 计入快照基线），昇腾先拍快照后初始化 HCCL（HCCL buffer 不计入基线）；
  设备初始化拆成 `_init_device` + `init_device` 两层（基座一个 init_device），便于 XliteWorker
  只 override init_device 即可换 ModelRunner。

## 横切点（点名不展开）
- `profiler`：类型 `TorchNPUProfilerWrapper`（torch_npu profiler 的薄包装），__init__ 懒初始化。
- `xlite/`（XliteWorker / XliteModelRunner）：平行于 NPUWorker/NPUModelRunner 的轻量执行路径。
- `NPUModelRunner`：execute_model 的派发目标、profile_run/profile_cudagraph_memory/capture_model
  的实际执行者；本章只点名，细节留后续 ModelRunner 章。
