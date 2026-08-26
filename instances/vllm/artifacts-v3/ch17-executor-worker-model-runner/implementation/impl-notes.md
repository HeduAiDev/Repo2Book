# ch17 精简版 impl-notes — 执行三层：Executor / Worker / ModelRunner（Part V：GPU 不等 Python）

- **Pin**：vLLM v0.27.1（`6e448d0ea`）。全部 `# SOURCE:` 行号对当前 pin 现核（dossier code_spine
  的 22 个锚 + must_keep 66 符号逐个比对过真实源码；未照抄 v2 资产的 v0.21.0 旧行号）。
- **产物**：`implementation/`（包布局镜像源码树——`executor/{abstract,uniproc_executor,
  multiproc_executor}.py`、`worker/{worker_base,gpu_worker,gpu_model_runner}.py`、
  `serial_utils.py`、`platforms/cuda.py`、`utils/{gpu_sync_debug,import_utils,system_utils,
  gc_utils}.py`、`structured_output/utils.py` + 两个 seam 文件 `_host_seams.py` /
  `_shm_broadcast_seam.py`）+ `tests/test_execution_layers.py` + `tests/_worker_double.py`。
- **跑法**：`cd instances/vllm/artifacts-v3/ch17-executor-worker-model-runner && python -m pytest
  tests/ -q` → **84 passed**（~28s；三次连跑稳定）。host 可跑：真 torch（含 CUDA 上下文）/ 真
  pyzmq / 真 cloudpickle / 真 mp spawn 子进程（WorkerProc 经真实 make_worker_process /
  worker_main / READY 握手出生、真 ZMQ 广播 MQ / 逐 worker 应答 MQ、真 busy loop / 监控线程 /
  三级关停）；无 vllm 包、无 xgrammar（CPU 内核替身，容器内真内核优先 import）。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包（HOST SEAM 例外见 §Seam
  清单——每个 seam 行内标注并在此登记）。
- **lint**：`python scripts/lint_fidelity.py <本章目录>` → **全部通过（无 BLOCKING、无警告）**；
  must_keep 66 符号经 linter `over_subtraction` 项全数核在。

## 本章主题 = 全真部分

- **三层解耦的骨架（m1/m2）**：`Executor.get_class` **逐字** minus ray/external_launcher 分支
  （删除项 1；删除后 "ray" 落到自定义 qualname 分支——测试断言这个删除后果）；`__init__` 摊平
  config → `_init_executor()` 抽象钩子逐字。
- **控制面唯一入口（m10）**：`collective_rpc` 契约 docstring（『It is recommended to use this
  API to only pass control messages, and set up data-plane communication to pass data.』）**逐字**；
  `execute_model`/`sample_tokens`/`initialize_from_config`/`determine_available_memory`/
  `get_kv_cache_specs`/`compile_or_warm_up_model` 全是薄封装逐字；CompilationTimes 跨 worker
  取 max 汇回（L127-L137）逐字（e2e 用真 RPC 断言 max=2.0）。
- **uni 直调形态（m3）**：`AsyncOutputFuture` **逐字**（result() 只等 `async_output.get_output()`
  即 D2H 事件、二次 result 瞬时、timeout 未实现）；`UniProcExecutor.collective_rpc` **逐字**
  （not non_block 直调 / non_block 同步算完包 Future / 异常入 Future / AsyncModelRunnerOutput 包
  AsyncOutputFuture）；`execute_model` 覆写带 `single_value=True` + non_block 完成即抛
  （L117-L121）逐字；`run_method` 三分支派发（serial_utils.py:L486-L514）**逐字**。
- **mp 星形拓扑（m4/m11-m14）**：`_init_executor` 建 rpc_broadcast_mq（一次 enqueue 全 worker
  可见）→ spawn → `wait_for_ready` 握手 → 监控线程 → response_mqs → futures_queue **逐字**（仅
  删 DP 日志/跨节点装配/OMP/numa/fork-fd——删除项 2/4）；`collective_rpc` 广播 + get_response
  + FutureWrapper **逐字**（仅删 KV 聚合分支——删除项 3）；`_get_output_rank` 公式与 TP=8/PP=4
  算例注释**逐字**；`execute_model`/`sample_tokens` 覆写带 `unique_reply_rank=self.output_rank`
  + 超时**逐字**。
- **FutureWrapper FIFO（m12）**：`appendleft`/`pop` 配对 + `result()` 先排干先于自己的 future
  **逐字**——单元测试（三个 future、后收者替先收者收尸）与 e2e 双面验证。
- **WorkerProc 的一生（m13/m17/m18）**：`worker_main`（信号→构造→READY→busy_loop、
  ready_writer None 与否区分启动期/运行期失败）**逐字** minus 平台适配段；`worker_busy_loop`
  三分支派发 + output_rank 过滤 + add_note 转 FAILURE **逐字**；`enqueue_output`/`handle_output`/
  `async_output_busy_loop`（含 set_device 注释）**逐字**；`start_worker_monitor`（sentinel→
  is_failed→shutdown→callback）与 `register_failure_callback` **逐字**；`shutdown` 三级
  （graceful→SIGTERM→SIGKILL）与 death pipe EOF 自清理 **逐字**——e2e 全链真跑（含
  proc.kill() 死亡路径、EXECUTOR_FAILED 哨兵 replica（core.py:L1029-L1031 逐字，测试侧））。
- **WorkerWrapperBase 延迟初始化（m5）**：构造只记 rpc_rank/global_rank、`init_worker` 才
  resolve_obj_by_qualname（envs→插件→实例化顺序）、只收字符串明确拒绝传类、`__getattr__`
  透传、`execute_model` 先 `_apply_mm_cache` **逐字**（仅删 extension 注入体与 mm cache 构建
  ——删除项 8；init_worker 前属性访问递归是真实既定行为，测试断言之）。
- **Worker.init_device（m7）**：DP local_rank 修正→选卡→**分布式初始化刻意先于显存快照**
  （注释原文）→构造 runner **逐字**（仅删 assigned_physical_gpu_ids 块——elide 注授权）；
  测试以记录器断言 dist_init < snapshot 顺序。
- **三锚点（m8）**：`load_model`（tag=weights 池上下文）→ `determine_available_memory`
  （骨架保留，账本归 ch14——must_keep 原文『骨架保留，账本细节 ch14』）→
  `initialize_from_config`（先 ensure_kv_transfer_initialized，tag=kv_cache 池内分配）；
  CuMem 池的 nullcontext 快路径（cuda + cumem 关闭的默认部署）真实保留并测试。
- **compile_or_warm_up_model 编排（m9）**：warmup 从大到小（capture 列表内的 size 跳过）→
  kernel_warmup → capture_model → _dummy_sampler_run（**在 capture 之后**，NOTE 原文）→
  inductor 惰性初始化 → JIT 纠察 → freeze_gc_heap → enable_gpu_sync_check → 返回
  CompilationTimes **逐字编排骨架**（内部细节按删除项 7 裁除）；with_gpu_sync_check 的
  启动期不查/运行期才查双态（gate 原文注释）逐字。
- **两段式契约（m15）**：`WorkerBase.execute_model`/`sample_tokens` 两段 docstring **逐字**
  （『If this method returns None, sample_tokens should be called immediately after』+ 自注）；
  `ExecuteModelState` 十字段 NamedTuple **逐字**；runner 入口 State error 断言 + 打包暂存 +
  return None + 解包即清 + apply_grammar_bitmask 施加点 + _sample/_update_states 调用位
  **逐字**（前向深水按删除项 6 退化为注释占位，局部变量以 None 绑定——契约行为不变）；
  bitmaks 施加在真 torch logits 上可观察（禁位→-inf、argmax 翻转）。
- **PP 接力（m16）**：`AsyncIntermediateTensors` **逐字**（__getattribute__ 懒拦截 .tensors、
  wait_for_comm 幂等）；`Worker.execute_model` 的 _pp_send_work 收割→irecv→转调→isend 后
  return None 三段**逐字**（enable_sp 分支按删除项 5 删；TP=1/PP=1 下快路径测试）。
- **平台轴（m6）**：`platforms/cuda.py` 的 `CudaPlatformBase.check_and_update_config`：
  `worker_cls=='auto' → "vllm.v1.worker.gpu_worker.Worker"` **逐字**（Nvml/NonNvml 二分与
  L1014 别名删——平台全量面不在本章）；规范 qualname 经 sys.modules 别名解析到本包
  （HOST SEAM，`resolve_obj_by_qualname` 本体逐字保留）。
- **异步调度声明面（m20）**：`supports_async_scheduling` 基类 False / uni True / mp True 三处
  **逐字**；worker 侧 `async_output_copy_thread`（把 D2H 拷贝挪出 busy loop）**逐字**，
  e2e 以 AsyncProbeWorker + async_scheduling=True 配置真跑。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因；核心行）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `Executor.get_class` | vllm/v1/executor/abstract.py:L47-L92 | **逐字** minus ray 分支（L60-L68）/ external_launcher 分支（L77-L80）/ 文末兼容导入块（L371-L380） | must_keep；删除项 1（ray 归 ch34） |
| `Executor.__init__`/`_init_executor` | abstract.py:L94-L116 | 逐字（@instrument 装饰删——删除项 4 观测装饰） | must_keep×2 |
| `Executor.initialize_from_config`/`compile_or_warm_up_model`/`register_failure_callback`/`determine_available_memory`/`get_kv_cache_specs` | abstract.py:L118-L150 | 逐字（后者含 max 汇回） | must_keep×4 |
| `Executor.collective_rpc`（overload×2 + abstract） | abstract.py:L152-L202 | 逐字；docstring 契约逐字 | must_keep（L181-L183 控制面契约原文） |
| `Executor.execute_model`/`sample_tokens` | abstract.py:L209-L247 | 逐字薄封装 | must_keep×2 |
| abstract 其余指令面（profile/save_sharded_state/lora 族/sleep/wake_up/supported_tasks/…） | abstract.py:L249-L368 | 逐字（保留未删——它们是 collective_rpc 泛化面的证据） | 无删除批准；结构洞见 §已知偏差 3 |
| `AsyncOutputFuture` | uniproc_executor.py:L26-L42 | 逐字 | must_keep |
| `UniProcExecutor` 全类 | uniproc_executor.py:L45-L147 | 逐字 minus set_worker_net_device（L59-L60）/ elastic EP 分支（L65-L68） | must_keep×3；删除项 4 |
| `ExecutorWithExternalLauncher` | uniproc_executor.py:L150-L196 | **整类删** | 删除项 1 |
| `FutureWrapper` | multiproc_executor.py:L70-L100 | 逐字 | must_keep（m12） |
| `MultiprocExecutor._init_executor` | multiproc_executor.py:L110-L247 | 逐字 minus DP 日志（L140-L150）/ 跨节点装配（L215-L220）/ OMP 上下文（L174-L181）/ fork-fd 跟踪（L167-L172、L193-L195） | must_keep；删除项 2/4 |
| `get_response_mqs` | multiproc_executor.py:L249-L258 | **整方法删** | 删除项 2 |
| `start_worker_monitor`/`register_failure_callback` | multiproc_executor.py:L279-L319 | 逐字（weakref/inline 测试钩原样） | must_keep×2（m17） |
| `execute_model`/`sample_tokens`/`execute_dummy_batch`/`take_draft_token_ids` | multiproc_executor.py:L321-L352 | 逐字（unique_reply_rank + 超时 + aggregator 传参原样） | must_keep×2 + m14 |
| `MultiprocExecutor.collective_rpc` | multiproc_executor.py:L354-L416 | 逐字 minus KV 聚合分支（L375-L379）；get_response 闭包逐字 | must_keep；删除项 3 |
| `_ensure_worker_termination`/`shutdown`/`check_health`/`_get_output_rank`/`supports_async_scheduling` | multiproc_executor.py:L418-L527 | 逐字（三级关停 + 公式 + 注释算例） | must_keep×4；m18 |
| `UnreadyWorkerProcHandle`/`WorkerProcHandle` | multiproc_executor.py:L530-L565 | 逐字（peer 列表保留空承载） | must_keep×2 |
| `WorkerProc._init_message_queues` | multiproc_executor.py:L575-L605 | 单节点分支逐字；多节点 else 删 | 删除项 2 |
| `WorkerProc.__init__` | multiproc_executor.py:L607-L670 | 逐字 minus 标题装饰两处（L635-L637/L641-L644）/ elastic EP（L645-L648）；async 线程块逐字 | must_keep；删除项 4 + m20 |
| `make_worker_process`/`wait_for_response_handle_ready`/`wait_for_ready` | multiproc_executor.py:L672-L782 | 逐字 minus numa 上下文（L712-L716）/ fork-fd 并入（L688-L690）；isinstance 断言 win32 放宽（HOST SEAM） | must_keep×2；删除项 4 |
| `WorkerProc.shutdown`/`monitor_death_pipe` | multiproc_executor.py:L784-L818 | 逐字 | must_keep（m18） |
| `WorkerProc.worker_main` | multiproc_executor.py:L820-L944 | 逐字 minus gpu_ids 发布（L843-L851）/ net device（L853-L854）/ tracer（L871-L877）/ numa 日志（L881-L882） | must_keep；删除项 4 |
| `ResponseStatus`/`enqueue_output`/`handle_output`/`async_output_busy_loop`/`worker_busy_loop` | multiproc_executor.py:L946-L1022 | 逐字 | must_keep×5；m13/m20 |
| `setup_proc_title_and_log_prefix` | multiproc_executor.py:L1024-L1058 | **整方法删** | 删除项 4 |
| `set_multiprocessing_worker_envs` | multiproc_executor.py:L1061-L1089 | 函数面 + `_maybe_force_spawn()` 保留；OMP 调优体删 | 删除项 4 |
| `CompilationTimes`/`WorkerBase` 全类 | worker_base.py:L34-L184 | 逐字（两段式 docstring L142-L157 逐字） | must_keep×3 |
| `WorkerWrapperBase` 全类 | worker_base.py:L187-L358 | 逐字 minus extension 注入体（L261-L287）/ gpu_ids 透传（L289-L293）/ mm cache 构建（L309-L315→None 赋值） | must_keep×5；删除项 4/8 |
| `AsyncIntermediateTensors` | gpu_worker.py:L96-L125 | 逐字 | must_keep（m16） |
| `Worker.__init__` | gpu_worker.py:L129-L179 | 逐字 minus elastic EP（L149-L151）/ sentinel（L152-L154）/ sleep 缓冲（L155-L157）/ 权重引擎（L159-L162）/ profiler（L164-L172）/ sleep 后端（L178-L179） | must_keep；删除项 4/5 |
| `_maybe_get_memory_pool_context`/`_scoped_allocator_max_split` | gpu_worker.py:L256-L301 | 逐字（nullcontext 快路径真实保留） | m8 载体 |
| `Worker.init_device` | gpu_worker.py:L303-L427 | 逐字 minus gpu_ids 块（L328-L357，elide 授权）/ workspace（L404-L406）/ usage stats（L425-L427）；use_v2 分支保留（脚注证据） | must_keep（m7） |
| `Worker.load_model` | gpu_worker.py:L435-L450 | 逐字 minus weight_transfer（L444-L450） | must_keep；删除项 5 |
| `Worker.determine_available_memory` | gpu_worker.py:L459-L611 | docstring 逐字；账本主体（L472-L611）注释占位 + HOST SEAM 返回 0 | must_keep 原文『骨架保留，账本细节 ch14』 |
| `Worker.initialize_from_config` | gpu_worker.py:L649-L676 | 逐字（routed_experts/KV-zero 尾段保留，config seam 下自然 no-op） | must_keep（m8） |
| `Worker.compile_or_warm_up_model` | gpu_worker.py:L678-L853 | 编排骨架逐字；内部细节（compile_ranges L695-L703 / 对比日志 L719-L733 / startup_plan L735-L791 / V2 warmup L793-L795 / pooling L813-L814）删 | must_keep（m9）；删除项 7 |
| `Worker.sample_tokens`/`execute_model` | gpu_worker.py:L1010-L1107 | 逐字（装饰器双 @ 保留）minus SP 分支（L1035-L1062）/ annotate_profile 包装（L899-L1008 删） | must_keep×2；m16 |
| gpu_worker 其余（lora/check_health/save/take_draft/execute_dummy） | gpu_worker.py:L1109-L1205 | 逐字（weight 族 L1207-L1313 / sleep 族 L181-L254 / profiler L1112-L1163 删） | 删除项 5 |
| `Worker.shutdown`/`init_worker_distributed_environment` | gpu_worker.py:L1315-L1340/L1346-L1389 | 逐字（消费的分布式函数为 HOST SEAM） | must_keep |
| `ExecuteModelState`/`GPUModelRunner.execute_model`/`sample_tokens` | gpu_model_runner.py:L437-L451/L4166-L4535/L4552-L4592 | 契约面逐字；前向深水（L4180-L4505）注释占位、局部变量 None 绑定；桩方法（_sample 等）同签名最小桩 | must_keep×5；删除项 6（ch18/19/29/33 域） |
| `run_method` | vllm/v1/serial_utils.py:L486-L514 | 逐字 | must_keep（m3） |
| `CudaPlatformBase.check_and_update_config` | vllm/platforms/cuda.py:L208/L306-L313 | 逐字（Nvml/NonNvml 二分与其余平台面删——本章不展开） | m6 |
| `enable_gpu_sync_check`/`with_gpu_sync_check` | vllm/utils/gpu_sync_debug.py:L26-L33/L131-L165 | 逐字（抑制器补丁 L39-L89 删——ch19 域）；宿主走真实 non-CUDA no-op 分支 | must_keep |
| `resolve_obj_by_qualname` | vllm/utils/import_utils.py:L104-L110 | 逐字 | must_keep |
| `update_environment_variables`/`freeze_gc_heap` | vllm/utils/system_utils.py:L34-L44 / vllm/utils/gc_utils.py:L96-L108 | 逐字 | m5 顺序约束 / m9 收口 |
| `apply_grammar_bitmask` | vllm/v1/structured_output/utils.py:L86-L175 | 逐字 minus GPU async H2D 分支（L137-L149） | must_keep（施加点位） |

## 删除台账 — dossier subtraction_plan 8 项 delete（全部执行）

1. **ray 后端 + external_launcher** ✓ — get_class 两分支 + 文末兼容块 + ExecutorWithExternalLauncher
   全类；删除后 "ray" 落到 qualname 分支（测试断言该后果）。
2. **多节点路径** ✓ — DP-leader 日志、peer_worker_response_mqs 装配、_init_message_queues else、
   get_response_mqs；单节点 `MessageQueue(1,1)` 闭环自洽。
3. **kv_output_aggregator** ✓ — collective_rpc 聚合分支 + abstract.init_kv_output_aggregator；
   字段保留恒 None、默认恒等 aggregate。
4. **平台/可观测/特性装饰** ✓ — OMPProcessManager、inherited_fds fork 适配（保留空表行为）、
   numa_utils、set_worker_net_device×2、VLLM_ELASTIC_EP_SCALE_UP_LAUNCH×2（连同 elastic_ep_executor
   装配与 elastic_ep_execute——同特性机器）、maybe_init_worker_tracer、setup_proc_title_and_log_prefix
   全方法与两处调用、set_multiprocessing_worker_envs 的 OMP 体、@instrument 装饰×4。
5. **Worker 生产特性面** ✓ — sleep/wake_up/checkpoint/_get_sleep_mode_backend、weight_transfer
   全家（引擎字段/创建/热更族）、profiler/profile/annotate_profile、worker_sentinel/handle_ft_command、
   SP/enable_sp 分支、update_config/reload_weights/update_max_model_len、_set_draft_weight_update_target。
6. **GPUModelRunner 执行细节** ✓ — 除五个保留 span 外全部注释占位（同签名最小桩维持 Worker 侧
   骨架可驱动；打包局部变量 None 绑定，契约行为不变）。
7. **compile_or_warm_up 内部** ✓ — compile_ranges 补边界、cudagraph 对比日志、startup_plan 建议
   与落盘、V2 warmup_kernels 分支、pooling 分支；编排骨架（从大到小→kernel_warmup→capture→
   sampler 预热→inductor→JIT 纠察→freeze_gc→开闸）完整。
8. **worker_extension_cls 注入体 + mm_receiver_cache 构建** ✓ — 注入体整段删；构建分支退化为
   None 赋值（warning 早退路径保留，_apply_mm_cache 恒 no-op）。

（另：elide 注授权的三处小裁剪——init_device 的 assigned_physical_gpu_ids 块/workspace manager/
usage stats，行内 SUBTRACTED 标注并引 elide 注。）

## Seam 清单（行内标注 HOST SEAM / win32 SEAM）

| Seam | 位置 | 是什么 | 为什么仍忠实 |
|---|---|---|---|
| `_shm_broadcast_seam.MessageQueue/Handle` | 独立文件（ch09 `_msgspec_seam` 同款惯例） | 真 ZMQ XPUB/SUB（loopback tcp）承载广播 + inproc PAIR cancel 仿 SpinCondition | 可观察控制面契约一致：一次 enqueue 全读者 FIFO 收到 / TimeoutError / wait_until_ready 集体握手 / shutdown 以 RuntimeError("cancelled") 唤醒阻塞读者（真实 acquire_read 同款）；偏差：载荷走消息体不走 SHM 环形缓冲（无写侧背压——控制面流量微小）、ipc://→tcp://（ch05/09 先例）、LINGER=0 + shutdown 关 socket（防解释器退出 context.term 卡死） |
| `current_platform` | _host_seams | Platform 接口子集（is_cuda_alike/set_device/…） | 消费面行为一致；设备谓词按宿主实况（测试可 monkeypatch 走 CUDA+cumem 池路径——真实判定梯逐字保留） |
| `MessageQueue` 消费侧 wait_for_ready 的 isinstance 断言 | multiproc_executor.py | win32 上 PipeConnection 非 Connection 子类 → 放宽到二元组 | unix 上原断言不变；语义同为「wait 返回 pipe 对象」 |
| VllmConfig 族 / SchedulerOutput / GrammarOutput / ModelRunnerOutput / DraftTokenIds 等载体 | _host_seams | 字段子集 dataclass | 装配线是 ch03 的产品；保留代码触及的字段全在 |
| `install_vllm_module_aliases` | _host_seams→包 `__init__` | 真实 vllm 缺席时把 "vllm.v1.…"-前缀 qualname 预置到 sys.modules 指向本包同名模块 | `resolve_obj_by_qualname` 本体逐字保留（importlib 命中 sys.modules）；真实 vllm 在场时不劫持 |
| distributed 族（init_distributed_environment/ensure_model_parallel_initialized/get_pp_group/…） | _host_seams | 记录 world/rank/local_rank + 单机退化组 | NCCL/分布式全貌归 ch34；TP=1/PP=1 下组语义一致；PP 传输面显式 NotImplementedError（结构洞） |
| `MemorySnapshot`/`request_memory`/CuMem 池记录仪 | _host_seams | 显存快照/请求内存/池 tag 记录 | 账本归 ch14；三锚点的**顺序与 tag** 可观察（测试断言） |
| `kernel_warmup`/`activate_jit_monitor`/`trigger_inductor_lazy_init`/`TensorizerLoader`/`load_general_plugins`/`set_current_vllm_config`/`set_random_seed`/`maybe_attach_gc_debug_callback` | _host_seams / utils | 调用面 no-op / 近似实现 | ch19 编译域/生产面的调用位是本章编排对象；行为=默认关闭路径 |
| `xgr`（_XgrammarSeam） | structured_output/utils.py | apply_token_bitmask_inplace 的 CPU 位解码内核（bit t 允许位→非 -inf） | xgrammar 内核文档语义（禁位→-inf）；容器内真内核优先 import 生效 |
| `envs` | _host_seams | 类属性替身，默认值对 pin（300s/5s/16MB/…） | 环境变量面未入本章；被测路径全部走默认值 |
| win32 `get_mp_context`/`_maybe_force_spawn` | multiproc_executor.py 文末 | spawn on win32 | 观测契约一致（WorkerProc 在独立进程）；ch09 同款先例 |
| `_worker_double.ProbeWorker 族` | tests/ | 测试侧 worker 替身（经真 qualname 机制解析） | 测试脚手架非实现；vLLM 自家测试同款手法 |

## 已知偏差（reviewer 重点）

1. **determine_available_memory 返回 0（HOST SEAM）**：真实返回 profile 定出的可用 KV 字节；
   账本归 ch14 的精简版。锚点的**位置与调用链**（executor.determine_available_memory →
   collective_rpc → 每 worker）经 e2e 真跑（测试侧 worker 返回 12345 验证链路）。
2. **runner 前向是注释占位**：execute_model 内 L4180-L4505 深水退化为 None 绑定 + 占位注释
   （删除项 6 批准）；`_sample` 返回占位 SamplerOutput（真采样栈归 ch29）。两段式协议
   （断言/打包/None/解包/清/bitmask/调用位）全部真实可观察。
3. **abstract 的 sleep/wake_up/profile/lora 族保留但 worker 侧对应面已删**：调用会以
   FAILURE（NotImplementedError 文本）回包——『删掉批准分支后的真码』的机械后果（ch09 结构洞
   惯例）；真实引擎在首个 FAILURE 后即停机，不会走到这些洞。
4. **weakref.finalize(self, self.shutdown) 是退出兜底而非 gc 钩**：绑定方法经 finalizer 注册表
   持强引用（真实 vLLM 同款写法、同款语义）——测试直接调 `_finalizer()` 验证兜底触发。
5. **FAILURE/超时会毒化应答 MQ**（真实 FIFO 设计：失败后引擎停机、不再发 RPC）；测试对错误
   路径用独立执行器，与真实引擎的停机纪律一致。
6. **`get_compilation_match_table`/`get_model_inspection`/`save_sharded_state` 的惰性真实导入
   保留**：宿主调用会 ImportError（结构洞）——未获删除批准的代码逐字保留，不删不加。
7. **mp e2e 在 win32 spawn 上跑**：fork 相关分支（inherited_fds）按删除项 4 裁除，空表行为
   保留；unix fork 路径不在本宿主验证面内。

## 测试面（84 passed；断言 pin 可观察行为，非自洽）

- **m1/m2**：get_class 四分支分发 + type/qualname/unknown 三种拒绝 + 删除后果（"ray"→
  ValueError）+ 源面无 Ray/ExternalLauncher 类引用。
- **m3**：uni 直调（blocking list / single_value / non_block 完成 Future / 异常 Future /
  AsyncOutputFuture 只等 D2H / 二次 result 瞬时 / timeout 未实现 / execute_model 完成即抛）+
  run_method 三分支（str/missing→NotImplementedError/bytes cloudpickle/callable）。
- **m5**：构造只记 rank（实例 dict 为证）+ init_worker 前属性访问递归（真实既定行为）+
  qualname 解析（含规范 vllm qualname→本包 Worker）+ 拒绝传类 + __getattr__ 透传 +
  wrapper.execute_model 先 _apply_mm_cache。
- **m6**：'auto'→"vllm.v1.worker.gpu_worker.Worker" 逐字 + resolve 双向解析。
- **m7**：非 cuda 设备 RuntimeError 原文；dist_init 先于 snapshot（记录器断言）；DP 修正
  local_rank += dp_local_rank×tp×pp；V1 runner 构造。
- **m8**：tag=weights / tag=kv_cache 池上下文 + nullcontext 快路径（cuda+cumem 关闭的默认
  部署）+ num_gpu_blocks 回填 + 两段 docstring 逐字。
- **m9**：warmup 从大到小 + capture 列表跳过 + 次序链（loras→kernel→capture→sampler→
  jit→freeze→开闸）+ CompilationTimes 形状 + gate 双态（未配置不开 / 配置了开且 non-CUDA
  no-op 分支真实走到）。
- **m15**：State error 原文双向 + 十字段 NamedTuple + 解包即清 + 真 torch logits 上 bitmask
  禁位→-inf、argmax 翻转 + 空槽早退分支 + Worker 层两段委托 + PP=1 快路径 + 0-token 委托。
- **m16**：懒 wait（首次访问 .tensors 触发、二次不重触发、postprocess 执行）+ 幂等。
- **m12**：FIFO 排空（后收者替先收者收尸、三者全 done）+ 异常转发 + timeout 未实现。
- **m17/m18**：EXECUTOR_FAILED 哨兵 replica（core.py:L1029-L1031 逐字）经 register 回调入队；
  e2e——RPC 级 boom→FAILURE 文本 / slowpoke→TimeoutError 原文 / proc.kill()→monitor→
  is_failed+callback / shutdown 后 MQ 关闭 + worker 退出 / _finalizer 兜底触发。
- **m4/m11/m13/m14/m20 e2e**：真 spawn 拉起（握手后 loaded=True、双进程双 pid）+ 广播全员
  收到 + callable cloudpickle 整函数下发 + execute_model 单点收割（dict 非 list）+
  output_rank 公式（TP8/PP4→24）+ 两段式 ②None→④dict 配对 + 异步调度 D2H 线程（Async 输出
  在子进程内解包后回包）。
- **生命周期 e2e**：determine_available_memory/initialize_from_config/compile max 汇回/
  check_health 全走真 collective_rpc。

## 收工审计（2026-08-22）

- `python -m pytest tests/ -q` → **84 passed**（28s；三连跑稳定：27.8/27.4/28.0s）。
- `python scripts/lint_fidelity.py 本章目录` → **全部通过**（must_keep 66 符号 over_subtraction
  空账；无发明标记）。
- 行尾：implementation/ 与 tests/ 全部 LF-only（`data.count(b'\r\n') == 0`）。
- 全部 `# SOURCE:` 锚经 linter「def/class 跨度内含锚」判据核过（含嵌套闭包/装饰器函数）。
