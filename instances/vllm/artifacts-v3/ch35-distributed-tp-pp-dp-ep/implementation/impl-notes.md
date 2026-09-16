# ch35《分布式 TP/PP/DP/EP》impl-notes —— 只做减法精简版

对应真实源码 pin **vLLM v0.27.1（6e448d0ea）**，行号全部现核（2026-09-14，
`instances/vllm/source`）。**不是** v2 资产的 v0.21.0 旧行号。

运行：`cd instances/vllm/artifacts-v3/ch35-distributed-tp-pp-dp-ep && python -m pytest tests/ -q`
→ **41 passed**（全 host：单进程单元 + spawn gloo 进程池 1/2/4/8 进程形态；
无 vllm 包安装、无 CUDA——NCCL 专有面以 pynccl/torch.distributed seam 承载）。
⚠ 上一轮崩溃残留 `tests/.tmp/` 的 FileStore 文件会毒化 rendezvous（gloo file
init 读到旧 world 数据 → 挂死）：store 路径已带进程号防跨轮复用，残留目录仍建议先删。

**验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包（HOST SEAM
例外见 §Seam 清单——每个 seam 行内标注真实源锚与偏离声明）。SUBTRACTED 标记
逐条挂 dossier.subtraction_plan.delete[0..8] 批准项编号；章界外域段以「→ chN」
注记（ch18 起的切面惯例）。

**lint**：`python scripts/lint_fidelity.py <本章目录>` → **全部通过（无
BLOCKING）**；must_keep 62 符号经 linter `over_subtraction` 项全数核在。
锚点另经自建 AST 校验器对 pin 现核（每个 def/class 的 `# SOURCE:` 行号与
真实文件 AST 行号一一对表）。

**锚点双置惯例**（ch22/ch25/ch26 同款）：每个 def/class 的 `# SOURCE:` 锚点在
声明上方（供读者）与 def/class 体内复置一行——后者是 `lint_fidelity.
_spans_missing_source` 的跨度判据（span=def 行上一行到函数尾）。

## 四段主线 ↔ 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `vllm/distributed/parallel_state.py` | 同名 | **主文件 1**（站 1-5/m1/m2/m3/m4/m5/m8/m11）：Handle Protocol + 模块级集合算子+fake（L152-L197）+ direct_register_custom_op 三注册（L352-L368）+ GroupCoordinator（双群组/__init__ L409-L527 minus 删除项 1 的 split_group/xpu 臂、用户面 L662-L748、broadcast_tensor_dict L885-L965、_should_use_all_gather、isend/irecv_tensor_dict L1019-L1198 minus CPU 旁路、barrier/send/recv/dispatch/combine）+ get_*_group 单例族 + init_distributed_environment（DP rank 偏移段 L1608-L1638 逐字）+ initialize_model_parallel（5 维张量四刀 L1812-L1950 逐字 minus 弹性 EP 覆写）+ destroy/in_the_same_node_as/_node_count |
| `vllm/distributed/communication_op.py` | 同名 | **主文件 2**（m7 接缝）：整文件逐字——tensor_model_parallel_* 自由函数族 |
| `vllm/v1/worker/gpu_worker.py` | 同名 | **主文件 3**（站 5/m9/m10/m12）：AsyncIntermediateTensors（L96-L125 逐字）+ Worker 载体（_pp_send_work L175-L176 + execute_model L1019-L1107 逐字 minus 删除项 8）+ init_worker_distributed_environment（L1347-L1389） |
| `vllm/v1/engine/core.py` | 同名 | **主文件 4**（站 6/m13/m17/m19/m20）：EngineCoreProc 收窄载体（EngineCore 基类 L103-L1000 的 add_request/capture/shutdown/resume/execute_dummy 按原行号并入；IO 双线程 → ch05）+ run_engine_core DP 出生分叉（L1272-L1360）+ DPEngineCoreProc 全景（L1918-L2188 minus 删除项 6：两阶段暂停/32 步共识/-1 哨兵/stale wave 上报逐字） |
| `vllm/v1/engine/core_client.py` | 同名 | **主文件 5**（站 8-10/m14-m17）：make_async_mp_client 三分支（L116-L139 逐字）+ AsyncMPClient（MPClient L503-L777 按 ch05 域扁平并入，identity 表段 L632-L672 原行号）+ DPAsyncMPClient（统计订阅/FIRST_REQ 转投 L1249-L1428 逐字 minus SCALE_ELASTIC_EP）+ DPLBAsyncMPClient（**score 主式 L1468-L1519 v0.27.1 重写版逐字**——⚠ v2 的 waiting*4+running 已废）+ abort 按引擎路由（L1587-L1607） |
| `vllm/v1/engine/coordinator.py` | 同名 | **主文件 6**（m18/m19/m20）：DPCoordinator/EngineState/DPCoordinatorProc（三 socket 主循环 L189-L455 逐字 minus SCALE_ELASTIC_EP 处理块；wave 语义 docstring L23-L57 逐字；_send_start_wave L458-L467） |
| `vllm/v1/worker/dp_utils.py` | 同名 | **主文件 7**（m21/站 15）：_run_ar（4×dp AR）+ _post_process_dp_padding/cudagraph_mode + _synchronize_dp_ranks + coordinate_batch_across_dp（L164-L225 逐字 minus 阈值面） |
| `vllm/distributed/device_communicators/cuda_communicator.py` | 同名 | **主文件 8**（m6/站 11）：__init__（L30-L207 minus 后端家族分支）+ all_reduce 七级回退链（L275-L341 逐字）+ all_gather/reduce_scatter(v)（symm-mem 臂注为省略）+ all_gatherv（L604-L668）+ dispatch/combine 转发（L710-L769） |
| `vllm/distributed/device_communicators/all2all.py` | 同名 | **主文件 9**（m22/站 14）：AgRsAll2AllManager（L44-L153 逐字：dispatch=all_gatherv/combine=reduce_scatterv）；DeepEP/MoRI/nixl/flashinfer 全家族 → ch26/ch40 |
| `vllm/distributed/device_communicators/base_device_communicator.py` | 同名 | 整文件逐字（All2AllManagerBase 契约 + DeviceCommunicatorBase 用户面；仅 ray CACHE 键清理/checkpoint 面按域注记） |
| `vllm/model_executor/layers/linear.py` | 同名 | 站 11/m7：UnquantizedLinearMethod（裸 Parameter+形状断言=nn.Linear 面；ModelWeightParameter → ch23）+ LinearBase/ColumnParallel（__init__ 装配子集）+ RowParallelLinear.__init__/forward（**L1748-L1774 逐字**——bias 只在 rank0） |
| `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py` | 同名 | m22 消费现场：MoEPrepareAndFinalizeNaiveDPEPModular（prepare 尾段 **dispatch 调用位 L158-L164 逐字**、finalize L187-L209 逐字）；量化细节 → ch27、Monolithic → 删除项 9 |
| `vllm/model_executor/layers/fused_moe/{modular_kernel,topk_weight_and_reduce,config}.py` | 同名 | 契约面：FusedMoEActivationFormat/TopKWeightAndReduce ABC/PrepareResultType + TopKWeightAndReduceContiguous（moe_sum 以等义 torch.sum 承载，行内标注） |
| `vllm/config/parallel.py` | 同名 | 站 3/7/16 配置面：DP/EP 字段族 + stateless_init_dp_group（L621-L662 逐字）+ use_all2all/use_sequence_parallel_moe（L690-L695/L673-L687 逐字）+ **has_unfinished_dp/sync_dp_state（L727-L762 逐字——MAX≡OR / 2 元素 SUM 双共识）** |
| `vllm/config/{vllm,utils,__init__}.py`、`vllm/{envs,logger,sequence,forward_context}.py`、`vllm/platforms/__init__.py`、`vllm/utils/*`、`vllm/v1/{outputs,serial_utils,utils}.py`、`vllm/v1/metrics/stats.py`、`vllm/v1/core/sched/{interface,output}.py`、`vllm/v1/engine/{utils,__init__}.py`、`vllm/distributed/utils.py`、各 `__init__.py` | 各自真实路径 | 载体/接缝（见 §Seam 清单）：needs_dp_coordinator L661-L681 逐字、IntermediateTensors L12-L43 逐字、set/get_current_vllm_config、@config 装饰器、DPMetadata、stateless gloo 构造族、split_tensor_along_last_dim、direct_register_custom_op、make_zmq_socket、launch_core_engines 的 coordinator 出生段 L1087-L1110 |

## HOST SEAM 清单（真源码锚 + 偏离声明，全部行内标注）

| seam | 真实源锚 | 偏离 |
|---|---|---|
| `PyNcclCommunicator`（pynccl.py） | pynccl.py:L60-L434 | NCCL C API → torch.distributed（组内 gloo）实现同一集合语义；all_gatherv=逐 root broadcast、reduce_scatterv=逐 root reduce（同通信图）；异步 stream 语义退化为同步完成（gloo 无 stream；调用方随后必然同步，观察结果等价）。`disabled=False` 恒可用 |
| CustomAllreduce/QuickAllReduce/FlashInferAllReduce/AiterCustomAllreduce/SymmMemCommunicator/symm-mem 谓词 | custom_all_reduce.py:L56+/quick:L44+/flashinfer:L322+/aiter:L19+/symm_mem.py:L25+/all_reduce_utils.py:L112+ | 构造/判定面 only、never enabled（宿主无对应硬件/库）——真实部署由平台决定；回退链对调用方透明 |
| `MessageQueue`（shm_broadcast.py） | shm_broadcast.py:L465-L732 | SHM 环形缓冲+ZMQ XPUB → torch.distributed 对象广播（同『src=writer 的全员一致』可观察契约）；create_from_process_group 保留集合握手步 |
| `_HostPlatform`（platforms/__init__.py） | platforms/interface.py Platform 谓词族 | 宿主无加速器：is_cuda* → False、device → cpu；get_device_communicator_cls 仍解析到本章 CudaCommunicator 切面（pynccl seam 恒走 torch.distributed 兜底=真实源码自述的 testing 路径） |
| envs/logger/divide/is_moe_layer 等 | envs.py/logger.py/utils/__init__.py | 接口位 stand-in（默认值对 pin；消费面仅属性访问） |
| `EngineCoreProc` 收窄载体 | core.py:L1008-L1915 + EngineCore L103-L1000 | EngineCore 基类按 ch09 域扁平并入（保留下游 DPE 触碰方法的原始行号）；scheduler/model_executor 由测试注入；_process_input_queue（L1404-L1433）→ ch09 域以注入承载 |
| `AsyncMPClient` 扁平载体 | core_client.py:L974-L1246 + MPClient L503-L777 | 引擎发射/握手/ready 等待 → ch05；identity 表段/ensure_alive 族按原行号并入；outputs IO 任务以注入承载 |
| `VllmConfig`/`SchedulerOutput`/`ModelRunnerOutput`/`SchedulerStats`/`EngineCoreRequest(Output s)` 字段子集 | vllm.py:L331+/sched/output.py:L193-L283/outputs.py:L261-L321/stats.py:L186-L214/__init__.py:L97-L154 等 | 本章消费面字段子集（差量协议/logprobs/观测族 → ch18/ch08/ch30） |
| `distributed_executor_backend` 类型 | parallel.py:L243-L246 | `type[Executor]` 联合臂与 None 回填（L909-L941 ray/平台判定）→ ch17，类型收窄为 `str | DataParallelBackend | None` |

## 1:1 Source Map（核心行；改动=减法或 seam，原因=批准条/章节边界）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| 5 维张量 + 四刀 | parallel_state.py:L1812-L1950 | 逐字 minus enable_elastic_ep 各覆写/DCP/EPLB 段 | must_keep（initialize_model_parallel）；站 1-3——m2 手推 8 GPU 例见 tests |
| GroupCoordinator.__init__ | parallel_state.py:L409-L527 | 逐字 minus split_group 门（L434-L445）/xpu·oot 臂（L492-L497）/use_cpu_custom_send_recv（L529-L533） | must_keep（cpu_group/device_group/device_communicator/use_all2all/use_custom_op_call）；删除项 1 |
| 模块级算子+fake+注册 | parallel_state.py:L152-L197, L352-L368 | 逐字（patched_fused_scaled_matmul 族 L200-L249/L370-L377 裁） | must_keep（all_reduce_fake/direct_register_custom_op）；m4 |
| all_reduce 用户面双路径 | parallel_state.py:L662-L748 | 逐字 | must_keep（all_reduce/all_gather/reduce_scatter/_all_reduce_out_place）；m5 |
| isend/irecv_tensor_dict | parallel_state.py:L1019-L1074, L1114-L1198 | 逐字 minus CPU 旁路 | must_keep；m8/m11——metadata 走 cpu_group、TP 切片优化 |
| barrier | parallel_state.py:L1200-L1207 | 逐字 | must_keep——『NCCL barrier 陷阱』注释原文 |
| run_engine_core 出生分叉 | core.py:L1272-L1360 | 逐字 minus tracer/numa（L1274-L1293）、kv 后缀 | must_keep；m13——'treat like DP=1' 注释原文 |
| DPEngineCoreProc.run_busy_loop | core.py:L2102-L2169 | 逐字 minus eep 推进段（L2112-L2122）/iteration_details 分支 | must_keep（ignore_start_dp_wave/pending_pause）；m19 |
| _has_global_unfinished_reqs | core.py:L2171-L2188 | 逐字 | must_keep——32 步门 + 共识后置 ignore |
| _maybe_publish_request_counts | core.py:L2075-L2090 | 逐字 | must_keep——-1 哨兵 + step/wave 盖章 |
| make_async_mp_client | core_client.py:L116-L139 | 逐字 | must_keep；站 8 三分支 |
| get_core_engine_for_request（DPLB） | core_client.py:L1468-L1519 | 逐字 minus late-interaction 分支（L1470-L1474） | must_keep；m15——**v0.27.1 score 重写版**，注释原文保留 |
| add_request_async（DPAsync） | core_client.py:L1410-L1425 | 逐字 | must_keep（current_wave/client_index/START 抢先）；F3/F4 回收现场 |
| sync_dp_state / has_unfinished_dp | config/parallel.py:L727-L762 | 逐字 | must_keep；m19/站 16 |
| DPCoordinatorProc.process_input_socket | coordinator.py:L189-L455 | 逐字 minus SCALE_ELASTIC_EP（L311-L345） | must_keep（DPCoordinator/DPCoordinatorProc/_send_start_wave/START_DP_WAVE）；m18/m20 |
| CudaCommunicator.all_reduce | cuda_communicator.py:L275-L341 | 逐字 | must_keep（CudaCommunicator）；m6 七级回退链 |
| AgRsAll2AllManager.dispatch/combine | all2all.py:L101-L150 | 逐字 | must_keep；m22 |
| RowParallelLinear.forward | linear.py:L1748-L1774 | 逐字 | 站 11——TP 消费现场；weight.shape 断言按 implementer 契约（`list(shape)==[out,in/tp]`——torch.Size 是 tuple 子类，`==[...]` 恒 False） |
| coordinate_batch_across_dp | dp_utils.py:L164-L225 | 逐字 minus 阈值判定 | must_keep（num_tokens_across_dp）；m21 |
| stateless_init_dp_group | config/parallel.py:L621-L662 + distributed/utils.py:L576-L684 | 逐字（listen_socket 直建臂 → 弹性 EP 域注裁） | must_keep（stateless_init_dp_group）——引擎级 gloo dp_group |
| execute_model（PP 接力） | gpu_worker.py:L1019-L1107 | 逐字 minus 删除项 8（SP 预备块/pooling）+ @with_gpu_sync_check（L1018，ch17） | must_keep（AsyncIntermediateTensors/_pp_send_work/_pp_broadcast_prev_sampled_token_ids）；m9/m10/m12 |

## 与测试的对表（TDD：先写测试后实现；测试锚定真实源码行为）

- **m2 手推修正**：dossier m2 注记的 8 GPU worked example 中 PP 组值有误——真实
  源码（逐字）对 (dp,pp,pcp,tp)=(2,2,1,2) 切出的 **PP 组是 [[0,2],[1,3],[4,6],[5,7]]**
  （每引擎各自的 tp0/tp1 两条流水 lane），不是 [[0,4],[1,5],...]（后者是 DP 组员——
  跨引擎）。DP 组 unbind 行序为 (tp,pp,pcp) 行主序：[[0,4],[2,6],[1,5],[3,7]]（集合
  与直觉一致）。EP 组 [[0,1,4,5],[2,3,6,7]] 与 dossier 一致。**测试断言按真实源码
  行为锁定**——writer 写 worked example 时务必以代码输出为准（本表已给正确值）。
- sync_dp_state 的 `has_unfinished_global`：全员 pending_pause（count==dp_size 整除
  完成）且无未完请求 → False（『部分 rank 还在等共识』才算有活）——docstring
  原文语义，测试按真公式锁定。
- combine 归位是 **owner 视角**：reduce_scatterv 按 sizes[rank] 切回本 rank 的行，
  不是全网行（naive_dp_ep 消费现场的 output 形状即 [local, hidden]）。

## 环境/宿主注记

- 全部测试 host 可跑：gloo 后端；`make_zmq_socket`/coordinator 用真 ZMQ
  （XPUB/XSUB/PULL/PAIR）——XSUB 订阅帧 b"\x01"（win32 libzmq 上
  setsockopt(SUBSCRIBE) 对 XSUB 报 EINVAL，测试脚手架已按真实订阅帧语义走）。
- 多进程测试经常驻 spawn 池（tests/_pool.py）复用 worker；DP>1 作业的引擎内坐标
  拆分（global = dp_rank×engine_size + engine_rank）在池侧按站 5 公式实现。
