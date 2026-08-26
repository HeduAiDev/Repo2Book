# ch10 精简版实现笔记 —— 连续批处理与 chunked prefill

只做减法的忠实精简版：与真实 `vllm/v1/core/sched/`（基线 **vLLM v0.27.1, 6e448d0ea**）
同名同结构同控制流，仅删除 `dossier.subtraction_plan.delete` 批准的子系统。删除点全部
`# SUBTRACTED:` 标注（附原行号）；每个 def/class 带 `# SOURCE: vllm/...:Lxxx` 现核行号。

## 文件构成

| 精简版文件 | 对应真实 vLLM | 角色 |
|---|---|---|
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 主角：schedule() 两阶段分账主线 + _update_after_schedule + _make_cached_request_data + add_request 主路径 + _preempt_request（因果层） |
| `interface.py` | `vllm/v1/core/sched/interface.py` | SchedulerInterface 契约（迭代级调度一句话定义）+ PauseState |
| `output.py` | `vllm/v1/core/sched/output.py` | NewRequestData（首次全量）/ CachedRequestData（增量）/ SchedulerOutput |
| `request.py` | `vllm/v1/request.py` + `vllm/sampling_params.py` | Request 账本字段群 + RequestStatus + 迷你 SamplingParams |
| `request_queue.py` | `vllm/v1/core/sched/request_queue.py` | FCFSRequestQueue（四操作）+ create_request_queue 工厂 |
| `kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | KVCacheManager **接口契约面**（黑盒：allocate_slots 满则 None / get_computed_blocks 命中折算 / full_sequence_must_fit 整序列准入门） |
| `scheduler_config.py` | `vllm/config/scheduler.py` | 预算与切块旋钮真相源（2048/128 基线、threshold、chunked 开关、reserve_full_isl、watermark） |
| `arg_utils.py` | `vllm/engine/arg_utils.py` + `vllm/usage/usage_lib.py` + `vllm/utils/mem_constants.py` | get_batch_defaults 硬件/场景仲裁表（A100 反例 #17885） |

## 1:1 Source Map（精简版 ↔ 真实 vllm@v0.27.1 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实位置 | 改动 | 原因 |
|---|---|---|---|
| `Scheduler.schedule` | `scheduler.py:L439-L1253` | 删 async 剪枝 L488-L502 / next_decode_eligible L504-L508 / defer_prefills L510-L514 / PRIORITY 抢占分支 L590-L613 / spec 登记 L640-L656 / LoRA L673-L681 / 阻塞态提升 L700-L711 / stale 跳过 L713-L722 / connector 外部命中 L739-L826 / load_kv_async L866-L873+L1023-L1053 / spec pad L881-L897 / encoder L916-L963 / reserved_blocks L965-L971 / use_v2 分支 L1132-L1142 / CoW+dynamic-sd L1165-L1206 / connector meta L1231-L1244 | 全部 dossier.delete 第 1/5/6/7/8/9/11/12/13 条批准；删后单机文本调度的两阶段控制流完整自洽 |
| 追赶公式（RUNNING） | `scheduler.py:L516-L532` | 原样保留（含 threshold/预算双钳制与 max_model_len 保险） | 「不分 prefill/decode 相」的核心；must_keep |
| num_new==0 continue | `scheduler.py:L557-L573` | 原样保留（含 woosuk 的 continue-not-break 注） | m6：不严格 FCFS；must_keep |
| 抢占重试环 | `scheduler.py:L576-L629` | 删 PRIORITY 分支，保留 FCFS `self.running.pop()` 抢队尾 | dossier.delete 第 6 条；FCFS 已足以讲清「RUNNING 可抢占」 |
| WAITING 守卫 | `scheduler.py:L683-L684` | 原样保留 `if not preempted_reqs and UNPAUSED` | m8：本拍抢占过就整拍不收新；must_keep |
| 前缀命中折算 | `scheduler.py:L744-L766` | 保留无 connector 的 else 分支（3 元组解包含 shared_prefix_boundary） | m9 黑盒调用；链式哈希归 ch15 |
| 切块三闸 | `scheduler.py:L874-L914` | 删 spec pad，保留 num_tokens−computed → threshold → chunked 开关 break → min(budget) | m10 核心；spec pad 是 WC2 代价证据（正文引原文） |
| 准入 allocate_slots | `scheduler.py:L973-L994` | 删 connector/encoder 四参，保留 `full_sequence_must_fit=self.scheduler_reserve_full_isl` 与 None→break | m11/WC4：整序列准入门；WAITING 绝不抢占 |
| 出队入 running | `scheduler.py:L1022-L1082` | 删 load_kv_async/LoRA/pad/encoder，保留状态分流 + `num_computed_tokens=命中数` + `_inflight_prefills` | m12 must_keep |
| 守恒断言 | `scheduler.py:L1108-L1119` | 原样保留 | m13：Σ≤预算、running≤max_num_seqs、调度数≤running 数 |
| new/cached 二分组装 | `scheduler.py:L1131-L1163` | 删 use_v2 分支；保留 from_request 全量 + _make_cached_request_data 增量 + prev_step 刷新 | m14/WC5 must_keep |
| SchedulerOutput 落袋 | `scheduler.py:L1208-L1229` | 删 encoder/mamba/connector/spec 六参，保留 num_scheduled_tokens/total/preempted/finished | must_keep；差量协议深挖归 ch18 |
| `_update_after_schedule` | `scheduler.py:L1317-L1365` | 删 structured 累计、defer_block_free 栅栏、routed experts 快照；保留乐观推进 + is_prefill_chunk + 集合换新不 clear | m15 must_keep；「账本先记、GPU 后算」 |
| `_make_cached_request_data` | `scheduler.py:L1410-L1467` | 删 PP 回传分支；保留 prev_step 差量判定 + resumed_req_ids | m14 must_keep |
| `add_request` | `scheduler.py:L2213-L2235` | 删重复 id 流式续跑分支与 connector/record_event，保留 `_enqueue_waiting_request` + requests 登记 | 第 1/10/11 条批准；else 主路径即完整正确 |
| `_preempt_request` | `scheduler.py:L1274-L1315` | 删 encoder free 与 async stale 标记；保留 free 块 → PREEMPTED → computed=0 → 回 waiting 队头 → 记 reset_preempted_req_ids | 本章只到「抢占因果层」；recompute-only/stale 深挖归 ch11 |
| `_enqueue_waiting_request` / `_is_blocked_waiting_status` / `_select_waiting_queue_for_scheduling` | `scheduler.py:L2050-L2074` | 保留原判与 FCFS 选队；删 PRIORITY 比较 | m17：双队列结构保留，阻塞态细节归 ch11 |
| `__init__` 约束装配 | `scheduler.py:L108-L123/L177-L199/L305-L307` | VllmConfig 装配换裸标量；保留 max_num_scheduled_tokens 回落解析、三容器、prev_step、reserve_full_isl、_pause_state、_inflight_prefills | m19；被删子系统字段随批删 |
| `SchedulerConfig` | `vllm/config/scheduler.py:L42-L80/L99-L105/L130-L141` | pydantic 换 dataclass；保留全部本章旋钮与 docstring 原文 | m4 预算真相源 |
| `EngineArgs.get_batch_defaults` | `arg_utils.py:L2514-L2596` | 平台探测换显式参数（默认值=探测失败回退档）；保留 GPU 决策表 L2541-L2563 逐字 | m4：A100 反例 #17885 的仲裁表 |
| `KVCacheManager`（契约面） | `kv_cache_manager.py:L229/L344/L567/L703/L876` | 分页/哈希/CoW/水位全删，按「空闲块计数 + 每请求持有块账」实现同一签名与满则-None 语义；full_sequence_must_fit 门保留 | dossier.scope_note「当黑盒契约面用」；ch13/14/15 的主角 |

## 测试覆盖（tests/test_scheduler.py，25 项，纯单元无 import vllm）

复现的真实可观测行为：追赶公式 decode 恒 1 / chunk 续切 4 拍走 RUNNING 阶段 /
threshold 双侧钳制 / chunked 关闭整拍 break / RUNNING 先吃预算·Σ=2048 守恒 /
PAUSED_ALL 短路 / max_num_scheduled_tokens 回落 / 整序列准入门开 vs 关（首 chunk
装得下≠整条装得下）/ 前缀命中折算（打桩 64 命中→只排 192）/ KV 耗尽抢队尾 +
被抢者 computed=0 回队头 + 抢占拍不收新 / 恢复请求走 resumed 分流 + 上拍没调度过
补全量 all_token_ids / WAITING 准入 None→break 在场请求毫发无损 / max_num_seqs 上界 /
num_new==0 不阻塞低优先 / NewRequestData 全量 vs CachedRequestData 增量 /
乐观推进 + is_prefill_chunk + _inflight_prefills 生命周期 / FCFS 四操作 /
config 基线 2048/128 + arg_utils 仲裁表（H100 档/A100 反例/小卡档）/ 256×1+1×8192
混相批算术。⑤拍生命周期（完成/free）在测试中以真实调用手工模拟（append_output_
token_ids / running.remove + kv_cache_manager.free），对应 ch9/ch11 的精简版范围。

## 桩说明

- `kv_cache_manager.py` 是**接口契约面**而非分页器：保留真实方法签名、返回 None 语义、
  full_sequence_must_fit 整序列检查与命中块挂账；前缀哈希（ch15）、块池/CoW（ch13）、
  watermark（ch14）删除。`get_computed_blocks` 默认无命中（等价关前缀缓存），测试可打桩
  注入命中数验证调度器的消费语义。
- `record_function_or_nullcontext` 以空上下文顶替（dossier.delete 第 11 条），控制流与
  缩进不变。
