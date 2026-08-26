# ch11 精简版实现笔记 —— 抢占与请求的一生

只做减法的忠实精简版：与真实 `vllm/v1/core/sched/`（基线 **vLLM v0.27.1, 6e448d0ea**）
同名同结构同控制流，仅删除 `dossier.subtraction_plan.delete` 批准的 11 类子系统。
删除点全部 `# SUBTRACTED:` 标注（附原行号）；每个 def/class 带 `# SOURCE: vllm/...:Lxxx`
现核行号（对 v0.27.1 现核，非 v2 资产旧行号）。

两条主线全保留：**段一 抢占与恢复**（RUNNING 抢占重试环 → _preempt_request 六件事 →
守卫关闸 → 双队列遍历 → 前缀重命中 → 水位准入 → resumed 回流）；**段二 一生的收尾**
（update_from_output 热循环 → 逐 token → check_stop 五连判 → 停止分流 → 终点 free →
外部 abort）。ch10 留下的两个钩子（allocate_slots None 的抢占内幕、守卫的 why）在本章
精简版里全部兑现。

## 文件构成

| 精简版文件 | 对应真实 vLLM | 角色 |
|---|---|---|
| `scheduler.py` | `vllm/v1/core/sched/scheduler.py` | 主角：抢占环 + _preempt_request（stale 协议全保留）+ 守卫 + 双队列遍历 + 水位准入 + resumed 回流 + update_from_output 热循环 + 停止分流 + finish_requests + 终点 free |
| `request.py` | `vllm/v1/request.py` + `vllm/sampling_params.py` | 一生的账本字段群（四计数器/num_preemptions/block_hashes）+ RequestStatus 全状态 + get_finished_reason + 迷你 SamplingParams/RepetitionDetectionParams |
| `utils.py` | `vllm/v1/core/sched/utils.py` | check_stop 五连判 + check_sequence_repetition + remove_all（130 行全保留，四个函数全在路径上） |
| `request_queue.py` | `vllm/v1/core/sched/request_queue.py` | FCFSRequestQueue 十操作（prepend=appendleft 被抢者回队头靠它；remove_requests 过滤重建） |
| `kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` + `vllm/v1/core/kv_cache_utils.py:L691-L748` | 契约面 + **watermark 水位两处原文** + **free 不清哈希 → get_computed_blocks 前缀重命中**（F2）+ get_request_block_hasher 增量哈希器 |
| `output.py` | `vllm/v1/core/sched/output.py` + `vllm/v1/outputs.py:L261-L271` | SchedulerOutput（preempted/finished 两个生死通告面）+ ModelRunnerOutput（⑤ 拍输入面） |
| `engine.py` | `vllm/v1/engine/__init__.py:L43/L184/L230` | FinishReason + EngineCoreOutput（四字段版）+ EngineCoreOutputs |
| `scheduler_config.py` | `vllm/config/scheduler.py` | scheduler_reserve_full_isl + watermark 两个旋钮真相源（+ ch10 已立的预算旋钮） |
| `interface.py` | `vllm/v1/core/sched/interface.py` | update_from_output / finish_requests 两段契约原文 + PauseState |

## 1:1 Source Map（精简版 ↔ 真实 vllm@v0.27.1 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实位置 | 改动 | 原因 |
|---|---|---|---|
| 抢占重试环 | `scheduler.py:L575-L629` | 删 PRIORITY 分支（L590-L613：max((priority, arrival_time)) 选择+本拍已领 token/块/预算/encoder 回滚）与 drop_stale_output=self.requires_kv_delivery 实参（L620）；保留 FCFS `self.running.pop()` 抢队尾 + preempted_req==request 自我放弃 break + new_blocks None 整拍 break | dossier.delete 第 1/3 条；FCFS 队尾必是本拍未调度者，无需回滚——PRIORITY 的回滚逻辑随分支删 |
| `_preempt_request` | `scheduler.py:L1274-L1315` | 删 encoder_cache_manager.free（L1291）与 log_stats record_event（L1310-L1311）；**六件事其余全保留**：free 块/PREEMPTED/computed=0/清 spec/stale 标记（assign 不累加 + drop 旗标继承，L1297-L1308 原文）/num_preemptions+1/回 waiting 队头 | 本章绝对核心（m3/m4）；docstring 原文保留（drop_stale_output 的 reset_prefix_cache 语义是第 9 条只删调用方不删协议的注脚） |
| WAITING 守卫 | `scheduler.py:L683-L684` | 原样保留 `if not preempted_reqs and self._pause_state == PauseState.UNPAUSED` | m5 核心：本拍抢占过=内存紧张信号→整拍不收新；PAUSED_ALL 预算短路（L460-L462）随第 11 条删 |
| 双队列遍历 | `scheduler.py:L685-L737` | 保留 step_skipped_waiting 收集、阻塞态 `_try_promote` 失败跳过（L700-L711，删 REMOTE_KVS debug 日志）、**stale 在途推迟一拍（L713-L722 原文）**、LoRA max_loras（L724-L737，未获删除批准，原样保留）；删 num_waiting_for_streaming_input 计数（L690） | m6/m4 核心；ch10 删掉的这三个块本章全数恢复 |
| 前缀重命中 | `scheduler.py:L744-L766` | 保留无 connector 的 else 分支（三元组解包含 shared_prefix_boundary）；删 connector 分支（L749-L759）与外部命中合并（L768-L826）/prefill_stats（L846-L853） | m7/F2 核心；链式哈希内部归 ch15 |
| 恢复准入 | `scheduler.py:L973-L994` | 保留 `full_sequence_must_fit=self.scheduler_reserve_full_isl` + `has_scheduled_reqs=bool(self.running)` + None→break；删 connector/encoder/lookahead/reserved 五参（第 1/2/6 条） | m8/m5：水位作用开关 + WAITING 侧绝不抢占的反差闭环 |
| 回流落位 | `scheduler.py:L1055-L1082` | 保留 running.append + WAITING→new / PREEMPTED→resumed 分流 + computed=命中数 + _inflight_prefills；删 record_event(SCHEDULED)（L1056-L1059）、LoRA 三行原样保留、pad_spec_decode（L1076-L1079） | m9；全序列放得下才收的参数含义归 ch14 |
| 步末重排 | `scheduler.py:L1099-L1101` | 原样保留 `skipped_waiting.prepend_requests(step_skipped_waiting)` | m6：不饿死的关键（测试核过 extendleft 反转的重试序） |
| `update_from_output` | `scheduler.py:L1670-L2048` | 保留 woosuk 瓶颈自注（L1728-L1730）、定位采样行/扣 in_flight/stale 锁步 drain（L1733-L1743）、abort 期完成 continue（L1747-L1755）、drop-mode 丢弃（L1757-L1759）、_update_request_with_output 调用（is_stale 透传）、should_emit、finish_reason 先抓再 handle（L1895-L1907）、stopped_running/preempted 分流、EngineCoreOutput 四字段装配（L1922-L1941）、remove_all 批量摘除（L1946-L1952）、EngineCoreOutputs 装配（L2012-L2017）；删 defer 栅栏/perf/failed_kv_load/routing_offsets（L1684-L1726）、spec 拒绝回扣（L1766-L1791）、encoder/pooling/structured/routed/logprobs/nans（L1793-L1921）、error 攒批/connector 统计/events/finished_req_ids_dict/make_stats（L1954-L2046） | m11-m16 核心；回扣分支归 ch12/spec 章 |
| `_update_request_with_output` | `scheduler.py:L2094-L2111` | **逐字保留**（含 is_stale 只被 AsyncScheduler 覆写用的注释与截断 del） | m12 |
| `_handle_stopped_request` | `scheduler.py:L2076-L2092` | 坍缩为 `return True`（resumable/streaming_queue/_update_request_as_session 随第 5 条删） | dossier.delete 第 5 条明示『简化为直接 return True』；非流式请求恒真终点 |
| `_try_promote_blocked_waiting_request` | `scheduler.py:L2678-L2712` | 三提升分支全删（REMOTE_KVS L2682-L2693 第 1 条 / GRAMMAR L2695-L2703 第 4 条 / STREAMING L2705-L2707 第 5 条——真实该分支本就无条件 return False）→ 坍缩 `return False` | must_keep：方法与『阻塞态的出口』语义保留；三态运行时来源随子系统删——精简版提升永不成功=仍在等，遍历方跳过 |
| `finish_requests` | `scheduler.py:L2237-L2298` | 保留两遍法全部结构（str/set/None 归一化、三队列摘除、置态+free、幂等 no-op）；删 STREAMING 计数（L2274-L2275）与 REMOTE_KVS delay 判定（L2287-L2293） | m17；abort 双投递的引擎侧落点 |
| `_free_request`/`_free_blocks`/`_free_request_blocks` | `scheduler.py:L2300-L2354` | 保留 finished_req_ids 登记 + del requests + kv free 直接路径；删 connector/ec_connector/encoder/finished_req_ids_dict（L2306-L2321）与 defer 栅栏（L2345-L2354） | m16 终点；delay_free_blocks 参数保留原签名恒 False |
| `_update_after_schedule` | `scheduler.py:L1317-L1365` | 保留乐观推进（computed+=n、in_flight+=n）+ is_prefill_chunk + 集合换新不 clear；删 defer 栅栏/structured 累计/routed experts | num_in_flight_tokens 是 stale 协议的赋值来源，必须保留 |
| `KVCacheManager.allocate_slots`（水位） | `kv_cache_manager.py:L463-L488` + `L510-L527` | **两处水位逻辑原文保留**：`watermark_blocks=0`；`has_scheduled_reqs and status∈{WAITING,PREEMPTED}` 才取 `self.watermark_blocks`；full-ISL 门与分配门都 `required = need + watermark_blocks` 对比 free；删 connector/encoder 四参与 reserved_blocks | m8 机制本体（本章 introduces）；coordinator 分组块计算换单组 ceil 算术（ch13 黑盒边界） |
| `KVCacheManager.free`（哈希不清） | `kv_cache_manager.py:L567-L578`（证：`block_pool.py:L719-L742`） | docstring 逆序归还原文保留；块归还空闲池（-1 命中占位不计）+ **cached_block_hashes 不清** | F2 事实锚：真实 free_blocks 只动 ref_cnt/自由队列、从不 touch 哈希 |
| `KVCacheManager.get_computed_blocks` | `kv_cache_manager.py:L229-L295` | 换沿 request.block_hashes 的连续命中计数（max_cache_hit_length=num_tokens-1 的 NOTE 原文保留）；删 find_longest_cache_hit 链式匹配/LRU/events | m7 可观测语义（重命中自己前缀）保留；链式哈希归 ch15 |
| `get_request_block_hasher` | `kv_cache_utils.py:L691-L748` | 保留链式父哈希增量结构（早停/while 全真）；删 mm/LoRA extra_keys 与 caching_hash_fn（换 tuple 哈希） | append_output_token_ids 连带增量块哈希（must_keep）的哈希器来源；真实装配 core.py:L220-L227/L983 |
| `RequestStatus`+计数器群 | `request.py:L348-L390` + `L150-L162` | **全状态+map 原样保留**（顺序即语义）；四计数器/num_preemptions/block_hashes 字段原文保留 | m10 状态机账本+ m4 载体；get_finished_reason 是 finish_reason 时序约束的取用面 |
| `check_stop` 等 | `utils.py:L10-L130` | **无删除**（四函数全在路径上） | m13 五连判+m20 remove_all 快路径 |
| `SchedulerConfig`（两旋钮） | `config/scheduler.py:L130-L141` | scheduler_reserve_full_isl + watermark docstring 原文保留 | 本章 introduces 的配置真相源 |

## 与 ch10 精简版的关系（同一文件的两个裁剪深度）

ch10 按其 dossier 删掉了 `step_skipped_waiting`/阻塞态遍历/stale 协议/水位（其 delete 第
12/13 条），本章按本章 dossier 全数恢复并展开（stale 标记 L1297-L1308、阻塞跳过 L700-L711、
stale 推迟 L713-L722、步末重排 L1099-L1101、kv 侧 watermark 两处）；ch10 删的
`get_finished_reason`/`block_hashes`/`update_from_output`/`finish_requests`/`_free_request`
同样在本章恢复为主角。反向地，ch10 保留的 `PAUSED_ALL` 短路与 `arg_utils` 仲裁表随本章
delete 第 11 条与 scope（预算地形归 ch10）不再出现。

## 测试覆盖（tests/test_scheduler.py，42 项，纯单元无 import vllm）

复现的真实可观测行为：状态机全序 + >PREEMPTED 一比较 + FinishReason 映射（含
STREAMING→STOP 特殊映射）/ 抢占环抢 FCFS 队尾六件事逐项核（含清 spec、stale=in_flight
赋值、free 块归还、回 waiting 队头、preempted_req_ids、守卫整拍不收新）/ 自我抢占整拍放弃 /
WAITING 侧绝不抢占 / **前缀重命中：被抢者恢复只补 1 token（64 命中 + 1 补算 vs 全量 65
重算）**、满前缀命中按块对齐向下取（cap=num_tokens-1）/ **水位三限定**（仅
WAITING/PREEMPTED、须 has_scheduled_reqs、RUNNING 增长不吃）+ watermark_blocks 取整 /
stale 协议（assign 赋值、锁步 drain + 仍送达、drop-mode 整段丢弃、排空前推迟一拍、同步版
自中和）/ 双队列（阻塞态路由 skipped、阻塞队头不饿死 ready、步末重排的重试序）/ 热循环
（定位采样行、扣在途、mid-prefill chunk 空行不外送、EOS→STOP→free→下拍 finished_req_ids
通告 worker、stop_token 记 stop_reason、停止后截断、min_tokens 先于 EOS 的顺序、
max_tokens/max_model_len 双封顶、重复检测）/ 被抢当拍完成的罕见路径（stopped_preempted
从 waiting 摘除）/ finish_requests（abort RUNNING/WAITING/PREEMPTED 三态摘除、幂等、
str/list/None 三形）/ 执行期 abort 的 update_from_output 幂等 continue / remove_all
快路径 / append_output_token_ids 连带增量块哈希。

## 桩说明（契约面边界）

- `kv_cache_manager.py` 仍是**接口契约面**而非分页器（ch10 同款口径，dossier.key_classes
  「当黑盒契约面用」），但相对 ch10 多实现了两块真语义：①watermark 两处扣减（原文行）；
  ②满块哈希表（登记于 allocate 的 cache 提交点、free 不清、get_computed_blocks 重命中）——
  这两块是本章 m7/m8 的可运行载体。哈希器是内容+父链哈希（无 cache_salt/extra_keys），
  无 LRU 驱逐（最坏情况=块被逐出后全量重算的另一半叙事归 ch15 正文，精简版哈希不逐出）。
- `record_function_or_nullcontext` 以空上下文顶替（第 10 条），控制流与缩进不变。
- LoRA 三处分支（L673-L681/L724-L737/L1067-L1068）未获删除批准，原样保留；
  `lora_config` 恒 None 时全为旁路——精简版行为不受影响，正文可一句话带过。
