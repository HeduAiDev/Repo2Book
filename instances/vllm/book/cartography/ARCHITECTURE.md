# vLLM v1 架构真相源（v3 · Phase 0 产出）

> 本文 = v3 重写的**架构底座**：六域深读卡片（`deepread/*.json`）+ 8 处交叉核对裁决的综合。
> 读者假设：懂后端、不懂 AI 推理的工程师。每个设计决策都回答「旧设计→痛点→方案→代价」。
> 源码 pin：**v0.27.1（6e448d0ea，2026-08-15 用户定最新 release）**。本文全部锚点已按 v0.27.1 **全量重核**
> （六域深读重跑 6/6 OK + 交叉核对 6/6 面零冲突）；旧 pin v0.21.0 深读卡归档 `deepread-v021/` 仅供演进对照。
> 配套：L0 架构图 `L0-architecture.svg/png`（全书唯一权威图，见 §1）。
> （v2 的素材收集式地图归档于 `ARCHITECTURE.v2.md`，仅供对照。）

## 0. 一句话主线

**一个请求的一生**：用户发来文本 → API 进程切成 token（`Renderer`）→ 跨进程扔给 EngineCore（ZMQ）→ 引擎逐拍循环「调度→前向→采样→回收」，每拍只问「这批 GPU 能吃多少 token」→ 新 token 跨进程回 API 进程 → 拼回汉字 → SSE 流给用户。

**全系统的第一设计原则：GPU 是最贵的员工，一切 CPU 活都不能让它等。** 三段式进程解耦、异步调度（v0.27.1 起默认开启）、两段式 execute_model、CUDA Graph、采样留 GPU——全部服务于这一句。第二原则：**显存是共享账本，一切调度都先对账**（token 预算 + KV 块池 + 水位护栏）。

### 0.5 v0.22→0.27.1 演进速览（相对本书 v2 所写 v0.21.0 的重大变化，写作必读）

| # | 新事实 | 落点 | 影响 v3 章 |
|---|--------|------|-----------|
| 1 | **Renderer 正式入三件套**：前端三件套 = Renderer+InputProcessor / EngineCore / OutputProcessor；raw prompt 的 tokenize/多模态跑 Renderer **线程池**不下事件循环（PR #49608, input_processor.py:L77-82） | ch3/ch5 | 三件套口径全面更新 |
| 2 | **异步调度默认开启**（#27614, 2025-12-29 翻转默认；async+PP 全支持 #3e440786a）：v0.27.1 的心跳就是重叠版 | config/vllm.py:L1095-1143 | ch11 叙事从「可选优化」改为「默认心跳+如何关」 |
| 3 | **水位（watermark）回归**：watermark_blocks 留 headroom 抑制抢占抖动（默认关，用户按负载开）——v1 曾删 v0 的 watermark，如今以新形态复活 | memory-kv 卡 | ch13 显存账本 |
| 4 | **partial prefix cache + 块内 CoW**（#45939/#46384, 2026-06/07）：cache_partial_block 在块内前缀边界注册哈希不分配新块、部分命中 CoW 拷贝——「只缓存满块」的旧口径已破 | block_pool.py:L445-544 | ch14 前缀缓存 |
| 5 | **Marconi 式共享前缀钉住 + 稀疏驻留**（#37898/#45845/#47782）：hybrid 场景 junction 检测+跨请求复用不被稀疏缓存杀死 | kv_cache_coordinator | ch14/ch24 hybrid |
| 6 | **LRU 双不变量**：free 逆序 + **无哈希块先于缓存块驱逐**（第二条为新增澄清） | kv_cache_manager.py:L567-574 | ch14 |
| 7 | **DPCoordinator 独立进程**（XPUB/XSUB+PULL 三 socket）：数据面/控制面彻底分离，请求输出完全不经过它；统计 100ms 快照、wave 只服务 MoE lockstep | coordinator.py:L23-256 | ch33 分布式 |
| 8 | **多 API server 水平扩展**（--api-server-count）：client_index 由前端盖章进 EngineCoreRequest、引擎按它选 PUSH socket 回发 | core_client.py:L1145-48 | ch3/ch36 服务面 |
| 9 | **vllm/models/<name>/ 硬件隔离新布局**：旗舰架构（DSV4 等）走目录级隔离，平台实现与模型定义分层 | vllm/models/ | ch21/ch27 模型层 |
| 10 | **MLA 三代演化**：DSV2/V3 潜向量 576 维 → DSV4 上下文压缩 + fp8_ds_mla 自定义布局 | mla_attention.py | ch23/ch24 primer |
| 11 | msgpack 零拷贝工程化：小张量（<256B）内联免拷、引擎输出侧「复用 bytearray+首帧 tracker」替客户端保活 | serial_utils.py | ch4 |
| 12 | 注意力后端**逐 KV 组混布**、新后端入表（TOKENSPEED_MLA R1 dims+FP8 KV、FLASHINFER_MLA 等） | selector.py/platforms | ch20 |
| 13 | AsyncMPClient 派生 **DP/LB 子类**（current_wave 盖章） | core_client.py:L1413 | ch33 |

## 1. L0 总图（唯一权威图）

```
┌──────────────────── API 进程（frontend，零 GPU）────────────────────┐
│                                                                    │
│  user ──HTTP──> OpenAI server / LLM(离线) / AsyncLLM(在线)          │
│                    │                          ▲                    │
│              Renderer.render               SSE / iterator          │
│              (tokenize 在这里!)                │                    │
│                    ▼                          │                    │
│              InputProcessor                  RequestOutputCollector│
│         (校验+id 双轨+EngineCoreRequest)      (单槽+Event+合并)     │
│                    │                          ▲                    │
│                    │                 OutputProcessor               │
│                    │              (detokenize+logprobs+组装)        │
└────────────────────┼──────────────────────────┼────────────────────┘
                     │ EngineCoreRequest         │ EngineCoreOutputs
                     │ (只有 token ids,          │ (每步整批聚合)
                     │  无文本!)                 │
┌────────────────────▼──── ZMQ 边界 ────────────┴────────────────────┐
│  下行: ROUTER(bind) ──> DEALER(connect, identity=rank)             │
│  上行: engine PUSH(connect) ──> client PULL(bind)   [HWM=0 无反压]  │
│  载荷: msgpack(array_like) + tensor 零拷贝独立帧 + OOB 共享内存旁路  │
└────────────────────┬──────────────────────────▲────────────────────┘
┌────────────────────▼── EngineCore 进程（busy loop）┴───────────────┐
│  input_queue → EngineCore.step() 逐拍:                              │
│    ① schedule()        ← 慢决策全放 GPU 启动前                      │
│    ② execute_model(non_block=True) → Future                        │
│    ③ get_grammar_bitmask()   ← CPU 活藏在 GPU 算的窗口里            │
│    ④ future.result() → sample_tokens(bitmask)                      │
│    ⑤ update_from_output()                                          │
│                                                                    │
│  Scheduler ◄────对账────► KVCacheManager / BlockPool / 前缀缓存     │
│  (只认 token 数,     (固定块池+逻辑块表+链式哈希;                   │
│   无 prefill/decode    分配失败→抢占 recompute-only)               │
│   相位之分)                                                        │
│        │ SchedulerOutput (差量协议: 新请求全量/老请求 diff)         │
│        ▼                                                            │
│  Executor(进程拓扑) → Worker(设备生命周期) → GPUModelRunner(批次执行)│
│    InputBatch(持久批次) + CpuGpuBuffer(预分配固定地址)               │
│    → piecewise torch.compile(注意力处切图) → CUDA Graph 回放         │
│    → 模型层(DecoderLayer 拼装; Attention=插座, MLA/GQA=变体)        │
│    → compute_logits(只在需要的位置物化)                              │
│    → Sampler 9 步管线 (GPU 全程; 位掩码约束; spec decode)           │
└────────────────────────────────────────────────────────────────────┘
```

## 2. 一个请求的一生（file:line 级走读，全部对 v0.27.1 核验）

**前端进程段**：
1. `POST /v1/chat/completions` → `create_chat_completion`（entrypoints/openai/api_router/chat_completion.py:L51-53，`@with_cancellation` 挂断连监听竞速）→ 建 `SamplingParams`（output_kind=DELTA，protocol.py:L722-724）。
2. `Renderer.render_chat`（online_renderer.py:L117）→ `_tokenize_prompt`（renderers/base.py:L472-487）——**tokenization 在前端、在过线之前**，async 跑在 renderer 线程池（base.py:L96-98，PR #49608：原始 prompt 的阻塞预处理不下事件循环）。
3. `AsyncLLM.add_request`（async_llm.py:L283-418）：已渲染 EngineInput 走同步快路径（L352-365「no blocking preprocessing needed」）；`InputProcessor.process_inputs`（input_processor.py:L323-394）校验/克隆 params/整理 mm 特征/**双轨 request_id**（assign_request_id 随机后缀，L231-249）→ 构造 EngineCoreRequest。
4. `_add_request` 双登记（async_llm.py:L420-435）：先本进程 `output_processor.add_request`（L429，回程消息到达时表已存在）‖ 后 `add_request_async` 过线。
5. `AsyncMPClient.add_request_async`：**盖章 client_index**（core_client.py:L1145-1147，DP 内部 LB 子类再盖 current_wave 按负载分选引擎 L1410-1423）→ `_send_input`：msgpack 多帧 → `(engine_identity, b' ', *bufs)` → ROUTER `send_multipart(copy=False)`（L1104-1123，零拷贝帧由 zmq 引用链保活）。

**引擎进程段**：
6. IO 线程 `preprocess_add_request`（core.py:L1718）→ 忙循环 `add_request`（L1514-1518）→ `Scheduler.add_request`（scheduler.py:L2213-2235）status=WAITING 入队尾。
7. `EngineCore.step()`（core.py:L584-614，v0.27.1 默认重叠版）→ `schedule()`（scheduler.py:L439）：RUNNING 先行（L484-671）→ WAITING（L684 守卫：本拍抢占过不收新）→ `get_computed_blocks` 查前缀缓存（kv_cache_manager.py:L229；混合模型走不动点+**块内细粒度 phase 2** single_type:L741-762）→ `allocate_slots`（scheduler.py:L973-985）：全序列准入门（kv_cache_manager.py:L472-488）+ **free−reserved−watermark 三重预算**（L510-527，水位抑制抢占抖动）→ 不够则抢占 `_preempt_request`（scheduler.py:L1274-1315，recompute-only：块不清哈希回 waiting 队头，恢复时重撞缓存）。
8. 组装 `SchedulerOutput`（scheduler.py:L1208-1229，差量协议）→ `_update_after_schedule` 乐观推进（L1317，num_in_flight_tokens L1327-1331——GPU 还没算，账先记上）。
9. `executor.execute_model(scheduler_output, non_block=True)`（core.py:L596）：mp 走 SHM 广播（multiproc_executor.py:L388）→ `Worker.execute_model`（gpu_worker.py:L1017-1081）→ `GPUModelRunner.execute_model`（gpu_model_runner.py:L4166）：`_update_states` 差量调和（L1192-1520：新块 GPU 置零/finished 移除/老请求 append 块/压实）→ `_prepare_inputs`（L1960：**第一句 commit_block_table 让 H2D 与 CPU 重叠** L1977-1979）→ logits_indices=query_start_loc[1:]-1 → 组注意力元数据 → `set_forward_context` → piecewise 图内前向（Attention 插座：`unified_kv_cache_update` 写 KV L775-798 → `unified_attention_with_output` L819-846，attention.py）→ CUDA Graph 按 BatchDescriptor 查表回放。
10. **与 GPU 前向并行**：主线程 `get_grammar_bitmask`（core.py:L597 → scheduler.py:L1646-1666 → structured_output/__init__.py:L212-359）`fill_next_token_bitmask` 填位掩码——CPU 活藏在 GPU 算的窗口里。
11. 前向完成 → `sample_tokens(bitmask)`（两段式契约，worker_base.py:134-149 一带）→ `compute_logits` 只在采样位物化 → `Sampler.forward` 9 步管线（GPU 全程）→ `ModelRunnerOutput`。
12. `update_from_output`（core.py:L605-614 → scheduler.py:L1924）：请求状态推进（单 IntEnum，`>PREEMPTED 即 finished`）、**按 request.client_index 分桶**产出 `dict[client_index, EngineCoreOutputs]`（scheduler.py:L1924, L2015-2016）、free 块（**逆序 + 无哈希块先驱逐**双不变量，kv_cache_manager.py:L567-574）。

**回程段**：
13. 每步每客户端**一条** EngineCoreOutputs（整批聚合；engine 输出侧「复用 bytearray+首帧 tracker」管理零拷贝缓冲）→ 输出线程 `sockets[client_index].send_multipart`（core.py:L1785-1793，DP 统计走 client_index=-1 哨兵路给 DPCoordinator）。
14. API 进程 `output_handler` 单任务分发：按 `VLLM_V1_OUTPUT_PROC_CHUNK_SIZE`（默认 128）切片 + 片间 `await asyncio.sleep(0)`（async_llm.py 一带；防一批的 detokenize 独占事件循环伤 P99）。
15. demux 到每请求 `RequestOutputCollector`（单槽+Event+生产侧合并）→ RequestOutputKind 三态裁剪 → SSE。
16. 客户端断连 → 取消 generate → **反向 abort**（外部 id 展开为内部 id；引擎侧 abort 双队列投递保序+及时）→ 引擎抠掉幽灵请求。

## 3. 设计决策总纲（why 链精华，全链见 deepread/）

### 进程与并发
| 决策 | 痛点 | 代价（诚实） |
|---|---|---|
| 三段式进程解耦（PR #9826） | Llama-8B@H100 单 step ~5ms，CPU 活与 GPU 循环抢 GIL | 每请求 ≥2 次 IPC + 两侧序列化税 |
| 多进程默认开（连离线 LLM 也走 SyncMPClient） | inproc/mp 两套路径行为分叉、双测试矩阵 | 离线也付 IPC 税；spawn 要求可 pickle |
| ROUTER/DEALER 输入拓扑（PR #15906） | DP 下 PUSH 无寻址能力，每引擎一条 socket | ROUTER 信封每条多一帧；「对端必须先发言」写死进 ready-first 协议 |
| PUSH/PULL 输出 + HWM=0 | 应答式 socket 会被慢前端反压卡 GPU | 无反压=前端卡死时内存堆积自担；fire-and-forget 无确认 |
| msgpack + 零拷贝帧 + OOB 旁路 | pickle 慢+不安全；多模态 MB 级张量过 socket 吃 TTFT | 双端 MessageTracker 保活；OOB 句柄/数据两通道要重排 |
| API 进程零 GPU | HTTP/JSON/SSE 的 CPU 活与引擎抢 GIL；API 要能独立于 GPU 数水平扩展 | 边界两侧各持一份请求状态靠消息对账；stop-string 前端判定后要反向 ABORT |

### 调度与显存
| 决策 | 痛点 | 代价 |
|---|---|---|
| 迭代级调度（Orca 血统，36.9×） | request-level 短请求陪长请求空转 | 每生成一个 token 都过一遍 CPU 调度——调度本身成吞吐上限（源码自认瓶颈注释） |
| 只认 token 数、无相位之分（Sarathi-Serve 血统） | 同为 1 请求，256 decode=256 token vs 8K prompt=8192 token，工作量差 30 倍 | TTFT 换 TPOT；混相批削弱纯 decode 优化 |
| RUNNING 先于 WAITING + 抢占过不收新 | 「先收新再抢老」= thrash，TPOT 方差失控 | 高负载下 TTFT 无上界（无 admission control） |
| recompute-only 抢占（swap 彻底埋葬） | swap 要 CPU 1:1 镜像+PCIe 占满；v0 作者自认 bizarre | 恢复=一次可能全量 prefill 重算，O(prompt) GPU |
| 分页 KV（PagedAttention 血统，利用率 20-38%→~满） | 按最大长度连续预分配的三重浪费 | 注意力 kernel 必须穿块表间接寻址；尾部浪费 ≤block_size-1 |
| 前缀缓存=链式哈希**非 radix 树**（澄清常见误解） | radix 树 Python 指针密集，GC/不变量成本调度循环付不起 | 只缓存满块；故意不去重（append-only 换简单） |
| LRU 逆序 free（隐藏不变量） | 顺序错=驱逐拦腰斩断最长前缀，命中率静默劣化 | 靠约定维持，无断言保护 |
| KVConnector 双面契约 | P/D 分离、offload 各写各的无法组合 | 本地+外部双查找路径慢一拍；异步加载期占块空转 |
| **水位回归（v0.27.1）** | 抢占抖动：边界负载下反复抢占-恢复 thrash | headroom 空闲不接客=吞吐换稳定，默认关 |
| **块内 CoW 部分命中（v0.27.1，#45939/#46384）** | 「只缓存满块」浪费块内前缀（hybrid 尤甚） | 每次部分命中 +1 块+GPU 拷贝带宽；三套簿记叠加 |
| **Marconi 式钉住+稀疏驻留** | 稀疏缓存杀死跨请求共享前缀复用 | chunk 被截断在 junction=prefill 变碎；跨模块隐式协议 |

### 执行与采样
| 决策 | 痛点 | 代价 |
|---|---|---|
| Executor/Worker/ModelRunner 三层 | 进程编排轴 × 硬件适配轴被焊死 | 一次 execute_model 穿三层+RPC 抽象 |
| 持久批次 InputBatch + 差量调和 | 每拍从零重建整批张量 CPU 线性涨 | 按 max 预留大数组；两进程请求视图可能漂移 |
| execute_model 两段式（forward→None→sample_tokens） | grammar bitmask 是 CPU 活，串行则 GPU 空转 | worker 有状态化；跨进程时是两次完整 RPC |
| 异步调度（采样 token 不落 CPU） | 单拍=GPU+CPU+IPC 三段相加，GPU 每 10-20ms 空转一次 | 「上一拍」影子状态整套；调度器盲调度错了要回扣 |
| CUDA Graph 按形状查表回放 | 图录的是「对固定地址执行这串 launch」，形状/地址变=错的工作量 | 捕获 5-20s 启动；图池独占显存；未捕获形状回退 eager 性能突降 |
| piecewise 编译（注意力处切图） | 全图 compile 进不了 attention 副作用；全 eager CPU 是瓶颈 | 接缝仍 eager，新的 CPU 开销洼地 |
| compute_logits 独立（只物化需要的位置） | 4096×129280×4B≈2GB fp32 logits 纯浪费 | 模型接入面变大（两方法契约） |
| Sampler 9 步管线 + argmax 不变性二分 | multinomial 强制 CPU-GPU 同步；greedy 常态不该付 softmax 税 | 处理器分类错=静默破坏 greedy 语义 |
| 位掩码结构化输出（非重试非枚举） | vocab 13 万，枚举合法集 O(V) 无法批处理 | 引擎级单后端；每步 CPU FSM walk + H2D |
| spec decode（draft→排批→验证→拒绝采样） | decode 访存受限，小 batch 算力闲置 | 接受率低纯亏；采样语义收窄（min_p 等互斥） |
| **异步调度默认开（v0.27.1，#27614）** | GPU 每 10-20ms 空转等 CPU 调度 | 乐观推进的补偿机制全套（拒绝回扣/脏占位清理） |
| **DPCoordinator 独立进程（XPUB/XSUB+PULL）** | DP 控制指令与数据面混流 | 多一进程三 socket；统计 100ms 快照有 race |
| **vllm/models/<name>/ 硬件隔离布局（v0.27 新政）** | 平台胶水与模型定义耦合、互相拖累 | 新架构两处维护（旧 model_executor 并存） |

## 4. 数据所有权地图（谁写谁读——8 处交叉裁决沉淀）

| 数据 | 生产者（写） | 消费者（读） | 备注 |
|---|---|---|---|
| EngineCoreRequest | 前端 InputProcessor | 引擎调度器 | 只含 token ids；tokenization 在 Renderer |
| request_id 映射 | InputProcessor（双轨生成） | OutputProcessor（外→内表） | abort 发引擎只带内部 id |
| ZMQ 帧序 | client send_multipart | engine recv_multipart | (identity, type, *payload)——DEALER 收端剥信封 |
| SchedulerOutput | 调度器 schedule() | worker ModelRunner | **非不可变**：ngram-GPU spec 路径 worker 就地裁剪，调度器回读 |
| new_token_ids | 调度器（仅 PP 非 async） | PP 下一段 | 其余空表，runner 自缓存 |
| 块号（WAITING 新请求） | `get_blocks()` 全量重读 | SchedulerOutput | 与 RUNNING 的 allocate_slots「仅新增」分流 |
| block_table/slot_mapping | ModelRunner（Triton kernel GPU 算） | 注意力 kernel | slot = block_table[req][pos//bs]*bs + pos%bs |
| KVCacheBlock | BlockPool（ref_cnt 共享） | 调度器纯 CPU 账本 | free 逆序=LRU 策略 |
| logits（采样前） | compute_logits | Sampler | raw logprobs 用变换前 logits（惩罚不扭曲模型意见） |
| grammar bitmask | 调度器侧 FSM（GPU 窗口期算） | Sampler（H2D 后 mask） | 编译异步化，未就绪请求不进批 |
| client_index | 前端 add_request_async 盖章进请求 | 引擎输出线程选 PUSH socket | 「谁发的」随请求过线的跨进程契约 |
| DP 统计（request counts） | 引擎 _maybe_publish（client_index=-1 哨兵） | DPCoordinator 聚合→前端 LB | 100ms 快照，LB 需本地 in-flight 兜底 |

## 5. v1 设计哲学（给 Phase 1 的输入）

从全部 why 链提炼的五条元原则——v3 大纲的每个 Part 都应呼应其中至少一条：

1. **GPU 不等 CPU**：一切 CPU 活或挪出循环（tokenize/detokenize）、或藏进 GPU 窗口（bitmask）、或固化（CUDA Graph/固定地址 buffer）、或并行化（异步调度）。
2. **账本先行**：显存 profile 一次定池、每拍对账（token 预算/块池）、差量协议（IPC 只发 diff）。
3. **单循环单真相**：一个 busy loop、一个 IntEnum 状态机、一套三件套双驱动——消掉 v0 的双引擎/双相位/双队列。
4. **每个设计都有代价**：诚实记录（TTFT↔TPOT、内存↔延迟、简单↜效率）——这是叙事的道德义务，也是最好看的内容。
5. **演进是被逼出来的**：每个模块都有一段 v0→v1 的血泪史（git/PR 为证）——讲系统要先讲它杀了什么。

## 6. Phase 0 → Phase 1 输入

六域的 `first_read_suggestion` 共识（供 pedagogy-plan 起点）：
- IPC/进程边界是最适合的**进门大厅**（不需要 ML 知识就能完全读懂）；
- 先「一个请求的一生」鸟瞰（对应 v2 ch01-02 的意图但要合并做实）；
- 调度→显存→执行→采样 的深潜顺序与播客 5 台阶高度吻合（播客正是以此书 v2 为源做的路径设计）；
- 模型层/MLA/spec decode 等重 ML 概念放后段（依赖前段的执行上下文）。
