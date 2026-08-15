# vLLM v1 架构真相源（v3 · Phase 0 产出）

> 本文 = v3 重写的**架构底座**：六域深读卡片（`deepread/*.json`）+ 8 处交叉核对裁决的综合。
> 读者假设：懂后端、不懂 AI 推理的工程师。每个设计决策都回答「旧设计→痛点→方案→代价」。
> 源码 pin：**v0.27.1（6e448d0ea，2026-08-15 用户定最新 release）**。⚠️ 本文与 deepread/ 卡的行号锚点在
> 深读时（2026-08-15 前）核于旧 pin v0.21.0（ad7125a4）——跨 6 minor 行号大面积漂移、个别机制已重构
> （elastic EP、xgrammar 调用面等）。why 链与架构判断大体有效；**行号/符号引用前须对 v0.27.1 现核**，
> 重构域另出增量深读补丁。
> 配套：L0 架构图 `L0-architecture.svg/png`（全书唯一权威图，见 §1）。
> （v2 的素材收集式地图归档于 `ARCHITECTURE.v2.md`，仅供对照。）

## 0. 一句话主线

**一个请求的一生**：用户发来文本 → API 进程切成 token（`Renderer`）→ 跨进程扔给 EngineCore（ZMQ）→ 引擎逐拍循环「调度→前向→采样→回收」，每拍只问「这批 GPU 能吃多少 token」→ 新 token 跨进程回 API 进程 → 拼回汉字 → SSE 流给用户。

**全系统的第一设计原则：GPU 是最贵的员工，一切 CPU 活都不能让它等。** 三段式进程解耦、异步调度、两段式 execute_model、CUDA Graph、采样留 GPU——全部服务于这一句。第二原则：**显存是共享账本，一切调度都先对账**（token 预算 + KV 块池）。

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

## 2. 一个请求的一生（file:line 级走读）

**前端进程段**：
1. `AsyncLLM.add_request`（async_llm.py）→ **双登记**：本进程 `OutputProcessor.request_states` 建表（含 detokenizer 状态）+ 跨进程发 EngineCoreRequest。先本进程后跨进程——保证回程消息到达时表已存在。
2. `Renderer.render → _tokenize_prompt`（renderers/base.py:L401-412）：**tokenization 在前端进程**（EngineCoreRequest 诞生即 tokenized，文本不过 IPC——PR #11963 删掉了 prompt 文本字段）。input_processor.py:L273-286 的裸 prompt 回退路径已弃用（v0.18 移除警告）。
3. `InputProcessor.process_inputs`：校验/克隆 params/整理 mm 特征/**双轨 request_id**（外部 id + `{id}-{8位hex}` 内部 id）→ `AsyncMPClient.send_multipart`。
4. ZMQ 下行帧序（**裁决 #1**）：`(engine_identity, type_byte, *msgpack载荷帧)`——identity 第 0 帧（ROUTER 信封）、`ADD=b'\x00'` 第 1 帧；引擎收端 DEALER 剥掉信封后看到的"首帧"才是类型字节。

**引擎进程段**：
5. `EngineCoreProc` 循环取消息 → input_queue → `EngineCore.step()`（core.py:L406-435）四段+bitmask 窗口（见总图）。
6. `Scheduler.schedule()`：RUNNING 先于 WAITING；本拍抢占过就整拍不收新（延迟压力显式从 TPOT 转嫁给 TTFT）。
7. KV 对账：`get_computed_blocks` 返回 `(KVCacheBlocks, num_new_computed_tokens)`（**裁决 #4**：二元组，第二个是 **token 计数**非块计数）→ `allocate_slots` 失败 → 抢占（recompute-only：free 块**不清哈希**，回 waiting 队头，恢复时先撞前缀缓存）。
8. 组装 `SchedulerOutput`（差量协议）：`scheduled_new_reqs`（首次全量）/`scheduled_cached_reqs`（仅 diff）/`finished`。**裁决 #6/#7**：`new_token_ids` **仅 PP 且非 async 调度时携带**（其余空表——runner 自己缓存）；resumed（抢占恢复）请求的 `new_block_ids` 是**整体替换**而非追加（output.py:L114-117；worker 侧 L1330-1333 同款替换分支）。**裁决 #5**：WAITING 新请求进批的块号并非 allocate_slots 返回值，而是 `get_blocks(request_id)` 全量重读；RUNNING 才用 allocate_slots 的「仅新增块」。
   ⚠ **裁决 #6（可变性）**：SchedulerOutput **不是不可变**——ngram-GPU spec decode 路径 worker 会就地裁剪它（ngram_proposer_gpu.py:L499-503），调度器 update_from_output 回读同一对象。常规路径 worker 只读。
9. `Executor → Worker → GPUModelRunner`：InputBatch 差量调和 → CpuGpuBuffer 固定地址 → slot_mapping 用 **Triton kernel 在 GPU 算**（CPU 算要先 D2H 拉回 positions，会斩断异步调度）→ piecewise 编译图内前向（注意力=不透明自定义算子）→ CUDA Graph 按形状查表回放。
10. `compute_logits`（只在需要位置物化：decode 批每请求 1 个；4096-token prefill 全量物化 fp32 logits ≈2GB 是反例）→ `Sampler.forward` 9 步管线（见 §4 模型域）。
11. `update_from_output`：请求状态推进（单 IntEnum，`>PREEMPTED 即 finished` 一次整数比较）、free 块（**逆序 free**——LRU 隐藏不变量）、finished 请求当拍退出。

**回程段**：
12. 每步**整批聚合**成一条 EngineCoreOutputs（不逐 token 发——IPC 次数爆炸）；按 client_index 选 PUSH socket（多前端）。
13. API 进程单任务 `output_handler`：按 chunk_size 切片逐片 process_outputs + `await asyncio.sleep(0)` 让事件循环喘气（防一批 256 请求的 detokenize 独占循环、伤 P99）。
14. demux 到每请求 `RequestOutputCollector`（单槽+Event+生产侧合并，**刻意不用 asyncio.Queue**——慢消费者下无界队列滞留 O(len²) 内存）→ RequestOutputKind 三态契约（DELTA/CUMULATIVE/FINAL_ONLY，使用面入口声明，非流式=FINAL_ONLY 每 token 白传的浪费归零）→ SSE。
15. 客户端断连 → 取消 generate 协程 → **反向 abort**（外部 id 展开为全部内部 id；发引擎的 abort 只带内部 id——**裁决 #3**）→ 引擎抠掉幽灵请求（否则烧 GPU 到 max_tokens）。

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
