# v3 大纲草案（Phase 1 · 待用户审批）

> 设计依据：ARCHITECTURE.md（六域深读）+ vllm-podcast season-plan（hook 台阶法）+ 用户四要素
> （完全理解系统 / 整体视角 / 最佳学习路径 / 阶梯式认知管理）。
> 结构：**Part = 认知台阶 = 一个 hook 大问题**；**章 = 台阶内的缩放步进**（每章都是 L0 图某块的放大）。
> 主轴 = **一个请求的一生**——读者在任何一章都知道自己在主线的哪一段。
> **混合 primer**（2026-08-15 用户裁决：v2「先原理后代码降低 gap」的优秀设计要参考）：
> 论文级推导链的大体系 = 独立 primer 章配对在代码章**紧前面**（Part 内相邻，防间隔遗忘）；
> 小而即时的概念 = 主场章内当场讲透。primer 遵循 v2 ch21 重设计后的新哲学（设计过的数学表达+顿悟图，非生硬讲源码）。

---

## Part I — 全景与读法（2 章）
**hook：俩字「你好」发进 vLLM，凭啥要跑两趟跨进程快递？**
**呼应哲学：⑤ 演进是被逼出来的（v0→v1）+ ① GPU 不等 CPU（第一次点题）**

| 章 | 标题（占位） | L0 缩放位置 | 内容 | v2 映射 |
|---|---|---|---|---|
| 1 | 一张图看懂 vLLM v1：v0 是怎么被逼成 v1 的 | **L0 全图首次给出** | v0 单进程同步的形态→三个痛点（GIL 抢占/双引擎维护/调度耦合）→三段式解耦→全书读法。L0 图 = 贯穿全书的唯一地图 | ch01 骨架迁移 |
| 2 | 跟一个请求走完全程：主线快速通道 | L0 整图（动态版） | 真实源码走主线：每站一个 file:line+一句话（它是什么/收什么/吐什么/交给谁），点到为止+跳转预告。与 ch1 分工：ch1 讲「系统为什么长这样」（静态），ch2 讲「请求怎么流过它」（动态） | ch02 重做 |

## Part II — 分而治之：进程边界与消息（5 章）
**hook：一千个并发用户，怎么让 GPU 永远不等 CPU、前端再忙也碰不到引擎一根汗毛？**
**呼应哲学：① GPU 不等 CPU（解耦的完整意义）+ ④ 每个设计都有代价（IPC 税）**

| 章 | 标题（占位） | L0 缩放位置 | 内容 | v2 映射 |
|---|---|---|---|---|
| 3 | 两个使用面，一套三件套 | API 进程左上（使用面+双登记） | AsyncLLM/LLM 共享三件套、双登记、add_request 的完整解剖、LLM 离线也走 IPC 的取舍 | ch04+ch37 部分 |
| 4 | ZMQ 拓扑与消息协议 | 紫色 ZMQ 边界带（放大） | ROUTER/DEALER 演进史（为什么弃 PUSH/PULL）、帧序 (identity, type, *payload)、PUSH/PULL 回程+HWM=0 无反压的代价、msgpack+零拷贝帧+OOB 旁路 | ch07 深化 |
| 5 | 下行：从文本到 token（Renderer+InputProcessor） | API 进程下行泳道 | tokenize 为什么必须在前端（PR #11963 删文本字段的双轨 id）、mm 特征、EngineCoreRequest 诞生 | ch05+ch06 合并 |
| 6 | 上行：从 token 到文字（OutputProcessor+Collector+SSE） | API 进程上行泳道 | 增量 detokenize（BPE/byte-fallback 的必要性）、单槽 Collector vs 无界 Queue、三态契约、chunk 分发、断连=反向 abort | ch08+ch09 前半 |
| 7 | 输出的另一个维度：logprobs | 上行泳道（logprobs 支路） | top-k logprobs 的 raw 语义（惩罚不扭曲模型意见）、bytes 字段与多字节 token、采样前物化的成本边界 | ch10 深化 |

## Part III — 引擎的心跳：调度循环（4 章）
**hook：调度器不认请求数——它每一拍只问：这批 GPU 能吃下多少个 token？**
**呼应哲学：③ 单循环单真相 + ④ TTFT↔TPOT 的显式交易**

| 章 | 标题（占位） | L0 缩放位置 | 内容 | v2 映射 |
|---|---|---|---|---|
| 8 | EngineCore 的逐拍循环 | 循环框（放大） | 五拍解剖（schedule→execute→bitmask 窗口→sample→update）、慢操作出循环、busy loop 的进程形态（EngineCoreProc/启动握手） | ch11+ch12 重组 |
| 9 | 连续批处理：token 预算与两阶段调度 | 调度账本列上半 | Orca/Sarathi 血统、无相位调度、RUNNING 先于 WAITING、抢占过不收新（TTFT 换 TPOT 的显式交易） | ch13+ch14 合并 |
| 10 | 抢占与请求的一生 | 调度账本+状态机 | recompute-only（swap 的死法）、单 IntEnum 状态机、抢占恢复与前缀缓存的联动（伏笔：Part IV 回收） | ch14 深化 |
| 11 | 异步调度：让下一拍的决策发生在这一拍的执行期间 | 循环框+执行臂接缝 | num_output_placeholders、采样 token 不落 CPU、影子状态、盲调度的回扣 | ch12 深化+ch19 部分 |

## Part IV — 显存是主角：分页 KV（4 章）
**hook：显存总共就那么多，但每个请求的 KV cache 必须活到生成结束**
**呼应哲学：② 账本先行 + ④ 利用率 20%→满 的代价（间接寻址）**

| 章 | 标题（占位） | L0 缩放位置 | 内容 | v2 映射 |
|---|---|---|---|---|
| 12 | 分页 KV：虚拟内存思想进显存 | KV 账本列 | PagedAttention 血统（三重浪费 20-38%）、块池+逻辑块表、slot_mapping 公式、注意力 kernel 的间接寻址代价 | ch15 重做 |
| 13 | 显存账本：从 profile 到准入 | KV 账本列+启动 | determine_available_memory 三步定账、num_blocks 倒推、full_sequence_must_fit 准入门（#39734 死锁教训）、混合注意力的组化分配 | ch16+ch15 部分 |
| 14 | 前缀缓存：链式哈希，不是 radix 树 | KV 账本列（缓存区） | 常见误解正面纠偏、Merkle 链式哈希、LRU 逆序 free 隐藏不变量、故意不去重的 append-only、抢占恢复先撞缓存（回收 Part III 伏笔） | ch16 深化 |
| 15 | KVConnector：把外部的 KV 世界接进池子 | KV 账本+边界 | 双面契约（调度器侧/worker 侧）、P/D 分离的动机与代价、offload 组合 | ch35+ch36 合并 |

## Part V — GPU 不等 Python：执行管线（6 章）
**hook：你在 Python 里写了一个 attention 层，它最后变成 CUDA graph 里反复重播的融合 kernel——中间连解释器都不经过**
**呼应哲学：① GPU 不等 CPU（本 Part 就是这句话的全景展开）**

| 章 | 标题（占位） | L0 缩放位置 | 内容 | v2 映射 |
|---|---|---|---|---|
| 16 | 执行三层：进程拓扑 / 设备生命周期 / 批次执行 | GPU 执行臂上层 | Executor 分发（uni/mp/ray）、Worker 延迟初始化（Wrapper 魔法）、控制面/数据面分离 | ch17 重做 |
| 17 | 持久批次与固定地址 | 执行臂中层 | InputBatch 差量调和（新全量/旧 diff/resumed 替换）、CpuGpuBuffer 预分配、SchedulerOutput 差量协议（含可变性裁决） | ch18 重做 |
| 18 | 编译与捕获：piecewise + CUDA Graph | 执行臂中层 | CustomOp 双实现、注意力=不透明算子+forward context、切图位置的选择、CUDA Graph 形状全等才能回放的本质、启动期前移 | ch23 重做 |
| 19 | 【primer】Flash-Attention：在线 softmax 与分块遍历的数学 | （原理章，无代码主线） | softmax 的 naïve 两遍法→在线单遍重写的推导链、tiling 与显存访问的账、为什么它天然适配 paged KV、LSE 合并 | v2 ch24 重做（ch21 样板哲学） |
| 20 | 注意力后端：一张优先级表定生死 | 执行臂+模型层接缝 | 平台优先级表+validate 回退、metadata 构建、为什么可逐层混布（消费上章 FA 数学：后端=该数学的工程落地） | ch24+ch25 重组 |
| 21 | slot_mapping 与 block_table：GPU 端的地址换算 | 执行臂+KV 接缝 | Triton kernel 在 GPU 算的 why（D2H 会斩断异步调度）、PAD 值语义、混合块细分 | ch19+ch18 部分 |

## Part VI — 模型的形状（6 章）
**hook：接入一个新架构，为什么只需要「拼层」——Attention 是插座，不是实现**
**呼应哲学：③ 单循环单真相（模型层与引擎解耦的契约）**

| 章 | 标题（占位） | L0 缩放位置 | 内容 | v2 映射 |
|---|---|---|---|---|
| 22 | 模型定义层拼装术 | 模型层框 | DecoderLayer=积木清单、forward/compute_logits 两方法契约（logits 只在需要处物化的 2GB 反例）、接入 registry 清单 | ch22+ch28 合并 |
| 23 | 【primer】注意力变体的数学：MHA→GQA→MQA→MLA 的压缩之路 | （原理章） | KV 显存账（每 token 每层多少元素）→低秩压缩的直觉（576 维潜向量从哪来）→展开与吸收两种形态的取舍 | v2 ch26 前半重做 |
| 24 | 注意力变体：MLA 的压缩与两种展开 | 模型层框（MLA） | prefill MHA 展开 vs decode MQA 吸收（消费上章数学）、hybrid KV 的组化（回收 Part IV） | v2 ch26 后半深化 |
| 25 | DeepSeek 的索引器：从 NSA 到 DSA | 模型层框（indexer） | （融入式 primer：NSA 三支路谱系）DSA 打分器、KV 挑选如何进入块表、与 MLA 的配合、**IndexCache**（indexer 打分结果的缓存与多轮复用——每轮只对新增 token 算 index，`DeepseekV4IndexerCache` 的生命周期与失效） | ch27+ch29 合并 |
| 26 | 【primer】量化：把权重压进更低位的算法故事 | （原理章） | GPTQ/AWQ 的数学、FP8/FP4 的训练后量化、量化与 kernel 的耦合（消费 Part V 编译章） | v2 ch26 量化部分独立成章 |
| 27 | 实战：DeepSeek-V4 是怎么拼出来的 | 模型层全景 | capstone：MoE+MLA+DSA indexer+FP4，从接入清单走一遍真实新架构落地的每一步 | ch28 重做 |

## Part VII — 选一个 token 出门（6 章）
**hook：GPU 算出 128000 个 logits，选哪个 token 出门要过 9 道关卡**
**呼应哲学：①（multinomial 同步之死）+ ③（9 步管线=一套契约）**

| 章 | 标题（占位） | L0 缩放位置 | 内容 | v2 映射 |
|---|---|---|---|---|
| 28 | Sampler 的 9 步管线 | 采样出口列 | 逐步解剖、argmax 不变性二分（greedy 免税）、惩罚/温度/截断、Gumbel 技巧杀同步、raw logprobs 的语义 | ch30 重做 |
| 29 | 约束解码 I：语法编译与后端契约 | 采样列（结构化输出组） | 语法→FSM、xgrammar/guidance/outlines 生态对比与取舍、GrammarCompiler 缓存、异步编译+调度状态门（不挡批） | ch31 重做 |
| 30 | 约束解码 II：bitmask 如何落到 logits | 采样列+bitmask | 位掩码（非重试非枚举的 why）、两段式 execute 的 GPU 窗口、fill_next_token_bitmask→apply 的 H2D 链 | ch32 重做 |
| 31 | 【primer】投机解码的数学：拒绝采样为什么无损 | （原理章） | speculative sampling 的分布不变性证明、EAGLE/MTP/DFlash drafter 谱系对比、**DSpark 论文解读**（arXiv:2607.05147：半自回归架构=并行骨干+序列 Markov 头、置信度调度验证、V4 线上提速 60-85%） | v2 ch33 重做+ch41 素材（papers/ch41-primer-dspark 已深研） |
| 32 | 投机解码的 vLLM 落地：draft 排批、一次验证、接受回扣 | 采样列+spec | draft 当 prefill 排批、target 一次前向验证、RejectionSampler 的 Triton 实现、接受率低纯亏的诚实账、vLLM speculator 生态（消费上章数学） | ch34 重做 |

## Part VIII — 走向生产（6 章）
**hook：真实服务不止一个引擎——规模、容错与接口的最后一公里**
**呼应哲学：全部五条的合流**

| 章 | 标题（占位） | L0 缩放位置 | 内容 | v2 映射 |
|---|---|---|---|---|
| 33 | 分布式：TP/PP/DP 在三段式里的位置 | L0 多实例视角 | TP 的算子级切分、PP 与循环的两拍协议、DP 与多引擎拓扑（回收 Part II 的 ROUTER 寻址） | ch20+ch21 重组 |
| 34 | P/D 分离：把 prefill 和 decode 拆到两台机器 | L0 双实例+KV 边界 | 为什么拆（算力/带宽错配）、prefill 侧 save/decode 侧 load 的完整生命周期、等待远端 KV 的调度集成、NIXL 传输、disaggregator | ch35+ch36 重做 |
| 35 | KV 池化：把 cache 搬出 GPU——DRAM 上的分布式 KV 管理 | KV 边界+外部池 | 为什么池化（KV 是显存大头，DRAM 便宜 N 倍容量大 N 倍；前缀复用跨实例=全局缓存池）；Mooncake 架构（transfer engine/conductor/以 KV 为中心的分离存储）；LMCache 与 CPU offload；HBM/DRAM/SSD 分层策略；vLLM 侧怎么接（KVConnector 的 offload/共享实现，回收 ch15 契约） | 新增（v2 未覆盖的方向；researcher 深挖 Mooncake 论文+生态） |
| 36 | 服务面：OpenAI 协议与多轮会话 | API 进程+入口 | serving 层结构、chat/completion 的流式协议、harmony 三通道多轮（final/analysis/commentary）、断连处理（回收 Part II 的 abort） | ch37+ch38 合并 |
| 37 | 弹性与自愈：引擎运维的最后一课 | EngineCore 带（弹性区） | EEP 弹性扩缩状态机、scale_up 的 KV 迁移握手、通知驱动的状态推进、多引擎负载（DP coordinator） | ch39 重做 |
| 38 | 终章：站在整张 L0 图上回望 | L0 全图（点亮版） | 全书走完后的 L0 图完整复盘（每块都读过了）、设计哲学五条的印证、延伸阅读路线（vllm-ascend/编译器方向） | 新写 |

---

## 设计自检（对照用户四要素）

1. **完全理解系统**：六域深读卡为每章供素材（why 链/gity 历史/代价），写作时 dossier+deepread 双真相源。
2. **整体视角**：L0 图 ch1 首次给出、每章开篇 zoom-in 到本章块、ch28 收官全图点亮——**一张图贯穿全书**。
3. **最佳学习路径**：Part I-II 不需要任何 ML 知识（进程/消息/后端直觉即可入门）→ III-IV 引入调度/显存的系统思维 → V 需要 GPU 编程直觉 → VI-VII 才需要 ML 概念（且在主场融入讲透）。
4. **阶梯式认知**：每 Part 一个 hook（读者带着问题读）+ 每章是上一章某块的放大（不引入无准备的概念）+ 伏笔显式登记（如 ch9 抢占↔ch13 前缀缓存恢复、ch4 ROUTER↔ch26 DP）。

## 与 v2 的章数对比

v2 39 章 → v3 **38 章**（8 Part，其中 4 个 primer 章）。初版 28 章被用户抓到容量问题（P/D 分离/约束解码/DeepSeek-V4/服务面四块压缩过头）修正为 33 章；再补 KV 池化章（用户点名 Mooncake 类 DRAM 分布式 KV 管理，v2 未覆盖的新方向）：P/D 独立成章（v2 2→2）、约束解码拆回 2 章（v2 2→2）、DSA indexer 独立（v2 3→2+拼装重做）、服务面拆 2+弹性独立（v2 3→2+1）。其余合并仍成立：重复意图章（ch01+02→2 章做实分工）、同机制章（ch05+06、ch08+09 前半）、primer 融入主场（flash-attention→注意力后端章、量化→编译/MLA/V4 三处融入、eagle→投机解码章、NSA→indexer 章）。v2 全部 39 章的源码解读段均有映射去向（见各表末列），择优迁移；量化如需专席可加 1 章（见待裁决）。

## 待用户裁决的点

1. **章数 38**（容量修正+KV 池化+4 primer 回归）：Part VIII 6 章为全书最重收尾。
2. **混合 primer 已采纳**（2026-08-15 用户裁决）：4 个论文级原理章独立、与代码章紧配对（FA→后端、变体数学→MLA、量化→V4、投机数学→落地）；小概念仍主场融入。
3. **Part 顺序**：调度（III）在显存（IV）前——播客同序；但也可以先显存后调度（块账本是调度的前置）。我选了「先调度后显存」因为调度是读者已见过的循环骨架的自然深入。
