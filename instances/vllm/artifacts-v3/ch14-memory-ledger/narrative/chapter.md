# 第 14 章　显存账本

[第 13 章](../../ch13-paged-kv/narrative/chapter.md)把块池整个打开了：等大的块、每请求一张块表、引用计数、自由队列。可有一个数它从头到尾当进门参数用，就是 `num_gpu_blocks`：池子一共多少块。这个数是谁定的？vLLM 的做法听着莽：启动时先拿假数据跑一遍真前向，把权重、激活峰值、CUDA 图占的显存全称一遍，剩下的才分给 KV。一笔账凭什么一次算清？算小了并发白白浪费，算大了运行期炸给你看。更扎心的是门的问题：账定了，请求进门时凭什么担保整条序列装得下？vLLM 的历史上有过一条 10 万 token 的 prompt，守着一个使用率 0.0% 的池子永远进不了门，把后面所有请求一起堵死（issue #39734）——这道死锁怎么来的、怎么被根治的？还有混合模型：有的层要全部历史、有的层只要一个滑窗、有的层干脆不存 KV 只存一份固定状态，一个池子怎么切才不浪费？

三问连着答：**饼有多大**（启动三步定账）、**怎么切**（混合注意力组化）、**门多紧**（准入门与水位）。

[第 10 章](../../ch10-continuous-batching-chunked-prefill/narrative/chapter.md)到[第 12 章](../../ch12-async-scheduling/narrative/chapter.md)把 `allocate_slots`（调度器领块的入口）当黑盒用了三章；[第 13 章](../../ch13-paged-kv/narrative/chapter.md)打开了黑盒的下半，即块池内部；worker 侧 `init_device`（设备初始化）的完整装配序要到[第 17 章](../../ch17-executor-worker-model-runner/narrative/chapter.md)（Part V）才正面拆，其中「量显存」那一截本章先提前展开。本章补上中间缺的那层：**账本本身**。L0 图「调度 · 显存账本」列（[第 1 章](../../ch01-vllm-v1-in-one-map/narrative/chapter.md) Part 速览表里的「KV 账本列」指的就是它）里 Scheduler 之下的 KV 半区再分上下：上半是 KVCacheManager 的定账管线与两道运行期的门（本章打开），下半是 BlockPool 与块表（前一章刚打开）。本章从池往上走，连同它头顶那条启动装配带（EngineCore 出生时量家当的一截）。

## 你在这里

Part IV 的总问题只有一句：**显存就那么多，KV cache 必须活到最后**。[第 13 章](../../ch13-paged-kv/narrative/chapter.md)回答了「块长什么样」；本章回答它头顶的问题：「池多大、谁定的、门多紧」。

![L2 章图：显存账本：从定账到把门十二站](../diagrams/L2-ch14.png)

> *图注：本章放大的是[第 1 章](../../ch01-vllm-v1-in-one-map/narrative/chapter.md) L0 图「调度 · 显存账本」列（Scheduler 之下的 KV 半区）的上半，加上它头顶的启动装配带。[第 13 章](../../ch13-paged-kv/narrative/chapter.md)打开过这半区的下半（BlockPool 与块表），本章从池往上走：先看启动期 EngineCore 怎么量出 `num_gpu_blocks`，再看请求进门要过的门。图上三段读：北行四站是启动期一次性的测量（快照定预算 → dummy 前向测峰值 → CUDA 图估计入账 → 每层自报形状）；中排 ①-⑦ 是定账管线（混合组化 → 护栏 → 定块数 → 一份账喂两侧 → 账本就位 → 准入门 → 水位门，⑥⑦ 是每个请求入场都过的运行期门；①-⑦ 就是图上行内那七张卡片「① 混合组化」到「⑦ 水位门」的编号，即定账管线依次经过的七景；卡片在正文里各有放大图，放大图右上角印着「L2 拍片N」徽标，N 与这里的编号一一对应，「拍片」这组字样只出现在那几张放大图上，护栏四道不单占站号）；南行是 SWA 窗外回收、kernel 块细分、三条 why 注与一条邻章分界。站号 1-12 = 账本从诞生到把门流经代码的顺序（1-4 测量 · 5-8 定账 · 9-10 过门 · 11-12 落地），正文按讲解需要编排、不必照站号读。两套编号的对应：拍片①组化 = 站 5、②护栏与③定块数同占站 6（「护栏四道不单占站号」说的就是它）、④喂两侧 = 站 7、⑤账本就位 = 站 8、⑥⑦两道门 = 站 9-10。*

读法建议：想知道「池子的数怎么来的」，从[「饼有多大」](#饼有多大一次-dummy-前向定终身站-1-3)读起；关心装不下时的报错与自动适配，跳[「字节换块数」](#字节换块数护栏四道含站-46)；想知道死锁怎么回事，直奔[「门多紧」](#门多紧超收死锁与抖动换来的三道预算站-9-10)；用混合模型（Gemma、gpt-oss 这类）的读者重点看[「一个池子多张账」](#一个池子多张账混合注意力的组化站-58)；想知道这套组化规矩是怎么被模型潮一步步逼出来的，读[「组化的前史」](#组化的前史引擎是怎么被模型潮推着走的)；冲 DeepSeek V4 来的读者，直奔[「DeepSeek V4：先把这一层看明白」](#deepseek-v4先把这一层看明白)，层画面、页宽账、压缩记账、第三路分组四节连读；想跟全程，按序读。

照例交代取证环境，全章数值表都适用：本章实测来自配套精简版，按 v0.27.1 只做减法抽出的「定账 + 门 + 组化」三幕，host 上实跑纯控制流，不依赖 GPU 与 vLLM 运行时。它与真实引擎有三处刻意差别，后文碰到会就近再提：其一，host 无 CUDA，显存快照读数与 CUDA 图估计值是**注入的示教值**（算术路径与源码逐字一致，凡涉及设备读数的表格都会标明）；其二，精简版关掉前缀缓存跑（`enable_prefix_caching=False`，cache 配置里的正交开关，真实部署默认开，vllm/config/cache.py:L93），本章讲的定账与门都不依赖它；其三，多卡部署里调度器与 worker 分属两个进程（单卡默认两者同住一个进程，同一份 config 契约照样成立），本章按单进程视角讲、多卡差异在流水线并行处单独点出。

## 饼有多大：一次 dummy 前向定终身（站 1-3）

先站到 L0 图的启动装配带上。EngineCore 出生时要干一件事：把「这张卡上我能用多少显存、其中多少归 KV」一次算清，之后运行期不再重算。这本账的旧设计不存在：早期推理系统没有系统内的账本，「权重之外全给缓存」是直觉不是公式，没人能回答「到底能开多大 batch、几条并发」：分多了运行期 OOM，分少了池空转、并发被白白压住。痛点是被卡的并发容量与稳定性：KV 池是全部请求的共享预算，预算错 = 崩溃或浪费。[第 13 章](../../ch13-paged-kv/narrative/chapter.md)算过那笔 0.5 MB/token 的账（Llama-2-7B FP16）；放到 24 GB 的卡上，权重吃掉 14 GB 后，池只剩大约 8 GB，每一 GB 都金贵。v1 的方案是**测量式分配**：不让人算，启动时量出来。代价也直说：量的是快照不是保证，只能靠预算比例留头寸（预先留出的余量），本节末尾展开。

### 预算先于一切：总显存乘一个比例（站 1）

账本第一步不量任何东西，先定预算。`gpu_memory_utilization`（GPU 显存利用率，vLLM 最常被调的启动参数之一）就是那个比例：

```python
# vllm/v1/worker/utils.py:L409-L429
def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    """
    Calculate the amount of memory required by vLLM, then validate
    that the current amount of free memory is sufficient for that.
    """
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization   # L415
    )

    if init_snapshot.free_memory < requested_memory:                      # L418
        raise ValueError(
            f"Free memory on device {init_snapshot.device_} "
            # … 省略：报错正文五行（报出空闲/总量/预算三个数，
            #       建议降 utilization 或清掉同卡的其他进程）……
        )

    return requested_memory                                              # L429
```

三行各有讲究。**分母是总显存，不是空闲显存**：`total_memory × utilization`，初学者最容易读错成「空闲的 92%」。默认值 0.92（vllm/config/cache.py:L68-L75），官方 CLI 文档的原话是 "a per-instance limit, and only applies to the current vLLM instance"：同一张卡跑两个 vLLM 实例，各设 0.46，各圈各的互不越界（多实例共存的标准玩法）。80 GB 卡上一笔说明性账：单实例默认预算 = 80 × 0.92 = 73.6 GB，哪怕卡上还空着 79 GB，vLLM 也只按 73.6 这份预算定账；若另一个进程先占了 10 GB（free 只剩 70），`free < requested` 直接 raise，报错把三个数都打给你，让你降比例或清场。**预算先于一切**：还没量任何东西，先回答「你最多能用多少」。同类参数对照一句帮定位（这不是 vLLM 独有的怪癖）：SGLang 的 `--mem-fraction-static` 圈的是「权重 + KV 池」两块静态分配、激活吃圈外剩余（[SGLang 官方文档](https://docs.sglang.io/advanced_features/server_arguments.html)）；vLLM 把激活与图池也算进圈内再用差价扣。圈法不同、思想同源：先按总显存比例圈地，再在圈内做减法。

为什么 0.92 不是 1.0？留的 8% 不是浪费：CUDA context（驱动运行时，进程用 GPU 的固定开销）、系统占用，以及后文要讲的「快照不是保证」的余量，都在这份头寸里。

还有一个顺序问题值得停一秒：`init_snapshot`（启动时的显存快照）是干净的吗？源码把顺序写死了（这段装配序[第 17 章](../../ch17-executor-worker-model-runner/narrative/chapter.md)稍后细拆）：**分布式初始化刻意排在快照之前**，注释原话 "This ensures NCCL buffers are allocated before we measure available memory"，NCCL 通信缓冲（多卡时可达数百 MB，量级示意）属于「起了就一直在」的债主，先把它记进账里再拍照，测出的可用显存才不虚高。拍照前还 `gc.collect()` 加 `empty_cache()` 清碎片。本章直接用它的两件产物：`init_snapshot` 与 `requested_memory`。

### 三本显存账：为什么快照要拍三张（站 2）

预算定了，第二步量「KV 之前的债主一共占多少」。这事比听着难，因为一张卡上的显存其实有三本账，PyTorch 官方 CUDA 文档讲得很清楚：

1. **allocated（已分配）**：张量真正占的，`memory_allocated()` 读；
2. **reserved（已保留）**：PyTorch 缓存分配器圈走没还的（释放的张量内存留在自己的池里备复用），官方警告原话 "The unused memory managed by the allocator will still show as if used in nvidia-smi"；
3. **非 torch 分配**：NCCL 通信缓冲、cuBLAS workspace、CUDA context 这些库级分配，**不在上述任何 torch 统计里**（[PyTorch CUDA 官方文档](https://docs.pytorch.org/docs/2.13/notes/cuda.html)）。

nvidia-smi 看到的是三本之和。一笔说明性例：nvidia-smi 显示已用 10 GB、`memory_reserved()` 是 8 GB、`memory_allocated()` 是 6 GB：6 GB 活张量（权重+当前激活），2 GB 是分配器扣着备复用的池，剩 2 GB 是 torch 账外的库级分配。若直接拿 torch 统计当「全部占用」去定 KV 预算，池会定大 2 GB，运行期 NCCL 一发力就是 OOM。这正是 vLLM 的显存快照要**同时**读驱动级空闲（free memory）与 torch 统计、把差额记为「非 torch」的原因。

三本账一次对齐的机制是 `memory_profiling` 上下文管理器（vllm/utils/mem_utils.py:L233-L326）：基线那张直接复用启动时的 `init_snapshot`（记作 `before_create`，本实例创建前拍的），profiling 里再补拍两张：跑前一张（`before_profile`，权重与通信库落地后）、出来后一张（`after_profile`，垃圾回收后）。docstring 自带一个量化的三类显存例子，正好实跑一遍（示教注入的读数、真实的算术；一处取整偏差先挑明：docstring 例里 NCCL 先占 0.5 GiB、到峰时才涨成 1，下表从 before_profile 起即记 1，只影响中间快照的 free 读数，总量账分毫不变）。例子的三类按**占用主体**切，与前面 allocated/reserved/非 torch 那三本只是刀口不同，逐条对上：

- **cat2 = 本实例的 torch 圈内**：allocated 加上分配器圈走的 reserved 合并算；
- **cat1 = 同卡他进程**：由 before_create 基线一次性定住（那时本实例 torch 占用为零，卡上已用的全是他进程的）；
- **cat3 = 本实例 torch 外的库级分配**：其后非 torch 部分的增量（快照的算法：非 torch = 驱动级已用 − torch 保留）。

<!-- trace: m2 -->
| 时点 | cat1 他进程 | cat2 torch | cat3 非torch | 读数/账目 |
|---|---|---|---|---|
| before_create | 1 | 0 | 0 | free 9 GiB |
| before_profile（权重+NCCL 落地） | 1 | 2 | 1 | free 6 GiB |
| during peak（dummy 前向峰） | 1 | 4 | 1 | 激活峰 2 GiB（torch_peak_increase 2147483648 B） |
| after_profile（gc 后） | 1 | 3 | 1 | free 5 GiB → total_consumed = 4294967296 B（4.0 GiB） |
| 峰值账 | — | — | — | transient = torch 峰 4（during 峰行 cat2）− after torch 3 = 1.0 GiB（1073741824 B）；non_kv = 4294967296 + 1073741824 = 5368709120 B（5.0 GiB = 权重 2 + 峰 2 + 非torch 1；「峰 2」里 1 GiB gc 后仍常驻、已含在 total_consumed 4 里，纯路过的只有 transient 这 1 GiB） |

峰值账的两项怎么算的，源码一行是一行：

```python
# vllm/utils/mem_utils.py:L314-L326 · memory_profiling（上下文管理器）
    # Measure total consumption via mem_get_info() instead of
    # memory_reserved(), which goes negative when pluggable allocators
    # (e.g. cumem) bypass PyTorch's tracking.
    result.total_consumed = (
        result.before_create.free_memory - result.after_profile.free_memory
    )                                                                     # L319

    # total_consumed already covers persistent torch allocations; add only the
    # transient peak headroom to avoid double-counting.
    result.transient_peak_headroom = (
        result.after_profile.torch_peak - result.after_profile.torch_allocated
    )
    result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom   # L326
```

`total_consumed`（总消耗）走驱动级口径：基线空闲减去结束后空闲，权重、常驻激活、非 torch 分配全在内，连可插拔分配器绕开 torch 记账的部分也穿得透（注释里点名的正是这事）。`transient_peak_headroom`（瞬时峰值余量）是 torch 峰值高于常驻的部分，即「路过但最胖的」（口径对齐一句：表里 cat2 列是「torch 圈内」的合并读数，公式读的是快照原生字段 `torch_peak` / `torch_allocated`；docstring 例中两组账面相同，transient 里的 4 与 3 就是 during 峰行与 after 行的 cat2）。两项不重不漏：前者数「住下的」，后者数「峰值瞬间比常驻多占的」，`torch_peak ≥ torch_allocated` 保证第二项非负。**账本宁记峰值不记现状**：这 10 GB 卡的例子里若少记 1 GB 的 transient，运行期第一批真实请求的激活一起冲高时就会 OOM。

上下文里跑的正主是 `profile_run`，拿假数据跑一遍真前向：

```python
# vllm/v1/worker/gpu_model_runner.py:L6492-L6506 · GPUModelRunner.profile_run
        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True                        # L6494
        )
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                # … 省略：pooling 模型分支一行（嵌入/打分类模型面）……
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()
```

两个细节：假数据的规模是 `max_num_tokens`（一步能排进 batch 的最大 token 数）。激活峰值由**最大一步**决定，所以拿它当形状；采样器也一起跑（`_dummy_sampler_run`），因为采样缓冲同样吃显存。多模态模型还要另量 encoder 与编码缓存，纯文本主线不展开。

### 一行减法：连图池一起扣干净（站 3）

第三步做减法。先看全段，再拆两笔：

```python
# vllm/v1/worker/gpu_worker.py:L498-L548 · Worker.determine_available_memory
        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()                             # L504

        # Profile CUDA graph memory if graphs will be captured.
        # … 省略：ROCm/XPU 平台差异注释五行（ROCm 已并入、XPU 排除）……
        cudagraph_memory_estimate = 0
        if (
            current_platform.is_cuda_alike()
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()   # L517

        # Respect the opt-in flag as originally designed.
        cudagraph_memory_estimate_applied = (
            cudagraph_memory_estimate
            if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            else 0
        )

        # … 省略：三个字段写入结果（total_consumed / peak_activation_memory /
        #       cudagraph_memory_estimate）……

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            # … 省略：报错正文（同容器的其他进程在 profile 期间释放了显存，
            #       建议 "isolate vLLM in its own container"）……
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory
            - profile_result.non_kv_cache_memory
            - cudagraph_memory_estimate_applied
        )                                                                # L548
```

最后一行就是本章的主角：**available_kv = 预算 − 非 KV 占用 − CUDA 图估计**，一行减法定出 KV 池的全部本金。三笔里最值得讲的是第三笔：为什么 CUDA 图（把一段 GPU 调用序列录下来重放的机制，[第 1 章](../../ch01-vllm-v1-in-one-map/narrative/chapter.md)立过）的显存要吃 KV 的预算？因为图重放要求各步读写**同一批地址**，捕获期创建的张量被锁进一块「图私有显存池」，独立记账、随图对象常驻。这笔债若不入账，KV 池就会定大，捕获时 OOM。而它能被**预估**的依据是共享池语义：多个图可以共享一块私有池（前提是按捕获顺序回放，PyTorch 官方原话 "It's safe for a set of graphs to share a private pool if you know they'll always be replayed in the same order they were captured"）。vLLM 据此真捕取样：开一个**临时的共享图池**（共享语义与真实捕获相同、句柄独立，注释原话 "Use a temporary pool for profiling to avoid fragmentation in the main pool"，不污染真实捕获要用的主池），全部 wrapper（每张 CUDA 图的包装对象，`CUDAGraphWrapper` 一族的实例）的池句柄被临时换到它上面、finally 里换回；每种模式只真捕**前两个**形状，第一捕量出共享底座，第二捕量出每张图的边际占用，其余形状按「首捕 + 每图边际 × (n−1)」外推（每图下限 1 MiB，debug 日志把这笔账写作 "first-capture + (%d-1) × %.2f MiB per-graph"）。估计不是拍脑袋，是有实测锚点的外推。开关 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 默认开（vllm/envs.py:L295），日常说的「0.92 已经含了图池的账」就是它。图捕获本身的编排（warmup、逐形状捕获）归编译章，此处只认显存侧的归属。

中间那个 assert 是「快照不是保证」的防御：如果 profile 期间同卡其他进程**释放**了显存（after 的 free 反而变多），说明环境在动，测出的账不可信，直接拒跑。整段跑完，12.5 GB 假想卡上的一笔完整账（示教读数、真算术；这一景刻意取 util 0.8 而非默认 0.92，12.5 GiB 卡配 0.8 让三个数凑整好验算）：

<!-- trace: m1 -->
| 步 | 动作 | 关键算式 | 结果 |
|---|---|---|---|
| 一 | request_memory 定预算 | ceil(13421772800 × 0.8)（12.5GiB 卡 × util 0.8） | requested = 10737418240 B（10.0 GiB）；free=5 时直接 raise |
| 二 | memory_profiling 峰值账 | total_consumed 1342177280 + transient 268435456（权重 0.75 + 峰 0.25 + 非 torch 0.5） | non_kv = 1610612736 B（1.5 GiB） |
| 三 | 一行减法 | 10737418240 − 1610612736 − 536870912（cudagraph 估计 0.5 GiB） | available_kv = 8589934592 B（8.0 GiB） |
| 四 | 字节换块数 | 8589934592 // 262144 // 32（page = 2×16×32×128×2） | 1024 块；护栏需 2147483648 B（2.0 GiB）≤ 8.0 GiB → 过 |
| 读数 | 容量/并发核算 | 容量 = 1024 × 16；并发 = 1024 / 256（每 token KV 524288 B = 0.5 MiB） | 容量 16384 token；max_model_len 4096 下并发 4.0× |

![三步定账：预算、峰值账、一行减法](../diagrams/ch14-fig-boot-three-steps.png)

> *图注：L0 启动段的定账放大（对应 L2 章图北行站 1-3）。左四步管道：request_memory 按总显存 × util 定预算（free 不足直接 raise）→ memory_profiling 记峰值账 → 一行减法 requested − non_kv − cudagraph = available_kv → 除页除层换块数；右侧把 10 GiB 预算瀑布式剖成 1.5 + 0.5 + 8.0 三段，就是同一行减法的可视化。8 GiB 本金换成 1024 块、容量 16384 token、4096 长度下并发 4×。池多大，这一趟减法说了算；此后运行期的每道门都照这份账放行。*

（表中第四步的换块数与护栏，下一节展开。）

三步定账的代价也要说全：**profile 是快照不是保证**。dummy 前向的激活峰是「最大一步」的峰，真实负载里多模态编码、异常长度的采样缓冲都可能超出。防御只有两条：`gpu_memory_utilization < 1` 留头寸（默认 0.92），以及 assert 拦住「环境在动」的启动。v0.27 还加了一条复跑通道：把测得的 KV 内存建议落盘成 startup plan，复跑时用 `--kv-cache-memory-bytes` 直接取上次测值、跳过 profile 的漂移（旁路，主线不展开）。

## 字节换块数：护栏四道（含站 4、6）

有了字节数，还要换成块数、并保证这份账能开工。现在走到 L0 图启动带与「调度 · 显存账本」列的交界：`get_kv_cache_configs`（定账总控，vllm/v1/core/kv_cache_utils.py:L2094-L2242），它的 docstring 五步流程（合并全 worker 的 spec → 全模型分组 → override 折算与 auto-fit → 逐 worker 护栏出 config → 流水线取最小）差不多就是本章剩下内容的大纲。本节先走「换算与护栏」这半。

### 每层自报形状：spec 是全部原料（站 4）

换算需要两个输入：一页多大、一「组」多少层。前者每层自己报：

```python
# vllm/v1/worker/gpu_model_runner.py:L7800-L7837
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        # … 省略：ec_transfer（EC＝弹性上下文缓存的跨实例迁移特性；
        #       非消费者的实例不进 KV 账本）一行早退 ……
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # … 省略：跨层 KV 共享（该层借用目标层的缓存，
                #       不自报 spec，省显存的一条正交通道）……
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                # … 省略：后端 stride 索引能力回填七行 ……
                kv_cache_spec[layer_name] = spec
        return kv_cache_spec
```

`KVCacheSpec`（KV 缓存形状描述）是每层交上来的「自我介绍」：注意力层报 `AttentionSpec` 家族，最常见 `FullAttentionSpec`（全历史）、回收型的 `SlidingWindowSpec`（滑窗）、`ChunkedLocalAttentionSpec`（分块局部），Mamba 层报 `MambaSpec`（状态型，下节讲）。页大小的物理公式[第 13 章](../../ch13-paged-kv/narrative/chapter.md)嵌过：`real_page_size_bytes = 2 × block_size × num_kv_heads × head_dim × dtype`（vllm/v1/kv_cache_interface.py:L211-L226），一页装 16 个 token 的 K 一份、V 一份。**spec 是分组与字节换块数的全部原料**，这句后面反复用到：每层自己知道自己要多少历史，这是账本能精确到层的前提。

### 总算术：除两次，页和层（站 6）

字节换块数的总算术只有一行：

```python
# vllm/v1/core/kv_cache_utils.py:L993-L1010
def get_num_blocks(
    vllm_config: VllmConfig,
    num_layers: int,
    available_memory: int,
    page_size: int,
) -> int:
    # … 省略：docstring 七行（参数说明）……
    num_blocks = int(available_memory // page_size // num_layers)       # L1008
    num_blocks = max(num_blocks, 0)
    return may_override_num_blocks(vllm_config, num_blocks)
```

`available // page_size // num_layers`：先除一页的字节数得「页数」，再除每块的层数得「块数」，因为一个块 id 在**每一层**都要有一页（[第 13 章](../../ch13-paged-kv/narrative/chapter.md)的账：块表是每请求一张、物理页是每层一份）。uniform 模型（全部层同一个 spec）里 `num_layers` 就是全模型层数；混合模型按组算，下一节展开。两次整除的零头直接丢：账本向下取整，宁少勿超；又因两条都是非负整除，预算只增时块数只增不减。拿前一笔账验算：8 GiB ÷ 262144 B/页 ÷ 32 层 = 1024 块，容量 1024 × 16 = 16384 token；`max_model_len`（模型允许的最大序列长度）4096 下每请求 256 块，并发 1024 / 256 = 4.0 条。

### 装不下就明说：护栏一与护栏二的二分估长

账算出来若连**一条** `max_model_len` 的请求都装不下，这个引擎开不了工，池是死的。第一道护栏就是这条活性下限：

```python
# vllm/v1/core/kv_cache_utils.py:L751-L788
def _check_enough_kv_cache_memory(
    available_memory: int,
    get_needed_memory: Callable[[], int],
    max_model_len: int,
    estimate_max_model_len: Callable[[int], int],
):
    if available_memory <= 0:
        raise ValueError(
            "No available memory for the cache blocks. "
            "Try increasing `gpu_memory_utilization` when initializing the engine "
            # … 省略：报错尾两行（官网省显存指南链接）……
        )

    needed_memory = get_needed_memory()

    if needed_memory > available_memory:
        estimated_max_len = estimate_max_model_len(available_memory)
        estimated_msg = ""
        if estimated_max_len > 0:
            estimated_msg = (
                "Based on the available memory, "
                f"the estimated maximum model length is {estimated_max_len}. "
            )

        raise ValueError(
            f"To serve at least one request with the model's max seq len "
            f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
            # … 省略：报错尾六行（建议调 utilization 或降 max_model_len）……
        )
```

差 1 字节也拦：护栏一的边界场景里 `available = 33554431`、`needed = 33554432`，照样 raise。但真正的巧思在报错里的那半句 "the estimated maximum model length is …"：装不下时不光说不行，还算出**能行多长**，这一步就是护栏二（二分估长）。算法是二分：

```python
# vllm/v1/core/kv_cache_utils.py:L820-L851 · estimate_max_model_len
    # Save the original max_model_len to restore after estimation
    original_max_model_len = vllm_config.model_config.max_model_len

    # Define a function to check if a given model length fits in memory
    def fits_in_memory(model_len: int) -> bool:
        # Temporarily modify the max_model_len for this calculation
        vllm_config.model_config.max_model_len = model_len
        # Calculate memory needed for the given model length
        memory_needed = max_memory_usage_bytes(vllm_config, kv_cache_spec.values())
        return memory_needed <= available_memory

    try:
        # Binary search for the maximum model length
        left, right = 1, original_max_model_len

        # If even the smallest model length doesn't fit, return 0
        if not fits_in_memory(left):
            return 0

        # Binary search for the maximum model length that fits
        result = 1
        while left <= right:
            mid = (left + right) // 2
            if fits_in_memory(mid):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result
    finally:
        # Always restore the original max_model_len to avoid side effects
        vllm_config.model_config.max_model_len = original_max_model_len
```

二分能用的根是单调性：`fits(len)` 问「装下一条 len 长的请求需要多少字节」，需要量 = `cdiv(len, bs) × 页 × 层数`，随 len 只增不减：每层的 KV 只会越攒越多。在单调谓词上做标准 upper-bound 二分，O(log L) 收敛；`try/finally` 恢复原值，估算不留副作用。2 层小模型上实跑一遍（场景 available 只够 100 个 16-token 长度块；进表前先说明两件事：「探针 N」指二分搜索的第 N 次求值，与上段护栏一那个边界场景是两个用法；表首行的护栏一另起一景，available 卡在差 1 字节的临界值上，别拿它去对上面那 100 块）：

<!-- trace: m4 -->
| 护栏/探针 | 输入/试长度 | 需字节（长度块） | 判定与动作 |
|---|---|---|---|
| 护栏一 check_enough | available 33554431 B（差 1 字节） | needed(4096) = 33554432 B | raise：拦下并在报错里附二分估长提示 |
| 探针 1（fits 早退检查） | len 1 | 131072（1） | fits：result=1，进二分 |
| 探针 2 | len 4096 | 33554432（256） | 256 > 100 不装 → 右缩 |
| 探针 3 | len 2048 | 16777216（128） | 128 > 100 不装 → 右缩 |
| 探针 4 | len 1024 | 8388608（64） | 64 ≤ 100 装下 → 记 result=1024 |
| 探针 5 | len 1536 | 12582912（96） | 96 ≤ 100 装下 → 记 result=1536 |
| 探针 6 | len 1792 | 14680064（112） | 112 > 100 不装 → 右缩 |
| 探针 7 | len 1664 | 13631488（104） | 104 > 100 不装 → 右缩 |
| 探针 8 | len 1600 | 13107200（100） | 100 ≤ 100 装下 → 记 result=1600（终值） |
| 探针 9 | len 1632 | 13369344（102） | 102 > 100 不装 → 右缩 |
| 探针 10 | len 1616 | 13238272（101） | 101 > 100 不装 → 右缩 |
| 探针 11 | len 1608 | 13238272（101） | 101 > 100 不装 → 右缩 |
| 探针 12 | len 1604 | 13238272（101） | 101 > 100 不装 → 右缩 |
| 探针 13 | len 1602 | 13238272（101） | 101 > 100 不装 → 右缩 |
| 探针 14 | len 1601 | 13238272（101） | 区间收敛 → 返回 1600；max_model_len 原值 8192 无副作用恢复 |
| 护栏三 auto-fit | original_max_model_len=-1、available 13107200 B | 组口径二分 15 探针 | max_model_len 8192 → 1600（并在启动序里 collective_rpc 同步 worker） |
| 护栏四 override 折算 | override=3、profiled 13107200 B | 折算 available = 3 × 131072 = 393216 B | num_blocks=3：护栏/定块同按折算容量规划，账本不漂移 |
| 护栏四 PP 取最小 | 两 worker 200 / 90 长度块 | 张量 13107200 → 5898240 B（缩到 90 页；PP 各 worker 只持 1 层，此行 1 块是单层页、非探针行的两层整块） | 两 rank 同取 90 块，张量按比例缩不空耗 |

![二分估长：单调谓词上的对折试探](../diagrams/ch14-fig-binary-search-len.png)

> *图注：L0 启动段定账护栏的放大（对应 L2 章图中排拍片②）。左：护栏一差 1 字节也拦（33554431 < 33554432 → raise）；左列「输出去向」面板：估出的长度写进这条报错的提示（estimated maximum model length is 1600），替用户回答「那我最多能开多长」，这个数就是右边这张表的产物；右：二分在 8192 里折半试探，4096 要 256 块不装、1024 要 64 块装下、1600 恰好 100 块装下、1601 又要 101 块，循环 13 次、恰达 ⌈log₂ 8192⌉ 上界，加首个 fits(1) 早退检查共 14 次调用。fits(len) 随 len 单调不减（每层 KV 只增不减）是二分正确性的根；全程 try/finally 恢复 max_model_len 无副作用。*

### 人工指定与流水线取最小：护栏三、四

护栏三是 auto-fit：`max_model_len` 传 `-1` 表示「不指定，能开多长开多长」。启动时在**各 worker 的投影组**（全模型分组投到该 worker 所持层上的子集；PP 时各组按流水段切开，单卡时就是全部组）上分别跑二分、取最小，瓶颈 worker 说了算，把长度直接定在账面上；`-1` 时二分的搜索上界取模型配置里的原生上下文上限（本例 8192）。m4 表那行的「15 探针」是这么数的：二分共 14 次求值（含首个 fits(1) 检查），加定长 1600 后护栏一按新长度复核需求的一次。同一个求值函数既当探针又当复核，两个口径殊途同归到 1600。护栏四是 `num_gpu_blocks_override`（人工指定块数，测试里常用来制造小池子逼抢占）的**折算**：

```python
# vllm/v1/core/kv_cache_utils.py:L2159-L2179 · get_kv_cache_configs
    # If `num_gpu_blocks_override` is set, the cache size that will actually
    # be allocated is decoupled from the profiled `available_memory`:
    # `may_override_num_blocks` in `get_kv_cache_config_from_groups` clamps
    # `num_blocks` to the override. Reflect that in `available_memory` here so
    # auto-fit, the admission check, and the per-worker config builder all
    # plan against the same effective capacity.
    override = vllm_config.cache_config.num_gpu_blocks_override
    if override is not None:
        adjusted_memory: list[int] = []
        for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
            if not groups:
                adjusted_memory.append(avail_mem)
                continue
            bytes_per_block = _pool_bytes_per_block(vllm_config, groups)
            # … 省略：override 记日志两行 ……
            adjusted_memory.append(override * bytes_per_block)
        available_memory = adjusted_memory
```

关键在注释那句 "so auto-fit, the admission check, and the per-worker config builder all plan against the same effective capacity"：人工指定块数不是简单替换一个数，而是把 `available_memory` 也改写成 `override × 每块字节`：auto-fit、准入门、配置产出全部按折算后的容量规划。**账本不许有两套数**，这条纪律贯穿本章。

护栏四的后半在函数尾部，先铺适用场景：`kv_cache_configs` 是每 worker 一份的清单，单卡只有一份；流水线并行（PP，模型按层切到多张卡）时每个 rank 一份（代码注释里的 rank 即各卡的 worker 进程），各卡显存与所持层数不同，各自 profile 量出的可用字节不同，算出的块数也就不齐。所以这段代码没有一个 PP 字样、却无条件对每份配置跑：单卡时对一个数取 `min` 是恒等操作、无害，多 rank 块数不齐时才真正生效。

```python
# vllm/v1/core/kv_cache_utils.py:L2210-L2242 · get_kv_cache_configs
    # Change the num_blocks of each rank to the smallest among all ranks.
    # We also need to shrink the tensor size proportionally to avoid
    # allocating unused memory.
    min_num_blocks = min(
        kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
    )
    for kv_cache_config in kv_cache_configs:
        num_blocks_old = kv_cache_config.num_blocks
        kv_cache_config.num_blocks = min_num_blocks

        # Shrink tensor size proportionally
        for tensor in kv_cache_config.kv_cache_tensors:
            assert tensor.size % num_blocks_old == 0
            tensor.size = tensor.size // num_blocks_old * min_num_blocks   # L2223

        if len(kv_cache_config.kv_cache_groups) > 0:
            # … 省略：容量/并发核算与两条启动日志（下一节展开）……
    return kv_cache_configs
```

走读三步：第一步 `min` 取全场瓶颈（一个块的页分散在各 rank 所持的层上，最穷的 rank 给所有人封顶）；第二步把每份的 `num_blocks` 都改写成这个最小值，调度器一本账只认一个块数；第三步把张量字节等比缩下来。缩的对象 `kv_cache_tensors` 在这首次露面：它是 config 开出的物理显存分配清单，每个条目是一张将来要在 GPU 上真分配的张量，`size` 记字节数，真分配发生在后文「worker 侧落地」一节，此刻只是账面数字。为什么缩、又为什么这么缩：块数已降到全场最小，张量若仍按原块数的字节去分配，就是买用不上的显存；而 `size` 是字节不是块数，先 `// num_blocks_old` 除回每块字节、再乘上新块数，`assert` 整除保的是页对齐——每块必须是整数页，账被动过手脚、除出零头就当场拦下。

表末 PP 行就是第三步的实跑实例：2 层小模型按 PP 切成两个 worker、各持 1 层，A 量出 200 块、B 量出 90 块，全场取 90。这行的块口径也随 PP 换了：每 worker 只持 1 层，1 块就是一张单层页 65536 B，不是探针行那种两层共 131072 B 的长度块；65536 正是「每层自报形状」一节页公式的代入值：每头 128 维 × 8 个 KV 头 × 每页 16 token × K/V 各一份 × fp16（16 位半精度浮点）每数 2 字节。所以 A 的张量是 200 × 65536 = 13107200 B，缩到 90 页 = 5898240 B，与表内数字一致；它与探针行的 13107200 数值相同并非同一笔账：那边 100 个两层整块、这边 200 张单层页，同一字节数的两种块口径。这个口径差不能放着不管：调度器账本里只记一个块数，块折合多少字节全靠口径对齐；若调度器按 131072 B 的两层整块记账、worker 手里却是 65536 B 的单层块，同一个「100 块」在两边就是不同的字节数，调度器以为放得下的请求 worker 那边根本装不下，或者反过来白白拒单。所以块的字节口径必须全场唯一，而「定账单点产出一份账、同一份账喂调度器进程与全部 worker 进程」正是下一节『一份账喂两侧』要拆的事。

## 一份账喂两侧（站 7）

定账的产物是 `KVCacheConfig`（num_blocks + 分组 + 张量布局），它要去两个世界：调度器进程拿它建账本，worker 进程拿它真分配显存。开读前先交代一条改名链，免得三个名字读成三笔钱。这笔数的出生地在站 3：各 worker 用一行减法量出 `available_kv_cache_memory_bytes`。汇到 EngineCore，入参名换成 `available_gpu_memory`（多卡时是一张逐 worker 的清单）；再传进定账总控 `get_kv_cache_configs`，形参又写成 `available_memory`。一路下来三个名字、一个数，行文统称 available_kv。先看总览图，再进总编排：

![站 7 调用全景：一笔账从站 3、站 4 汇入 get_kv_cache_configs 单点定账，再分两版喂调度器与 worker](../diagrams/ch14-fig-init-call-panorama.png)

> *图注：L0 图「调度 · 显存账本」列启动带的放大，站 7 的调用关系一图看全：账从哪来（站 3 一行减法量出的可用字节、站 4 各层自报形状）→ 单点定账在哪（get_kv_cache_configs，护栏四道上一节已拆，这一景 40 MiB 算到 320 块）→ 喂到哪两侧（拍平版给调度器建账本，布局版给 worker 在 CuMem（vLLM 的显存分配器）显存池真分配）。先在图上认全这三段再进下面的代码块，正文按讲解需要逐段展开。*

```python
# vllm/v1/engine/core.py:L301-L330 · EngineCore._initialize_kv_caches
        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks   # L316
        kv_cache_groups = scheduler_kv_cache_config.kv_cache_groups
        if kv_cache_groups:
            vllm_config.cache_config.block_size = min(
                g.kv_cache_spec.block_size for g in kv_cache_groups
            )
            num_tokens, max_concurrency = get_kv_cache_capacity(
                vllm_config, scheduler_kv_cache_config
            )
            vllm_config.cache_config.kv_cache_size_tokens = num_tokens    # L325
            vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency  # L326

        vllm_config.validate_block_size()

        self.model_executor.initialize_from_config(kv_cache_configs)     # L330
```

三个动作值得逐个看。**其一，auto-fit 的善后**：worker 是在 profile **之前**拉起的（装配序[第 17 章](../../ch17-executor-worker-model-runner/narrative/chapter.md)细拆），它们缓存的是原始的较大 `max_model_len`；auto-fit 把长度缩小后要 `collective_rpc("update_max_model_len")` 同步过去，不然两边按不同长度算账。**其二，写回四件套**：`num_gpu_blocks` / `block_size` / `kv_cache_size_tokens`（池能装多少 token）/ `kv_cache_max_concurrency`（满长度请求能并发几条）写进 `cache_config`，前端日志与 API 看到的就是这些值。容量与并发怎么算的，两条布局各跑一遍（第二行有三个没见过的记号：SWA、在途、cap 公式，下一段立刻逐个交代，先扫一眼位置即可）：

<!-- trace: m15 -->
| 布局 | 每请求块数 | 并发 | 容量（token） |
|---|---|---|---|
| uniform Llama-2-7B | 256（=cdiv(4096,16)，单组） | 4.0（=1024/256） | 16384（=4.0×4096=1024×16） |
| 混合（in_flight 8192） | 513（full 256 + swa cap 257，cap=cdiv(min(511+8192,4096),16)+1） | 1.9961（=1024/513） | 8176 |

uniform 一行全是旧相识：序列全长 4096、每块 16 token、池 1024 块，就是「总算术」一节那笔 8 GiB 验算账的原班数字；每请求 256 块、并发 1024/256 = 4.0、容量 16384，照抄进表。混合一行要先认一个新角色：SWA（Sliding Window Attention，滑动窗口注意力）层只保留最近 W 个 token 的 KV（W 是窗口大小，这一景取 512），所以它每请求占的块数不随序列全长涨、有个封顶；full 组（全注意力层，按整序列记块）配上一个 SWA 组，就是表里第二行的布局。SWA 到底是什么、为什么按组记账，下一节「一个池子多张账」完整展开；这里只需要「它有个封顶」这一个事实，把封顶 257 逐因子算出来。

封顶 257 要代入三个数：窗、在途、长度上限。窗是 512：窗口盖住当前 token 加它前面的 511 个（窗减一），这 511 个历史 token 的 KV 是 SWA 层必须留住的；窗口里最新的那个 token 通常还在途上，归下一项计。

「在途」是哪些 token？调度器每一拍会把一批 token 排进 batch 交给模型去算，从交出去到结果回来之间有一段空档，这批 token 就处在「已经排进 batch、这一拍还没算完」的状态；书里把结果回来、数字定死这一步叫落账，落账之前它们都算在途。在途期间它们的 KV 块既不能回收、也不能挪给别人用，账上必须留着。上界参数 `max_in_flight_tokens` 就是为这段空档留的：一拍排进去的 token 数不超过 `max_num_batched_tokens`，而多卡流水线或异步调度时会有多个批同时在途，源码把它写作「同时在途的批数 × `max_num_batched_tokens`」（vllm/config/vllm.py:L553-L561）；本例单批、配置里 `max_num_batched_tokens` 取 8192，所以在途上界就是 8192。加减这个数的时机在「门多紧」一节正面拆，这里只需要这个上界。

两项相加 511 + 8192 = 8703，被 max_model_len 压到 4096，这个 4096 就是前文 uniform 例里的序列全长上限；4096 个 token 按每块 16 个正好切满 256 块，再 +1 顶着「窗口起点不在块首」的最坏错位，封顶 = 256 + 1 = 257。

每请求块数 = 各组之和：full 组按整序列记 256（就是 uniform 行那个数），SWA 组记封顶 257，合计 513。并发 = 同一个池 1024 块 ÷ 513 = 1.9961，容量 = int(1.9961 × 4096) = 8176（容量恒等式 tokens = 并发 × max_model_len）。这就是混合行的看点 **公式通用**：每请求块数按组求和之后，用的还是 uniform 行那套 `num_blocks / 每请求块和` 出并发、再乘 max_model_len 出容量，两条布局共用一条公式。启动日志那两行 "GPU KV cache size: %s tokens" 与 "Maximum concurrency for %s tokens per request: %.2fx"（kv_cache_utils.py:L2225-L2240）输出的就是这两笔。

257 这个值别过度解读：这一景在途 8192 太大，把封顶一路顶到了 max_model_len，SWA 组只比整序列的 256 多记 1 块，窗口语义近乎退化。窗口真正发力在小在途与长序列，那笔收益账留到「回收感知准入上限」一节末的收益合成再算。

**其三，喂两侧**。调度器侧拿到的是拍平版：

```python
# vllm/v1/core/kv_cache_utils.py:L1855-L1874
def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """
    Generate the KV cache configuration for the scheduler.
    """
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    # All workers have the same kv_cache_config except layer names, so use
    # an arbitrary one to initialize the scheduler.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg
```

`generate_scheduler_kv_cache_config` 只做无损拍平：断言全部 worker 的 num_blocks 相等（PP 取最小已保证），代表 spec 任取一层。worker 侧拿到的是同一份 config 的张量布局，据此在 GPU 上真分配显存，落点在 CuMemAllocator（vLLM 的显存分配器）的 `tag="kv_cache"` 池里。这个分配器按显式 tag 分池记账，kv_cache 一池、weights 一池，互不混账。分池直接服务一个特性：「睡觉/唤醒」（sleep/wake，把整池显存让渡给同卡其他进程、需要时再收回）就是按池整批操作的，两本分池账正是它的记账单位（分配本体 vllm/v1/worker/gpu_worker.py:L649-L676，[第 17 章](../../ch17-executor-worker-model-runner/narrative/chapter.md)会嵌这段细拆）。装配序实跑（2 层小模型、40 MiB 可用）：

<!-- trace: m9 -->
| 装配步 | 动作 | 关键数 | 落账 |
|---|---|---|---|
| 收 spec | worker 每层自报形状 | page 65536 × 2 层 = 每块 131072B | — |
| 定账 | get_kv_cache_configs 单点产出 | available 41943040 B → 41943040 // 65536 // 2 | num_blocks = 320 |
| 写回 | cache_config 四件套 | block_size 16、容量 5120 = 320×16 | 并发 1.25 = 320 / 256（4096 长度每请求 256 块） |
| 喂两侧 | 拍平版喂调度器 / 布局版喂 worker | worker initialize 拿到同 config | 两侧 num_blocks 同 320（executor_got_same_config=true） |

![一份账喂两侧：单点产出的两次投影](../diagrams/ch14-fig-one-ledger-two-sides.png)

> *图注：L0 启动段到「调度 · 显存账本」列与 GPU 列的双喂线放大（对应 L2 章图中排拍片④）。profile 出 40 MiB 可用，定账函数单点算到 320 块，写回 cache_config 四件套（num_gpu_blocks=320、block_size=16、容量 5120 token、并发 1.25×），拍平版喂调度器建 KVCacheManager、张量布局版喂 worker 在 kv_cache 池内真分配。两侧数字必然一致：它们是同一份 KVCacheConfig 的两次投影；PP 场景各 rank 先取最小再缩张量，单源即防漂，任何一侧想看到不同的块数都必须绕过这个函数，而装配序里没有第二条路。*

这里的不变量值得一句论证：`get_kv_cache_configs` 是 `KVCacheConfig` 的**唯一产出点**（engine/core.py:L304-L306 单点调用），调度器侧只做无损拍平、worker 侧只做按布局分配、PP 时显式取最小，两侧 num_blocks 永远相等，靠的是**结构**，不是运行时对账。

## 一个池子多张账：混合注意力的组化（站 5、8）

到目前为止都假设全部层是同一个 spec，一张块表、`num_layers` 除一次了事。多数模型确实如此，但真实的主力模型早就不是了。现在走进 L0 图「调度 · 显存账本」列的**池内**：一个池子怎么伺候好几种层。

### 层的类型不一样了：滑窗、分块、Mamba

先立三个外部概念，它们是本节的载荷。

**滑动窗口注意力**（Sliding Window Attention，SWA；站 7 的混合行只借了它「有封顶」这一个事实，这里正式介绍它）：每个 token 只「看得见」前面 W 个 token（W 就是窗口大小），位置 i 的 query 只对 `[i−W+1, i]` 的 key/value 算注意力。动机就是 KV 账：全注意力层每生成一个 token 要为**全部历史**存 K/V，序列越长池越大；SWA 层的 KV 需求封顶在 W，与序列长度无关。省显存的理由值得说到根上：既然位置 i 的注意力只扫这一个窗口，比窗口起点更早的 token 就不在任何**未来** query 的窗口里了，没有人会再读它们的 K/V，而没人再读的 KV 不必继续占着显存，可以整块还给池子。全注意力层没有这条性质（每个历史 token 对每个后续 query 都可见），它的块一个都不能扔。这条对比是下一小节「共用一张表的两难」与后面「SWA 的还账方式」两处共同的根。这路数的工程化出自 Mistral 7B（arXiv:2310.06825）：W=4096，配「滚动缓冲区」，论文原话 "The cache has a fixed size of W, and the keys and values for the timestep i are stored in position i mod W of the cache"，i 超过 W 后老位置直接被覆写，32k 序列上省 8 倍缓存显存。常被问的「窗口截断了信息怎么传远」：答案是层层接力，第 k 层每个位置能看到上一层 `[i−W, i]` 的隐状态，信息逐层向前搬，k 层之后理论可达 k × W（论文按 W=4096 算出 32 层约 131K token 的理论跨度）。vLLM 的实现与 Mistral 的环不同**粒度**：Mistral 是 token 级的环（覆写），vLLM 是**块级**的回收，窗外整块 free 归池、原位换 null 占位（本节末与下一节都有实跑），没有环形覆写。差异不是风格：分页是 vLLM 一切显存操作的地基，回收也不例外。

**混合注意力模型**（hybrid attention）：省 KV 的层与管全局的层按固定比例掺着排，这是 Gemma、Llama、gpt-oss 这批模型的真实架构。Gemma 3 技报（arXiv:2503.19786）说得直白："A challenge with long context is the memory explosion of the KV cache during inference. To reduce this issue, we interleave multiple local layers between each global layer"，5 个局部层配 1 个全局层（5:1），局部层窗口 1024，消融显示纯全局布局的 KV 开销约 60%、混合后压到 15% 以下。LLaMA 4 是 3 local : 1 full，局部层是「分块注意力」（chunked attention，块大小 8192，块内互看）。gpt-oss 每两层一块交替 dense 与 sliding-128（128 token 的窗口）；配上 EAGLE（一种投机解码方案：小草稿模型先猜几个 token、大模型一次验证，采样篇展开）的草稿层后正是 12 个滑窗层 + 13 个全注意力层，这个 12+13 马上会再见到。对推理引擎，这一切意味着一件事：**一个模型各层自报的 KVCacheSpec 不再全同**，一张块表伺候不了。（vLLM 官方的混合 KV 管理设计文档是这条线讲得最全的参考：[docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)。）

**Mamba 与状态空间模型**（SSM）：把「记住全部历史」从「每 token 存一对 K/V」换成「把历史压进一个固定形状的状态张量」，像 RNN 一样边走边压缩，序列再长它的「缓存」也不长一个字节（Mamba 论文 arXiv:2312.00752 的摘要账：5× 于 Transformer 的推理吞吐、序列长度线性伸缩）。主流落地是混合：Jamba（arXiv:2403.19887）按 attention : Mamba = 1:7 掺层，账面收益 "an 8x smaller KV cache compared to a vanilla Transformer"（256K 上下文 4 GB 对纯 Transformer 32 GB）。对账本的意义：Mamba 层进账本时报的是 `MambaSpec`，它的一页装的是一份固定形状的状态张量（卷积状态与 SSM 状态合起来算），大小由**状态形状**决定，既不随 block_size 缩放，也不随序列长度涨——序列跑到十万 token，它要留的还是这一份。这正是它省显存的本钱，也是它在池里最别扭的地方：别的层的页按 token 数算，它按状态形状算，两边的页宽天生对不上。注意它不是 KV cache，账本科目不同，这个差别马上在「页统一」处收账。

这一批模型里还有一位要单独交代：**DeepSeek V4**。它是本书 Part VI（模型层）后半程逐章拆读的主角：[第 24 章](../../ch24-primer-attn-variants/narrative/chapter.md)立注意力变体的数学，[第 28 章](../../ch28-deepseek-v4-principles/narrative/chapter.md)把它的原理账逐件算清，[第 26 章](../../ch26-deepseek-indexer-nsa-dsa/narrative/chapter.md)拆它怎么挑要看的 token，[第 29 章](../../ch29-deepseek-v4-assembly/narrative/chapter.md)把它整个拼装起来。本章不展开那条线，只拿它给通用组化规矩做一次压力测试：V4 一层里同时挂着四类缓存、账本上四种页宽并存；真实在产的旗舰模型就是这种量级，通用规矩得接住它。

### 组化的前史：引擎是怎么被模型潮推着走的

上一节讲的是模型侧为什么要把层掺着排；引擎侧接住这批模型，也有一段被逼着走的路。今天的分组规矩不是一次设计出来的，把这段来龙去脉立住，后面每条规矩长什么样才说得清。

旧世界先看清楚。v0（vLLM 第一代引擎，[第 1 章](../../ch01-vllm-v1-in-one-map/narrative/chapter.md)立过 v0/v1 两代的分野）没有为「各层缓存形状不同」留位置，Jamba 这批最早的混合模型靠手工补丁挤进来：Mamba 状态存成独立张量，按 `max_num_seqs`（调度器允许的最大并发序列数）预分配，并发猜大了启动直接 OOM、猜小了请求排队容量白扔（[PyTorch 官方博客](https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/)，vLLM 团队撰文，2025-11-05）。比 OOM 更深的问题是账本割裂：状态张量游离在块池之外，前缀缓存（把已算过的公共前缀 KV 留在池里给后续请求复用）、KV 传输（把算好的 KV 搬进别的进程，[第 16 章](../../ch16-kv-connector/narrative/chapter.md)的正题）、PD 分离（prefill 与 decode 两阶段拆开部署，[第 37 章](../../ch37-pd-disaggregation/narrative/chapter.md)的正题）这三件依赖统一块账本的特性，对混合模型全部无效。解法是 v1 的统一 KV 分配器：不管什么形状的缓存，都从同一个分配器领块，三件事一起解锁。官方博客把 2025-11-05 称为混合模型成为「一等公民」的日子；代价也直说——V0 的手工补丁退役，换来的是一整套要伺候多种页形状的分配机制，本章后面的全部复杂度都是这份学费。

统一分配器立住之后，组的形态分三步长出来（社区深读博客[《vLLM KVCache 分组》](https://mengqingcao.github.io/posts/vllm-kvcache-grouping/)把它梳理成三种形态，本节的图按它画）。第一步，全同 spec 组：所有层页形状一模一样（Qwen2.5、DeepSeek V3.1），全层一组，不需要任何混合机制，多数模型至今停在这一步。第二步，同型异宽组：spec 不全同、但都是注意力缓存、只是每 token 字节不同。第一个现实案例是 DeepSeek V3.2 的索引器缓存（132 B/token，对主 MLA 的 656 B/token），MTP 层（multi-token prediction，多 token 预测：DeepSeek 系模型自带的投机解码头，一次预测多个 token 供大模型验证）与 MiniCPM 4.1（面壁智能的开源小模型，索引器缓存同样偏小）跟进。vLLM 为此造了 UniformTypeKVCacheSpecs（同型 spec 的打包容器：把每块要的 token 槽数相同的层捆成一个分配单位），落在 [PR #25101](https://github.com/vllm-project/vllm/pull/25101)（2025-09）。第三步，混合 spec 组：跨类型的层要共享同一个物理池，全注意力对滑窗、对线性注意力都算跨类型，Gemma3 与 Qwen3-Next（阿里的混合架构模型，全注意力：线性注意力按 1:3 掺层）把它变成刚需。社区把这套跨类型共池机制叫 HMA（Hybrid Memory Allocator，混合内存分配器）；同期的 [PR #24486](https://github.com/vllm-project/vllm/pull/24486)（2025-09 开、10 月合）把 kernel 认的块大小与 KV 页大小解耦，高性能内核不再被小页层的页形状绑架（这个解耦的下游就是本章末「大块拆小块」那节的换算）。

为什么第三步必须共享一个池、而不是一类层一个池？Qwen3-Next 的账给出现实的浪费：它的线性层每请求只需要固定 1 个状态块，若按页统一朴素分配，每个线性层被摊到与全注意力层同样的块数，而线性层的数量还更多，浪费直接放大。可反过来给线性层单开一个池也不行——前缀缓存立刻否掉：块命中数要到运行时才知道，社区深读转述这段设计取舍时的原话是「无法预知运行时会有多少个 block 被命中」，静态分池必炸。共享一个池、池内按类型分账，是唯一活路；至于池内怎么分，下一节从「一张表为什么不行」讲起。

把这条线的时间坐标钉一下：2024-11，Marconi 论文（[arXiv:2411.19379](https://arxiv.org/abs/2411.19379)，MLSys 2025）先诊断了混合模型前缀缓存这道难题（状态不可回滚、缓存条目量级失衡），学界先于工程看到了坎；2025-09/10，上面两个 PR 把组化机制立起来；2025-11-05，一等公民；2026-04-24，DeepSeek V4 与 vLLM 官方支持博客同天发布，把这套机制推到压力测试的极限——本章后文的 V4 几节，就是那场测试在账本侧的记录。

![组化前史：三种组形态是一条演进线](../diagrams/ch14-fig-grouping-lineage.png)

> *图注：L2 章图站 5（混合组化）的前史带，图面自带「站 5 前史」指北签。左：演进前的旧世界，V0 手工补丁（Mamba 状态独立张量、按 max_num_seqs 猜并发：猜大 OOM、猜小排队）到 V1 统一分配器（前缀缓存 / KV 传输 / PD 分离三件事一起解锁）；右：三张演进卡，①全同 spec 组（多数模型，无需混合机制）→ ②类型统一组 UniformTypeKVCacheSpecs（PR #25101，动机是 MTP 投机层与 MiniCPM 4.1 的小页索引器缓存）→ ③混合 spec 组 HMA（Gemma3、Qwen3-Next full:linear = 1:3 共享同一物理 buffer，多块池被否的引文「无法预知运行时会有多少个 block 被命中」写在卡内）；下方时间线五节点：2024-11 Marconi 论文 → 2025-09-09 PR #24486 → 2025-09-17 PR #25101 → 2025-11-05 混合模型一等公民 → 2026-04-24 DSV4 双落地；底部橙色落点条：本章后文 V4 节拆的第三条分组路径，是形态②的打包容器接住形态③的跨类型共池——四种页宽装进多个 UniformType 组。日期与 PR 号为社区史料（官方博客与 GitHub PR，一手出处），分组代码以本章 pin 为准。*

### 共用一张表的两难：浪费还是损坏

「一张块表伺候不了」值得拆开论证，不是懒洋洋的工程取舍：本节整套规矩（分桶、等量、页统一）都由它倒推。直觉一句：一栋楼装一套总水表，按最省水的住户停水，24 小时热水的高层当场断供；按最费水的住户供水，省水住户的账单白交。块表同理，一张表只有一种回收语义，全按全历史或全按窗口，两个方向各有各的坏法。

全按全历史（都不回收）：窗外 KV 白占到请求结束。它的现实形态下一小节就会实跑：`--disable-hybrid-kv-cache-manager` 把滑窗层全部提升成全注意力，warning 原话自认 "we do not enable any optimizations for saving KV cache memory"。白占有实数（场景取自「门多紧」的混合门：窗 512、块 16、4096-token 序列，SWA 层按 full 语义要 256 块、回收感知只要 33 块；33 就是准入上限公式在在途 0 时的取值，即 cap = cdiv(511, 16) + 1，推导见「回收感知准入上限」一节）：一张表的世界里每请求白占 223 块，省的只剩计算侧。浪费，但账是对的。

全按窗口（都回收）：这一侧坏的是正确性。full 层每个历史 token 对每个后续 query 都可见（全注意力的定义），把它的旧块也当窗外账收走，两笔坏账同时发生：仍在读的 token 落到 NULL 占位页，丢历史；块已归池转租给别的请求，跨请求串写。回收安全的前提是「被收块对后续任何计算不可见」（后文「SWA 的还账方式」立这条不变量），全注意力类型把这条前提焊死在代码里：管家（每组一个的类型专属缓存管理员，按 spec 查注册表选，装配见「两把尺子与装配」）报跳过的 `get_num_skipped_tokens`（报「多少 token 已滑出窗口、不再被读」的方法）在全注意力管家那里不覆写、基类恒返 0，从不还账。强行统一成窗口语义，等于把它没有的「窗外」强加给它。

<!-- trace: m17 -->
| 假想世界 | 一张块表的样子 | 后果 | 证据锚 |
|---|---|---|---|
| 都不回收 | 全模型按 full 语义一张表（`--disable-hybrid-kv-cache-manager` 的现实形态） | 窗外 KV 白占到请求结束，省的只剩计算侧；SWA 层 4096 序列白占 223 块（256−33） | kv_cache_utils.py:L1568-L1589 warning 原话；下一小节分组表末行已实跑 |
| 都按窗口回收 | 滑窗语义统治全表，full 层的旧块也当窗外账收走、原位换 NULL | 仍在读的 token 落到 NULL 页＝丢历史；块归池转租他人＝跨请求串写，正确性当场损坏 | 回收安全的前提是「被收块对后续任何计算不可见」（后文回收一节的不变量）；full 管家 get_num_skipped_tokens 恒 0（基类默认 return 0）正是这条前提的代码形态 |
| 分表分管家（现实） | 每组一张私有块表＋类型专属管家：full 组全长持有、SWA 组窗外整块回收 | 账目两清、各语义无损；新代价是跨类型命中语义 | 管家按 spec 查注册表选（get_manager_for_kv_cache_spec）；跨类型命中只干净支持 full＋恰好一种其它（kv_cache_utils.py:L1189-L1194），[第 15 章](../../ch15-prefix-caching/narrative/chapter.md)展开 |

结论：**分表是唯一无损解**。可见性不同就必须分表；分了表的组还要共享同一个块池，于是有等页、等量一整套规矩，下一小节起逐条立。新代价也随即而生：多张表之后「哪些请求能认领哪些块」的命中语义变复杂，跨类型命中官方只干净支持全注意力加恰好一种其它类型（原文见下一小节末六条假设的第 6 条），完整机制在[第 15 章](../../ch15-prefix-caching/narrative/chapter.md)。

### 分桶等量：为什么每组层数必须相同

类型不一的层要共用一个池，规矩是什么？`get_kv_cache_groups`（kv_cache_utils.py:L1781-L1852）进门先把层分四路，依次问四个问题。全部层交的是同一个 spec 吗？是就一组了事（多数模型）。spec 不全同、但每块要的 token 槽数一样？也并成一组。是 DeepSeek V4 那种一层里几种页宽同时在场的情况？转交专门处理多元组打包的第三条分组路径。四种页宽怎么被它收编，[「DeepSeek V4：先把这一层看明白」](#deepseek-v4先把这一层看明白)起的四节完整拆。四个问题都不中，才落到本小节的通用等页路径；Gemma3、gpt-oss 这类页宽全等的混排走的就是它，规矩一句话：**类型相同的层并成一个桶，再把桶切成层数相等的组**。

为什么组要等大？上一小节刚论证可见性不同必须分表；但分了表的组，池还是共用那一个 BlockPool，块号全池通用，任何一个空闲块都可能被分给任何一个组。而一个块号落到组里代表什么，取决于这个组有几层：块号的物理含义是「组内每层各占一页」，组里 N 层，一个块号就要兑现 N 页。于是两组层数不等时，同一个块号在两边的字节数就不等，池里「一个块有多大」会随你从哪个组看而变。麻烦在于池的记账从头到尾只认块数：`num_blocks`、空闲队列长度、使用率分母、启动日志里的容量与并发，全是「块数 × 每块字节」；块大小不唯一，这些数就集体失真，调度器按块数放行的请求到了 worker 那边可能多占或少占一批页。源码把这条写作假设 1，给的理由是碎片：不同大小的块混在一个自由队列里，没法互换着用。所以硬约束是**每组每块物理字节数相等**：先统一页宽（否则同一组内层与层之间的页都不等，下一小节「页统一」专讲），再凑齐每组层数（本小节的 padding 就是补这个）。这条硬约束的落地就是下面这段代码：

```python
# vllm/v1/core/kv_cache_utils.py:L1233-L1280 · _get_kv_cache_groups_uniform_page_size
    # Split each group into smaller groups, to make the number of layers in each
    # group identical. Add padding to the last group of each type if necessary.
    # E.g., (full.0, full.1), (sw.0, sw.1, sw.2)
    # split to 3 groups with 2 layers each:
    # (full.0, full.1), (sw.0, sw.2), (sw.1, padding).
    # FIXME(Chen): At the moment of writing this code (2025-06-02), all
    # open-source hybrid model follows a n:1 pattern between different attention
    # types (e.g., Gemma3 5:1 between sw and full, LLaMA4 3:1 between local and
    # full), so we can use the "1" in the n:1 pattern as the group size, which
    # is the minimum number of layers among all attention types. Need a better
    # strategy if we want to support more complex patterns (e.g., 20 full + 30
    # sw, where the group size should be 10).
    min_num_layers = min([len(layers) for layers in layer_buckets])       # L1245
    group_size = min_num_layers
    max_num_layers = max([len(layers) for layers in layer_buckets])
    if max_num_layers < min_num_layers * 1.5:
        # If the number of layers is not much larger than the minimum number of
        # layers, use the maximum number of layers as the group size to avoid
        # too many padding layers. A typical example is gpt-oss-20b + eagle,
        # with 12 sw + 13 full. We pad it to (13 sw, 13 full) instead of
        # (12 sw, 24 full). 1.5 is a heuristic to avoid too many padding
        # layers while accommodating speculative decoding drafters that add
        # extra layers to one attention type.
        group_size = max_num_layers                                      # L1256
    grouped_layers = []
    for layers in layer_buckets:
        num_padding_layers = group_size - len(layers) % group_size
        if num_padding_layers != group_size:
            logger.warning(
                "Add %d padding layers, may waste at most %.2f%% KV cache memory",  # noqa
                num_padding_layers,
                num_padding_layers / len(layers) * 100,
            )
        num_groups = cdiv(len(layers), group_size)
        # In PP case, say if we have
        # - stage 0: full.0, sw.0, sw.1
        # - stage 1: full.1, sw.2, sw.3
        # We should have 3 groups: (full.0, full.1), (sw.0, sw.2), (sw.1, sw.3)
        # It can't be (full.0, full.1), (sw.0, sw.1), (sw.2, sw.3) because
        # … 省略：反例的四行展开（连续切会让某 stage 出空组、补 padding 更浪费）……
        # To avoid this, we assign layers[i::num_groups] to the i-th group
        # instead of layers[i * group_size: (i + 1) * group_size]
        for i in range(num_groups):
            grouped_layers.append(layers[i::num_groups])                 # L1279
    return create_kv_cache_group_specs(kv_cache_spec, grouped_layers)
```

四个决策。**组大小默认取各类型层数的最小值**：开源混合模型都是 n:1 模式，那个「1」（全局层的层数）天然是组大小。**1.5 启发式**：层数比不大时取 max 免得 padding 过多，12 SW + 13 full 补成 13/13（padding 1 层），比按 min=12 切（full 桶要补 11 层到 24）划算得多；padding 层白占显存，warning 原话 "may waste at most N% KV cache memory" 把账打给你。**PP 交错分派** `layers[i::num_groups]`：按步长切片入组，让流水线每个 stage 都有活干，不出空组。**落点**是 `KVCacheGroupSpec`，每组 = 同型层名表 + 合并后的代表 spec。四个场景实跑：

<!-- trace: m5 -->
| 场景 | 分桶层数 | 组大小 | 分组结果 | 代价/账 |
|---|---|---|---|---|
| uniform 对照 | 32 full | 32 | 1 组 × 32 层（多数模型） | 无 padding、单块表 |
| Gemma3 式 10+2 | SWA 10 / full 2 | 2（=min） | 6 组 × 2 层；SWA 第 0 组拿 layers 0,5（layers[i::n] 交错） | 无 padding；PP 交错避免某 stage 出空组 |
| 12 SW + 13 full | SWA 12 / full 13 | 13（13 < 12×1.5=18 → 取 max 非 min） | 2 组：SWA 12 层 + 1 padding、full 13 层 | 补 1 层 padding，浪费上界 8.33%（warning 原话 may waste at most 8.33% KV cache memory） |
| disable 回退 | SWA 5 / full 1 | — | 1 组全按 full 分配，sliding_window=512 只记录不生效 | 窗外 KV 白占显存（warning：we do not enable any optimizations for saving KV cache memory） |

![混合分桶与等量化组](../diagrams/ch14-fig-hybrid-groups.png)

> *图注：L0「调度 · 显存账本」列池内的组化层放大（对应 L2 章图中排拍片①）。左：Gemma3 式 10 SWA + 2 full 以组大小 2 切成 6 组，SWA 层按 layers[i::5] 交错入组（[0,5] / [1,6] / …），PP 时每个 stage 组数均衡；右：gpt-oss 式 12 SW + 13 full 因为 13 < 12×1.5=18，组大小取 13：SWA 桶补 1 个 padding 层凑 13/13（浪费上界 8.33%），比按 min=12 切（full 桶要补 11 层到 24）划算得多。每组层数必须相同，这是一池共享等大块的硬约束。*

最后一行「disable 回退」是这条路的退路，今天仍在：`--disable-hybrid-kv-cache-manager` 时 `unify_hybrid_kv_cache_specs`（kv_cache_utils.py:L1568-L1589）把滑窗 spec 全部提升成全注意力：一张表、实现简单，代价是 warning 原话直说的："we do not enable any optimizations for saving KV cache memory (e.g., dropping the KV cache outside the sliding window). The compute of layers like sliding window is still saved."，窗口外的 KV 白占显存（省下的只剩计算侧）。

等字节这条硬约束的完整清单在 `_get_kv_cache_groups_uniform_page_size` 的 docstring 里（kv_cache_utils.py:L1169-L1198），六条假设：每组每块物理字节相等（不同大小的块混住会碎片化）、组内同 block_size、每 token 每层字节由模型定（当前只支持各层相同）、每组层数相同（padding 补齐）、组内同注意力类型、以及官方自认的第六条：跨类型的最长命中决策 "only supports one attention type or two types of full-attention plus exactly one another type"（那套决策的细节在[第 15 章](../../ch15-prefix-caching/narrative/chapter.md)）。作用域也要说破：这六条管的是等页路径（docstring 就挂在 `_get_kv_cache_groups_uniform_page_size` 上），那四条分支在更早处就把不满足的层引去了别路。

### 页统一：各层页宽不一样时怎么办

等量化的前提是各层页字节已经相等；不等时先统一。`unify_kv_cache_spec_page_size`（kv_cache_utils.py:L1070-L1132）以最大页为基准，每层三条出路：

```python
# vllm/v1/core/kv_cache_utils.py:L1091-L1132 · unify_kv_cache_spec_page_size
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        elif isinstance(layer_spec, MambaSpec):
            # MambaSpec's page size is determined by its state shapes and does
            # not scale with block_size, so pad the page instead. This is the
            # … 省略：注释后半（与平台对齐 Mamba 页的同一 pad 机制，
            #       草稿模型页更大时会走到这里）……
            new_spec: KVCacheSpec = replace(layer_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif (
                isinstance(layer_spec, AttentionSpec)
                and layer_spec.indexes_kv_by_block_stride
            ):
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                raise NotImplementedError(
                    f"Layer {layer_name}: page size is not divisible by the "
                    "maximum page size and cannot be padded. Padding is only "
                    "supported for attention layers whose backend indexes KV "
                    "pages by the block stride (indexes_kv_by_block_stride is "
                    "True)."
                )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec
```

三条出路各有一笔账。**调大 block_size**：页 = 2 × bs × heads × dim × dtype 随 bs 线性放大，小页层把块大小翻倍、页就翻倍，每 token 字节不变，**容量零损失**；代价是切得粗了，这一层的一页从装 16 个 token 变成装 32 个，序列尾部不满一块的零头最多从 15 个 token 变成 31 个（后文「两把尺子与装配」一节会看到它怎么牵动调度尺与哈希尺）。**物理 pad**：Mamba 状态页和「后端按 stride 索引」的层只能垫高。Mamba 为什么走不通上一条路：它的一页装的是状态张量，大小由状态形状算出、跟 block_size 无关，把块大小翻倍，它的页一个字节都不会变；而它又要跟注意力层共用同一个块号空间，出路就只剩把页垫到最大页。垫高只动分配的字节数（`page_size_padded` 改了，状态形状一个字节没动），所以只花显存、不坏正确性，表里 4096 → 65536 那行就是它。「按 stride 索引」指这类注意力后端定位一页用「基址 + 块号 × 固定步长」，不问页内真实字节排到哪，垫高的页尾根本不会被寻址，pad 同样安全。两路的垫出字节都照付显存。**拒收**：既不整除又垫不了的直接 `NotImplementedError`，宁可拒收，不给错账。五个场景实跑：

<!-- trace: m6 -->
| 输入 | 页字节（前→后） | 出路 | 结果/代价 |
|---|---|---|---|
| 等页 {a,b} 同 65536 | 65536 → 65536 | 原样返回（同一 dict） | 零开销 |
| 小层 heads 4（32768） | 32768 → 65536 | block_size 16 → 32（×2 线性放大） | 每 token 字节不变，容量零损失 |
| Mamba 状态页 4096 | 4096 → 65536（pad） | page_size_padded 物理垫高，block_size 16 不变 | 每页浪费 61440B（93.75%） |
| 畸形页 40000（非 stride） | 40000 → 拒收 | 65536 % 40000 = 25536（不整除 → 拒收） | NotImplementedError |
| 畸形页 40000（stride 索引） | 40000 → 65536（pad） | indexes_kv_by_block_stride=True → 允许 pad | 后端按 stride 寻址吃得下 pad |

Mamba 那行 93.75% 的浪费是极端例（状态页远小于注意力页时的代价）；真实混合模型靠 Mamba 状态够大让页自然接近，pad 的浪费小得多。

### DeepSeek V4：先把这一层看明白

等页路径的三条出路都指向同一个终点：所有层落到同一个最大页。可真有一种模型，四种页宽互不凑整、又不是 stride 索引层，会怎样？按上一节的规矩答案该是 NotImplementedError 拒收，但这个模型是 DeepSeek V4（DeepSeek 的混合注意力旗舰，本章混合组化的主打实例），vLLM 没有拒收它，而是为它单开了第三条分组路径。要懂这条路径为什么长这样，得先认识这个模型：本节立画面，接着两节数账（页宽与写入），最后一节拆分组。

Gemma3、gpt-oss 的混排只是「窗口多宽、几层一换」，每个层的内部还是同一副注意力骨架；V4 往前再走一步，把压缩、滑窗、检索三件事装进同一个层。名字先对齐，免得两套文献对不上号：DeepSeek 官方叫 CSA（Compressed Sparse Attention，压缩稀疏注意力）与 HCA（Heavily Compressed Attention，重压缩注意力），vLLM 代码里叫 c4a/c128a（压缩比 4 与 128 的注意力层族，下文简称 c4/c128），两套名字指同一批层。这么折腾值不值，官方给过总账：1M token 设定下 V4-Pro 的单 token 推理 FLOPs（浮点运算次数，算力开销的计量）只要 V3.2 的 27%、KV cache 只要 10%（[DeepSeek 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)）；vLLM 侧的口径：1M 上下文每序列 KV 9.62 GiB，对 V3.2 形态的 83.9 GiB，约 8.7 倍（[vLLM 官方博客](https://vllm.ai/blog/2026-04-24-deepseek-v4)）。这两笔账是谁省出来的、代价是什么，[第 28 章](../../ch28-deepseek-v4-principles/narrative/chapter.md)逐件算清；本章只把这一层的静态画面立住，然后进账本。

一个 c4 层里，KV 沿三股流走。**滑窗流**：最近一个窗口内的 token 原样存 KV、不压缩，近处的原文永远直读，粗粒度但便宜。**压缩流**：压缩器每凑满 4 个 token，把它们的 KV 加权压成一份（c4 的压缩窗带重叠，8 个 token 以步长 4 滑出两份），远处的细粒度记忆全在这条流上。**检索流**：indexer（索引器，给压缩 KV 配的检索头）在压缩键上为每个 query 打分、挑出 top 条目，注意力真正读的就是挑出来的这些，每个 query 只看 512 条——1M token 压到 4:1 仍是 25 万条的量级，不挑读不动。主 KV 沿用 MLA（multi-head latent attention，多头潜在注意力：训练学一对下/上投影，把每 token 的 K、V 联合压进一个低秩潜在向量再缓存，DeepSeek-V2 起的省显存路数；多头的数学与 584 B 特形归[第 24 章](../../ch24-primer-attn-variants/narrative/chapter.md)展开）。c128（HCA）是同一副骨架的重压缩版：每 128 个 token 才压一份、不挑，压缩序列短到 1M token 也只剩约 8k 条，直接全看（它的 topk 上限 8192 恰好盖满）。还有两种角色不在主线上。c1 层不压缩、只有滑窗流（MTP 草稿层恒为这种；实发模型的逐层表里写 0 就表示纯滑窗层，下一节的 max(1,·) 折叠说的就是它）。mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）改的则不是注意力而是残差：残差从「加回去」扩成多条并行流、由学出来的门控调和，落在模型代码的 hc_pre / hc_post 两个钩子上（每个半层前降投、算完抬升拼回残差），算力开销不大（细账归[第 28 章](../../ch28-deepseek-v4-principles/narrative/chapter.md)），换前层信息不被深堆叠冲淡。

![DeepSeek V4 一个 c4 层的解剖：三股流与五挂缓存](../diagrams/ch14-fig-v4-layer-anatomy.png)

> *图注：L2 站 5（混合组化）的放大，把一个 CSA 层拆开看它向账本报的缓存户口。左（数据视角）：token 流分三股，滑窗流近窗原文直存、压缩主 KV 流每 r=4 个 token 压一份、indexer 检索流在压缩键上打分挑 top 条目；右（账本视角）：五挂缓存各是一条 spec，滑窗缓存与压缩主 KV 同页 37440 B（刻意共享同一张物理张量）、indexer k_cache 与 indexer 内嵌压缩器状态同页 8640 B、注意力压缩器状态 32832 B；底：户口数由 compress_ratio 唯一决定——c1 层（含 MTP）只报 1 类（主 KV 不报账，get_kv_cache_spec 返回 None）、c4 层 5 类、c128 层 3 类（去 indexer 两挂，主 KV 独占 1728 页）。三股流各挂各的账：一层之内就已是混合缓存。*

### 一层报几本账：七种形态、四种页宽

对着这张图数账。理论画面里的三股流，落到账本上是四类缓存：主 KV（压缩流的正式账，全历史）、索引器 KV（检索流自己的 KV 账）、每层一挂的滑窗缓存（每层构造一个，attention.py:L319-L325）、压缩器状态（注意力与 indexer 各带一份自己的）。四类缓存、七种形态、四种页宽，全压在这一个模型的账上。取证口径先交代（census、写入路径、第三路分组三节的数字同源）：页宽 census 与分组结果都来自 pin 真码的 host 桩跑，vllm.config 一类配置模块用桩替换，spec 构造与分组代码零改动；逐层压缩比表取官方回归测试的 5 层玩具表 [0, 0, 4, 128, 0]（源码逐层读 config 表、max(1,·) 把 0 折成 1，attention.py:L210-L213）；窗口取 8192（源码只读 config.sliding_window、实值随权重发布；页宽与切组不随窗口值变）。spec 家族只有两支：MLAAttentionSpec（MLA 型，主 KV 与索引器报它）与 SlidingWindowMLASpec（V4 新设的滑窗型，滑窗缓存与压缩器状态报它）。

谁报几条，代码一句话定一半：

```python
# vllm/models/deepseek_v4/attention.py:L655-L674 · DeepseekV4Attention.get_kv_cache_spec
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None                                                        # L659
        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,                    # L665
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,                                # L669
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),             # L673
        )
```

c1 层的主 KV 不报账：返回 None，注释明说滑窗那半另册处理，所以 c1 层唯一的缓存面就是那个滑窗缓存。c4/c128 层才报主 KV，spec 带三个 V4 专属字段 compress_ratio / alignment / model_version（kv_cache_interface.py:L389-L395 的注释明写 "DeepseekV4 only fields"）。c4 层还有两挂 indexer 侧的账：indexer.k_cache 报一份不带 model_version 的 MLAAttentionSpec（attention.py:L698-L710），indexer 自己又内嵌一个压缩器、带出第二个压缩器状态缓存（attention.py:L799-L809），七种形态里最容易被漏数的就是这一挂。逐层 census：

<!-- trace: v4-cache-spec-census -->
| 层 | compress_ratio | spec 类数 | 自报的缓存面 | 页宽贡献 |
|---|---|---|---|---|
| 层 0 | 0（c1：max(1, 0) 折成 1） | 1 | swa | 37440 |
| 层 1 | 0（c1） | 1 | swa | 37440 |
| 层 2 | 4（c4） | 5 | swa, main, indexer, indexer_state, attn_state | 37440 / 8640 / 32832 |
| 层 3 | 128（c128） | 3 | swa, main, attn_state | 37440 / 1728 / 32832 |
| 层 4 | 0（c1） | 1 | swa | 37440 |
| 合计 | — | 11 条 spec | 4 种页宽 | 37440 / 8640 / 32832 / 1728 |

表格读法：census 是 compress_ratio 表的确定函数——c1 层恰 1 类、c4 层恰 5 类、c128 层恰 3 类（合法压缩比仅 1/4/128，越界 raise），全模型 spec 总数 = 1·n(c1) + 5·n(c4) + 3·n(c128)；真实模型把玩具表换成实际逐层表即可，公式不变（MTP 层恒折成 1）。列里的 swa / main / indexer / indexer_state / attn_state 是 census 记的形态名，对应正文的滑窗缓存、主 KV、索引器 KV、indexer 压缩器状态、注意力压缩器状态。页宽的底数在 spec 类里：

```python
# vllm/v1/kv_cache_interface.py:L403-L413 · MLAAttentionSpec
    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio                           # L405

    @property
    def real_page_size_bytes(self) -> int:
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584                            # L413
```

页 = storage_block_size × 每槽字节，再向上取整到 alignment（576，fp8_ds_mla 布局；`_apply_alignment_padding`，kv_cache_interface.py:L353-L359）。每槽 584 B = 448 B NoPE（不带位置编码的潜在分量）+ 128 B RoPE（带旋转位置编码的分量）+ 8 B fp8 scale（块量化的缩放字节）；indexer 那挂每槽 132 B（128 B fp8 + 4 B fp32 scale）。四笔账算下来：c4 主 KV 64 槽 × 584 B = 37376 → 37440，滑窗缓存同式同页；c128 主 KV 2 槽 × 584 B = 1168 → 1728；indexer 64 槽 × 132 B = 8448 → 8640；压缩器状态每槽 8192 B（c4，4 槽）或 4096 B（c128，8 槽）= 32768 → 32832。storage_block_size 这条除法下一节还要当主角，先记住它是「管理块折物理行」的换算尺。

七种形态只占四种页宽，靠的是三对刻意同页，每对都有源码注释作证。最扎眼的一对：滑窗缓存与 c4 主 KV（源码注释叫 C4A，后缀 A 即主 KV，这组后缀名下文沿用）同页 37440。同字节、异窗口，这不是巧合是设计——两类块共享同一张物理张量，所以必须同页，C4A 的块形 [256//4, 584] 反过来把滑窗块的 block_size 定在 64 token：

```python
# vllm/v1/attention/backends/mla/sparse_swa.py:L77-L82 · DeepseekV4SWACache.__init__
        # Block size is constrained by tensor sharing between SWA and C4A KV blocks.
        # Since both block types share the same physical tensor, they must use the
        # same page size. The C4A KV block shape [256//4, head_dim] = [64, head_dim]
        # determines the SWA block size of 64 tokens per block.
        # TODO(yifan): make SWA block size automatically determined and configurable.
        self.block_size = 64
```

TODO 注释自认这层耦合是待解的硬编码。第二对在压缩器状态上。状态缓存自己也是一本滑窗账，窗宽与块大小都由压缩比硬推：

```python
# vllm/models/deepseek_v4/compressor.py:L173-L189 · CompressorStateCache.__init__
        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]                                      # L174
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio                            # L176
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4                                                # L185
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")
```

c4 到窗 8、块 4，c128 到窗 128、块 8；注释给的是同一约束的另一面：状态块与 KV 块共物理张量、块形被页宽反推（c4 块形 [4, 8192]、c128 块形 [8, 4096]）。这个状态存的是什么：压缩器要凑满一个压缩比才产出一份正式压缩 KV，凑的过程中原料得暂存，状态宽度是 2 × coff × head_dim（coff 就是嵌入片段里的重叠系数，c4 取 2、c128 取 1）——社区深读技报的拆解把它读成两半，攒下的 KV 原料与 softmax 分数各占一半（[DeepSeek V4 KVCache Management](https://mengqingcao.github.io/posts/deepseek-v4-kvcache-management/)）。两族注意力压缩器状态（c4 与 c128）就这样同页 32832。第三对最隐蔽：indexer 内嵌的第二个压缩器状态（head_dim 128，页宽算到 8640）恰好与 indexer KV 同页 8640，它不独立成桶也不独立成页宽，全模型四种页宽里跟着 indexer 走。对账本的意义：页宽的多样性不是待统一的麻烦，是模型侧显存复用的刻意设计，等页路径的 unify 三条出路一条都接不住它；这张账单两节之后（「第三条分组路径」）兑现，下一节先把压缩块的写入账记清。

![DeepSeek V4 全模型 census：七种形态收进四种页宽](../diagrams/ch14-fig-v4-census.png)

> *图注：站 5 组化的 census 侧放大（配上一张的层解剖：那张拆一层，这张看全模型的页宽账）。左列七种形态经同页配对收进右列四档页宽：37440 由 c4 主 KV 与滑窗缓存刻意同页（同一张物理张量，C4A 块形 [256//4, head_dim] 反定滑窗块 64 token）、8640 由 indexer 与其内嵌压缩器状态同页、32832 由 c4/c128 两族注意力压缩器状态同页、1728 由 c128 主 KV 独占；右条给官方算例形状（11 c4 + 10 c128、共 85 条 spec）的页宽计数：37440×32、8640×22、32832×21、1728×10。页宽多样性是共享物理张量的刻意设计，等页 unify 的三条出路（调大块、pad、拒收）一条都接不住它。*

### 压缩块的记账口径：管理面 256、物理面 64 行

页宽数完，接着问写入：一个 token 算完，它的压缩 KV 落到物理货架哪一格？V4 用两套坐标系记这本账。**管理面**按 256-token 的块发块号，块表、块分配、前缀哈希全按这个口径，调度器眼里一块就是 256 个 token；**物理面**每块只有 256//r 行（r＝压缩比，kernel 里的 COMPRESS_RATIO），c4 一块 64 行、c128 一块 2 行——上一节 storage_block_size 的那条除法就是两套坐标系的换算尺。块大小 256 也不是写死的：用户没指定时取后端偏好，V4 后端族声明 256（`get_preferred_block_size`，sparse_swa.py:L119-L121，协商链归执行篇）。

两套坐标系靠槽位映射缝合，kernel 主体十来行：

```python
# vllm/v1/attention/backends/mla/compressor_utils.py:L24-L50 · _compressed_slot_mapping_kernel
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len                                               # L31

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)                      # L37
        is_valid = (pos + 1) % COMPRESS_RATIO == 0                                 # L38
        pos_after_compress = pos // COMPRESS_RATIO                                 # L39

        block_ids = pos_after_compress // block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid,
        )
        slot_ids = block_numbers * block_size + pos_after_compress % block_size    # L46

        # NOTE
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)                            # L49
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)
```

三行承重。`is_valid = (pos+1) % r == 0`（L38）：只有凑满 r 个 token 的最后一位才产生要写的压缩条目，每 r 个位置恰一位有效；`pos_after_compress = pos // r`（L39）：压缩后的行号，随 pos 严格单调；`slot = 块号 × block_size + 压缩位 % block_size`（L46）：物理行坐标。注意 kernel 形参 block_size 传的是 storage_block_size（调用方 sparse_mla.py:L208-L216 与 indexer.py:L797-L805 都这么传），kernel 眼里的「块」是物理块 64，不是管理块 256；不满 r 的位置整体置 PAD_ID=-1（L49），外层封装函数 get_compressed_slot_mapping 还会先把输出 fill_(-1)（compressor_utils.py:L62-L68），未覆写位天然是 -1。

写入侧对 -1 的保证在 V4 自己的融合 kernel：全部四个 triton（vLLM 写 GPU kernel 用的 Python 框架，kernel 代码长得就像普通 Python）入口都有同款双保险，逐字摘一段：

```python
# vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:L157-L165 · 融合压缩写入 kernel 入口
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:                                                            # L160
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:                                   # L164
        return
```

两道独立早退：槽位映射的 -1 一道、位置判定一道。两道拦的是同一件事的两种表述，「这一位不归我写」是账面表达、「这一位物理上没有格」是语义事实，任一道失守另一道仍挡住脏写。官方回归测试（tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py:L37-L118，上一节 census 的玩具层表就出自它）把这个映射连同跨块界场景钉成了期望张量：单请求已算 240 个 token、本拍 40 个 query（序列长 280）、块表 [5, 7]。取证口径照实说：该测试标了 CUDA-only，本机无 GPU 未实跑，下表是期望张量的逐行转录，并已按 kernel 公式在 host 逐位手推复核（两套算法对上了每一个数）：

<!-- trace: v4-cache-compress-write-path -->
| token 绝对位 pos | (pos+1)%4==0？ | 压缩位 pos//4 | 压缩位//64 → 查块表 | slot = 块号×64 + 压缩位%64 |
|---|---|---|---|---|
| 240、241、242 | 241/242/243 % 4 = 1/2/3 → 否 | — | — | 三个 -1（不凑整不写） |
| 243 | 244 % 4 = 0 → 是 | 60 | 60//64 = 0 → 块表[0] = 5 | 5×64 + 60 = 380 |
| 247 / 251 / 255 | 是（每 4 个恰一位） | 61 / 62 / 63 | 均 0 → 5 | 381 / 382 / 383（块 5 的最后三格） |
| 259 | 是 | 64 | 64//64 = 1 → 块表[1] = 7 | 7×64 + 0 = 448（跨存储块界换块） |
| 263、267、271、275、279 | 是 | 65 / 66 / 67 / 68 / 69 | 均 1 → 7 | 449 / 450 / 451 / 452 / 453 |
| 40 位合计 | 10 是 + 30 否 | 60..69 连续 10 位 | 前 4 位块 5、后 6 位块 7 | 10 个有效槽 380-383、448-453；30 个 -1（写入侧双保险跳过） |

![压缩块的记账口径：管理面 256、物理面 64 行](../diagrams/ch14-fig-v4-compress-accounting.png)

> *图注：站 5 组化的写入路径侧放大（接上一图的 census：页宽是静态账，这张看 token 怎么落进物理行）。上（管理面）：块表第 k 项管 256 token，本拍 40 个 token（pos 240-279）跨在管理块 0/1 上，块表 [5, 7] 映到物理块 5/7；中（槽位映射）：40 个位置逐一过筛，is_valid=(pos+1)%r==0 的恰 10 位（压缩位 60..69 连续），其余 30 位 -1；下（物理面）：压缩货架每块 64 行，压缩位 60..69 恰跨存储块界 64，前 4 位落块 5（slot 380-383）、后 6 位落块 7（slot 448-453）；右栏：写入 kernel 的双保险（slot<0 与非凑整位各 return 一次）与 storage_block_size 换算（c4 每块 64 行、c128 每块 2 行），块大小 256 来自后端偏好而非写死。压缩发生在 block_size 维：管理面块数不变、物理行数被 r 除掉。*

表里最有信息量的是跨块界那一行：压缩位 60..69 连续 10 位，恰跨 64 的倍数这条块界，前 4 位落物理块 5 的行 60-63（slot 380-383），后 6 位落物理块 7 的行 0-5（slot 448-453）——管理面上这 40 个 token 只跨了块表 [5, 7] 两项，物理面上落点自动换块。这个回归测试防的 bug 正是沿用未压缩的 slot_mapping：40 个位置全当有效、压缩位错 4 倍。管理面与物理面的分工一句话收束：**压缩发生在 block_size 维**，块数（管理面）不变，每块的行数（物理面）被 compress_ratio 除掉。

### 第三条分组路径：装包、近似 GCD 与重叠布局

分组的入口是一串四路 if/elif，前文「分桶等量」一节交代过 `get_kv_cache_groups` 依次问的四个问题；现在逐路走读（每条判什么、接住谁、产出什么），源码在前：

```python
# vllm/v1/core/kv_cache_utils.py:L1802-L1819 · get_kv_cache_groups（四条分支）
    if is_kv_cache_spec_uniform(kv_cache_spec):
        # KV cache of all layers are the same, which is true for
        # most models. Allocate the same amount of memory for
        # each layer.
        return _get_kv_cache_groups_uniform_spec(kv_cache_spec)
    elif uniform_spec := UniformTypeKVCacheSpecs.from_specs(kv_cache_spec):
        # All layers need the same number of token slots (e.g., all layers are
        # full attention, or all layers are sliding window attention with the
        # same window size). Put all layers into one group.
        return _get_kv_cache_groups_uniform_type(uniform_spec)
    elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):
        # DeepseekV4 case: All layers need the same number of token slots,
        # yet some layers are full attention while others are sliding window
        # attention in different sizes. Need to group layers into multiple
        # UniformTypeKVCacheSpecs.
        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
        _annotate_eagle_groups_deepseek_v4(vllm_config, kv_cache_spec, kv_cache_groups)
        return kv_cache_groups
```

逐路走读：

- **第一路·全同 spec**（L1802-L1806）：全部层交上来一模一样，`_get_kv_cache_groups_uniform_spec` 一组了事。多数单型模型的日常，本章开头到「一份账喂两侧」的 uniform 例子走的都是它。
- **第二路·同型异宽**（L1807-L1811）：spec 不全同但同型（每块要的 token 槽数相同），`UniformTypeKVCacheSpecs.from_specs`（就是前史一节立过的那个打包容器，kv_cache_interface.py:L837-L849）把全部层并成 **一个元组组**：组内每层各占一页、组的页＝成员页之和（kv_cache_interface.py:L848）。走第二路的是 DeepSeek V3.2：主 KV（656 B/token 的 MLA 特形，kv_cache_interface.py:L414-L416）与索引器 KV（132 B/token）同为 compress_ratio 1 的 MLA 型、只是每 token 字节不同，并成一组照样一张表，「spec 异型就得多组」的直觉由它证伪。
- **第三路·V4 专属元组路**（L1812-L1819）：判据是「SlidingWindowMLASpec 在场＋页宽多于一种」，目前只有 DeepSeek V4 走这条路。装包三步：`group_and_unify_kv_cache_specs`（L1592-L1632）分桶（MLA 全家一桶、滑窗族按 (block_size, window) 分桶）→ `_get_kv_cache_groups_uniform_groups`（L1670-L1754）按层元组打包 → `_approximate_gcd`（L1635-L1667）对齐各桶元组数，下文逐步拆。
- **第四路·通用等页**（L1821-L1841）：前三路都不中才落到这里，先抽出 hidden-state 层（HiddenStateCacheSpec，缓存的不是 K/V 而是隐状态的一族层）再 unify 页宽、分桶等量，就是「分桶等量」「页统一」两小节拆的机制；Gemma3、gpt-oss 这类页宽全等的混排走它。

[第 15 章](../../ch15-prefix-caching/narrative/chapter.md)讲跨类型命中时会回来对这张全貌。V4 在梯子上逐级下落：第一路显然不中（七种形态、两族家族）；第二路也不收，MLA 全家（压缩比大于 1，每块只要少量槽）与滑窗族（每 token 一槽）本就不同型，滑窗族还占着四种形态，「窗口大小不同的滑窗不是同一型」是该类 docstring 的原文（kv_cache_interface.py:L840-L842）。（消歧一句：第三路注释首句 "All layers need the same number of token slots" 不能按字面读成「各层每块槽数相等」：c4 主 KV 与索引器 64 槽、c128 主 KV 2 槽、各族压缩器状态 4/8 槽并不等；分路的真正依据在后半句，同为 token 槽型缓存的层里混了全历史与三种窗口宽度，装不进第二路的「同型」单组。）落到第三路的判据，两个 guard 都是先验检查：

```python
# vllm/v1/core/kv_cache_utils.py:L1599-L1607 · group_and_unify_kv_cache_specs
    if not any(
        isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()
    ):
        return None

    # SlidingWindowMLASpec models with uniform page sizes don't need tuple packing.
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return None
```

没有 SlidingWindowMLASpec 就 return None 落回等页路径；页宽只有一种也 return None（uniform 页宽的滑窗模型不必打包）。V4 必然双条件命中：七种形态里四种报 SlidingWindowMLASpec（滑窗缓存与三挂压缩器状态）、页宽实测四种（函数 docstring 自报家门："Currently, this is only used for DeepseekV4"）。反过来问：若硬把 V4 塞进等页路径会怎样？三种小页 32832、8640、1728 没有一个整除最大页 37440，unify 的出路只剩 pad，而 pad 只留给 MambaSpec 与按 stride 索引的层；即便 unify 抛出 NotImplementedError，分配兜底 `_try_get_full_allocation_fallback_groups` 见到 SlidingWindowMLASpec 直接 return None（kv_cache_utils.py:L1540-L1544）。第三路不是优化，是唯一可行解。

第三路怎么分组？`_get_kv_cache_groups_uniform_groups`（kv_cache_utils.py:L1670-L1754）按注释自带的示例层数走，注释值得逐字读：

```python
# vllm/v1/core/kv_cache_utils.py:L1691-L1697 · _get_kv_cache_groups_uniform_groups
    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
```

元组（layer tuple）＝一组不同页宽的层捆成的最小分配单元，整组一起分、不再拆开。装包先分桶（`group_and_unify_kv_cache_specs`，L1592-L1632）：MLA 全家一桶，C4I、C4A、C128A（依次是 c4 层的索引器 KV、c4 层的主 KV、c128 层的主 KV，后缀名沿用源码注释的叫法）共 32 份 spec 全体并入，按 11 个 [C4I, C4A, C128] 元组打包（元组第三格的 C128 是注释对 C128A 的简写；C128A 只有 10 层、天然短一截），组的页＝32 份 spec 页之和 524160 B（= 11×37440 + 11×8640 + 10×1728）；一个块号在这个组里，就是组内每层各占一页。滑窗族按 (block_size, window) 键切三桶（L1613-L1618）：21 个滑窗缓存 (64, 8192) 一桶；(4, 8) 桶 22 份——注意力压缩器状态 11 份 + indexer 压缩器状态 11 份，两种页宽（32832 与 8640）同桶，桶内每页宽层数恰等（11 = 11）；c128 状态 (8, 128) 10 份一桶。四桶元组数 [11, 21, 11, 10] 不齐，对齐靠近似 GCD（GCD＝最大公约数；这里求的是「近似」公约数，为什么近似马上看到）：

```python
# vllm/v1/core/kv_cache_utils.py:L1635-L1648 · _approximate_gcd
def _approximate_gcd(values: Sequence[int], *, lower_bound: int | None = None) -> int:
    """Pick a chunk size that minimizes total upward padding.

    Each x is rounded up to a multiple of d:

      x -> ceil(x / d) * d

    Total padding is:

      pad(d) = sum_i (ceil(x_i / d) * d - x_i)

    We brute-force d in [lower_bound, max(values)] (fine for small lists / small
    maxima) and return the d with minimum padding. Ties prefer larger d.
    """
```

真公约数不存在（21 本就不是 11 的倍数），所以叫「近似」：成本函数把每个桶的元组数向上取整到候选 d 的倍数，pad(d) 数的就是这些取整多出来的层数，暴力枚举 d、取总 padding 最小者。实证：d=11 的成本口径最省，只多计 2（滑窗桶 21 计到 22、c128 状态桶 10 计到 11）；对照 d=12 要多计 7、d=21 要多计 31。要紧的分野：这个「对齐」只活在择 d 的计数里，实际切组不添任何实体层——滑窗桶按 `layer_tuples[i::num_tuple_groups]` 交错切成 11+10 两组（分桶那节 `layers[i::num_groups]` 的同一手法），(8, 128) 桶一组 10 层，短的那截与 MLA 桶里 C128A 天然短一截同理；对照等页路径的 padding：那边垫的是真层、白占显存，这边 11+10=21 一个不多。(4, 8) 桶 22 份 = 11 元组 × 每元组 2 层，整桶 1 组。终 5 组：MLA 元组组、滑窗两组、(4,8) 状态组、(8,128) 状态组。组间断言是打包正确性的哨兵：MLA 桶必须纯 MLA（L1681-L1685）、滑窗桶必须纯 SlidingWindowMLASpec（L1711-L1715）、桶内每个页宽的层数必须相同（L1726-L1729）。全模型实跑对账：

<!-- trace: v4-cache-uniform-groups-verify -->
| 装包桶（group_and_unify） | 层数 / 成员 | 页宽构成 | 元组数 | 近似 GCD 对齐后 | 终组 |
|---|---|---|---|---|---|
| 入场 census | 85 条 spec | 37440×32、8640×22、32832×21、1728×10 | — | — | 页宽 4 种 |
| MLA 桶（第一组限定纯 MLA） | 32 | 37440×11、8640×11、1728×10 | 11 | 11（已齐） | G0：块 256、组页 524160 |
| SWA 桶 (64, 8192) | 21 | 37440×21 | 21 | 对齐 22（实切 11+10） | G1 + G2 交错切 11 + 10 |
| 状态桶 (4, 8) | 22 | 32832×11、8640×11 | 11 | 11（已齐） | G3：一组 22 层 = 11 元组 × 每元组 2 层 |
| 状态桶 (8, 128) | 10 | 32832×10 | 10 | 对齐 11（实为一组 10） | G4 |
| _approximate_gcd | values [11, 21, 11, 10]、lower_bound 11 | d=11 → pad=2 全场最小（d=12 → 7、d=21 → 31） | chosen 11 | rounded_up [11, 22, 11, 11] | 终 5 组 |
| 块尺寸两把尺 | 组块 [256, 64, 64, 4, 8] | — | — | — | scheduler LCM 256 / hash GCD 4（prefix_match_unit 未设） |
| packed 布局 | block_stride 524160、70 个偏移 | offset 0 被 5 个组共享 | — | — | 页统一若硬走：37440%8640=2880、%1728=1152、%32832=4608 全除不尽 |

硬约束也从「每组每块字节相等」换成了「块号在重叠打包布局里各有落点」：五组共用一条块号轴，物理块步长 block_stride = 524160 B，取的是最宽一组的组页和（MLA 组的密排宽度，恰为全场最大组页和：装包前的滑窗桶页和更大，但已被切成 11+10 两组、终组页和反而小）；每组在自己偏移上密排、组间布局允许重叠，因为一个块 id 同一时刻只归一组（`_get_packed_kv_cache_layout`，L1283-L1305；70 个不同字节偏移里 offset 0 被 5 个组共享是常态；张量侧落点下一节「张量怎么共享」的第三型布局再接）。调度器侧还有一笔实数：五组的组块尺寸 [256, 64, 64, 4, 8] 送进「两把尺子与装配」那两把尺（调度尺取 LCM＝最小公倍数、哈希尺取 GCD 或用户以 prefix_match_unit 覆盖的更细粒度，V4 没设覆盖），得 scheduler 尺 256 / hash 尺 4——V4 的前缀哈希每 4 token 一枚、命中对齐 256，[第 15 章](../../ch15-prefix-caching/narrative/chapter.md)的粒度视图会再用到这两个数。

边界三句话收束：584 B 特形与 compress_ratio 的数学归[第 24 章](../../ch24-primer-attn-variants/narrative/chapter.md)与[第 28 章](../../ch28-deepseek-v4-principles/narrative/chapter.md)展开，本章只消费结论；模型侧三件套（索引器、压缩器、滑窗缓存）与打包张量布局，归[第 29 章](../../ch29-deepseek-v4-assembly/narrative/chapter.md)的拼装实战；开 MTP 时，最后一层的滑窗缓存所在组会被标注 `is_eagle_group`（`_annotate_eagle_groups_deepseek_v4`，kv_cache_utils.py:L1757-L1778），草稿层怎么混编进宿主组归投机解码线。账本侧本章已给全：四类缓存、七种形态、四种页宽、五组归属，第三路讲完了。

顺带把视野拉远一步：同一道「混合层的缓存怎么管」，隔壁引擎给出了镜像的解。SGLang 走分离双池——Mamba/线性层的状态按请求级另立一池、KV 按 token 级一池，配树形的 MambaRadixCache 管状态命中，命中到的状态必须拷贝一份、不能共享（[SGLang 官方博客](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)，2025-12）。LMSYS（SGLang 背后的研究组织）的 Unified Radix Cache 把「每类层一条复用规则」做成树组件，DSV4 在它那里被建模成全注意力（FULL）与滑窗（SWA）两个组件、压缩池做成 sidecar（伴生小池）（[LMSYS 博客](https://lmsys.org/blog/2026-08-11-unified-radix-cache/)，2026-08）。vLLM 的答案是本章这条：单池调和，所有形态挤进一个块池、按 spec 分账。两条路线互为镜像，都被同一批模型压力测试过；孰优孰劣要看负载，本书的立场只到「同一问题存在多种自然的分解」为止。

### 张量怎么共享：三种布局

组切好了、页统一了（或像 DeepSeek V4 那样换了条路分组），剩下的物理问题是：块池张量怎么摆？`get_kv_cache_config_from_groups`（kv_cache_utils.py:L1361-L1443）有三种布局，通用分支的方案注释里那张 ASCII 图值得逐字读：

```python
# vllm/v1/core/kv_cache_utils.py:L1411-L1437 · get_kv_cache_config_from_groups
    else:
        # General case:
        # We will have group_size memory pools, each is shared by one layer from
        # each group. As layers of different groups have different block table,
        # they will use different parts of the shared Tensor.
        # The memory layout for 3 groups (full.0, full.1), (sw.0, sw.2),
        # (sw.1, padding) will be: (group_size = 2)
        # full.0, sw.0, sw.1: share a Tensor with size=available_memory//2
        # full.1, sw.2: share another Tensor with size=available_memory//2
        group_size = max(len(group.layer_names) for group in kv_cache_groups)

        page_size = get_uniform_page_size(
            [group.kv_cache_spec for group in kv_cache_groups]
        )
        assert group_size > 0, "group_size must be greater than 0"
        num_blocks = get_num_blocks(
            vllm_config, group_size, available_memory, page_size
        )
        kv_cache_tensors = []
        for i in range(group_size):
            shared_by = []
            for j in range(len(kv_cache_groups)):
                if i < len(kv_cache_groups[j].layer_names):
                    shared_by.append(kv_cache_groups[j].layer_names[i])
            kv_cache_tensors.append(
                KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
            )
```

方案一句话：**group_size 个内存池，每个池由每组各出一层共用**。上面例子里 full.0 / sw.0 / sw.1 共享一张张量、full.1 / sw.2 共享另一张（sw.1 组只有一层，第二池就没有它）。为什么敢共享？注释原话给了答案："As layers of different groups have different block table, they will use different parts of the shared Tensor"——**一个 block_id 同一时刻只归一个组用**（块一经分给某组的请求，就只出现在那个组的私有块表里），共享的是房间面积，不是钥匙。`num_blocks = available // page // group_size` 在这里再除一次组内层数。单组异宽布局（同型但页宽不同的层挤在一组，下文表第二行的「单组异宽」就是它）走「逐层各一张张量」按页分账；还有一族 packed 重叠布局让多层的缓存物理上别名重叠，上一小节 DeepSeek V4 五组共轴的落点正是它（一个块 id 同一时刻只归一组、组间布局可重叠，张量的完整刀法归拆读 DeepSeek-V4 的专章）。三组两池的实跑：

<!-- trace: m7 -->
| 布局 | group_size | 张量数 | 每张量共享者 | 尺寸 |
|---|---|---|---|---|
| 通用（3 组） | 2 | 2 | 张量一 [full.0, sw.0, sw.1]；张量二 [full.1, sw.2] | 各 655360B（=65536×10 页） |
| 单组异宽（l0 heads 8 / l1 heads 4） | 1（2 层） | 2 | 逐层各一：[l0] / [l1] | 327680B / 163840B（按各自页分账，5 块） |

![张量共享：每池由每组各出一层](../diagrams/ch14-fig-tensor-sharing.png)

> *图注：L0 的「调度 · 显存账本」列与 GPU 列接缝处放大（对应 L2 章图中排拍片④的 worker 半边）。左：三组 × 两池的指派矩阵（组 3 只有 sw.1 一层，第二池没有它的位）落到两张 10 页的物理张量（各 655360 B），三组各持一张私有块表；右：单组异宽对照，逐层各一张张量按页宽分账。共享不冲突的原因写在源码注释里：一个 block_id 同一时刻只归一个组使用，共享的是物理面积，钥匙（块表条目）各自私有。*

### 两把尺子与装配：LCM、GCD、每组一个管家

组化之后，调度器面对的是「各组块大小可能不同」的世界，需要两把对齐的尺子：

```python
# vllm/v1/core/kv_cache_utils.py:L626-L688
def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Resolve (scheduler_block_size, hash_block_size).

    - ``scheduler_block_size`` is the token-alignment invariant used by the
      scheduler (e.g. for ``num_computed_tokens`` rounding). Single group:
      ``cache_config.block_size * dcp``. Multiple groups: LCM of every
      group's effective block size. Attention groups are scaled by DCP;
      Mamba groups keep their full per-rank state and are not scaled.
    - ``hash_block_size`` is the granularity at which ``Request.block_hashes``
      is computed. Single group: equals scheduler block size. Multiple groups:
      ``cache_config.prefix_match_unit`` override if set, else the GCD of
      group block sizes; every group's block size must be divisible by it.
    # … 省略：docstring 尾三行（哈希不活跃或 mamba 非 align 时回退 scheduler 粒度）……
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    group_block_sizes = [
        g.kv_cache_spec.block_size * dcp
        if isinstance(g.kv_cache_spec, AttentionSpec)
        else g.kv_cache_spec.block_size
        for g in groups
    ]
    scheduler_block_size = math.lcm(*group_block_sizes)                  # L659

    # Block hashes are only consumed by prefix caching and KV connectors
    # (P/D, offloading); when neither is active, keep hash_block_size equal
    # to the scheduler block size.
    connector_enabled = vllm_config.kv_transfer_config is not None
    if not (cache_config.enable_prefix_caching or connector_enabled):
        return scheduler_block_size, scheduler_block_size

    # … 省略：mamba 组块大小 ≠ cache 块大小（注释所谓非 align）时
    #       回退 scheduler 粒度的判定九行 ……
    requested = cache_config.prefix_match_unit
    hash_block_size = (
        requested if requested is not None else math.gcd(*group_block_sizes)
    )                                                                    # L681
    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(
            f"Invalid prefix_match_unit={hash_block_size}; all KV cache group "
            f"block sizes must be divisible by prefix_match_unit. "
            f"Got group block sizes={group_block_sizes}."
        )
    return scheduler_block_size, hash_block_size
```

**调度尺取 LCM（最小公倍数）**：`num_computed_tokens`（请求已算 token 数，[第 10 章](../../ch10-continuous-batching-chunked-prefill/narrative/chapter.md)立的账）按这把尺取整，它必须是所有组块边界的公共倍数，否则取整后总有一组停在半块（docstring 里乘的 dcp 是解码上下文并行的缩放因子，单卡恒 1，多卡叙事不展开）。**哈希尺取 GCD（最大公约数）** 或 `prefix_match_unit` 覆盖：前缀哈希按这把尺切（每个组的块大小都必须整除它，否则 ValueError）。GCD 是「每个组都认的刻度」，取它保前缀复用粒度（哈希的消费者就是摘录注释点名的两类：前缀缓存，与 KVConnector。KVConnector 是跨机搬运 KV cache 的接入组件，机制本体在 Part IV 末章 KVConnector 章正面拆）。六场景实跑：

<!-- trace: m8 -->
| 场景 | 组块大小 | scheduler / hash | 理由 |
|---|---|---|---|
| 单组 | 16 | 16 / 16 | 两把尺子合一（×dcp 恒 1） |
| 双组 + 前缀缓存 | 16, 32 | 32 / 16 | scheduler=lcm=32；hash=gcd=16 |
| prefix_match_unit 覆盖 | 16, 32 | 32 / 8 | 配置覆盖 GCD → 更细命中粒度 |
| unit 不整除 | 16, 32 | raise | 16 % 5 = 1、32 % 5 = 2 → ValueError |
| 无缓存无 connector | 16, 32 | 32 / 32 | 哈希没人消费 → hash 退回 scheduler 粒度 |
| mamba 非 align | 16, 64 | 64 / 64 | mamba 状态格 64 ≠ cache 块格 16，GCD 哈希边界取不出 mamba 状态 → 回退 |

末行「mamba 非 align」值得两句展开。Mamba 组的「块」计的不是 KV 页，而是**多少 token 存一份状态检查点**，`mamba_block_size` 是与注意力块大小无关的独立配置：前缀缓存开着时默认对齐到 cache 块大小（配置注释原话，对齐是「前缀缓存的基本粒度」），用户也可以另设。另设之后状态检查点落进自己的 64-token 格、与 16-token 的注意力块格分道：GCD=16 的哈希边界只有四分之一（64 的倍数）落得进任何状态检查点，其余边界上的前缀命中根本取不出 mamba 状态，细粒度哈希伺候不了它，于是回退 scheduler 粒度 64（两种格在那里同时对齐：注意力 4 块、mamba 1 块）。源码注释把这条禁令写作 "break divisibility"，破的不是「16 整除 64」这条算术（它成立），而是「每个哈希边界都能被所有组兑现」这条对齐。

`prefix_match_unit=8` 那行的代价与收益（哈希条目更多、换更细的前缀命中）是[第 15 章](../../ch15-prefix-caching/narrative/chapter.md)的前置，这里先把两把尺立住。

装配收尾在启动序里三步走（vllm/v1/engine/core.py:L156-L168）：`resolve_kv_cache_block_sizes` → 构造 `Scheduler`（内部建 `KVCacheManager`，`watermark=`（水位参数，下一节的主角）在这里注入）→ coordinator 装配。装配形态一句话：**一个 BlockPool，每组一个类型专属管家**：

```python
# vllm/v1/core/kv_cache_coordinator.py:L90-L120 · KVCacheCoordinator.__init__
        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            enable_caching=enable_caching,
            hash_block_size=hash_block_size,
            # … 省略：事件与指标两行 ……
        )

        # … 省略：EAGLE（投机解码草稿）组标注七行 ……
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.block_pool,
                # … 省略：缓存/组号/对齐粒度等参数五行 ……
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )
```

池只有[第 13 章](../../ch13-paged-kv/narrative/chapter.md)拆过的那一个（等大块、自由队列、引用计数），管家的类型按组内 spec 查注册表选：全注意力组用 `FullAttentionManager`、滑窗组用 `SlidingWindowManager`。而 `get_manager_for_kv_cache_spec`（vllm/v1/core/single_type_kv_cache_manager.py:L1861-L1878）构造滑窗/分块管家时还顺手干了一件本章后半场最重要的事：把准入上限注入进去，下节正面拆。

## 门多紧：超收、死锁与抖动换来的三道预算（站 9-10）

账定了，镜头切到运行期。每个请求进场要过 `allocate_slots` 的门，门的所有预算参数都来自启动账本，**运行期不再重算**。现在走到 L0 图「调度 · 显存账本」列内 Scheduler 与 KVCacheManager 的接缝。这道门的设计史是 vLLM 的一堂公开课：两个月里连踩了两个方向相反的坑。

### 两幕史：门太松会抖，门太紧会死

**第一幕：门太松。** 2026 年 3 月之前的默认行为，PR [#37307](https://github.com/vllm-project/vllm/pull/37307) 的描述原话是 "The default behaviour of vLLM is to schedule requests if the first chunk of a request fits"：chunked prefill（切块预填充）下，准入只看**第一个 chunk** 装不装得下。一条比池还长的请求照样进门：prefill 推进到池满被抢占、释放重排、再调度又只查第一 chunk、又放行，无限循环。PR 给的重现例：4 条 100k ISL（input sequence length，输入序列长度）请求已占 KV 池约 90%，第 5 条仍被放进来，形成 "a continuous prefill overhead that starves decode requests"，吞吐从约 100 tok/s/GPU 跌到 1.5 tok/s/GPU。修复就是加**整序列准入门**：准入前按完整输入长度算块数、不够直接拒。基准战报（Qwen3-235B-FP8，40k 入 / 1k 出）：吞吐 271 → 387 tok/s、TTFT（首 token 延迟）159s → 106s、TPOT（每输出 token 耗时）159ms → 103ms。评审里 mgoin 的意见干脆利落："I actually think we shouldn't have a config at all, this should just be the scheduler's behavior"，于是 `scheduler_reserve_full_isl` 默认 True。

**第二幕：门太紧。** 门立起来一个月后，issue [#39734](https://github.com/vllm-project/vllm/issues/39734)（2026-04-13）：Gemma-4-31B-it 混合模型、2×H100 双卡张量并行（TP，tensor parallel，把一层切到多卡算），KV 池只装得下约 76,640 token，而 `max_model_len = 262,144`，一条约 100k 的 prompt 永远卡在 WAITING，日志反复打印 "Waiting: 1 reqs, GPU KV cache usage: 0.0%"，零吞吐。根因（贡献者 he-yufeng 的分析原话）："the deadlock happens because can_fit_full_sequence() checks against currently free blocks"：**运行期准入门按全长计块，而启动期池大小器按回收感知计**（滑窗层的窗外块会回收，稳态占用远小于全长）。两个公式不一致：启动说「池够开这个 max_model_len」，运行说「全长装不下」，这条请求永远过不了门，又永远排在队头，把后面所有人堵死（head-of-line blocking，队头阻塞）。修复 PR [#40946](https://github.com/vllm-project/vllm/pull/40946)（2026-04-27 合入）的思路不是加运行时校验，而是**单源化**：给滑窗/分块 spec 加一个共享方法，让 "startup pool sizing and runtime admission use the same recycling-aware bound"，启动定池和运行放行用**同一个公式**算上限。同一个机制后来镜像进了 vllm-ascend（vLLM 的华为昇腾 NPU 移植版，#9548）。

两幕合起来的教训就一句话：**门与账本若用两套公式，向松漂是抢占循环或 mid-prefill OOM（装到一半炸），向紧漂是队头阻塞死锁**。pin 源码里那条注释 "Drift between the two would re-introduce the deadlock from issue #39734 or, worse, mid-prefill OOM"，两个方向都在这句话里。顺带一个对照系帮定位：#37307 的描述里点名 TRT-LLM（NVIDIA 的 TensorRT-LLM 推理引擎）更保守，"unlike TRT-LLM's more conservative full-ISL + max-new-tokens reservation"（连最大输出长度都一并预留）；vLLM 取中：输入全长预留，输出长度未知不预留，增长靠水位与抢占兜底。这正是本节第三道门的由来。

### 第一道：整序列准入门

门的参数与调用点：

```python
# vllm/config/scheduler.py:L130-L134 · SchedulerConfig
    scheduler_reserve_full_isl: bool = True
    """If True, the scheduler checks whether the full input sequence length
    fits in the KV cache before admitting a new request, rather than only
    checking the first chunk. Prevents over-admission and KV cache thrashing
    with chunked prefill."""
```

```python
# vllm/v1/core/sched/scheduler.py:L965-L985 · Scheduler.schedule
                reserved_blocks = 0
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    # … 省略：前缀命中与外部已算 token 等参数六行 ……
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,   # L982
                    reserved_blocks=reserved_blocks,
                    has_scheduled_reqs=bool(self.running),
                )
```

调用点把门要用的几样预算显式传入：`full_sequence_must_fit`（整序列检查开不开）、`reserved_blocks`（异步 KV 加载时其它在途 prefill 的预约块数；那条线归 Part IV 末章 KVConnector，此处只认「新来的请求不能占掉在途 prefill 已经预约的块」）、`has_scheduled_reqs`（本步是否已有调度请求，水位门用）。门本体：

```python
# vllm/v1/core/kv_cache_manager.py:L463-L488 · KVCacheManager.allocate_slots
        watermark_blocks = 0
        # The watermark is applied to waiting/preempted requests only, and only
        # when there's at least one request already scheduled.
        if has_scheduled_reqs and request.status in (
            RequestStatus.WAITING,
            RequestStatus.PREEMPTED,
        ):
            watermark_blocks = self.watermark_blocks                    # L470

        if full_sequence_must_fit:
            # First check and fail if the full request sequence won't fit.
            full_num_tokens = min(request.num_tokens, self.max_model_len)

            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=full_num_tokens,
                # … 省略：已算/编码 token 等参数四行 ……
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks  # L486
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None
```

门内三步：按整条序列算块数（`apply_admission_cap=True` 这个开关马上讲）、加上水位、与空闲块比，不够返回 None，请求留在 WAITING。10 块小池上四轮实跑（200 token 的请求、chunked prefill 首 chunk 16 token；free 为什么是 9 不是 10：池里恒有一块 null 占位块占着一个块位，下一小节「SWA 的还账方式」细讲）：

<!-- trace: m10 -->
| 轮 | 门配置 | 需求计算 | 判定 |
|---|---|---|---|
| 一 | full-ISL 门开（默认） | 整序列 cdiv(200,16)=13 块（首 chunk 只 1 块） | 13 > free 9 → None 拒之门外 |
| 二 | 门关（旧行为=只查第一 chunk） | 首 chunk cdiv(16,16)=1 块 | 1 ≤ 9 → 放进；chunk1 后实持 1 块、free 8。200 token 的请求在 160 token 的池里，prefill 到中途必然装不下 |
| 三（换 64-token 请求，演示 reserved） | 门开 + reserved 7 | cdiv(64,16)=4 块 | 可用 = 9 − 7 = 2；4 > 2 → None |
| 四（换 32-token 请求） | 门开 + reserved 7 | cdiv(32,16)=2 块 | 2 ≤ 2 → 放行（给在途 prefill 留足预约） |

门为什么灵：**封顶论证**。准入时每个在场请求的「封顶需求」都过了门检，Σ(封顶) ≤ 池容量，此后每步的增量需求都封顶于整序列总需求，不存在「中途才发现不够」的时刻。旧行为只查第一 chunk，缺的就是这条封顶。门的代价也直说：整序列检查比首 chunk 保守，第一条长请求可能占着门不放、batch 装不满，吞吐换活性。

### SWA 的还账方式：窗外块回收与 null 占位（站 11）

第一幕的修复靠「按全长算」，但第二幕告诉我们：对**回收型层**（滑窗、分块局部），全长算得太狠，它们的稳态占用远小于全长。要理解修复，先看这些层怎么「还账」。每个管家类型自己知道要多少历史：

```python
# vllm/v1/core/single_type_kv_cache_manager.py:L1057-L1083
    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.
        # … 省略：docstring 尾两行（Args/Returns）……
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)     # L1083
```

一行公式 `max(0, computed − window + 1)`：已算 7 个 token、窗口 4，下一个 token 的窗口盖住 4~7，窗外 0~3 共 4 个 token，**从此任何注意力计算都不再读它们**，块可以还。全注意力管家不覆写这个方法、基类直接 `return 0`（注释原话 "The default behavior is to not skip any tokens"，等价于同一算式取窗口无限的极限），从不还账。这是「每层类型自知历史」在账本侧的落点。还账的动作：

```python
# vllm/v1/core/single_type_kv_cache_manager.py:L622-L659
    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.
        # … 省略：docstring 中段五行（依赖 get_num_skipped_tokens，
        #       各注意力类型各自实现）……
        """
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)
```

只收**整块**（窗外 token 数整除块大小），块表开头一段逆序 free 归池、**原位换 null_block 占位**。为什么必须占位而不是把表缩短？位置不变量：块表第 i 项恒对应第 i×block_size 个 token（[第 13 章](../../ch13-paged-kv/narrative/chapter.md)的槽位恒等式靠它）。占位保住对齐，注意力 kernel 照表读、读到 NULL 的位置本来就在窗外、根本不会读。这套还账还是**每组各演各的**：组化给了每组一张私有块表，同一个请求在 full 组的表里全长在册、一个 NULL 都没有，在 SWA 组的表里开头早已是一排 NULL，两张表各按各的语义记账、互不越界（「共用一张表的两难」里「分表是唯一无损解」的兑现正在此处；DeepSeek V4 那五组同理，full 管家与滑窗管家各收各的账）。null 块在池里也要单独照顾：

```python
# vllm/v1/core/block_pool.py:L187-L191 · BlockPool.__init__
        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
```

引用计数不维护（占位不租给任何人），释放、清零、使用率分母三处都要为它单写一条分支。16-token 请求（持 4 块）在窗口 4、块 4 的设定下推进（池 8 块）：

<!-- trace: m13 -->
| 轮 | processed（已算） | 窗外 token / 整块回收 | 块表形态 | 实持 / 池 free |
|---|---|---|---|---|
| 初始 | 0 | 0 / 0 | [b1, b2, b3, b4] | 4 / 3 |
| 推进一 | 7 | 4 / 1 | [NULL, b2, b3, b4] | 3 / 4 |
| 推进二 | 11 | 8 / 1 | [NULL, NULL, b3, b4] | 2 / 5 |
| 推进三 | 15 | 12 / 1 | [NULL, NULL, NULL, b4] | 1 / 6 |
| 稳态 | 60（64-token 序列、池 64 块，window 8） | 53 / 13 | [NULL×13, …] | 3 / 60（3 块 = 12 个 token 槽位，每块 4 槽，窗口那 8 个 token 从 53 起、落在这 3 块里；3 块里最早的那个槽位已经出窗，但与窗内同块、被整块保留；按 token 数算真正未回收的是 64 − 53 = 11 个，12 是槽位数不是 token 数） |
| 对照 full | 10000 | 0 / 0（get_num_skipped_tokens 恒 0） | 从不回收 | 全长持有到请求结束 |
| 对照 chunked | 13 / 8 / 7 | 8 / 8 / 0（按 chunk 对齐） | 整 chunk 回收 | computed 13→收 8、7→0 |

![窗外回收与 null 占位](../diagrams/ch14-fig-swa-null-swap.png)

> *图注：L0「调度 · 显存账本」列池内动态的放大（对应 L2 章图南行站 11）。左：窗口 4、块 4 的请求算到第 7 个 token 时 tokens 0-3 落到窗外，整块 b1 归还块池、该块位换成立 NULL 占位；推进到 11、15，b2、b3 依次离场，实持 4→3→2→1、池 free 3→4→5→6。SWA 的显存回收是连续小步，不是一次性。右：稳态 64-token 序列算到 60 时 13 块已归池、实持仅 3 块（tokens 52-63）。块位号不变：第 i 块永远是第 i×4 个 token 起，注意力 kernel 照表读、读到 NULL 的位置本来就在窗外不读；full attention 对照全长持有到请求结束。*

末两行对照补两句场景。「稳态」行换了景：64-token 序列、池 64 块（free 60 = 64 − 实持 3 − null 1）。「对照 chunked」行是分块局部注意力的同款还账，只是它的「窗口」是当前 chunk：chunk 8、块 8 时已算 13 个 token，注意力只看第 8~15 那一块，前面 0~7 共 8 个 token 整块归还（`get_num_skipped_tokens(13) == 8`，docstring 自带此例）；已算恰 8 个时同样收 8（刚跨进新 chunk），已算 7 个时还在第一块内、收 0。回收永远对齐到 chunk 边界，与 SWA 对齐到窗外整块同理，只是取整的尺子从窗口换成了 chunk。

注意时点：`remove_skipped_blocks` 在每个 chunk 的分配预测**之前**先跑（kv_cache_manager.py:L504-L508 的调用），且基准是 processed（已落账）而非乐观的 computed：在途步的注意力窗口还在读的块不能收。这个时序是下一道门的正确性前提。

### 第二道：回收感知准入上限——单源铁律

有了回收，滑窗层的**稳态**占用就封顶了：窗口内 + 在途。上限的公式写在 spec 上：

```python
# vllm/v1/kv_cache_interface.py:L587-L618
    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        """Per-request admission cap, in blocks.

        Single source of truth for both startup pool sizing
        (`max_memory_usage_bytes`) and the runtime admission gate. Per-request
        real-held blocks plateau at this bound because
        `SlidingWindowManager.remove_skipped_blocks` runs from `allocate_slots`
        before each chunk's `get_num_blocks_to_allocate`.
        # … 省略：docstring 尾两行（max_in_flight_tokens 的定义指引）……
        """
        # During chunked prefill, we hold KV for the last `sliding_window-1`
        # computed tokens plus the in-flight tokens (frees happen on the
        # processed-token basis); never more than `max_model_len`.
        num_tokens = min(self.sliding_window - 1 + max_in_flight_tokens, max_model_len)   # L604
        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1                     # L608

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # … 省略：DCP 断言两行（上下文并行不支持滑窗）……
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes
```

三个部件。`max_in_flight_tokens` 是「已排进 batch、还没落账的 token 数」的上界（前文算 257 用的那个 8192，口径一致）。`sliding_window − 1` 里的减一：窗口 W 个位置里最新的那个 token 通常还堵在途上（已经算在在途项里了），落账在册的只占 W−1，两项相加恰好盖满窗口。这两项为什么要加在一起，源头在回收的时点：窗外块的回收按 processed（已落账）基准做，一拍里在途的那批 token 的块还收不回来，所以最坏一瞬实持 = 窗口内已落账的 W−1 个 + 整批在途的 token。`+1` 顶着「窗口起点不在块首」的最坏错位（注释原例：块大小 4、6-token 窗口 [CDEF] 要占 [XXCD][EF] 两块）；docstring 第一句就是本章的铁律原文——**"Single source of truth for both startup pool sizing and the runtime admission gate"**：同一个方法喂启动期池大小器（`max_memory_usage_bytes` 直接拿它乘页大小）与运行期准入门。运行侧的夹取：

```python
# vllm/v1/core/single_type_kv_cache_manager.py:L178-L191 · SingleTypeKVCacheManager.get_num_blocks_to_allocate
        num_required_blocks = cdiv(num_tokens, self.block_size)
        if apply_admission_cap and self._max_admission_blocks_per_request is not None:
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            # chunked-local counterpart). `remove_skipped_blocks` runs from
            # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            # so per-request peak real-held blocks <= this cap, which keeps
            # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            # Drift between the two would re-introduce the deadlock from
            # issue #39734 or, worse, mid-prefill OOM.
            num_required_blocks = min(
                num_required_blocks, self._max_admission_blocks_per_request   # L190
            )
```

注意这段推理链，它回答为什么「按上限放行」不会超收：`remove_skipped_blocks` 在每个 chunk 的预测前先跑，所以每请求**峰值实持** ≤ cap；于是「预约之和 ≤ 池」⟹「峰值实持之和 ≤ 池」，放行安全（反向不必成立：预约比峰值保守，正是要的）。而这一切**没有运行时校验**，靠单源 + 注释防御：上一节整序列门里那个 `apply_admission_cap=True` 就是打开这个夹取的开关（只有整序列门打开它：准入要按封顶算，运行中的增量预测不用夹）。装配点在管家工厂：`get_manager_for_kv_cache_spec` 构造滑窗/分块管家时把这个上限算好注入（single_type_kv_cache_manager.py:L1861-L1878，注释再强调一遍 single source of truth）。窗口 8、块 4、在途 8 的推进实跑（混合门那行另起一景：窗口 512、块 16、在途 0 的 SWA 组配 full 组，池 1000、4096-token 请求）：

<!-- trace: m11 -->
| 轮 | 场景 | 关键数 | 判定/落账 |
|---|---|---|---|
| 公式 | SWA cap | cdiv(min(8−1+8, 64), 4)+1 = cdiv(15,4)+1 = 5 | cap = 5（+1 顶着窗口不在块首的最坏错位 [XXCD][EF]） |
| 公式对照 | chunked cap（无 +1） | cdiv(8+0, 4) = 2 | chunked 窗口从块首开始 |
| 推进 1 | computed 0 → 8 | 首 chunk 需 2 块 | 实持 2 ≤ cap 5 |
| 推进 2 | computed 8 → 16 | 窗外 1 token（<1 块，不收）→ 需补 2 | 实持 4 ≤ 5 |
| 推进 3 | computed 16 → 24 | 窗外 9 token → 收 2 块、补 2 | 实持 4 ≤ 5（稳态） |
| 推进 4..8 | computed 24…56 | 每步窗外多 2 块、补 2 块 | 实持稳在 4 ≤ 5，池 free 回升 |
| 混合门 | 4096 请求过 full-ISL 门 | full 组 cdiv(4096,16)=256；SWA 组不夹 256 / 夹到 33（=cdiv(511,16)+1） | 总需求 256+33=289 ≤ free 999 → 放行；不夹则 512，并发白丢一半 |

![SWA 实持封顶与混合门夹取](../diagrams/ch14-fig-swa-cap-plateau.png)

> *图注：L0「调度 · 显存账本」列内 Scheduler 与 KVCacheManager 接缝的准入门放大（对应 L2 章图中排拍片⑥）。上：SWA 请求逐 chunk 推进，窗外块每步先回收归池，实持块从 2 涨到 4 就封顶（cap=5 留了 +1 的块首错位余量），序列再长实持也不涨；下：混合模型过准入门，full 组按整序列 256 块、SWA 组被夹到 33 块，总 289 ≤ 999 放行；不夹则要按 512 算，1000 块的池并发从约 3.4 条掉到约 2 条。cap 由 spec 的同一个方法算出，启动期定池大小、运行期放请求进门共用，单源，漂移即 #39734 死锁。*

混合门那行就是第二幕的修复现场：夹取后同一池子的并发从约 2 条救回约 3.4 条。拿 #39734 的场景心算（示意推演，非 issue 原文数字）：一条 100k prompt 按全长要 cdiv(100000,16) ≈ 6250 块；窗口 1024、块 16、在途按一个 16-token chunk 计的 SWA 层，回收感知稳态只要 cdiv(1023+16,16)+1 = 66 块。6250 与 66 的鸿沟，被一个公式抹平。与前文混合布局表那行的 257（cap=cdiv(min(511+8192,4096),16)+1）对照，差别只在 `max_in_flight_tokens` 一个代入值：在途 0 时窗口项生效得 33，在途 8192 时 max_model_len 顶住得 257。

门立完了，把「一个池子多张账」的收益合成一笔看。一张表的世界里，每请求的账必须按最保守的语义（全历史）摊给全部层；分组后是各组各按各的语义封顶再求和，差距由回收型组的「封顶不随长度涨」单调拉开。三个静态口径加一个动态口径，各兑现一次：

<!-- trace: m18 -->
| 口径 | 一张表（最保守语义说了算） | 组化后（各组各按各的语义） | 落账 |
|---|---|---|---|
| 4096-token 请求过准入门（窗 512、在途 0、池 1000） | full 256 + SWA 按全长 256 = 512 | full 256 + SWA 夹到 33 = 289 | 并发 1.9531 → 3.4602（同池 1000） |
| 死锁修复场景 100000-token prompt（窗 1024、在途 16） | 按全长 cdiv(100000,16) = 6250 块 | 回收感知 cdiv(1023+16,16)+1 = 66 块 | 6250 → 66：鸿沟被单源公式抹平 |
| 混合容量核算（在途 8192 顶到 max_len 4096、池 1024） | —（uniform 无此形态） | full 256 + cap 257 = 513 | 并发 1.9961（对照 uniform：每请求 256 → 4.0） |
| 运行期还账（decode 推进中） | full 管家从不还账（skipped 恒 0） | SWA 组每步窗外整块归池：实持 4→3→2→1 | 池 free 回升 3→4→5→6（前文推进一~三） |

第三行别过度解读：在途 8192 顶住 max_model_len 时 cap 257 只比全长 256 多 1，这一景分组几乎没省（并发 1.9961 对 uniform 的 4.0）；窗口真正发力在两头：在途小时（准入门那行的 33），序列长时（32768 长度 full 组要 2048 块、SWA 组封顶 545）。最后一行补上动态半边：full 组从不还账，SWA 组每步把窗外整块还回池里，省显存不是准入时的一笔折扣，是运行期每拍都在发生的还账。512 对 289、6250 对 66、513 对 256：三个口径三个数，背后是同一个公式，这就是 PR #40946 单源化的全部含义。

### 第三道：水位，吞吐换稳定

整序列门管住了输入长度，但有一类超收它管不着：**输出长度未知**。准入时只预留输入，decode 每拍都在长。decode 密集负载（输出远大于输入）高并发下会发生什么，官方 benchmark 脚本的头部注释写得像教材（benchmarks/kv_cache_watermark.sh）：

```text
# Why this workload triggers thrashing:
#   Requests are admitted based on the KV cache they need *at admission time*.
#   With `--scheduler-reserve-full-isl` (default) the input length is reserved up
#   front, but the *output* length is unknown and unreserved. A decode-heavy
#   workload (output >> input) at high concurrency therefore over-admits while
#   requests are short, then runs out of KV cache as they all grow during decode
#   -> the scheduler preempts (recompute) recently-admitted requests, re-prefills
#   them later, and repeats. The watermark keeps a block of KV cache free so
#   running requests can grow into it instead of triggering this churn.
#
# … 省略：脚本机制五行（KV 约束配置下启动 vllm serve、扫水位取值、
#       收抢占/吞吐/延迟指标并画图）……
# Default workload: concurrency 200, input ~300 tokens, output ~4000 tokens
```

抖动环六步：准入只预留输入 → 短时超收 → 全体增长 → 池尽 → 抢占刚准入者 → 重 prefill → 回到第一步（抢占环的内景[第 11 章](../../ch11-preemption-request-lifecycle/narrative/chapter.md)拆过：v1 只有 recompute 一条路，被抢者全部块释放、num_computed_tokens 归零、回队头）。脚本的复现口径：并发 200、输入约 300、输出约 4000、KV 池压到均值需求的约 1.5 倍。水位的回答是给「增长」留一块垫片（垫片 = 水位预留的那块空闲，下文与 headroom 同义；区别于 0.92 预算里留的头寸：那是启动期的余量，这是运行期的）：

```python
# vllm/config/scheduler.py:L136-L141 · SchedulerConfig
    watermark: float = Field(default=0.0, ge=0.0, lt=1.0)
    """Fraction of total KV cache blocks to keep free (the watermark) when
    admitting waiting or preempted requests into the running queue. This headroom
    helps avoid frequent KV cache eviction and the resulting repeated preemption
    of requests when GPU memory is scarce. Must be in the range [0.0, 1.0); 0.0
    (the default) disables the watermark."""
```

块数在 KVCacheManager 构造时就算死：`watermark_blocks = int(watermark × num_blocks)`（vllm/v1/core/kv_cache_manager.py:L171）。计入的条件在门段开头已经见过（L463-L470）：**两个条件同时成立**才把水位算进 required：本步已有调度请求（`has_scheduled_reqs`），且来者是 WAITING 或 PREEMPTED。随后与稳态判定合流（L521-L527）：

```python
# vllm/v1/core/kv_cache_manager.py:L521-L527 · KVCacheManager.allocate_slots
        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks          # L524
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None
```

这笔 `num_blocks_to_allocate` 已不是第一道门那笔整序列数：两道门之间源码按本步重算了一遍——`num_tokens_need_slot`（本步要落槽的 token 数：已算 + 新排入的 token + lookahead，封顶 max_model_len；kv_cache_manager.py:L490-L519），不夹准入上限，L524 比的是这笔；上一节「运行中的增量预测不用夹」说的就是它。一次 `allocate_slots` 调用里两笔先后各查各的：第一道门查整条序列装不装得下，这一道查本步增量够不够。

这个 None 与整序列门的 None 同一个出口：WAITING 侧等下一拍，RUNNING 侧进抢占环。10 块池、水位 0.5（`watermark_blocks = 5`）三轮判定：

<!-- trace: m12 -->
| 轮 | 请求状态 / has_scheduled_reqs | required 计算 | 判定 |
|---|---|---|---|
| 一 | WAITING / True | 5 块 + 水位 5 = 10 | 10 > free 9 → None（headroom 留给 running 长大） |
| 二 | WAITING / False（首拍空转） | 5 + 0 = 5 | 5 ≤ 9 → 放行（池全空时再保守就永远开不了工） |
| 三 | RUNNING / True | 5 + 0 = 5（精修版只对 WAITING/PREEMPTED） | 5 ≤ 9 → 放行（在座长个不受垫片约束） |

![水位门：垫片只管新客进门](../diagrams/ch14-fig-watermark-gate.png)

> *图注：L0「调度 · 显存账本」列准入位的放大（对应 L2 章图中排拍片⑦）。左（水位关，默认 0.0）：decode 密集负载的六步抖动环，准入只预留输入、短时超收、集体增长、池尽、抢占刚准入者、重 prefill，循环往复；右（水位 0.5）：同一个 80-token 请求在 10 块的池前被暂缓（5 块需求 + 5 块水位 = 10 > free 9），把 headroom 留给已就座的请求加椅子。首拍空转与 RUNNING 涨块不扣水位：垫片只管「新客进门」，不管「在座长个」。吞吐换稳定，默认关闭交给用户按负载调。*

两个条件各自防一种笨：首拍不算水位（轮二），池全空时若还算水位，第一条请求永远进不来，系统开不了工；RUNNING 不算水位（轮三），垫片是给「增长」留的，在座的请求自己就是增长本身。这套「精修版水位」对照的是 v0 的老办法：v0 的 BlockSpaceManager 用全局静态垫片（默认 1%），不管有没有 running、不管来者是谁一律垫，保守浪费；v1 裸奔两代后在 #44594（2026-06-11）把它以精修形态请回来，默认 0.0 关闭。代价始终是那句：**headroom 空闲不接客，吞吐换稳定**——交给用户按负载调，官方脚本就是调参的复现台。

## worker 侧落地：每组一张表、大块拆小块（站 12）

账本侧的故事讲完了，最后一块拼图在 GPU 半边：worker 拿到 config 真分配张量时，混合布局要每组一张块表；而**账本的块大小**与**注意力 kernel 认的块大小**还可能不同：调度器按 32-token 的块记账，而注意力 kernel 只认 16-token 的块。换算是纯乘法加法：

```python
# vllm/v1/worker/block_table.py:L220-L248
    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )                                                                   # L246

        return kernel_block_ids.reshape(-1)
```

双射换算 `kernel_id = kv_id × k + j`：大块与其 k 个小块覆盖的 token 集合逐字相等，无重号无漏号（整除性由构造期校验兜底）。`MultiGroupBlockTable`（block_table.py:L270-L336）给每个 KV 组一行，各行按需细分：同一个请求在不同组有不同行宽。kernel 块大小由**注意力后端的能力**决定：每个 KV 组各自协商，取该组全体后端都支持、且整除该组账本块大小的最大块，源码注释还举了 256 拆 4×64 的例（vllm/v1/worker/gpu_model_runner.py:L7644-L7651）。谁声明支持什么、怎么协商，归执行篇注意力后端章。实跑：

<!-- trace: m14 -->
| 轮 | 输入 | 换算 | 结果 |
|---|---|---|---|
| 纯算术 | 块 id [0, 1, 2]（kv 32 / kernel 16） | kernel_id = kv_id × 2 + j | [0, 1, 2, 3, 4, 5] |
| 恒等 | 块 id [5, 9]（不拆分） | blocks_per_kv_block=1 原样返回 | [5, 9] |
| 块表行 | append [0, 1]（32 → 2×16） | 行宽 8 → 16（×2） | 行首 [0, 1, 2, 3]（余位补 0 占位） |
| 多组 | 组 0 [0,1] / 组 1 [2] | 每组一张表，各自按需细分 | 组 0 行 [0,1,2,3]；组 1 行 [2] 原样 |
| 后端协商 | 组 0 后端认 16 / 组 1 后端认 32,16 | 取全体后端都支持的最大公因子块 | kernel_block_sizes = [16, 16]（组 0 拆、组 1 直认） |

[第 13 章](../../ch13-paged-kv/narrative/chapter.md)在读腿上埋过一个问号（注意力 kernel 穿块表读的代价），那个账归执行篇结算；本章只把「账本块号 → kernel 块号」的换算对齐：账本说的 3 号块、kernel 眼里的 6、7 号小块，指的是同一段 token。

## 总结：「调度 · 显存账本」列上半与启动带点亮

本章点亮了 L0 图「调度 · 显存账本」列 KV 半区的上半，从启动装配带的测量到 `KVCacheManager` 的两道门，加上 worker 侧每组一张块表、大块拆小块的落地。与[第 13 章](../../ch13-paged-kv/narrative/chapter.md)（池的内部）合起来，这列从上到下全通；[第 10 章](../../ch10-continuous-batching-chunked-prefill/narrative/chapter.md)到[第 12 章](../../ch12-async-scheduling/narrative/chapter.md)三章里那个当参数用的 `num_gpu_blocks`，从此有了完整的出生证明。开篇三问的答案：**饼多大**：预算（总显存 × 0.92，分母是总量、per-instance）减去峰值账（dummy 前向量出的 non_kv = 常住 + 瞬时峰）再减图池估计，一行减法定本金，`// 页 // 层`换块数，护栏四道保证这份账开得了工；**怎么切**：组化规矩是被模型潮一步步逼出来的（v0 手工补丁退役、v1 统一分配器解锁前缀缓存与 KV 传输、三种组形态渐次长出，多块池被前缀缓存否掉）；混合模型按 spec 分桶、等量化组（padding 是账）、页统一（调大块字节不变但切得粗、pad 有代价、垫不动拒收）、每池每组出一层共享张量，两把尺 LCM/GCD 对齐调度与哈希；DeepSeek V4 七种形态、四种页宽走单开的第三条路（元组装包＋近似 GCD＋重叠布局），压缩块按「管理面 256 / 物理面 64 行」两套坐标系记账；**门多紧**：整序列门堵超收（第一幕）、回收感知上限堵门账漂移（第二幕）、水位留头寸堵输出未知的抖动，三道预算全部来自启动账本、运行期零重算。带走三件事：

1. **测量式分配是一次性契约**。启动量一次、算一次、写回 cache_config，此后运行期只有照账放行、没有重新定账。这意味着 profile 的错（真实负载峰超 dummy run）运行期没有补救，防御全在启动侧：util 留头寸、assert 拒环境漂移、护栏拦死账。代价与收益是同一枚硬币：少一分运行期开销，多一分对启动测量的信任要求。
2. **单源是防漂移的唯一手段，且贯穿始终**。一份 KVCacheConfig 单点产出喂两侧（调度器拍平、worker 按布局分配、PP 取最小）；准入上限是 spec 上的同一个方法喂启动定池与运行放行；override 折算把 available_memory 一并改写。三处都没有运行时对账：结构上让漂移不可能，比校验便宜也可靠。
3. **等字节硬约束是混合组化的总纲**。一个池共享等大块，推出页统一、组等量、张量按池共用一整套；它的代价清单也诚实：padding 层白占显存、Mamba pad 极端时浪费 93.75%、六条假设里官方自认「跨类型命中只干净支持 full + 恰好一种其它」。

还有一条暗线值得点名：本章的还账（窗外块回收）与[第 13 章](../../ch13-paged-kv/narrative/chapter.md)的还块（free 不清哈希）在同一块土壤上——被回收的满块，哈希还留在表上。哪些请求能认领这些块、哈希按哪把尺切（本章立住的 `hash_block_size`）、LRU 驱逐怎么挑人，下一章《前缀缓存》把这半本账接着记。

（完）
