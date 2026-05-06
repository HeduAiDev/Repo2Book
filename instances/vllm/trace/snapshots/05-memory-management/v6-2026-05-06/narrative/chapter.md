# 第5章：GPU 显存管理系统 — 从 80 GiB 到 35,148 个 KV block

> 本章涉及的 vLLM 源码：
> - `instances/vllm/source/vllm/utils/mem_utils.py:L60-L275`（`MemorySnapshot`、`memory_profiling`）
> - `instances/vllm/source/vllm/v1/worker/gpu_worker.py:L352-L505`（`determine_available_memory`）
> - `instances/vllm/source/vllm/v1/kv_cache_interface.py:L80-L205`（`KVCacheSpec`、`AttentionSpec`）
> - `instances/vllm/source/vllm/v1/core/kv_cache_utils.py:L113-L370`（`KVCacheBlock`、`FreeKVCacheBlockQueue`）+ L930-L947（`get_num_blocks`）
> - `instances/vllm/source/vllm/v1/core/block_pool.py:L130-L510`（`BlockPool`，前缀缓存 + LRU）
> - `instances/vllm/source/vllm/v1/core/sched/scheduler.py:L952-L972`（recompute 抢占路径）
>
> 本章源码 commit：`98661fe`。
>
> **第 4 章里 `kv_cache_manager.allocate_slots(req, n)` 返回 `None` 是触发 preempt 的唯一信号——但这个 `None` 是怎么算出来的？80 GiB 的 H100 怎么变成 35,148 个 KV block？为什么是 35,148，不是 50,000？这章是把这条路走完。**

---

## 这章要讲什么？

第 4 章里我们对 KV cache 做了最粗糙的抽象：一个 free list、一个 `allocate_slots` 返回 `None` 表示 OOM。但生产 vLLM 的 KV cache 管理器是一个 600 行的类——它要回答几个朴素却致命的问题：

1. **8% 的安全边界从哪来？** 为什么 `gpu_memory_utilization` 默认是 0.92？
2. **`num_gpu_blocks = 35148` 这个数字是怎么算的？** 哪些 byte 算 KV，哪些不算？
3. **空块队列为什么不能用 `deque`？** 一个看起来人畜无害的数据结构选型，为什么是前缀缓存性能的命门？
4. **抢占的时候，为什么 vLLM v1 选 recompute 而不是 swap？** 明明 swap 在延迟上快 2.6 倍。

打开 `instances/vllm/source/vllm/v1/worker/gpu_worker.py:L411-L445` —— 引擎启动时计算可用 KV cache 的核心几行：

```python
# vllm/v1/worker/gpu_worker.py:L411-L415
non_kv_cache_memory = (
    profile_result.non_torch_increase
    + profile_result.torch_peak_increase
    + result.weights_memory
)
# vllm/v1/worker/gpu_worker.py:L441-L445
available_kv_cache_memory_bytes = (
    int(total_gpu_memory * self.cache_config.gpu_memory_utilization)
    - non_kv_cache_memory_bytes
    - cudagraph_memory_estimate
)
```

四项相减——一个简单减法，背后是 vLLM 对 GPU 显存做的一次"三类切分"。本章就是把这个减法每一项的 byte 来源、测量方法、设计取舍讲清楚。

学完这章你能：

- 在白纸上写出 `available_kv_cache = requested - weights - peak_activation - non_torch - cudagraph`，并说出每一项是怎么测的（NVML 还是 `torch.accelerator.memory_stats`？）。
- 用 demo 的 Llama-3.2-1B / H100 配置做一遍 page_size 计算和 `num_gpu_blocks` 推导，得到 35,148。
- 说清楚为什么 `gpu_memory_utilization=0.92` 不是拍脑袋的安全边界——它要吸收四类具体的"NVML 看不见"的开销。
- 解释 `BlockPool` 用手写双向链表的根本原因（`touch()` 的 O(1) 中间删除），以及 null block 这个零号"占位符"存在的目的。
- 用 8K-token 请求的 KV size = 1 GiB 数据，对比 recompute（164 ms）vs swap（62 ms），并解释 vLLM v1 为什么选了延迟更慢的那一种。

---

## 5.1 问题：KV cache 是显存的"主力消费者"

### 5.1.1 80 GiB 不是给你的

打开 `instances/vllm/source/vllm/v1/worker/gpu_worker.py:L364-L382` —— `Worker.determine_available_memory()` 的开头：

```python
total_gpu_memory = current_platform.get_device_total_memory()
# ...
requested_memory = int(total_gpu_memory * self.cache_config.gpu_memory_utilization)
```

H100 的 80 GiB 显存里，vLLM 默认只用 0.92 × 80 ≈ **73.6 GiB**。剩下的 6.4 GiB（**8%** 的安全边界）不能直接给你做生意——它要去吸收四类一旦被占满就立刻 OOM 的开销，每一类我们 5.2 都会一个个看：

- **PyTorch caching allocator 的内部碎片**：torch 自己的内存池切块对齐导致的浪费，约 2-3%。
- **profile run 没看到的激活峰值**：用户真实输入的形状比 profile 时大就会戳穿这个余量。
- **CUDA context overhead**：torch 看不到、但 NVML 看得到的那部分。
- **采样时段的瞬时峰值**：sampling 的临时张量。

这 6.4 GiB 不进入预算，但它**保护**预算。理解这一点是理解 5.2 那条 5 项减法的前提。

### 5.1.2 73.6 GiB 也不全是 KV

73.6 GiB 不是 KV cache 的预算——它是**整个引擎进程**的预算。在它里面，KV cache 必须排在最后：

```
73.6 GiB requested
├── 权重    (weights)              ← 必须
├── 峰值激活 (peak activation)     ← 必须
├── CUDA graph 内存                ← 可选（默认 OFF）
├── non-torch (NCCL、attn workspace、CUDA context)  ← 你不主动分配，但它真实占着
└── KV cache                        ← 剩下这点，给你
```

跑一下 demo（`python3 -m instances.vllm.artifacts.05-memory-management.implementation.demo`），第一段输出告诉你 Llama-3.2-1B 在 H100 上的实际分配：

```
[1] Llama-3.2-1B on H100 (80 GiB) — memory layout
  Total GPU memory                        80.00 GiB
  Requested (util=0.92)                   73.60 GiB
  ├─ Model weights                         2.40 GiB
  ├─ Peak activation                       1.80 GiB
  ├─ CUDA graph memory                     0.25 GiB
  ├─ Non-torch (NCCL etc.)                 0.50 GiB
  └─ Available for KV cache               68.65 GiB
  Util margin (unused safety)              6.40 GiB

  Page size                                64.0 KiB / layer
  Num layers                                 32
  num_gpu_blocks                          35148
  Wasted (rounding)                         1.6 MiB
```

权重只有 2.4 GiB——**KV cache 拿走了 68.65 GiB，是权重的 28 倍。** 这就是为什么"GPU 显存管理"约等于"KV cache 管理"在 vLLM 的语境里，反过来讲不对。下一章你会反复用到这个事实。

### 5.1.3 68.65 GiB 能装多少并发请求？

KV cache 容量决定**并发**，不是吞吐。看 demo 第二段：

```
[2] Max concurrent requests at avg_seq_len=2048
    275 concurrent requests fit in the KV cache
```

275 个并发——每个请求 2K token——这是 H100 在 Llama-3.2-1B 上的物理上限。比这个多，下一个请求就只能 `kv_cache_manager.allocate_slots → None`，触发第 4 章的 preempt 路径。换句话说：**第 4 章那个 `None` 信号的真正阈值，就是本章计算出来的 35,148**。

---

## 5.2 NVML 和 torch 之间的"gap" — 三类显存

### 5.2.1 一个 cudaMalloc 的真实归属

打开 `instances/vllm/source/vllm/utils/mem_utils.py:L204-L235`——这是 `memory_profiling` 的 docstring，里面藏着 vLLM 内存模型的全部世界观：

> 1. memory used by anything OTHER THAN the current vLLM instance
> 2. memory used by torch in the current vLLM instance
> 3. memory used by the current vLLM instance but NOT by torch
>    (NCCL buffers, attention-backend workspaces, CUDA context)

这三类切分**不是表观分类——它是测量方式上的真实分裂**。NVML 看的是整张卡（cuda 字段：`total_memory - free_memory`）；`torch.accelerator.memory_reserved()` 看的是 torch 自己的内存池。两者相减就是第 3 类：

$$
\mathrm{non\_torch\_memory} = \mathrm{cuda\_memory} - \mathrm{torch\_memory}
$$

**为什么这个 gap 不能忽略？** 我们的 demo 给的数字：non_torch = 0.5 GiB。这是 NCCL 通信库分配的 buffer + flash attention 的 workspace + CUDA driver 自己的 context。这部分**对 PyTorch 透明**——`torch.cuda.empty_cache()` 不能释放它，PyTorch 的内存统计也看不见它。但它真实占用了显存，所以 KV cache 必须把它扣掉。

### 5.2.2 `MemorySnapshot` — 一次拍照就是一组 5 个数

打开 `instances/vllm/artifacts/05-memory-management/implementation/mem_snapshot.py:L29-L86`——`MemorySnapshot` 数据结构和它的 `__sub__`：

```python
# implementation/mem_snapshot.py:L38-L51
@dataclass
class MemorySnapshot:
    torch_peak: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0       # = total - free, 整卡占用
    torch_memory: int = 0      # = torch.accelerator.memory_reserved()
    non_torch_memory: int = 0  # = cuda_memory - torch_memory
    timestamp: float = 0.0
    device: str | None = None
    auto_measure: bool = False
```

vLLM 真实的 `MemorySnapshot.measure()`（`mem_utils.py:L96-L126`）只做两件事：
1. 调 `current_platform.mem_get_info(device)` —— 一次 NVML 读，拿 free 和 total。
2. 调 `torch.accelerator.memory_stats(device)` —— 拿 `allocated_bytes.all.peak` 和 `reserved`。

这两次读耗时几微秒，所以可以在引擎启动的关键路径上反复拍照。我们的实现把 `auto_measure=False` 默认掉，让测试可以**手填**字段——demo 不需要真 GPU。

### 5.2.3 `memory_profiling` 上下文管理器 — 三张快照定方位

打开 `implementation/mem_snapshot.py:L106-L148`：

```python
# implementation/mem_snapshot.py:L106-L148
@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        weights_memory=weights_memory,
    )
    yield result

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp
    result.non_kv_cache_memory = (
        result.non_torch_increase
        + result.torch_peak_increase
        + result.weights_memory
    )
```

这个上下文管理器在 vLLM 启动时围绕 **profile_run**（一次最大批次的假前向）使用，三张关键快照：
- `before_create`：vLLM 进程**还没创建**前的状态——记录"别人已经占了多少"，未来减掉。
- `before_profile`：模型权重**已加载**、profile run **还没开始**——这一刻的 `torch_peak` 是"权重加载完的稳态"。
- `after_profile`：profile run **跑完**后——`torch_peak` 一定 >= `before_profile.torch_peak`，差值就是 peak activation。

vLLM 源码 `mem_utils.py:L209-L235` 给了一个示例：

```
Before vLLM creation:        cat1=1, cat2=0,           cat3=0          GiB
After model load:            cat1=1, cat2=2 (weights), cat3=0.5 (NCCL)
Peak during profile:         cat1=1, cat2=4 (acts +2), cat3=1
After profile (gc'd):        cat1=1, cat2=3,           cat3=1
```

带入上下文管理器的 yield-后逻辑：
- `weights_memory` = 2 GiB（参数传入）
- `torch_peak_increase` = 4 - 2 = 2 GiB（profile 内部 peak - profile 前的 peak = 激活峰值）
- `non_torch_increase` = 1 - 0 = 1 GiB（after profile.non_torch - before create.non_torch）
- `non_kv_cache_memory` = 2 + 2 + 1 = 5 GiB

这正是 `tests/test_mem_snapshot.py` 的 `test_memory_profiling_worked_example` 在断言的算术。

我们的实现去掉了 vLLM 在 `__enter__` / `__exit__` 调用的 `gc.collect()` 和 `torch.accelerator.empty_cache()`（`mem_utils.py:L249-L260`）——它们对教学没帮助，但会让 demo 失败（没真 GPU）。caller 端手填快照即可。

---

## 5.3 从 73.6 GiB 到 35,148 个 block：逐项减法

### 5.3.1 `determine_available_memory` 的核心算术

打开 `instances/vllm/artifacts/05-memory-management/implementation/memory_layout.py:L87-L144` —— `determine_available_memory`：

```python
# implementation/memory_layout.py:L107-L129
def determine_available_memory(
    init_snapshot, profile_result, cudagraph_memory,
    gpu_memory_utilization, spec, num_layers,
) -> MemoryLayout:
    total = init_snapshot.total_memory
    requested = int(total * gpu_memory_utilization)              # ① 80 GiB × 0.92 = 73.6 GiB

    non_kv_cache = (                                              # ② 三项相加
        profile_result.non_torch_increase
        + profile_result.torch_peak_increase
        + profile_result.weights_memory
    )

    available_kv_cache = requested - non_kv_cache - cudagraph_memory  # ③
    available_kv_cache = max(available_kv_cache, 0)                    # ④ 负值防御

    page_size = spec.page_size_bytes                              # ⑤
    num_blocks = get_num_blocks(available_kv_cache, num_layers, page_size)
    used_bytes = num_blocks * page_size * num_layers
    wasted = available_kv_cache - used_bytes                      # ⑥
    ...
```

逐点解释：

1. `requested = int(total * util)` — `int()` 截断在这里。M02：`int(80 * 1024**3 * 0.92) == 78920663040` 精确等于 73.6 GiB（小数点后两位）。如果你担心浮点漂移，**记住引擎里这个值是 int 唯一的真值**，所有后续减法都建立在它上面。
2. `non_kv_cache` — 三项加法对应 vLLM `gpu_worker.py:L411-L415`：non_torch 增量 + torch_peak 增量 + 权重。
3. `available_kv_cache = requested - non_kv_cache - cudagraph_memory` —— vLLM 同款一行，`gpu_worker.py:L441-L445`。这个数字就是"还能给 KV cache 用多少 byte"。
4. `max(..., 0)` —— 模型大到溢出 requested 的极端情况，避免负数把后面的整除搞坏。这是 OOM-at-startup 的 sentinel，看到 `available_kv_cache == 0` 引擎启动期间就会拒绝起飞。
5. `page_size = spec.page_size_bytes` —— 5.3.3 详细讲。先告诉你它是 64 KiB（block_size=16, 8 KV head, head_size=128, fp16）。
6. `wasted = available_kv_cache - used_bytes` —— 整除后剩下的 byte，放不下一个 block 就丢了。demo 报告 1.6 MiB——在 68 GiB 的尺度上，0.0023%，可以忽略。但这个值需要监控：如果它突然变成几 GiB，说明你的 page_size 配错了。

⚠️ **Ch20 forward-pointer**：我们的实现里 `cudagraph_memory` 总是减——是无条件的。vLLM 真实代码（`gpu_worker.py:L417-L423`）只有当环境变量 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 被设置时才减；默认 OFF 时让 cudagraph 内存从那 8% 安全余量里出。第 20 章引入 ModelRunner 时会引入这条 env-var 派发逻辑——一行 `if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:` 守卫。Ch05 跑 demo 时永远传 256 MiB 显式扣除，pedagogically 更清楚。

### 5.3.2 page size 公式 — KV cache 的"最小账单单位"

打开 `instances/vllm/artifacts/05-memory-management/implementation/kv_cache_spec.py:L54-L67` —— `AttentionSpec.real_page_size_bytes`：

```python
# implementation/kv_cache_spec.py:L54-L67
@property
def real_page_size_bytes(self) -> int:
    return (
        2                              # K and V interleaved per-token
        * self.block_size              # tokens per block (default 16)
        * self.num_kv_heads            # GQA: 8 for Llama-3.2-1B
        * self.head_size               # 128 for Llama-3.2-1B
        * self.dtype_bytes             # 2 for fp16/bf16
    )
```

5 个数字相乘。每一项的含义：
- **2**：K 和 V 是分开存的，但每一对在一起算账。这个 2 不是冗余，它是 attention 算法里 K/V 必须配对的物理体现。
- **block_size**：一个 block 装 16 个 token 的 KV（vLLM 默认）。第 2 章解释过这个粒度的取舍——更小则 prefix cache 命中率高、内部碎片小，但 indexing overhead 高。
- **num_kv_heads**：GQA 之后 K/V 头数远少于 Q 头数（Llama-3.2-1B 是 8 KV / 32 Q）。这一栏直接乘以 4 倍内存节省，是 GQA 设计的关键。
- **head_size**：每个 head 的维度（hidden_dim / num_q_heads）。Llama 系列大多是 128。
- **dtype_bytes**：fp16/bf16 是 2，fp32 是 4，nvfp4 走另一条 packed layout（`kv_cache_interface.py:L154-L163`）。

代入 demo：`2 * 16 * 8 * 128 * 2 = 65,536` byte = **64 KiB**。这是**一层、一个 block 的 byte 数**。还要乘以层数（32）才得到一个 block 在整模型上占的总 byte：64 KiB × 32 = **2 MiB / block**。

### 5.3.3 `get_num_blocks` — 双重整除

打开 `implementation/kv_cache_spec.py:L105-L112`：

```python
# implementation/kv_cache_spec.py:L105-L112
def get_num_blocks(available_memory: int, num_layers: int, page_size: int) -> int:
    return max(int(available_memory // page_size // num_layers), 0)
```

两次整除——`// page_size` 然后 `// num_layers`。

为什么不是 `available_memory // (page_size * num_layers)`？数学上等价；写成两次整除有两个工程理由：

1. **可读性**：每一次 `//` 都说一件事。第一次整除告诉你"按 page_size 切，能切多少 page"；第二次整除告诉你"这些 page 平摊到多少层，每层能分多少 block"。读者扫一眼就知道每一项是什么。
2. **多 group 路由**：vLLM 真实代码里这个函数（`kv_cache_utils.py:L930-L947`）会被异构层模型（mamba 混 attention）多次调用——每个 group 的 page_size 不同但 num_layers 也不同，分开传两个参数比合并预乘一次更易于多 group 适配。

代入 demo 数字：available_kv_cache = 73,656,750,243 byte（73.6 GiB - 5.0 GiB - 0.25 GiB = 68.65 GiB），page_size = 65,536，num_layers = 32：

$$
\mathrm{num\_blocks} = \left\lfloor \frac{68.65 \,\mathrm{GiB}}{64 \,\mathrm{KiB} \cdot 32} \right\rfloor = 35{,}148
$$

`tests/test_memory_layout.py::test_demo_num_gpu_blocks_reproduces` 对这个数字做精确断言（不是"在某个范围内"，而是 `== 35148`）。

### 5.3.4 block_size 的甜蜜点扫描

demo 还跑了一个 block_size 扫描（`implementation/demo.py:L146-L155`）：

```
[5] page_size sensitivity (block_size sweep)
    block_size=  8: page= 32.0 KiB, num_blocks= 70297, wasted=0.60 MiB
    block_size= 16: page= 64.0 KiB, num_blocks= 35148, wasted=1.60 MiB
    block_size= 32: page=128.0 KiB, num_blocks= 17574, wasted=1.60 MiB
    block_size= 64: page=256.0 KiB, num_blocks=  8787, wasted=1.60 MiB
```

**block 数和 block_size 严格反比**：每翻一倍 block_size，num_blocks 砍半。这不是巧合——`num_blocks = available / (page * layers)`，page 与 block_size 线性，所以 num_blocks 与 block_size 反比例。

第 12-13 章讨论 prefix cache 时会回到这个表：block_size 太大，前缀必须**完全相同**才能复用——不利命中率；太小，indexing 开销线性放大。vLLM 默认 16 是经验值，不是理论最优。

---

## 5.4 KV block 的元数据：`KVCacheBlock` + 双向链表

到这里，"73.6 GiB 变成 35,148 个 block" 这条路走完了。但 vLLM 还没分配真正的 byte——它只决定了 block 的**数量**。真正的 GPU tensor 是这一行：

```python
# 概念示意：每层一个 flat int8 tensor
torch.zeros(num_blocks * page_size_bytes, dtype=torch.int8, device='cuda')
```

每个 block 在这片 byte 上有一个 **slot index**（block_id）。但 byte 自己不会管自己——你需要一个**元数据**结构跟踪每个 block 是谁占着、什么时候被释放、是不是命中了 prefix cache。这就是 `KVCacheBlock`。

### 5.4.1 `KVCacheBlock` — 一个 block 的全部"档案"

打开 `instances/vllm/artifacts/05-memory-management/implementation/kv_cache_block.py:L28-L59`：

```python
# implementation/kv_cache_block.py:L28-L59
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: Optional[bytes] = None

    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    is_null: bool = False

    @property
    def block_hash(self) -> Optional[bytes]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, value: bytes) -> None:
        assert self._block_hash is None, "Block already has a hash; this is a bug."
        self._block_hash = value
```

5 个字段一组属性。每一个都很重要：

- **block_id**：这个 block 在 GPU 大 tensor 里的 slot index。模型 forward 时拿这个 index 去 attention kernel 里查（第 3 章讲过 `block_table[logical] → physical_block_id`）。
- **ref_cnt**：有多少 running request 在用这个 block。0 表示空闲；>= 1 表示在用。
- **_block_hash**：当 block 被填满（16 个 token 都写完）时，按它存的 token 内容哈希一下，这个 hash 是它的 prefix cache key。关键性质：**ref_cnt 归 0 后 hash 不立即清掉**——下个新请求如果 prefix 哈希和它一样，可以直接复用，省一次 prefill 计算。
- **prev_free_block / next_free_block**：双向链表指针。**只在 ref_cnt == 0 时有意义**（block 在空闲队列里时）。
- **is_null**：5.4.3 解释。

setter 上的 assert（`_block_hash is None` 才能写）是个看起来废话、实则保命的检查：vLLM 假设每个 block 一辈子最多被 hash 一次（写满那一刻），重复 hash 是 bug。

### 5.4.2 `FreeKVCacheBlockQueue` — 双向链表，不是 deque

这是本章最重要的数据结构选型。打开 `implementation/kv_cache_block.py:L62-L155`：

```python
# implementation/kv_cache_block.py:L81-L102
class FreeKVCacheBlockQueue:
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # 把传入的 blocks 串成双向链表
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # fake head/tail —— 消除 null-pointer 边界讨论
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
```

**fake head / tail 技巧**：真实 block 永远夹在两个 sentinel 节点之间。`popleft()` 永远是"head 的 next"，`append()` 永远是"tail 的 prev"——不需要在每个操作里写"如果队列空就……否则……"的边界讨论。这是教科书级别的 doubly-linked list 实现。

**LRU 顺序**：head 是最早进来的（最老空闲）= 下一个被分出去的。tail 是最近释放的。`get_new_blocks` 从 head 拿，`free_blocks` 往 tail 推。这个顺序很重要——5.6 节讲为什么。

#### 为什么不是 `collections.deque`？

`deque` 也支持 O(1) 的 popleft 和 append。如果只是这两个操作，deque 完美。但 vLLM 还需要**第三个操作**：

```python
# implementation/kv_cache_block.py:L129-L136
def remove(self, block: KVCacheBlock) -> None:
    """O(1) removal from the middle. Called when a freed block is re-touched."""
    if block.prev_free_block is None or block.next_free_block is None:
        raise RuntimeError(f"remove() called on an invalid block: {block}")
    block.prev_free_block.next_free_block = block.next_free_block
    block.next_free_block.prev_free_block = block.prev_free_block
    block.prev_free_block = block.next_free_block = None
    self.num_free_blocks -= 1
```

**从中间 O(1) 删除任意节点**。`deque` 做不到这个——`deque.remove(x)` 是 O(n)。

为什么需要这个？看 5.6 的 prefix cache touch 路径：当一个 already-freed-but-still-cached block 被新请求 prefix-hit，它必须从 free queue 里**消失**（变回 in-use 状态）——但它不在 head 也不在 tail，可能在中间任意位置。如果用 deque，这个 `remove(b)` 是 O(n)，意味着 prefix cache fast path 是 O(n)。整个 prefix-cache 的设计就这么一个数据结构选型给毁了。

vLLM 的工程师知道这点，所以宁愿手写 30 多行 doubly-linked list 也不用现成的 deque。这个细节是面试题里"为什么不用 deque"的标准答案。

### 5.4.3 null block：被永久占用的 block 0

打开 `implementation/block_pool.py:L62-L67`：

```python
# implementation/block_pool.py:L62-L67
# The null block: a placeholder for sliding-window padding. Block 0
# is reserved and pinned (popped out of the free queue at startup).
# REFERENCE: instances/vllm/source/vllm/v1/core/block_pool.py:L173-L177
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True
```

**`BlockPool.__init__` 的最后一行就是把 block 0 popleft 掉**。它从此再也回不到 free queue。这是一个看起来浪费 64 KiB × 32 = 2 MiB 的设计——为什么要这么干？

答案在 sliding-window attention（第 1 章末尾、第 3 章 GQA 部分都提过）。Sliding window 限制每个 token 只看最近 K 个 token，超出窗口的 block 在 block_table 里要被填上一个**占位 ID**——既不能算 attention，也不能塞别人在用的 block。这个占位 ID 必须是合法 `int` 类型——不能是 `None`，否则 block_table 张量的 dtype 没法定义。

解决：**预留 block 0 当占位符**。它永远在 GPU 上、永远是合法 block_id、但永远没人在它上面读写。代价：损失 1 个 block 的空间（H100 上 0.003%）。收益：block_table 是 `int32` 数组，CUDA kernel 不用做 nullable 检查。这是一笔划算的交易。

实际效果：第一个用户请求的第一个 block 拿到的 `block_id == 1`，不是 0。`tests/test_block_pool.py::test_first_user_block_is_id_1` 就是断言这个行为。`get_usage()` 也对应做了减法：`total = num_gpu_blocks - 1`（`block_pool.py:L156`）——把 null block 从分母里扣掉，否则一个空池子的 usage 也是 1/16 ≠ 0。

---

## 5.5 BlockPool：把 block 借出去和收回来

`KVCacheBlock` 是数据，`FreeKVCacheBlockQueue` 是容器，`BlockPool` 才是**对外的接口**。scheduler 通过 `block_pool.get_new_blocks(N)` 和 `block_pool.free_blocks(blocks)` 管理 KV cache，整个 5.5 就讲这两个函数和它们的 prefix cache 路径。

### 5.5.1 `get_new_blocks` — 借

打开 `implementation/block_pool.py:L70-L86`：

```python
# implementation/block_pool.py:L70-L86
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

    ret = self.free_block_queue.popleft_n(num_blocks)

    for block in ret:
        if self.enable_caching:
            self._maybe_evict_cached_block(block)
        assert block.ref_cnt == 0
        block.ref_cnt += 1
    return ret
```

三步：
1. **容量检查**：不够就 raise。注意这是 raise 不是返回 None——和第 4 章的 `allocate_slots → None` 不一样，因为 `BlockPool` 是更底层的原语。`KVCacheManager.allocate_slots` 在调用 `get_new_blocks` 前**自己**做了容量检查，raise 在这里就是断言契约被违反。
2. **从 LRU head 弹出 N 个**：`popleft_n` 是 5.4.2 提到的双向链表的 bulk popleft。
3. **处理每个被弹出的 block**：如果开启了 caching，调用 `_maybe_evict_cached_block` 把它从 prefix cache hash 表里清出去（这个 block 要被新请求接管了，旧 hash 必须作废）；然后 `ref_cnt += 1`。

`_maybe_evict_cached_block`（`block_pool.py:L88-L100`）的逻辑：拿 block 的 hash，从 `cached_block_hash_to_block` 字典里 pop 出去，调用 `block.reset_hash()`。这是"LRU 驱逐"的物理动作。

### 5.5.2 `free_blocks` — 还

打开 `implementation/block_pool.py:L114-L134`：

```python
# implementation/block_pool.py:L115-L134
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks = list(ordered_blocks)
    for block in blocks:
        block.ref_cnt -= 1
    # Only blocks that hit ref_cnt 0 (and aren't the null block) go back.
    self.free_block_queue.append_n(
        b for b in blocks if b.ref_cnt == 0 and not b.is_null
    )
```

两步：
1. **每个 block 的 ref_cnt 减 1**：如果还有别的 request 在用（COW 共享），block 不会立刻进 free queue。
2. **ref_cnt 归 0 的 block 进 free queue 的 tail**：tail 即 LRU 的"最近释放"端。这个顺序保证下次 `popleft` 拿的是最早释放的——LRU 语义。

**关键性质**：block 的 `_block_hash` **没有被清掉**。它还坐在 `cached_block_hash_to_block` 字典里。它现在是"freed but cached"——既在 free queue 又在 prefix cache 索引里。这是 prefix cache 复用的物理基础。

`ordered_blocks` 是 reverse-allocation-order 传入的（caller 在 `kv_cache_manager.free` 里 `reversed(blocks)`，`vllm/v1/core/kv_cache_manager.py:L420-L427`）。这样最新分配的 block 最先回到 LRU 的"最近释放"端——使后到达的请求复用顺序更接近时间局部性，prefix cache 命中率更高。这是软件工程里"看似无关的细节"实际上是核心优化的典型例子。

### 5.5.3 `touch` — 复用

打开 `implementation/block_pool.py:L102-L112`：

```python
# implementation/block_pool.py:L103-L112
def touch(self, blocks: Iterable[KVCacheBlock]) -> None:
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

这是 prefix cache hit 的关键路径。当一个新请求计算它的 prefix hash，发现某个 hash 已经在 `cached_block_hash_to_block` 里，新请求**不重新分配 block**——它调 `touch` 把现有 block 接管：
1. 如果该 block 当前在 free queue（ref_cnt == 0）：从队列**中间**移除它（5.4.2 的 O(1) `remove`）。
2. ref_cnt += 1（接管了）。

如果 ref_cnt 当时 > 0，说明**别的 request 已经在用这个 block**（同一个 prefix 被多个并发请求复用，不少见）——那就只是 ref_cnt += 1，本质上是 copy-on-write 的引用计数。

### 5.5.4 走一遍 demo 第 [3] 段

demo 用 16 个 block 演示了一个完整的生命周期（`demo.py:L96-L131`）。复刻输出：

```
[3] BlockPool LRU + prefix cache (mini scenario, 16 blocks)
    Initial: free=15/15, usage=0.0%, null_block_id=0
    A allocates 4: ids=[1, 2, 3, 4], free=11, usage=26.7%
    A caches blocks 0,1 under hashes h1,h2: cache_size=2
    A frees 4: free=15, cache_still_holds=2
    B hits prefix h1 via touch(): block=1, ref_cnt=1, free=14
    C allocates 10: free=4, cache_after_evictions=2
```

逐步追踪：

- **Initial**：16 个 block 构造完成，block 0 是 null 被 popleft；free=15，ID 范围 [1, 15]，usage=0%（分母 = 16 - 1 = 15）。
- **A allocates 4**：从 head 弹 4 个 LRU block（按构造顺序就是 ID 1, 2, 3, 4）。free=11，A.ref_cnt 都是 1。
- **A caches blocks 0, 1**：A 把它前两个 block（block_id=1, 2）的 hash 注册到 prefix cache 字典。注意这里"blocks 0, 1"是 A 内部索引（a_blocks[0]、a_blocks[1]），物理 block_id 是 1 和 2。
- **A frees 4**：A 完成。每个 block.ref_cnt -= 1（从 1 归 0）。4 个都进 free queue 的 tail。free=15。**关键：cache_still_holds=2**——hash 字典没被动，物理 block 1 和 2 是 "freed but cached"。
- **B hits prefix h1 via touch(`)**：B 算它的第一个 block hash 等于 h1，从字典 lookup 拿到物理 block_id=1。`touch` 发现 block 1 在 free queue 里（ref_cnt == 0），用 O(1) middle-remove 把它弹出来；ref_cnt = 1。free 从 15 减到 14。**B 跳过了一次 prefill 算 K/V**——这就是 prefix cache 的全部价值。
- **C allocates 10**：C 要 10 个 block。free queue 里现在是 [3, 4, 5, ..., 15]（block 1 被 B 拿走了，block 2 还在 cache 里），共 14 个。popleft 10 个；前面那个 cached block 2 一旦轮到它被 popleft，`_maybe_evict_cached_block` 把它从 hash 字典里清掉（因为它的物理身份要给新请求接管）。free=4。**cache_after_evictions=2**——这里 demo 输出说"还剩 2 个"，因为 h1 对应的 block 1 还在 B 手上（受 ref 保护），h2 对应的 block 2 已被 evict——但等等，输出是 2 不是 1。

让我重看一遍。h1（block 1）被 B touch 走了，`touch` 没动 hash 字典。h2（block 2）是被 popleft 时 evict 掉的吗？看 demo `c_blocks = pool.get_new_blocks(10)`——队列是 [3,4,...,15]+[1...?  实际上 A 的 4 个 block 都被 free 进 tail，顺序是 [5,6,7,...,15] + [被 free 的顺序，reversed]。这里 demo 没有传 reversed 顺序，就是按 a_blocks 原序 free。所以 free queue 在 free 之后是 [5, 6, ..., 15, 1, 2, 3, 4]（前 11 个原本就在的，加 4 个新进来的）。然后 B touch 把 1 移走，剩 [5, ..., 15, 2, 3, 4]（13 个）。然后 C 拿 10 个：popleft 10 个就是 [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]——**block 2 还在 free queue 里没被 popleft 到**！所以 cache 仍然有 h1 (block 1, 在 B 手里) 和 h2 (block 2, 在 free queue 里)，size=2。

啊对——这就是 "caching survives free" 的真实意义：**只要 cached block 还没被新请求 popleft 掉，它就一直在那等命中**。这就是 prefix cache 的价值——只要有空间，被释放的 prefix 永远站着。

---

## 5.6 抢占的两条路：recompute vs swap

第 4 章我们看到 `kv_cache_manager.allocate_slots → None` 触发 preempt。但 preempt 时具体做什么？vLLM v1 的答案是 **recompute**——丢掉 KV、把 `num_computed_tokens` 归零、prefill 重做。还有另一条历史上的路：**swap**——把 KV 拷到 CPU，腾出 GPU，恢复时再拷回来。两条路都成立。我们这一节的目的是把"为什么 v1 选了 recompute"讲透。

### 5.6.1 KV bytes 的精确测量

打开 `instances/vllm/artifacts/05-memory-management/implementation/recompute.py:L62-L76`：

```python
# implementation/recompute.py:L62-L76
@property
def kv_bytes(self) -> int:
    return (
        2  # K and V
        * self.num_layers
        * self.num_kv_heads
        * self.head_size
        * self.prompt_tokens
        * self.dtype_bytes
    )
```

跟 5.3.2 的 page_size 公式一模一样的形状，只是把 `block_size` 换成了 `prompt_tokens`——因为现在算的是**整个请求**的 KV，不是一个 block。代入 demo 的 8K 请求（32 layers, 8 KV heads, head_size 128, fp16）：

$$
\mathrm{KV} = 2 \cdot 32 \cdot 8 \cdot 128 \cdot 8192 \cdot 2 = 1{,}073{,}741{,}824 \mathrm{B} = 1.0 \, \mathrm{GiB}
$$

**整整 1 GiB**。这是一个看起来不大但意义重大的数字——它说明 long-context 推理的内存压力主要在 KV 而不在权重。Llama-3.2-1B 权重 2.4 GiB，一个 8K 请求的 KV 1 GiB——并发 3 个 8K 就和模型权重一样大了。

### 5.6.2 两条路的 latency

```python
# implementation/recompute.py:L88-L93
@property
def recompute_seconds(self) -> float:
    return self.prompt_tokens / self.prefill_throughput_tokens_per_sec

# implementation/recompute.py:L100-L110
@property
def swap_bytes_moved(self) -> int:
    return 2 * self.kv_bytes  # round-trip: out + in

@property
def swap_seconds(self) -> float:
    return self.swap_bytes_moved / self.pcie_bandwidth_bytes_per_sec
```

代入 demo 的默认参数（prefill 50K tok/s，PCIe Gen4 x16 = 32 GB/s）：

| 项 | 计算 | 结果 |
|----|------|------|
| recompute | 8192 / 50000 | **163.84 ms** |
| swap one-way | 1 GiB / 32 GB/s | 31.25 ms |
| swap round-trip | 2 × 31.25 | **62.50 ms** |

**recompute 比 swap 慢 2.62 倍**。看起来应该选 swap。但 vLLM v1 选了 recompute——为什么？

### 5.6.3 latency 不是唯一的轴

打开 `implementation/recompute.py:L13-L23` —— trade-off 矩阵注释：

```
                       Recompute              Swap-to-CPU
    GPU ↔ CPU PCIe     0 bytes               O(KV bytes)
    Compute redo       O(prompt_len)          0
    Bandwidth need     0                     PCIe ~32 GB/s
    Latency cost       prefill time          KV / PCIe bandwidth
    Code complexity    1 path                 2 paths + cudaMemcpyAsync
    Determinism        same numerical result  bit-identical replay
    OOM safety         always works           fails if CPU also full
```

100 ms 的 latency 差距是真实的，但它换来了：

1. **Code complexity 减半**：swap 需要两条独立路径（preempt 时拷出，resume 时拷回）+ 异步 cudaMemcpy 配合；recompute 复用 prefill 路径，零新代码。
2. **OOM safety**：CPU 内存如果也满了，swap 直接失败——你的引擎在 OOM 边缘多了一个失败模式。recompute 的"没有 CPU 依赖"意味着它**永远可用**。
3. **Determinism**：recompute 重新算 prefill，结果和首次运行**字节相同**（同样的 input、同样的权重、同样的 kernel）。swap 实际上也是——但前提是 cudaMemcpy 路径不引入精度损失（一般不会，但需要保证）。
4. **CPU memory budget tracking**：swap 要在 CPU 端跟踪一个独立的 "swapped KV pool"——如果 CPU 也吃紧呢？又是一个独立的 OOM/管理问题。recompute 没这块。

所以 vLLM v1 的设计选择是：**用 100 ms 的延迟买 4 类复杂度的消除。** 这是工程上"可读、可维护、可推理"压倒"原始性能"的典型案例。`scheduler.py:L952-L972` 的 `_preempt_request` 逻辑因此可以是 5 行：free blocks、reset computed_tokens、prepend 到 waiting——干净利落。

⚠️ 注意 `vllm/v1/kv_offload/` 这个子系统**仍然**做 GPU↔CPU KV 传输（`kv_offload/cpu/gpu_worker.py:L319` 有 `swap_blocks_batch`），但它的目的是 **prefix cache offload**（把已完成请求的 prefix 缓存到 CPU/disk 供未来命中），**不是 preempt 的备选路径**。两件事，不要混淆。第 12-13 章讲 prefix cache 时会详细回到 kv_offload。

### 5.6.4 Kwon et al. 2023 的"swap"为什么是 v0 的事

如果你读过 vLLM 的原始论文（Kwon et al., SOSP 2023），里面提到 swap-to-CPU 是 PagedAttention 的一项优势。那是 vLLM v0 的设计。v1（2024 年起的重写）撤销了这个决定，理由就是上面 5.6.3 那张表。一句话：**v0 用 swap 做 preempt，v1 用 recompute；v1 不用 swap 做 preempt，但仍然用 swap 做 prefix cache offload。** 读论文时认清版本号，避免混淆。

---

## 5.7 我们的实现 vs vLLM 源码：1:1 对照表

| 我们的代码 | vLLM 源码 | 我们改了什么 | 为什么 |
|-----------|----------|-------------|--------|
| `MemorySnapshot` (`mem_snapshot.py:L29-L86`) | `vllm/utils/mem_utils.py:L70-L157` | `auto_measure` 默认 False；`measure()` 是 stub | demo 没真 GPU |
| `MemorySnapshot.__sub__` (`mem_snapshot.py:L75-L86`) | `mem_utils.py:L128-L145` | 一字不差 | diff 数学 |
| `MemoryProfilingResult` (`mem_snapshot.py:L90-L102`) | `mem_utils.py:L160-L187` | 字段一致，去掉 `__post_init__` device 路由 | caller 手填快照 |
| `memory_profiling` ctx mgr (`mem_snapshot.py:L106-L148`) | `mem_utils.py:L190-L275` | 去掉 `gc.collect` / `empty_cache` | pedagogical |
| `KVCacheSpec` 基类 (`kv_cache_spec.py:L23-L38`) | `kv_cache_interface.py:L80-L127` | 一字不差 | — |
| `AttentionSpec` (`kv_cache_spec.py:L41-L73`) | `kv_cache_interface.py:L129-L170` | 去掉 nvfp4 + kv-quant 分支 | 第 26 章再讲量化 |
| `AttentionSpec.real_page_size_bytes` | `kv_cache_interface.py:L153-L170` | fp16/bf16 路径一字不差 | 核心公式 |
| `FullAttentionSpec` (`kv_cache_spec.py:L77-L101`) | `kv_cache_interface.py:L173-L205` | 去掉 DCP/PCP context-parallel 因子 | 单 GPU |
| `get_num_blocks` (`kv_cache_spec.py:L105-L112`) | `kv_cache_utils.py:L930-L947` | 去掉 `may_override_num_blocks` config hook | 数学一致 |
| `KVCacheBlock` (`kv_cache_block.py:L28-L59`) | `kv_cache_utils.py:L113-L159` | 一字不差 | — |
| `FreeKVCacheBlockQueue` (`kv_cache_block.py:L62-L155`) | `kv_cache_utils.py:L162-L370` | `popleft_n` 用循环（vLLM 优化为单次链表切断） | 清晰 > 极限性能 |
| `BlockPool` (`block_pool.py:L31-L186`) | `block_pool.py:L130-L510` | `dict` 替代 `BlockHashToBlockMap`；去掉 KV events | demo 不需要 distributed event 发布 |
| `BlockPool.get_new_blocks` | `block_pool.py:L322-L352` | 去掉 `metrics_collector` 调用 | optional path |
| `BlockPool.touch` | `block_pool.py:L391-L406` | 去掉 `metrics_collector.on_block_accessed` | optional path |
| `BlockPool.free_blocks` | `block_pool.py:L408-L422` | 一字不差 | — |
| `BlockPool.evict_blocks` | `block_pool.py:L424-L441` | 一字不差 | — |
| `BlockPool.get_usage` | `block_pool.py:L486-L497` | 一字不差 | — |
| `determine_available_memory` (`memory_layout.py:L87-L144`) | `gpu_worker.py:L352-L505` | 接受预构建 `MemorySnapshot` / `Result`；cudagraph 无条件减 | 没真 GPU；env-var 派发推到 Ch20 |
| `MemoryLayout` (`memory_layout.py:L37-L83`) | — | 教学新增 | 把 worker 散字段聚合便于叙事 |
| `estimate_max_concurrency` (`memory_layout.py:L147-L163`) | `kv_cache_utils.py:L872-L890` | 数学一致 | — |
| `PreemptionScenario` (`recompute.py:L42-L128`) | — | 解析模型 | 让读者插入自己的数 |

**故意砍掉的内容**（每项都对应一个后续章节）：

- `BlockHashToBlockMap` 的"多 block 共享一个 hash" 分支（`block_pool.py:L34-L127`）——12-13 章 prefix cache 深入。
- `KVCacheManager` 完整签名（9 参数 of `allocate_slots`）和 `KVCacheCoordinator` 多 group 路由（`kv_cache_manager.py:L107`+）——12-13 章。
- nvfp4 packed layout、per-token-head kv-quant scales（`kv_cache_interface.py:L143-L163`）——第 26 章 quantization。
- DCP/PCP context-parallel sharding factor（`kv_cache_interface.py:L198-L203`）——第 11 章 DCP/PCP。
- KV events、metrics collector（`block_pool.py` 多处）——监控与可观测性，不在书的 scope。
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` env-var 派发（`gpu_worker.py:L417-L423`）——第 20 章 ModelRunner。
- v0 的 swap-to-CPU 抢占路径——v1 撤销，本章 5.6 解释。

---

## 验证

### 跑测试

```bash
cd instances/vllm/artifacts/05-memory-management
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

预期输出：

```
74 passed in 0.16s
```

74 个测试覆盖 7 个模块：

| 模块 | 测试数 | 验证什么 |
|------|------:|---------|
| `test_mem_snapshot.py` | 7 | 字段相减、`auto_measure=False` 不调 measure、worked-example 算术 |
| `test_kv_cache_spec.py` | 11 | page_size 公式、`get_num_blocks` 双重整除、`max_memory_usage_bytes` 向上取整 |
| `test_kv_cache_block.py` | 16 | `popleft_n(0)` no-op、`remove` 头/中/尾 全 O(1)、append-then-remove 重链 |
| `test_block_pool.py` | 20 | null block 隔离、`get_new_blocks` LRU、prefix cache 跨 free 存活、`touch` 中间删除 |
| `test_memory_layout.py` | 8 | demo 的 35,148 精确复现、负值 KV 截零、block-size sweep 精确匹配 |
| `test_recompute.py` | 9 | KV bytes 公式、`recompute_is_faster` 在 8K 是 False（K06）、PCIe / prefill 比例 |
| `test_integration.py` | 3 | 端到端 demo workflow、`AttentionSpec` 流入 `MemoryLayout`、Ch04 跨章节 import |

### 跑 lint

```bash
python3 scripts/lint_formulas.py instances/vllm/artifacts/05-memory-management/narrative/chapter.md
python3 scripts/lint_source_grounding.py instances/vllm/artifacts/05-memory-management/
```

两个都必须 PASS。

### 跑 demo

```bash
python3 -m instances.vllm.artifacts.05-memory-management.implementation.demo
```

对照 5.1.2 / 5.3.4 / 5.5.4 节的输出。结尾应该看到 `block_size= 16: page= 64.0 KiB, num_blocks= 35148`、`275 concurrent requests`、`Recompute (no IO) : 163.8 ms / Swap round-trip : 62.5 ms`。

---

## 总结

1. **80 GiB 不是给你的，73.6 GiB 也不全是 KV cache 的。** 8% 安全余量吸收四类 NVML 看不见的开销；剩下的 73.6 GiB 还要扣权重、峰值激活、CUDA graph、non-torch 开销。Llama-3.2-1B 在 H100 上最终 **68.65 GiB 给 KV cache，35,148 个 block，275 个并发**——这些是 demo 的精确数字，不是估计。

2. **三类显存模型来自测量方式的真实分裂。** NVML 看 `total - free`，torch 看自己的 `memory_reserved`，两者相减就是 non-torch 这一类。它对 PyTorch 透明，但对 KV cache 预算可见——`empty_cache()` 救不了你。`memory_profiling` 用三张快照定位激活峰值和 NCCL 增量。

3. **page_size = 2 × block_size × num_kv_heads × head_size × dtype_bytes。** 一层、一个 block 的 byte 数，五个数字相乘。`get_num_blocks` 用双重整除分配到层——不是为了数学，是为了未来异构层模型能多 group 路由。

4. **空块队列必须能 O(1) 中间删除。** `deque` 不行——它的 `remove` 是 O(n)。手写双向链表 30 行代码、用 fake head/tail 消除边界讨论；prefix cache 的 `touch` 路径靠这个 O(1) 移除支撑整个性能模型。null block (block 0) 是 sliding-window 的占位符，永久占用、永远是合法 int，换 block_table 不需要 nullable。

5. **Free 不删 cache hash，touch 才接管。** 一个 block 释放后仍坐在 prefix cache 字典里，等下个请求 prefix-hit 时被 `touch()` 拉回——这个"freed but cached"中间态是 prefix cache 复用的物理基础。LRU 顺序保证最早释放的最早被新请求接管。

6. **vLLM v1 用 recompute 抢占，不是 swap。** 8K 请求 KV = 1 GiB，recompute 164 ms vs swap 62 ms——recompute 慢 2.6 倍。但选 recompute 换来：单代码路径、零 CPU 内存依赖、bit-determinism、无 CPU-also-OOM 失败模式。100 ms 买四类复杂度消除，划算。`kv_offload/` 不是 preempt 的备选，它是 prefix cache offload，第 12-13 章讲。

### 下章预告

第 4-5 章把"调度怎么决定"和"内存怎么算"都讲清楚了。但 vLLM 还有一类资源争夺：**when GPU is free but the policy says wait, or when the policy says go but GPU is full**——更细的调度策略层。第 6 章 `Scheduling Policies` 把 FCFS、priority、chunked-prefill、long_prefill_token_threshold 这些刻意做粗的细节抠开，给你完整的策略选项空间。

---

← 第 4 章：Continuous Batching | 第 6 章：Scheduling Policies →
