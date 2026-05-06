# 第8章：Tensor Parallelism — 没有 `class TensorParallel` 的张量并行

> 本章涉及的 vLLM 源码：
> - `instances/vllm/source/vllm/distributed/parallel_state.py:L290-L1136`（`class GroupCoordinator` — 包住 `torch.distributed.ProcessGroup` 的一进程一 rank wrapper）+ `L502-L530`（`GroupCoordinator.all_reduce` 主入口，`world_size==1` bypass 在 L518-L519）+ `L1229-L1235`（`get_tp_group()`）+ `L1494-L1599`（`initialize_model_parallel` 把 TP/PP/DP rank 切成 device-mesh）+ `L1586`（`_TP = init_model_parallel_group(...)` 模块级单例）+ `L1837-L1845`（`get_tensor_model_parallel_world_size` / `get_tensor_model_parallel_rank` 两个所有 parallel layer 都调的 helper）
> - `instances/vllm/source/vllm/distributed/communication_op.py:L12-L40`（`tensor_model_parallel_all_reduce` / `_all_gather` / `_reduce_scatter` — 一行 wrapper，全部走 `get_tp_group().<op>`）
> - `instances/vllm/source/vllm/distributed/utils.py:L60-L66`（`divide` — 强制断言整除的 universal 除数）+ `L67-L92`（`split_tensor_along_last_dim` — RowParallelLinear `input_is_parallel=False` 路径用）
> - `instances/vllm/source/vllm/model_executor/layers/linear.py:L410-L608`（`class ColumnParallelLinear`：`weight_loader` 沿 OUTPUT dim narrow，`gather_output=True` 时调 `tensor_model_parallel_all_gather`）+ `L609-L976`（`class MergedColumnParallelLinear`：fuse N 个 output projection；`L767-L820` 的 per-segment narrow 循环是关键）+ `L977-L1393`（`class QKVParallelLinear`：head 维 column-parallel；`L1029-L1043` 的 GQA replication 分支）+ `L1394-L1577`（`class RowParallelLinear`：`weight_loader` 沿 INPUT dim narrow，`L1486-L1487` bias 是 FULL output_size，`L1557-L1559` bias 只在 rank 0 加，`L1562-L1563` 一次 all-reduce）
> - `instances/vllm/source/vllm/model_executor/models/llama.py:L81-L121`（`class LlamaMLP`：`gate_up_proj = MergedColumnParallelLinear` + `down_proj = RowParallelLinear` + `act_fn = SiluAndMul`）+ `L124-L233`（`class LlamaAttention`：`qkv_proj = QKVParallelLinear` + `o_proj = RowParallelLinear`）+ `L228-L229`（`qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)`）
> - 对照实现 `instances/vllm/artifacts/08-tensor-parallelism/implementation/`：`tp_math.py`、`comm_primitives.py`、`column_parallel.py`、`row_parallel.py`、`qkv_parallel.py`、`mlp_block.py`、`demo.py`
>
> 本章源码 commit：`98661fe`。第 7 章用"vLLM 没有 radix tree"开篇；第 8 章用"vLLM 没有 `class TensorParallel`"开篇——两个都是同一种课题：**outline 写的是教科书框架，源码呈现的是组合**。第 7 章用 chain hash + flat dict 替代 radix tree；第 8 章用 5 个文件 + 1 个模块级单例替代一个 TP framework class。

---

## 这章要讲什么？

打开 `instances/vllm/source/vllm/distributed/communication_op.py:L12-L14`，整个 vLLM 的 tensor model parallel all-reduce 入口长这样：

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)
```

**两行函数**。这就是 TP 的"用户 API"。再打开 `parallel_state.py:L502-L530` 的 `GroupCoordinator.all_reduce`，第 518-519 行有一个 `if self.world_size == 1: return input_` 的 bypass——也就是说 tp_size=1 时这个 all-reduce 根本不发生，就把 tensor 原地返回。

整个 vLLM v1 源码树里**搜不到一个 `class TensorParallel`**：

```
$ grep -rE "class TensorParallel|class TPCoordinator|class ParallelReplica" \
    instances/vllm/source/vllm/
(zero matches)
```

——但是网上文献和 issue 讨论里"vLLM 怎么实现 TP" 这种说法又特别常见。这就是本章要解决的第一个语言陷阱：**TP 在 vLLM 里不是一个 class，而是 5 个文件的协同**。

vLLM 的 TP 是**组合模式**：

- `parallel_state.py` 提供 group + collective 抽象（`GroupCoordinator`，模块单例 `_TP`）。
- `communication_op.py` 是一行 wrapper，把 `tensor_model_parallel_all_reduce` 等 API 翻译成 `get_tp_group().all_reduce()`。
- `linear.py` 提供 4 个 TP linear 类：`ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear`、`MergedColumnParallelLinear`——这是 TP 的"具身"。
- `vocab_parallel_embedding.py` 提供 vocab-parallel embedding（同 column-parallel 模式，沿 vocab 维切）。
- `models/llama.py` 是真实使用现场：`LlamaMLP` 用 `MergedColumnParallelLinear` + `RowParallelLinear`，`LlamaAttention` 用 `QKVParallelLinear` + `RowParallelLinear`——就这样把 TP 拼起来了。

所以**没有"vLLM 怎么实现 Megatron framework"这个问题**——vLLM 不是 Megatron 的 framework copy。vLLM 算法上**是** Megatron-style（column-parallel → row-parallel pair，每 pair 一次 all-reduce），但实现上是直接构建在 `torch.distributed` 之上的干净 Python，没有任何 Megatron 框架代码的 dependency。这是本章的 §X 重构。

学完这章你能：

- 在白板上推导 column-parallel 和 row-parallel 的数学等价性（demo §1 给出 `col_tp{2,4,8}_max_abs_diff = 0` 的存在证明）。
- 用 ring all-reduce 的 α-β 公式

$$
T = 2 \cdot \frac{P-1}{P} \cdot \left( \alpha + \frac{S}{P} \cdot \beta \right)
$$

  解释为什么 **TP=2 在 1 KB payload 上比 P=8 快**（demo §2 NVLink 表格：P=8 是 P=2 的 1.75× **慢**）——这是 Trap-A 的反直觉一半。
- 解释为什么一个 Llama transformer block 有 **2 次** all-reduce（attn 的 o_proj 一次 + MLP 的 down_proj 一次）而**不是 1 次**——一次 all-reduce per col→row pair，不是 per block。
- 用 `divide` 的整除断言契约推 GQA × TP 的 KV 节省 cap：Llama-3-70B 8 KV heads，tp=8 节省 8×，tp=16 仍然只节省 8×（K cap at total_num_kv_heads）。
- 区分 **5 个语言陷阱**：TP=2 ≠ 2× 吞吐？✗（demo §2 实测 1.75× 慢 / 2.17× 快两端都不对得上 2×）；QKV column-parallel 是按 feature 列切？✗（按 head 切）；TP 必然减半 KV？✗（GQA 情况会被 KV head 数 cap 住）；MLP TP 需要 all-gather + all-reduce？✗（**一次** all-reduce）；RowParallelLinear input 自动切？✗（默认 `input_is_parallel=True`，假设上游已切）。

接下来 6 节按 outline 走，但 §8.2 已经从 "Megatron-style TP 在 vLLM 中的实现" 重构成 "vLLM 没有 `class TensorParallel`：5 文件协同"——源码里就没有这个 class，硬讲就是失真，跟 Ch07 §7.2 的"为什么 vLLM 选了 dict 不选 radix tree"一脉相承。

---

## 8.1 数学：column-parallel 和 row-parallel 的等价性证明

### 8.1.1 打开 column-parallel 入口

源码定位：`instances/vllm/source/vllm/model_executor/layers/linear.py:L410-L608`，`class ColumnParallelLinear`：

```python
# vllm/model_executor/layers/linear.py:L410-L460 (节选)
class ColumnParallelLinear(LinearBase):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(
        self, input_size: int, output_size: int, bias: bool = True,
        gather_output: bool = False, ...,
    ):
        ...
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.output_size_per_partition = divide(self.output_size, self.tp_size)
```

三件事一起进来：**`tp_size` / `tp_rank` 从模块单例取**（L1837-L1845 的两个 helper）、**`output_size_per_partition = divide(output_size, tp_size)`**（L454）、**`gather_output` flag 决定 forward 后做不做 all-gather**（默认 False）。

`divide` 是 `utils.py:L60-L64` 的强制整除断言——这是 vLLM 的契约：**所有 shard 大小都过 `divide`**：`output_size_per_partition`、`input_size_per_partition` (L1447)、`num_heads` (L1030)、`num_kv_heads` (L1035)、`num_kv_head_replicas` (L1033)。如果 `tp_size` 不能整除某个维度，构造时就 assert 失败——不会让你跑到 GEMM 深处才出错（K07 知识：整除契约由 `divide` 在层构造时强制）。

### 8.1.2 大白话：为什么"列切"和"行切"是 GEMM 的两种自然分解

先用大白话讲直觉。一个 GEMM 是 $Y = X A$，其中 $A$ 是 `[in, out]` 形状的权重矩阵。把这个矩阵想成一摞列向量——你有两种自然的方式把它"切"给多个 GPU：

- **按列切**（column-parallel）：把权重矩阵 A 按它的第二维（输出维）切成 p 份——

$$
A = [A_1 \,|\, A_2 \,|\, \cdots \,|\, A_p]
$$

  每份 `[in, out/p]`。每个 rank 拿到全部输入 X，但只算自己那 `out/p` 列的输出 $Y_i = X A_i$。最后把 p 份输出**拼起来**就是完整的 Y。

- **按行切**（row-parallel）：把权重 A 按它的第一维（输入维）切成 p 份（注：这里"行切"指的是把 input 维分给不同 rank）。每个 rank 拿到对应的输入分片和权重分片——形状是完整 `out`，但是是 partial sum。最后把 p 份**加起来**才是完整的 Y。

这两种方式数学上**完全等价于** $Y = XA$——一种用拼接（concat），一种用求和（sum）。下面把这件事写成正式定理。

### 8.1.3 等价性引理

**Column-parallel 引理**（输出维分块）：

$$
Y = X A = X [A_1 \,|\, A_2 \,|\, \cdots \,|\, A_p] = [X A_1 \,|\, X A_2 \,|\, \cdots \,|\, X A_p]
$$

证明：矩阵乘法定义

$$
(XA)_{ij} = \sum_k X_{ik} A_{kj}
$$

把 A 的列切成 p 份不影响每列的计算独立性，每列只看到 A 自己的那一列——计算和不切完全一致，只是分到不同的 rank。**没有通信**。这是 demo §1 column-parallel 测出来 `col_tp{2,4,8}_max_abs_diff = 0`（**bit-for-bit 零**）的根本原因——column-parallel forward 不引入任何 floating-point 加法，仅是 partition + concat。

**Row-parallel 引理**（输入维分块）：

$$
Y = X A = \sum_{i=1}^{p} X_i A_i
$$

其中 X 沿最后维切成 $X = [X_1, X_2, \ldots, X_p]$，A 沿第一维切成

$$
A = \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_p \end{bmatrix}
$$

证明：把矩阵乘的求和

$$
(XA)_{kj} = \sum_l X_{kl} A_{lj}
$$

按 input 维分组：

$$
\sum_l = \sum_{i=1}^{p} \sum_{l \in S_i}
$$

其中 $S_i$ 是第 i 个 rank 拥有的 input 维 indices。每个 rank 算的就是

$$
\sum_{l \in S_i} X_{kl} A_{lj} = (X_i A_i)_{kj}
$$

**最后那次外层加法 $\sum_i$ 就是 all-reduce**。

Demo §1 row-parallel 测出来 `row_tp{2,4,8}_max_abs_diff` 在 7.629e-06 ~ 9.537e-06 量级——不是 0。为什么？因为 float32 加法**不满足结合律**，row-parallel 的 partial 求和顺序和单次完整 BLAS 调用内部累加的顺序不同，会产生 ULP 级别的差。这是 row-parallel 必然的特征（K10：column-parallel 用 `array_equal` 测，row-parallel 用 `allclose(atol=1e-5)` 测——这个区分本身就能抓到 regression）。

### 8.1.4 Megatron pair：col→row 复合需要**几次** all-reduce？

现在把 column-parallel 和 row-parallel **串起来**：第一层是 column-parallel（输出 `out/p` 份），第二层是 row-parallel（input 维已经切好，正好对应上）。直觉上"中间应该 all-gather 一下"——把 column 的 sharded 输出拼回完整再喂给 row layer。

**但是不需要**。看清楚：column-parallel 的 sharded 输出 $Y_i^{(\mathrm{col})} = X A_i$ 形状是 `[..., out/p]`，正好就是 row-parallel 需要的 $X_i$（如果它的 `input_is_parallel=True`，第二层的 input 维 = 第一层的 output 维 = `out` = `out/p × p`，按 rank 已经切好了）。所以中间**没有** all-gather，row-parallel 直接吃 column-parallel 的 sharded 输出。整个 col→row pair **只有一次** all-reduce（row-parallel 出口的 sum）。

形式化：

$$
Y^{(\mathrm{block})} = \mathrm{rowparallel}(\mathrm{colparallel}(X)) = \sum_{i=1}^{p} (X A_i^{(\mathrm{col})}) A_i^{(\mathrm{row})}
$$

中间没有 all-gather。Demo §1 里 `colrow_tp{2,4,8}_num_collectives = 1`（一字不差），三档 tp 都是 **1**。这是**Megatron 论文**的核心 insight，本章的载量不变量。**Tip 1 反白**：「一次 all-reduce per pair」**不是**「一次 all-reduce per transformer block」——一个完整的 Llama transformer block 有 attn pair + mlp pair = **两次** all-reduce per block（详见 §8.6）。

### 8.1.5 我们的实现：`tp_math.py`

`implementation/tp_math.py:L70-L101` 把 column-parallel forward 写成了纯函数：

```python
# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L410-L432
def column_parallel_forward(
    X: np.ndarray, A: np.ndarray, tp_size: int, gather_output: bool = False
) -> list[np.ndarray] | np.ndarray:
    out = A.shape[-1]
    out_per_rank = divide(out, tp_size)
    # REFERENCE: linear.py:L454 — output_size_per_partition = divide(output_size, tp_size)
    A_shards = [
        A[..., r * out_per_rank : (r + 1) * out_per_rank] for r in range(tp_size)
    ]
    Y_shards = [X @ A_r for A_r in A_shards]
    if gather_output:
        # REFERENCE: linear.py:L589-L591 — tensor_model_parallel_all_gather
        return np.concatenate(Y_shards, axis=-1)
    return Y_shards
```

`A_shards` 沿 output 维切——这是对应源码 `weight_loader` 的 `narrow(output_dim, ...)`（`linear.py:L561`）。每个 rank 算自己的 `Y_i = X @ A_i`。`gather_output=True` 路径走 `np.concatenate` 模拟 `tensor_model_parallel_all_gather`（L589-L591）；常规路径返回 list（下游是 row-parallel，吃 list）。

`row_parallel_forward`（`tp_math.py:L129-L175`）对应：

```python
# REFERENCE: linear.py:L1543-L1577 — forward
if input_is_parallel:
    # linear.py:L1547-L1548 — `if self.input_is_parallel: input_parallel = input_`
    X_local = X_shards
else:
    # linear.py:L1549-L1553 — split_tensor_along_last_dim
    X_local = list(split_tensor_along_last_dim(X_shards, tp_size))
A_shards = [A[r * in_per_rank : (r + 1) * in_per_rank, :] for r in range(tp_size)]
Y_partials = [X_local[r] @ A_shards[r] for r in range(tp_size)]
if reduce_results:
    return np.sum(Y_partials, axis=0)  # ← 这是 all-reduce
return Y_partials
```

注意 `A_shards` 是沿**第一维**（input 维）切——这是 `weight_loader` 的 `narrow(input_dim, ...)`（`linear.py:L1524`）——和 column-parallel 的 narrow output_dim **方向相反**。这是 Trap-F（也是 W01 易错点）。

源码差异：vLLM 的 `weight_loader` 是 lazy 的——每个 rank 只从 checkpoint 里 narrow 自己那一份并加载到 GPU；我们一次性把全权重输入然后切，是为了讲清楚切的方向。

### 8.1.6 §8.1 mini 映射表（数学等价部分）

| 我们的代码 | vLLM 源码 | 一致 / 简化 |
|-----------|----------|-----|
| `divide` (`tp_math.py:L45-L49`) | `utils.py:L60-L64` | 一字不差整除断言 |
| `ensure_divisibility` (`tp_math.py:L37-L41`) | `utils.py:L53-L64` | 同 |
| `split_tensor_along_last_dim` (`tp_math.py:L53-L63`) | `utils.py:L67-L92` | numpy 实现，签名一致 |
| `column_parallel_forward` (`tp_math.py:L71-L101`) | `linear.py:L410-L607` forward | numpy 切片替 `narrow` |
| `column_parallel_weight_loader` (`tp_math.py:L105-L121`) | `linear.py:L534-L569` | 一致 narrow output_dim |
| `row_parallel_forward` (`tp_math.py:L129-L175`) | `linear.py:L1543-L1577` forward | `np.sum` 替 all-reduce |
| `row_parallel_weight_loader` (`tp_math.py:L179-L196`) | `linear.py:L1499-L1532` | 一致 narrow input_dim |
| `column_then_row_block` (`tp_math.py:L204-L239`) | `llama.py:L94-L121` LlamaMLP | 复合 + 计 collective |
| `verify_column_parallel_equivalence` (`tp_math.py:L246-L266`) | — | 新增（demo §1 用） |
| `verify_row_parallel_equivalence` (`tp_math.py:L269-L288`) | — | 新增 |
| `verify_column_then_row_block` (`tp_math.py:L291-L313`) | — | 新增 |

---

## 8.2 vLLM 没有 `class TensorParallel`：5 文件协同

### 8.2.1 源码事实先讲（§X 重构 — Ch07 "no radix tree" 风格）

第 7 章开篇说"vLLM v1 没有 radix tree"。第 8 章开篇说"vLLM 没有 `class TensorParallel`"——同一种 outline-vs-source 错位课题。先把搜证据列出来：

```
$ grep -rE "class TensorParallel|class ParallelReplica|class TPCoordinator" \
    instances/vllm/source/vllm/distributed/ \
    instances/vllm/source/vllm/model_executor/
(zero matches)
```

——零匹配。既不存在 `class TensorParallel`，也不存在某个一站式的 TP framework class。但是 vLLM 的 TP **是** Megatron-style 的（column → row 一对，每对一次 all-reduce）。怎么"是"的？通过 5 个文件的**组合**：

**(1) `parallel_state.py` — group + collective 抽象层**（2132 行）

```python
# vllm/distributed/parallel_state.py:L290-L330 (节选)
class GroupCoordinator:
    """A coordinator wrapping a torch.distributed ProcessGroup."""
    rank: int
    world_size: int
    device_communicator: DeviceCommunicatorBase
    ...
```

`GroupCoordinator` 包了一个 `torch.distributed.ProcessGroup`，提供 `all_reduce` / `all_gather` / `reduce_scatter` 三个集体通信方法。每个方法的实现都是"world_size==1 bypass + 委托给 device_communicator"——backend 抽象，CUDA 上委托给 NCCL。

**(2) 模块级单例 `_TP`** — TP 不是某个层持有的实例，而是模块全局变量

```python
# vllm/distributed/parallel_state.py:L1494-L1599 (节选)
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: str | None = None,
) -> None:
    global _TP
    ...
    _TP = init_model_parallel_group(...)   # L1586

# vllm/distributed/parallel_state.py:L1229-L1235
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, ...
    return _TP

# vllm/distributed/parallel_state.py:L1837-L1845
def get_tensor_model_parallel_world_size() -> int:
    return get_tp_group().world_size

def get_tensor_model_parallel_rank() -> int:
    return get_tp_group().rank
```

注意 **L1586** 里的 `_TP = init_model_parallel_group(...)`——这是一个**模块级 global 变量**（T02 知识）。每个 parallel layer 都通过 `get_tp_group()` 拿到这个共享实例，**没有任何层持有自己的 TP 实例**。这也意味着：在 layer 构造之前，必须先调 `init_distributed_environment` + `initialize_model_parallel`——**顺序错了，所有层都炸**（这是新人最容易踩的语言陷阱：以为 `ColumnParallelLinear()` 自己就能跑）。

**(3) `communication_op.py` — 1 行 wrapper 暴露给 layer**

```python
# vllm/distributed/communication_op.py:L12-L40 (一字不漏)
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().all_reduce(input_)

def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return get_tp_group().all_gather(input_, dim)

def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    return get_tp_group().reduce_scatter(input_, dim)

def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    return get_tp_group().gather(input_, dst, dim)
```

**这就是 layer 调的 API**。`linear.py` 的 ColumnParallelLinear / RowParallelLinear 在 forward 里只 import 这几个名字。注意每个函数都是 1 行——所有的复杂度都被 **`get_tp_group()` 取单例 + `.all_reduce(input_)` 委托** 这两个动作吸收了。

**(4) `linear.py` — 4 个 TP linear 类**

```
linear.py:L410-L608   class ColumnParallelLinear(LinearBase)
linear.py:L609-L976   class MergedColumnParallelLinear(ColumnParallelLinear)
linear.py:L977-L1393  class QKVParallelLinear(ColumnParallelLinear)
linear.py:L1394-L1577 class RowParallelLinear(LinearBase)
```

这是 TP 的"具身"。每个类负责一种切法（output dim、N 个 segment、head 维、input dim）。

**(5) `models/llama.py` — 真实使用现场**

```python
# vllm/model_executor/models/llama.py:L94-L121 (节选)
class LlamaMLP(nn.Module):
    def __init__(self, ...):
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,  # gate + up
            ...
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            ...
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x
```

这就是把 4 个 TP layer 拼起来的"vLLM 的 TP 用法"。看清楚——**没有任何 `TensorParallel(...)` 这种 wrapper class**——直接 import + 实例化 + 调 forward。

### 8.2.2 模块级单例 `_TP`：方便和坑都在这里

T02 知识入口。模块级单例**好处**：

- 每层不必传 `parallel_group` 参数，构造时调 `get_tp_group()` 即可——节省了几百个层的 constructor 参数管线。
- 测试时可以用 `monkeypatch.setattr('parallel_state._TP', mock_group)` 注入假的 group。

**坑**：

- 必须先 `init_distributed_environment(...)` + `initialize_model_parallel(tp_size=...)`，这两个**任一**没调，`get_tp_group()` 在 L1230 的 `assert _TP is not None` 就 fire——而且错误信息常常出现在 layer 构造时，新人会以为是 layer 配置问题。
- `_TP` 是 process-local 的——每个 rank 有自己的 `_TP`。production vLLM 是**一进程一 rank**；如果你想在单进程里模拟多 rank（像我们的实现这样），就不能用 `_TP`，得自己持有 `rank_states`。

### 8.2.3 我们的简化：`rank_states` 列表

我们的 `column_parallel.py:L98-L100` 这样设计：

```python
# rank_states[r] holds the rank-r view; in real vLLM each process owns one.
self.rank_states: list[dict] = [
    {"weight": None, "bias": None, "tp_rank": r} for r in range(tp_size)
]
```

把 `tp_size` 个 rank 的状态都塞进一个 list，每个元素是一个 rank 的视图。这样我们能在**单进程**里跑教学 demo——但代价是失去了真正并发的好处（K17 honest demo caveat：demo §3 的 ms 数完全不能代表真实 TP 性能，详见 §8.5）。

### 8.2.4 §8.2 mini 映射表（5 文件协同）

| 我们的代码 | vLLM 源码 | 一致 / 简化 |
|-----------|----------|-----|
| `rank_states[r]` 字典列表 | `_TP` 模块单例 + 每进程一份状态 | 单进程模拟多 rank |
| `column_parallel.py:ColumnParallelLinear` | `linear.py:L410-L608` | numpy 替 torch.Parameter |
| `row_parallel.py:RowParallelLinear` | `linear.py:L1394-L1577` | 同 |
| `qkv_parallel.py:QKVParallelLinear` | `linear.py:L977-L1393` | 同 |
| `column_parallel.py:MergedColumnParallelLinear` | `linear.py:L609-L976` | per-segment 一致 |
| `mlp_block.py:LlamaMLPTP` | `llama.py:L81-L121` LlamaMLP | 一致 wiring |
| `comm_primitives.py:simulate_all_reduce` | NCCL ring + `parallel_state.py:L502-L530` | numpy 替 NCCL |
| `comm_primitives.py:HARDWARE_PROFILES` | — | 新增（α-β 表） |

---

## 8.3 ColumnParallelLinear + RowParallelLinear：Megatron pair 的 1 次 all-reduce

### 8.3.1 ColumnParallelLinear 的 forward 路径

源码定位：`linear.py:L579-L607`：

```python
# vllm/model_executor/layers/linear.py:L579-L607 (节选)
def forward(
    self, input_, ...
) -> tuple[torch.Tensor, ...]:
    bias = self.bias if not self.skip_bias_add else None
    output_parallel = self.quant_method.apply(self, input_, bias)
    if self.gather_output:
        # All-gather across the partitions.
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```

三步：

1. **`quant_method.apply(self, input_, bias)`**（L587）——这一行是**真正的 GEMM**。`quant_method` 是 `LinearMethodBase` 的实例，对 unquantized 路径就是直接的 `F.linear(input_, self.weight, bias)`。注意 `F.linear` 的 weight 形状是 `[out, in]`（这是 W01：F.linear shape `[out, in]`，Trap-F 的源头）——但因为 `self.weight` 此时是 column-parallel 后的 sharded weight `[out/p, in]`，每个 rank 的输出是 `[..., out/p]`。
2. **`gather_output` 分支**（L589-L591）——默认 `False`，因为下游通常是 RowParallelLinear 直接吃 sharded 输出。
3. **bias 输出**——`output_bias` 在 `skip_bias_add=True` 时单独返回（这是 fused bias-into-norm 的 hook）。

我们的 `column_parallel.py:L152-L169` 一致复刻：

```python
# REFERENCE: linear.py:L579-L607 — forward
def forward(self, X: np.ndarray) -> list[np.ndarray] | np.ndarray:
    Y_shards: list[np.ndarray] = []
    for r in range(self.tp_size):
        W_r = self.rank_states[r]["weight"]
        Y_r = X @ W_r  # REFERENCE: linear.py:L587 — quant_method.apply
        if self.has_bias:
            Y_r = Y_r + self.rank_states[r]["bias"]
        Y_shards.append(Y_r)
    if self.gather_output:
        return np.concatenate(Y_shards, axis=-1)  # REFERENCE: linear.py:L589-L591
    return Y_shards
```

我们循环 `tp_size` 个 rank（生产是每个进程跑自己那一次），其余完全对应。

### 8.3.2 RowParallelLinear 的 forward 路径 + bias-on-rank-0 戏法

源码定位：`linear.py:L1543-L1577`：

```python
# vllm/model_executor/layers/linear.py:L1543-L1577 (节选)
def forward(
    self, input_, ...
):
    if self.input_is_parallel:
        input_parallel = input_                           # L1547-L1548
    else:
        tp_rank = get_tensor_model_parallel_rank()
        splitted_input = split_tensor_along_last_dim(input_, ...)
        input_parallel = splitted_input[tp_rank].contiguous()   # L1549-L1553

    # bias is added on rank 0 only — see L1557-L1559
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)

    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)   # L1562-L1563
    else:
        output = output_parallel
    ...
    return output, output_bias
```

四件事：

1. **`input_is_parallel` 分支**（L1547-L1553）——`True` 表示上游已经按 rank 切好了（默认值，Megatron pair 用法）；`False` 时调 `split_tensor_along_last_dim` 把 input 沿最后维切成 `tp_size` 份，每个 rank 取自己那一份。
2. **bias 在 rank 0 only**（L1557-L1559）——`bias_ = None if self.tp_rank > 0 ...`。每个 rank 的 weight 形状是 `[out, in/p]`（input 维切了，output 维完整），bias 形状是**完整的 `[out]`**（**没有切**——这是 T06 知识）。如果每个 rank 都加 bias，all-reduce sum 出来就是 `tp_size × bias`。所以只在 rank 0 加。
3. **GEMM**（L1561）——这步 partial output `[..., out]`，但每个 rank 数值不同（partial sum）。
4. **all-reduce**（L1562-L1563）——`tensor_model_parallel_all_reduce`，这是这个 layer 的**唯一通信**（注意 L1562 还有 `self.tp_size > 1` 守卫——单 rank 时 bypass）。

**Tip 4 反白**：bias 这件事必须两半都讲清楚。

- **bias 在每个 rank 上是完整 `[out]`**——**RowParallelLinear 的 bias 不分 shard**（与 ColumnParallelLinear 相反，column 的 bias 是和 weight 一起 sharded 的）。这告诉读者：**TP 不能减少 bias 的内存**——bias 在每个 rank 上都要存完整。
- **bias 只在 rank 0 上加**——避免 all-reduce 后变成 `tp_size × bias`。

T13 钉死的测试模式：**weight 全 0、bias 非零**。这时 GEMM 贡献 0，all-reduce 后输出应该等于 bias（加了一次）。如果 buggy 的实现在每个 rank 加了 bias，all-reduce 后就是 `tp_size × bias`——一个 4× 的 silent off-by-tp_size 错误。这个 trick 能在不靠 numerical tolerance 的情况下 pin 住 bias 语义。

我们的 `row_parallel.py:L137-L140` 一致：

```python
# REFERENCE: linear.py:L1557-L1560 — fuse bias on rank 0 ONLY
bias_for_rank = (
    self.rank_states[r]["bias"] if (self.has_bias and r == 0) else None
)
Y_r = X_local[r] @ W_r
if bias_for_rank is not None:
    Y_r = Y_r + bias_for_rank
```

### 8.3.3 MergedColumnParallelLinear 的 per-segment narrow——一个真实 bug 的故事

源码定位：`linear.py:L767-L820`，这段是 weight loader：

```python
# vllm/model_executor/layers/linear.py:L767-L820 (节选)
shard_offsets: list[tuple[int, int, int]] = []
current_shard_offset = 0
for i, output_size in enumerate(self.output_sizes):
    shard_offsets.append((i, current_shard_offset, output_size))
    current_shard_offset += output_size

for shard_id, shard_offset, shard_size in shard_offsets:
    # narrow per-rank within this segment
    shard_size = divide(shard_size, self.tp_size)
    shard_offset = ... + shard_offset // ... + ...
    # ...
    param_data = param_data.narrow(output_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.narrow(output_dim, ..., shard_size)
```

这里有一个**真实存在过的 bug**——这个 bug 我们在 implementer 第一稿里复现过，被 Tip 5 钉住的那个。

**故事**：MergedColumnParallelLinear fused 了 `gate_proj` 和 `up_proj` 两个 output，权重形状是 `[hidden, 2*ffn]`——前 `ffn` 列是 gate，后 `ffn` 列是 up。看似可以**朴素地**沿最后一维切给 `tp_size` 个 rank：rank 0 拿 `[..., 0:2*ffn/p]`、rank 1 拿 `[..., 2*ffn/p:4*ffn/p]`……

**但这样切 rank 0 拿到的是什么？**——是 gate 的前 `2*ffn/p` 列，即 gate 的**前两个 1/p**——**完全没有 up 的任何一列**。其它 rank 也类似——某些 rank 全是 gate、某些 rank 全是 up。然后跑 SiluAndMul，**每个 rank 内的 `gate * up` 完全不对**——一个 silent 但实质性的 bug。

**正确做法**（per-segment narrow）：每个 segment（gate 段和 up 段）**独立**沿 tp 切。rank 0 应该拿 `[gate_前 ffn/p 列 | up_前 ffn/p 列]`，concat 起来形状是 `[..., 2*ffn/p]`——这才是和单 GPU 等价的切法。

**可观测性**：

- 朴素切法在 tp=4 时 end-to-end MLP 的 max-abs-diff 是 **~7.7e-4**（demo §5 的"naive narrow"路径）。
- 正确切法是 **~1e-7**——**4 个量级**的差距。
- shape-only 测试两条路径都过：朴素切法的 weight shape 看起来正确，only 数值不对。

**Tip 5 反白**：**这是个真实 bug，不是教学戏剧化**。implementer 第一稿就踩到了，靠 Tester 的 end-to-end equivalence 测试（不是 shape 测试）抓出来。这是为什么 Tester 的测试设计强调"don't pass for the wrong reason"（W02）——shape-only 测试会错过这种 bug。

我们的 `column_parallel.py:L122-L134` 用了正确的 per-segment 循环：

```python
for r in range(self.tp_size):
    # REFERENCE: linear.py:L767-L820 — per-segment narrow loop.
    shards = []
    running_offset = 0
    for seg_size in segment_sizes:
        seg = A_full[:, running_offset : running_offset + seg_size]
        shard_size = divide(seg_size, self.tp_size)
        start = r * shard_size
        shards.append(seg[:, start : start + shard_size])
        running_offset += seg_size
    self.rank_states[r]["weight"] = np.concatenate(shards, axis=-1).astype(self.params_dtype)
```

每个 segment 独立切再 concat——和源码一致。Tester 在 `tests/test_column_parallel.py` 的 `test_per_segment_loader_avoids_naive_narrow_bug` 和 `test_chain_with_naive_narrow_would_be_wrong` 用 recognizable values（gate 100..115、up 200..215）pin 住了正确切法 vs 朴素切法的差异。

### 8.3.4 §8.3 mini 映射表（col + row）

| 我们的代码 | vLLM 源码 | 一致 / 简化 |
|-----------|----------|-----|
| `ColumnParallelLinear.__init__` (`column_parallel.py:L69-L101`) | `linear.py:L436-L505` | 一致 |
| `output_partition_sizes` LIST (`column_parallel.py:L86-L92`) | `linear.py:L455-L460` `hasattr(...,'output_sizes')` | 一致 MRO trick (T03) |
| `ColumnParallelLinear.load_weight` (`column_parallel.py:L104-L149`) | `linear.py:L534-L569` + `L767-L820` | per-segment narrow |
| `ColumnParallelLinear.forward` (`column_parallel.py:L152-L169`) | `linear.py:L579-L607` | numpy 替 quant_method.apply |
| `MergedColumnParallelLinear.__init__` (`column_parallel.py:L206-L227`) | `linear.py:L609-L725` | output_sizes 在 super 前设 |
| `MergedColumnParallelLinear.split_per_rank` (`column_parallel.py:L229-L246`) | `llama.py:L228-L229` `qkv.split([...])` | 一致 |
| `RowParallelLinear.__init__` (`row_parallel.py:L59-L92`) | `linear.py:L1429-L1497` | 一致；包括 bias+!reduce_results 报错 |
| `RowParallelLinear.load_weight` (`row_parallel.py:L95-L109`) | `linear.py:L1499-L1532` | narrow input_dim；bias 完整 (T06) |
| `RowParallelLinear.forward` (`row_parallel.py:L112-L150`) | `linear.py:L1543-L1577` | bias on rank 0 only (T13) |
| `silu_and_mul_per_rank` (`mlp_block.py:L57-L64`) | `activation.py:SiluAndMul` | 元素级，per-rank works |

---

## 8.4 QKVParallelLinear：head 维 column-parallel + GQA replication

### 8.4.1 打开 QKV 入口

源码定位：`linear.py:L977-L1393`，`class QKVParallelLinear(ColumnParallelLinear)`：

```python
# vllm/model_executor/layers/linear.py:L1029-L1043 (节选)
self.num_heads = divide(self.total_num_heads, tp_size)        # L1030

if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1                                     # L1032
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
                                                              # L1033
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)  # L1035
    self.num_kv_head_replicas = 1                             # L1036

input_size = self.hidden_size
output_size = (self.num_heads + 2 * self.num_kv_heads) * tp_size * self.head_size
self.output_sizes = [
    self.num_heads * tp_size * self.head_size,                # q segment (full)
    self.num_kv_heads * tp_size * self.head_size,             # k segment (full)
    self.num_kv_heads * tp_size * self.head_size,             # v segment (full)
]
```

注意几个关键点：

- **L1030**：`num_heads = divide(total_num_heads, tp_size)`——而**不是** `output_size // tp_size`。这是 Trap-C 的关键证据：QKV 是按 **head 维**切的，**不是** 按 feature 维任意切。
- **L1031-L1036 GQA 分支**：当 `tp_size >= total_num_kv_heads`（KV head 比 rank 还少），每个 rank 拿 `num_kv_heads = 1` 个 KV head，replication factor 是 `tp_size / total_num_kv_heads`（即每 `replicas` 个相邻 rank 共享同一个 KV head）。
- **`output_sizes` 三元组**：(q full, k full, v full)——MergedColumnParallelLinear 的每 segment 独立切（§8.3.3）正好处理这个 fused weight。

### 8.4.2 为什么 QKV 必须按 head 维切（Trap-C）

直觉：self-attention 的 head 是**独立**的——每个 head 有自己的 Q、K、V projection 和自己的 attention computation。两个 head 之间在 attention 计算里**互不通信**。

如果 QKV 按"任意 feature 列"切，rank 0 可能拿到 head 0 的前半 + head 1 的后半——这样 rank 0 算的"attention"既不是 head 0 的 attention 也不是 head 1 的 attention，是个**乱炖**。**数学上根本不等价**。

按 head 维切不一样：rank 0 拿 head 0..k 的**完整** Q、K、V projection 权重；它在自己 rank 上算的是**真正的** head 0..k 的 attention。每个 rank 独立做完，o_proj 用 row-parallel 聚合——这是数学等价。

这就是为什么 `linear.py:L1030` 写的是 `divide(total_num_heads, tp_size)` 而不是 `output_size // tp_size`。

### 8.4.3 GQA × TP 的 KV 节省 cap（Trap-D）

GQA（Grouped-Query Attention）让 KV head 数比 Q head 数少（典型：Llama-3-70B 有 64 Q heads、8 KV heads，每 8 个 Q heads 共享一个 KV head）。这给 KV cache 带来 8× 的内存节省（K shared 8× → KV 整体 8×）。

但 GQA 和 TP 复合时有个**边界**：如果 `tp_size > total_num_kv_heads`，每个 rank 仍然只能持有**一整个**KV head（你不能把一个 KV head 切两半给两个 rank——head 切就破坏 attention 计算独立性）。所以 `tp_size = 16, total_num_kv_heads = 8` 时，每两个 rank 共享一份 KV head 的 weight——也就是 KV 在 rank 之间**复制**了。

这就是 `num_kv_head_replicas` 的意义：每个 KV head 被复制 `tp_size / total_num_kv_heads` 次。结果是**KV cache 内存的 cap 在 `total_num_kv_heads`**——再增加 tp_size 不会进一步减少 KV/rank（每 rank 仍然存一份完整 KV head）。

Demo §4 的数字（Llama-3-70B-style：64 Q heads、8 KV heads、head_size=128）一字不改：

| tp_size | kv_heads/rank | replicas | KV/rank/token (B) | save factor |
|---|---|---|---|---|
| 2 | 4 | 1 | 2048 | 2.0× |
| 4 | 2 | 1 | 1024 | 4.0× |
| 8 | 1 | 1 | 512 | **8.0×** |
| 16 | 1 | 2 | 512 | **8.0× (cap)** |
| 32 | 1 | 4 | 512 | **8.0× (cap)** |

注意 save factor 是 **non-monotonic in step size**：2.0 → 4.0 → 8.0 → 8.0 → 8.0——前三档每加倍 tp_size 节省 2 倍，第四档之后**完全不增**。这个边界是 Trap-D 的核心证据。

完整 KV cache 每 token (fp16) = `2 × 8 × 128 × 2 = 4096 bytes`（K = 2、num_kv_heads = 8、head_size = 128、fp16 = 2 bytes）；tp=8 时每 rank 存 `4096 / 8 = 512` bytes/token——直接 8×。tp=16 时每 rank 仍是 `512` bytes/token——**没有进一步压缩**，但**通信**和**计算**继续涨（更多 rank 需要更多 all-reduce）。所以 GQA 模型的 TP "甜区"通常正好是 `tp_size = total_num_kv_heads`。

### 8.4.4 我们的 QKV 实现：head 维切 + KV replication

`qkv_parallel.py:L74-L83`：

```python
# REFERENCE: linear.py:L1030 — heads divided by tp_size
self.num_heads = divide(self.total_num_heads, tp_size)

# REFERENCE: linear.py:L1031-L1036 — GQA replication branch
if tp_size >= self.total_num_kv_heads:
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

这一段和源码 1:1。`load_qkv_weights` (`qkv_parallel.py:L122-L165`) 多了一个 KV replication 步骤：

```python
if self.num_kv_head_replicas > 1:
    def replicate(W):
        W3 = W.reshape(self.hidden_size, self.total_num_kv_heads, self.head_size)
        W3r = np.repeat(W3, self.num_kv_head_replicas, axis=1)
        return W3r.reshape(self.hidden_size, -1)
    Wk_eff = replicate(Wk_full)
    Wv_eff = replicate(Wv_full)
else:
    Wk_eff = Wk_full
    Wv_eff = Wv_full
W_fused = np.concatenate([Wq_full, Wk_eff, Wv_eff], axis=-1)
super().load_weight(A_full=W_fused, b_full=None)
```

`np.repeat` 沿 head 轴复制 KV——把 `[head0_kv, head1_kv, ..., head7_kv]` 变成 `[head0, head0, head1, head1, ..., head7, head7]`（replicas=2 时）。这样 fused weight 沿 output 维切给 16 个 rank 时，每两个相邻 rank 共享同一个 KV head。

`split_qkv` (`qkv_parallel.py:L170-L184`) 把每个 rank 的 fused 输出切成 (q, k, v) per-rank，head-major 布局：

```python
q_size = self.num_heads * self.head_size      # per-rank q size
kv_size = self.num_kv_heads * self.head_size  # per-rank kv size
offsets = [0, q_size, q_size + kv_size, q_size + 2 * kv_size]
for Y_r in qkv_per_rank:
    out["q"].append(Y_r[..., offsets[0] : offsets[1]])
    out["k"].append(Y_r[..., offsets[1] : offsets[2]])
    out["v"].append(Y_r[..., offsets[2] : offsets[3]])
```

对应 `llama.py:L228-L229` 的 `qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)`——同样的三段切法。

### 8.4.5 §8.4 mini 映射表（QKV）

| 我们的代码 | vLLM 源码 | 一致 / 简化 |
|-----------|----------|-----|
| `QKVParallelLinear.__init__` (`qkv_parallel.py:L55-L116`) | `linear.py:L1005-L1060` | 一致 |
| `num_heads = divide(...)` (`qkv_parallel.py:L74`) | `linear.py:L1030` | 一致 (T01) |
| GQA replication 分支 (`qkv_parallel.py:L77-L83`) | `linear.py:L1031-L1036` | 一致 (T04) |
| `output_sizes` 三元组 (`qkv_parallel.py:L95-L99`) | `linear.py:L1037-L1047` | 一致 |
| `load_qkv_weights` (`qkv_parallel.py:L122-L165`) | `linear.py:L1141-L1393` (`weight_loader_v2`) | 三 shard 合一 |
| KV `np.repeat` (`qkv_parallel.py:L141-L153`) | `linear.py:L1366-L1393` (`shard_rank // num_kv_head_replicas`) | 思路一致 |
| `split_qkv` (`qkv_parallel.py:L170-L184`) | `llama.py:L228-L229` | 一致三段切 |
| `per_rank_summary` (`qkv_parallel.py:L190-L201`) | — | 新增（demo §4 用） |

---

## 8.5 通信代价的 α-β 模型——为什么 P=8 在 1KB 上比 P=2 慢

### 8.5.1 打开 all-reduce 入口

源码定位：`parallel_state.py:L502-L530`：

```python
# vllm/distributed/parallel_state.py:L502-L530 (节选)
def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
    ...
    # Bypass when world_size == 1
    if self.world_size == 1:                               # L518
        return input_                                       # L519
    return self.device_communicator.all_reduce(input_)
```

**L518-L519** 是单 rank bypass——这是 T11：单 rank 时 cost = 0（我们的 `ring_all_reduce_cost` 也照搬了 `if P < 2: return 0.0` 的契约）。多 rank 时委托给 `device_communicator.all_reduce`（CUDA 上是 NCCL），NCCL 内部根据 payload 大小选择 ring / tree / double-binary-tree 算法。

**Tip 3 反白**：vLLM 这一行是 user-facing API。**所有的真实算法选择都在 NCCL 里**——vLLM 不复刻 NCCL 算法，只是调用。我们要建模的是 NCCL ring 在大 payload 上的性能。

### 8.5.2 ring all-reduce 的 α-β 公式

**直觉**：ring all-reduce 把 P 个 rank 摆成一个圈，每个 rank 把自己的张量切成 P 块。算法分两半：

- **reduce-scatter 半**：P-1 步。每步每 rank **传出**自己负责的块给下一个 rank，**收到**前一个 rank 传过来的块**累加**到自己。P-1 步之后，每个 rank 拥有"它负责的那块的全 reduced 版本"。
- **all-gather 半**：再 P-1 步。每个 rank 把它的"全 reduced 块"沿环形传播一圈。结束时每个 rank 都有所有 P 块的 reduced 版本——拼起来就是完整的 all-reduce 结果。

总共 `2*(P-1)` 步，每步传一个 `S/P` 大小的块。模型：

$$
T_{\mathrm{ring}}(P, S) = 2 \cdot \frac{P - 1}{P} \cdot \left( \alpha + \frac{S}{P} \cdot \beta \right)
$$

其中 $\alpha$ 是单步延迟（per link 启动开销），$\beta$ 是单字节传输时间（带宽倒数）。展开成两项：

$$
T_{\mathrm{ring}}(P, S) = T_{\mathrm{lat}}(P) + T_{\mathrm{bw}}(P, S)
$$

延迟项：

$$
T_{\mathrm{lat}}(P) = 2 \cdot \frac{P - 1}{P} \cdot \alpha
$$

带宽项：

$$
T_{\mathrm{bw}}(P, S) = 2 \cdot \frac{P - 1}{P^2} \cdot S \cdot \beta
$$

注意两项的渐近行为：

- 延迟项：P=2 时是 α，P 趋近无穷时是 2α。**延迟随 P 单调增**。
- 带宽项：P=2 时是 $S \beta / 2$，P 趋近无穷时趋近 0。**带宽 cost 随 P 减少**。

两项的**相对权重**取决于 payload 大小 S。S 小时（**α-bound**）延迟项占主，P 增大让 cost **变大**；S 大时（**β-bound**）带宽项占主，P 增大让 cost 变小但是 sub-linear。

### 8.5.3 demo §2：α-bound 区间是反直觉的（Tip 2 — 先讲 α-bound，再讲 β-bound）

**Tip 2 反白**：很多人对 Trap-A 的描述是 "all-reduce 大 payload 时是带宽限制，所以 TP=2 不会给 2× 吞吐"——这只对**一半**，而且是**不太反直觉**的那一半。**真正反直觉**的是：**小 payload 时 P=8 比 P=2 还慢**——加 rank 反而让通信变慢。

Demo §2 NVLink_HSXM4 profile（α=2μs，bandwidth=300GB/s）的预测表（一字不改）：

| payload (B) | P=2 | P=4 | P=8 |
|---|---|---|---|
| 1024 | 2.00 μs | 3.00 μs | **3.50 μs** |
| 16384 | 2.03 μs | 3.02 μs | 3.51 μs |
| 262144 | 2.44 μs | 3.33 μs | 3.69 μs |
| 4194304 | 8.99 μs | 8.24 μs | 6.56 μs |
| 67108864 | 113.85 μs | 86.89 μs | **52.43 μs** |

读法（先 α-bound）：

- **1 KB**：P=8 是 P=2 的 1.75×（3.50 / 2.00）——**慢** 75%。在小 payload 上加 rank 是**净亏**——多了 2(P-1)/P 倍的 α，带宽节省可以忽略不计。
- **16 KB、256 KB**：α-bound 区间继续往上延伸。要到 ~MB 级别 payload，β 项才开始追上 α 项。
- **4 MB**：开始平衡——P=8 略快于 P=2（6.56 vs 8.99 μs）。
- **64 MB**（β-bound 极端）：P=2 = 113.85 μs、P=8 = 52.43 μs——**P=8 快 2.17×**。注意：是 **2.17×**，不是 4×。即使在最 β-bound 的极端，**rank 数翻 4 倍只快 2.17 倍**——sub-linear。

这就是 Trap-A 的两端：

- **α-bound 端**（小 payload）：P 增加**让** all-reduce **变慢**。
- **β-bound 端**（大 payload）：P 增加让 all-reduce 变快但 sub-linear（比例随 (P-1)/P，从 0.5 增长到 1）。

无论哪端，**P=2 给 2× 吞吐都不可能**——前端是反直觉的"加 rank 反而慢"，后端是"加 rank 快但 sub-linear"。

### 8.5.4 fit α-β：从微基准恢复 (α, β)

Demo §2 还做了一件事：用合成噪声测量值反向**拟合**出 (α, β)：

```
fit_alpha = 4.32 μs        true_alpha = 5 μs
fit_bw = 144.56 GB/s       true_bw = 150 GB/s
```

——5% 噪声下偏差在 5-15%。`fit_alpha_beta` (`comm_primitives.py:L159-L178`) 用最小二乘解 $T = \alpha + \beta S$。

实战意义：你用任何一台 GPU 测一组 all-reduce 时间（`payloads = [1KB, 16KB, 1MB, 16MB, 256MB]`），把 (S, T) 喂给 `fit_alpha_beta`，得到的 (α, β) 就是这条互联**对 ring all-reduce 的有效模型参数**。然后 `ring_all_reduce_cost(payload, P, ab)` 给出任意 (payload, P) 的预测——**不用真测就能算 TP=2 vs TP=4 vs TP=8 哪个甜区**。

### 8.5.5 一个 Llama transformer block 的 all-reduce 数

Demo §3 用 `predict_block_overhead` 算 Llama-7B-shaped 一个 transformer block 的通信开销。**注意：一个 transformer block 有 2 次 all-reduce**——attn 的 o_proj 一次（payload `[B, S, hidden]`） + MLP 的 down_proj 一次（payload 同）。所以 `predicted_seconds_per_block = 2 × predicted_seconds_per_allreduce`：

```
weights_per_layer_MB_fp16 = 270.533
weights_per_rank_tp1_MB_fp16 = 270.533
weights_per_rank_tp2_MB_fp16 = 135.267
weights_per_rank_tp4_MB_fp16 = 67.633
predicted_AR_us_tp1_NVLink = 0
predicted_AR_us_tp2_NVLink = 8.99
predicted_AR_us_tp4_NVLink = 8.24
```

**weights/rank 干净对半**：tp=1 → 270.5 MB、tp=2 → 135.3 MB（halves cleanly）、tp=4 → 67.6 MB。这是 TP **真实**的好处——内存确实 1/p。

**predicted AR overhead** 用 α-β 表给出。这是 production-honest 的数（用 NCCL ring 在 NVLink 上的预测）——**和 demo §3 输出里的 ms wallclock 不是一回事**（详见下一节 K17）。

### 8.5.6 §8.5 mini 映射表（α-β model）

| 我们的代码 | vLLM 源码 | 一致 / 简化 |
|-----------|----------|-----|
| `AlphaBetaModel` (`comm_primitives.py:L45-L67`) | — | 新增（线性模型） |
| `ring_all_reduce_cost` (`comm_primitives.py:L71-L90`) | NCCL ring 内部 + `parallel_state.py:L502-L530` | 公式预测 |
| `simulate_all_reduce` (`comm_primitives.py:L94-L152`) | NCCL ring + `parallel_state.py:L502-L588` | numpy 步进模拟 |
| `world_size==1 bypass` (`comm_primitives.py:L121-L123`) | `parallel_state.py:L518-L519` | 一字不差 (T11) |
| `fit_alpha_beta` (`comm_primitives.py:L159-L178`) | — | 新增最小二乘 |
| `HARDWARE_PROFILES` (`comm_primitives.py:L185-L192`) | — | 5 种互联校准 |
| `predict_block_overhead` (`comm_primitives.py:L195-L216`) | — | 2 × per-AR cost |

---

## 8.6 系统影响 + 跨章衔接 + 主映射表 + 语言陷阱回顾

### 8.6.1 Demo §3 的 wallclock 必须配 K17 caveat（Tip 3 反白）

Demo §3 还输出了 `compute_per_forward` 的 ms 数字——但这些 ms 是**单进程串行模拟**——`tp_size` 个 rank 在**同一个 Python 进程里**串行执行 forward。这意味着 wallclock **随 tp_size 线性增长**——和真实 production TP **完全相反**（production 是每个 rank 在自己的 GPU 上 ~并行，wallclock 大体不变，加上 all-reduce 开销）。

这就是 **K17 honest demo caveat**——**写代码可以引用，但要钉一个 caveat**：

> 这些 wallclock 数字来自一个单进程 Python 模拟，所有 tp_size 个 rank 串行跑在同一进程里——所以 wallclock 随 tp_size 线性涨。真实 production TP 中每个 rank 跑在自己的 GPU 上，wallclock 大体不变（compute-bound），加上一次 all-reduce 的开销（α-β model 给的那个数）。**用 §8.5 的 α-β 预测来推 production cost，不要用 demo §3 的 ms 数**。

**production-honest 的 quote-safe** 数字（demo §3 / §4 / §5 里）：

- **weights/rank**（demo §3 的 `270.5 / 135.3 / 67.6 MB`）——cleanly halves（实际 production TP 确实 halve）。
- **predicted AR overhead from α-β model**（demo §3 的 `8.99 μs / 8.24 μs`，demo §2 的 ring 表）——production NCCL 的预测值。
- **collectives per forward**（demo §5 的 `mlp_tp{2,4,8}_collectives_per_forward = 1.0`）——这是 invariant，不依赖任何 wallclock。
- **GQA boundary**（demo §4）——内存数学，和模拟 wallclock 完全无关。
- **数学等价性 max diff**（demo §1 / §5）——float32 numerical fidelity，production 一致。

**不能 quote** 的：demo §3 的 `compute_per_forward` ms 数（除非配 K17 caveat 一起说）。

### 8.6.2 一个 Llama transformer block 实际有几次 all-reduce？

回答：**两次**——一次 attn pair（QKV-col → o_proj-row + all-reduce）、一次 MLP pair（gate_up-merged-col → down_proj-row + all-reduce）。

打开 `models/llama.py` 看到：

```python
# vllm/model_executor/models/llama.py:L94-L121 (节选, LlamaMLP)
self.gate_up_proj = MergedColumnParallelLinear(
    input_size=hidden_size,
    output_sizes=[intermediate_size] * 2,
    ...
)
self.down_proj = RowParallelLinear(
    input_size=intermediate_size,
    output_size=hidden_size,
    ...
)
```

```python
# vllm/model_executor/models/llama.py:L164-L179 (节选, LlamaAttention)
self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=self.head_dim,
    total_num_heads=num_heads,
    total_num_kv_heads=num_kv_heads,
    ...
)
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,
    hidden_size,
    ...
)
```

每个 RowParallelLinear 的 `reduce_results` 默认 `True`——一次 all-reduce。一个 block 有 attn + mlp = **2 次** all-reduce。Demo §5 的 `mlp_tp{2,4,8}_collectives_per_forward = 1.0` 是**单**MLP block 的数（一次），完整 transformer block (attn + mlp) 是 **1 + 1 = 2 次**——这是 Tip 1 的关键消歧。

Tester 的 `test_attn_then_mlp_two_collectives_per_block`（`tests/test_integration.py`）一字不漏地 pin 住这一点：构造完整 attn + mlp 链，断言 collective 总数 = **2**——既不是 1（误认为整 block 只有一次 all-reduce）也不是 4（朴素的 col→all-gather→col→row 架构需要 4 次）。

### 8.6.3 跨章衔接：Ch01 / Ch03 / Ch09 / Ch11 / Ch15+

Ch08 不是孤立的——它是 Ch01 head 结构 + Ch03 attention kernel 之上的一层并行抽象，又是 Ch09 EP / Ch11 DCP-PCP / Ch15+ 模型架构 wiring 的基础。

| 关系 | 章节 | 引用方向 |
|-----|-----|---------|
| **back-pointer** | Ch01 Self-Attention 基础 | head 结构（num_heads / head_size / Q/K/V projection）— Trap-C "head 切而不是 feature 切" 必须读懂 head 独立性才能消化 |
| **back-pointer** | Ch03 FlashAttention / PagedAttention | attention kernel 本身 TP-agnostic——它看到的 head 已经是 local rank 的 slice。FlashAttention 的输出 `[B, S, num_heads*head_size]` 正好对应 o_proj 的 row-parallel input |
| **forward-pointer** | Ch09 Expert Parallelism | EP 是 MoE 的 TP 类比；EP+TP 复合是真实大模型的常用 pattern（Ch26-28 用 frontier-model 部署） |
| **forward-pointer** | Ch11 DCP / PCP | Decode/Prefill Context Parallelism 共享 §8.5 的 collective primitives 和 α-β model |
| **forward-pointer** | Ch15 Llama 模型架构 | 真实把 TP-wrapped layer 拼接成 transformer 的现场——`llama.py:L81-L121 LlamaMLP`、`L124-L233 LlamaAttention` 是 canonical instantiation |

**Ch09 forward-pointer**：MoE 的 expert_parallelism 把 expert 分到不同 rank。EP 的 all-to-all 是另一种 collective，但 α-β model 同样适用——Ch09 复用本章的 cost 框架。EP+TP 的复合（device mesh）也来自本章建立的 GroupCoordinator 抽象。

**Ch11 forward-pointer**：DCP/PCP 把序列维切给多 rank。RingAttention 在每个 ring step 用 P2P send/recv，但聚合阶段还是 all-reduce——同样的 α-β。本章 §8.5 给的 fit_alpha_beta + ring_all_reduce_cost 在 Ch11 直接拿来用。

**Ch15 forward-pointer**：真实把 TP layer 装进 transformer block 的现场。Ch15 解释 `LlamaForCausalLM`、`LlamaModel`、`LlamaDecoderLayer` 怎么把 attention + MLP 串起来——本章的 `LlamaMLPTP` 是 Ch15 单 layer 的子集。

### 8.6.4 主映射表：Our code → vLLM source 1:1（≥25 行）

| 我们的代码 | vLLM 源码 | 我们改了什么 | 为什么 |
|-----------|----------|-------------|--------|
| `divide` (`tp_math.py:L45-L49`) | `utils.py:L60-L64` | 一字不差整除断言 | 契约 (T01) |
| `ensure_divisibility` (`tp_math.py:L37-L41`) | `utils.py:L53-L64` | 同 | 契约 |
| `split_tensor_along_last_dim` (`tp_math.py:L53-L63`) | `utils.py:L67-L92` | numpy `np.split` 替 `torch.chunk` | 单进程 |
| `column_parallel_forward` (`tp_math.py:L71-L101`) | `linear.py:L410-L432` docstring + `L579-L607` forward | 切片替 narrow | 教学 |
| `column_parallel_weight_loader` (`tp_math.py:L105-L121`) | `linear.py:L534-L569` | output_dim narrow | 一致 |
| `row_parallel_forward` (`tp_math.py:L129-L175`) | `linear.py:L1394-L1425` docstring + `L1543-L1577` | `np.sum` 替 all-reduce | numpy |
| `row_parallel_weight_loader` (`tp_math.py:L179-L196`) | `linear.py:L1499-L1532` | input_dim narrow (W01) | 易错点钉死 |
| `column_then_row_block` (`tp_math.py:L204-L239`) | `llama.py:L94-L121 LlamaMLP` | 复合 + 计 collective | 教学 |
| `verify_*_equivalence` (`tp_math.py:L246-L313`) | — | 新增（demo §1 用） | demo |
| `AlphaBetaModel` (`comm_primitives.py:L45-L67`) | — | 新增线性模型 | α-β |
| `ring_all_reduce_cost` (`comm_primitives.py:L71-L90`) | NCCL ring + `parallel_state.py:L502-L530` | 公式预测 | α-β |
| `simulate_all_reduce` (`comm_primitives.py:L94-L152`) | NCCL ring + `parallel_state.py:L502-L588` | numpy 步进 | 教学 |
| `world_size==1 bypass` (`comm_primitives.py:L121-L123`) | `parallel_state.py:L518-L519` | 一字不差 | T11 |
| `fit_alpha_beta` (`comm_primitives.py:L159-L178`) | — | 最小二乘新增 | demo §2 |
| `HARDWARE_PROFILES` (`comm_primitives.py:L185-L192`) | — | 5 种互联校准 | reference |
| `predict_block_overhead` (`comm_primitives.py:L195-L216`) | — | 2 × per-AR cost | demo §3 |
| `ColumnParallelLinear.__init__` (`column_parallel.py:L69-L101`) | `linear.py:L436-L505` | tp_rank/tp_size 由 ctor 传入 | 单进程 |
| `output_partition_sizes` LIST (`column_parallel.py:L86-L92`) | `linear.py:L455-L460` | hasattr MRO trick | 一致 (T03) |
| `ColumnParallelLinear.load_weight` (`column_parallel.py:L104-L149`) | `linear.py:L534-L569` + `L767-L820` | per-segment narrow | 一致 (T08) |
| `ColumnParallelLinear.forward` (`column_parallel.py:L152-L169`) | `linear.py:L579-L607` | numpy 替 quant_method.apply | 教学 |
| `MergedColumnParallelLinear` (`column_parallel.py:L184-L246`) | `linear.py:L609-L976` | output_sizes 在 super 前设 | MRO |
| `MergedColumnParallelLinear.split_per_rank` (`column_parallel.py:L229-L246`) | `llama.py:L228-L229` `qkv.split([...])` | 一致三段切 | 一致 |
| `RowParallelLinear.__init__` (`row_parallel.py:L59-L92`) | `linear.py:L1429-L1497` | bias+!reduce_results raise | 一致 |
| `RowParallelLinear.load_weight` (`row_parallel.py:L95-L109`) | `linear.py:L1499-L1532` | input_dim narrow + bias 完整 | T06 |
| `RowParallelLinear.forward` (`row_parallel.py:L112-L150`) | `linear.py:L1543-L1577` | bias 只在 rank 0 加 | T13 |
| `QKVParallelLinear.__init__` (`qkv_parallel.py:L55-L116`) | `linear.py:L1005-L1060` | 一致 | 一致 |
| `num_heads = divide(...)` (`qkv_parallel.py:L74`) | `linear.py:L1030` | 一字不差 | Trap-C |
| GQA replication 分支 (`qkv_parallel.py:L77-L83`) | `linear.py:L1031-L1036` | 一字不差 | Trap-D (T04) |
| `output_sizes` 三元组 (`qkv_parallel.py:L95-L99`) | `linear.py:L1037-L1047` | 一致 | 一致 |
| `load_qkv_weights` (`qkv_parallel.py:L122-L165`) | `linear.py:L1141-L1393` `weight_loader_v2` | 三 shard 合一 | 单进程 |
| KV `np.repeat` 复制 (`qkv_parallel.py:L141-L153`) | `linear.py:L1366-L1393` `shard_rank // num_kv_head_replicas` | 思路一致 | 一致 |
| `split_qkv` (`qkv_parallel.py:L170-L184`) | `llama.py:L228-L229` | head-major 三段切 | 一致 |
| `LlamaMLPTP.__init__` (`mlp_block.py:L86-L117`) | `llama.py:L82-L115` | 直接拼装 4 个 layer | 一致 |
| `LlamaMLPTP.forward` (`mlp_block.py:L140-L156`) | `llama.py:L117-L121` | 计 collective | 一致 |
| `silu_and_mul_per_rank` (`mlp_block.py:L57-L64`) | `activation.py:SiluAndMul` | 元素级 per-rank works | Trap-E |
| `reference_unsharded_mlp` (`mlp_block.py:L166-L173`) | — | 单 GPU reference | demo |

**故意砍掉的内容**（每项指向后续章节）：

- 真实 `torch.distributed` / NCCL：`simulate_all_reduce` 是 numpy stand-in，production 走 `device_communicator.all_reduce(input_)`（`parallel_state.py:L502-L530`）→ NCCL → 实际跑在 GPU 之间。我们 **建模 cost，不实现 kernel**。
- Quantization / weight loading from disk：`column_parallel.py:load_weight` 接 numpy；production `linear.py:L534-L569` 处理 `torch.Parameter`、HuggingFace checkpoint narrow。
- CUDA Graph capture：`parallel_state.py:L464-L500` 有 `graph_capture` context manager 用于 `cudaGraphCapture`-friendly TP。这是性能优化，不是 TP 算法本身。
- `custom_all_reduce`：vLLM 在 NVLink P2P 小 payload 时走 `device_communicators/custom_all_reduce.py`，是 production 的 α-bound "fast path"。我们只在 Trap-B 提一句。
- VocabParallelEmbedding：`vocab_parallel_embedding.py:L192-L502` 把 vocab 维 column-parallel 切——和 §8.3 的 column-parallel 同结构。本章为了控制篇幅没单独走读。
- PP / DP 复合并行：本章只讲 TP；Ch11 讲 CP；Ch26-28 讲 frontier-model 的 4D/5D 并行（TP+CP+EP+PP+DP）。

### 8.6.5 5 个语言陷阱回顾（Ch07 §7.6.4 风格）

读到这里，重新过一遍开篇的 5 个语言陷阱——现在每个都有源码 + math 兜底：

**陷阱 A：TP=2 doubles throughput.** 错。TP 切的是 weight，**通信是新增的开销**——一个 transformer block 多了 2 次 all-reduce（attn pair + mlp pair）。在小 batch / 短序列时（payload < ~4 MB）通信是 α-bound，加 rank **更慢**：demo §2 的 1 KB payload 上 P=8 是 P=2 的 1.75× **慢**。在大 batch / 长序列时（payload > 4 MB）是 β-bound，加 rank 更快但 sub-linear：64 MB payload 上 P=8 仅比 P=2 快 2.17×（不是 4×）。**两端都不会给 2× 吞吐**。源码证据：`linear.py:L1562-L1563` 的 all-reduce 是 sequential dependency。

**陷阱 C：QKV 是 column-parallel 沿 feature 列切。** 错。它沿 **head 维**切。head 在 self-attention 里独立，按 feature 任意切会让某个 rank 拿到"head 0 的前半 + head 1 的后半"——这种 frankenstein 在 attention 计算下不等价于原模型。源码证据：`linear.py:L1030` 写的是 `divide(total_num_heads, tp_size)` 而不是 `output_size // tp_size`。

**陷阱 D：TP 减半 KV 内存。** 错（条件性）。**仅当** `total_num_kv_heads >= tp_size` 时成立。GQA 模型（如 Llama-3-70B 8 KV heads）当 tp_size=16 时，每 rank 仍持有 1 个 KV head 的完整 weight + cache（2× replicas），KV 内存节省 cap 在 **8×**，加 rank 不再压缩。Demo §4 的 save factor 表非单调：2.0 → 4.0 → 8.0 → 8.0 → 8.0。源码证据：`linear.py:L1031-L1036` 的 `num_kv_head_replicas` 分支。

**陷阱 E：MLP TP 需要 all-gather + all-reduce.** 错。Megatron pair 的 col→row 复合**只需要 1 次 all-reduce**——column-parallel 的 sharded 输出直接 feed 给 row-parallel（element-wise activation 不破坏 sharding）。如果中间真的 all-gather，就**多了一次**通信，破坏 Megatron 的 win。Demo §5 一字不漏：`mlp_tp{2,4,8}_collectives_per_forward = 1.0`。源码证据：`llama.py:L94-L121` LlamaMLP 把 `gate_up_proj.gather_output` 默认 False，`down_proj.input_is_parallel` 默认 True——中间没有 all-gather。

**陷阱 F：RowParallelLinear 的 input 自动切。** 错（条件性）。**仅当** `input_is_parallel=False` 时（这是 explicit override）。默认是 `input_is_parallel=True`，因为上游通常是 column-parallel layer，已经按 rank 切好输出——RowParallelLinear 直接 consume，不再 split。如果你把这个 flag 调反，要么默默把 input 又切一次（数据错位），要么 broadcast 一份给每 rank（双倍通信）。源码证据：`linear.py:L1547-L1553` 的 `if self.input_is_parallel` 分支。

---

## 验证

### 跑测试

```bash
cd instances/vllm/artifacts/08-tensor-parallelism
python3 -m pytest tests/ --ignore=tests/_legacy -q
```

预期输出：

```
144 passed in 4.05s
```

144 个测试覆盖 7 个模块：

| 模块 | 测试数 | 状态 |
|---|---|---|
| `test_tp_math.py` | 29 | PASS |
| `test_comm_primitives.py` | 22 | PASS |
| `test_column_parallel.py` | 19 | PASS |
| `test_row_parallel.py` | 18 | PASS |
| `test_qkv_parallel.py` | 25 | PASS |
| `test_mlp_block.py` | 16 | PASS |
| `test_integration.py` | 15 | PASS |

### 跑 lint

```bash
python3 scripts/lint_formulas.py instances/vllm/artifacts/08-tensor-parallelism/narrative/chapter.md
python3 scripts/lint_source_grounding.py instances/vllm/artifacts/08-tensor-parallelism/
```

两个 linter 都应当 0 阻塞。

### 跑 demo

```bash
python3 -m instances.vllm.artifacts.08-tensor-parallelism.implementation.demo
```

5 段输出对应本章 5 个核心数字（一字不漏）：

- §1 数学等价：`col_tp{2,4,8}_max_abs_diff = 0`（**bit-for-bit**）；`row_tp{2,4,8}_max_abs_diff` 在 7.629e-06 ~ 9.537e-06；col→row block tp=2 diff=0 collectives=1，tp=4 diff=2.384e-07 collectives=1，tp=8 diff=2.980e-07 collectives=1
- §2 α-β fit：`fit_alpha = 4.32 μs vs true 5 μs`；`fit_bw = 144.56 GB/s vs true 150 GB/s`；NVLink ring 表 1 KB→64 MB on P=2/4/8（1 KB 时 P=8/P=2 = 1.75× 慢；64 MB 时 P=2/P=8 = 2.17× 快）
- §3 throughput sweep：`weights_per_layer_MB_fp16 = 270.533`；tp=1/2/4 weights/rank = 270.533 / 135.267 / 67.633 MB（halves cleanly）；predicted AR overhead tp=2 = 8.99 μs、tp=4 = 8.24 μs（α-β model 给的 production 数）。**`compute_per_forward` ms 不引用**——K17 caveat。
- §4 GQA boundary：Llama-3-70B-style（64 Q / 8 KV / head=128）save factor 表 2.0× → 4.0× → 8.0× → **8.0× (cap)** → **8.0× (cap)**
- §5 LlamaMLP：tp=1 max_abs_diff=0 collectives=0；tp={2,4,8} max_abs_diff 在 6.4e-10 ~ 8.1e-10、collectives_per_forward = **1.0**（每 MLP 一次；完整 transformer block = attn + mlp = **2 次**）

---

## 总结

第 8 章把 vLLM 的 tensor parallelism 拆开了。**最重要的一句话**：vLLM **没有** `class TensorParallel`——它用 5 个文件（`parallel_state.py` group + `communication_op.py` 1 行 wrapper + `linear.py` 4 个 TP linear 类 + `vocab_parallel_embedding.py` + `models/llama.py` 真实 wiring）**组合**实现了 Megatron-style TP，而且代码量 / 抽象层次比一个 framework class **轻一个数量级**。

五件值得记住的事：

1. **数学等价是 column 用 concat、row 用 sum**：column-parallel demo §1 给 `max_abs_diff = 0`（bit-for-bit），row-parallel 给 ~7e-6（float32 加法不结合律的 ULP 噪声）。一对 col→row 复合 = 1 次 all-reduce per pair（不是 per block）。

2. **一个 transformer block = 2 次 all-reduce**：attn pair 一次 + mlp pair 一次。Tester 的 `test_attn_then_mlp_two_collectives_per_block` 把 collective 总数 pin 在 2——既不是 1（误以为 block 整体一次）也不是 4（朴素 col→all-gather→col→row 架构）。

3. **TP=2 不会给 2× 吞吐**：α-bound 端（小 payload）P 增加**让** all-reduce **慢**（demo §2: P=8 在 1KB 是 P=2 的 1.75× 慢），β-bound 端（大 payload）P 增加快但 sub-linear（64 MB 上 P=8 仅比 P=2 快 2.17×）。**两端都不到 2×**。

4. **GQA × TP 有边界**：KV 节省 cap 在 `total_num_kv_heads`。Llama-3-70B（8 KV heads）tp=8 节省 8×、tp=16 仍只 8×（K replicate）。`linear.py:L1031-L1036` 的 `num_kv_head_replicas` 分支是一字不差的源码 evidence。

5. **bias 在 RowParallelLinear 是个语言陷阱**：bias 是**完整 `[out]`**（不切，所以 TP 减不了 bias 的内存）；只在 **rank 0** 加（避免 `tp_size × bias`）。weight=0、bias≠0 的零权重测试一次 pin 住这两件事——Tester 的 `test_bias_added_only_on_rank_zero`。

**5 个语言陷阱也别再犯**：TP=2 ≠ 2× 吞吐；QKV 不是 feature 列切（是 head 切）；TP 不一定减半 KV（GQA 有 cap）；MLP TP 不需要 all-gather + all-reduce（一次 all-reduce per pair）；RowParallel 的 input 默认不切（`input_is_parallel=True`，假设上游已切）。

### 下章预告

第 9 章 `Expert Parallelism` 把单 expert 的 TP 推广到多 expert 的 EP——MoE router 的 token 分发用 all-to-all（另一种 collective），但 §8.5 的 α-β model 直接复用。EP+TP 的复合（device mesh）也来自本章建立的 `GroupCoordinator` 抽象：每个 rank 同时属于一个 TP group 和一个 EP group。

更远处：第 11 章 `DCP/PCP` 把序列维切给多 rank（context parallelism）——RingAttention 在每 ring step 用 P2P send/recv，但聚合用 all-reduce，同样的 α-β model；第 15 章 `Llama Model Architecture` 把 TP-wrapped layer 真正装进 `LlamaForCausalLM`、`LlamaModel`、`LlamaDecoderLayer`——本章 `LlamaMLPTP` 是 Ch15 单 layer 的子集。Ch08 是它们的基础——每个细节本章已经埋好。

---

← 第 7 章：Prefix Cache 与 APC-Aware Allocation | 第 9 章：Expert Parallelism →
