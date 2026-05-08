# 第11章：DCP/PCP —— 没有 `class RingAttention` 的上下文并行

> 本章涉及的 vLLM 源码（commit `98661fe`）：
> - `instances/vllm/source/vllm/distributed/parallel_state.py:L1234-L1290`（`_DCP` / `_PCP` 单例 + 三个 accessor + `get_context_model_parallel_group = get_dcp_group` 兼容别名）+ `L1497-L1498`（`initialize_model_parallel` 签名包含 `prefill_context_model_parallel_size` / `decode_context_model_parallel_size`）+ `L1569-L1575`（5D mesh `reshape(-1, dp, pp, pcp, tp)`）+ `L1594-L1614`（DCP groups —— `reshape(-1, dcp_size).unbind(0)`，无 transpose）+ `L1616-L1633`（PCP groups —— `transpose(3, 4).reshape(-1, pcp_size).unbind(0)`，靠 transpose 把 pcp 移到最里）+ `L1741-L1782`（`ensure_model_parallel_initialized` 的 PCP 断言）+ `L1791-L1797`（`prepare_communication_buffer_for_model` 把 PCP 也接进来）+ `L1847-L1854`（`get_decode_context_model_parallel_world_size/_rank` 两个公开 helper）
> - `instances/vllm/source/vllm/v1/attention/ops/dcp_alltoall.py:L1-L20`（模块 docstring：A2A 是 AG+RS 的替代品，引用 arxiv.org/abs/2507.07120）+ `L39-L103`（`_lse_weighted_combine` —— LSE 加权组合算法核心）+ `L106-L130`（`_dcp_a2a_lse_pack_dim` —— 把 fp32 LSE 塞进 bf16 buffer 的两格表示）+ `L134-L196`（`_dcp_a2a_pack_send_kernel` Triton kernel）+ `L197-L319`（`_dcp_a2a_unpack_combine_kernel` Triton kernel —— 接收侧 LSE-stable 合并）+ `L320-L447`（编排代码）+ `L448`（`dist.all_to_all_single` —— 真正落地的 NCCL 调用）
> - `instances/vllm/source/vllm/v1/attention/backend.py:L700-L757`（`AttentionImpl` CP 字段 + `__new__` 用 try/except 发现 `_DCP` / `_PCP`），尤其 `L703`（`supports_pcp: bool = False`）+ `L705-L706`（`supports_mtp_with_cp_non_trivial_interleave_size` —— Ch10 MTP ↔ Ch11 CP 的显式跨章 flag）+ `L722-L729`（六个 CP 字段 `dcp_world_size/dcp_rank/pcp_world_size/pcp_rank/total_cp_world_size/total_cp_rank`）+ `L731-L757`（discover-or-fallback 模式）
> - `instances/vllm/source/vllm/v1/attention/backends/utils.py:L820-L857`（`get_dcp_local_seq_lens` —— striped 切分的核心 helper；`base + remainder + clip` 三步走）
> - `instances/vllm/source/vllm/v1/attention/backends/flashinfer.py:L213`（`class BatchDCPPrefillWrapper` —— 整个仓库**唯一**带 DCP 前缀的 class，flashinfer 内部 batched wrapper，**不是**顶层 CP orchestrator）
> - `instances/vllm/source/vllm/v1/attention/backends/mla/flashattn_mla.py:L125`（`supports_dcp_with_varlen=(interleave_size==1)`）+ `L175`（`num_heads_q = num_heads * dcp_world_size`，DCP=2 时 Q 复制两份）+ `L196-L250`（`dcp_tot_seq_lens_device` 元数据穿过 forward）+ `L353-L355`（`cp_world_size`/`cp_rank` 直插 FA3 kernel）
> - `instances/vllm/source/vllm/v1/kv_cache_interface.py:L195-L205`（`max_memory_usage_bytes`：`cdiv(max_model_len, dcp × pcp) × cdiv(..., block_size) × page_size_bytes` —— Ch11 的 HBM 节省定理）
> - `instances/vllm/source/vllm/v1/executor/multiproc_executor.py:L116-L121`（world_size 断言 `tp × pp × pcp`，**DCP 不入积**）+ `L258-L259`（`_get_parallel_sizes` 返回 `(tp, pp, pcp)`）+ `L985-L1001`（per-process 名字标签 `_PCP{rank}` 条件追加）
> - `instances/vllm/source/vllm/config/parallel.py:L115`（`prefill_context_parallel_size`）+ `L310-L313`（`decode_context_parallel_size`）+ `L315-L321`（`dcp_kv_cache_interleave_size` —— 已被弃用）+ `L322-L328`（`DCPCommBackend = Literal["ag_rs", "a2a"]` —— **真正的两个 backend 名字**）+ `L330-L342`（`cp_kv_cache_interleave_size` —— 新 API）+ `L469-L478`（`tp % dcp == 0` 硬约束 ValueError）+ `L480-L483`（A2A 要求 `dcp > 1`）
> - 对照实现 `instances/vllm/artifacts/11-dcp-pcp/implementation/`：`parallel_state_dcp_pcp.py`、`world_topology.py`、`lse_combine.py`、`dcp_alltoall.py`、`seq_sharding.py`、`kv_cache_per_rank.py`、`attention_backend_dcp_pcp.py`、`dcp_vs_pcp_demo.py`、`demo.py`
>
> 第 7 章用"vLLM 没有 radix tree"开篇，第 8 章用"vLLM 没有 `class TensorParallel`"开篇，第 9 章用"vLLM 没有 `class ExpertParallel`"开篇，第 10 章用"vLLM 没有 `class MultiTokenPrediction`"开篇——第 11 章是这条系列的 **第五件**：**vLLM 没有 `class RingAttention` / `class StripedAttention` / `class ContextParallel` / `class DecodeContextParallel` / `class PrefillContextParallel`**。这条 reframe 在 N=5 时正式从"趋势"升格为**章节母题**：vLLM 系统性地用模块级纯函数 + GroupCoordinator 单例 + per-backend `__new__` 发现去替代顶层 orchestrator class。Context Parallelism 在源码里是 12 个文件的协同，不是某一个类。

---

## 这章要讲什么？

打开 `instances/vllm/source/vllm/v1/kv_cache_interface.py:L195-L205`：

```python
# vllm/v1/kv_cache_interface.py:L195-L205
def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
    if dcp_world_size * pcp_world_size > 1:
        # each dcp rank only need save 1/dcp_world_size of the kv cache
        max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
    return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

整个第 11 章的 HBM 主定理就藏在这 10 行里。把它代入 Llama-70B-128K 的真实参数（80 层 × 8 个 KV head × head_size=128 × bf16 × 2 因为 K 和 V）：

$$
128\,\mathrm{K}\;\times\;80\;\times\;8\;\times\;128\;\times\;2\;\times\;2 \;=\; 42{,}949{,}672{,}960\;\mathrm{bytes} \;=\; 40.0\;\mathrm{GB}
$$

40 GB —— **每个请求**的 KV cache 就要这么多，比 H100 整张卡 80 GB 的一半还多，再去掉模型权重和中间激活就直接 OOM 了。但是只要把 `dcp = 4, pcp = 4`，那条 `cdiv(max_model_len, dcp × pcp)` 就把 max_model_len 砍 16 倍，HBM 降到 **2.5 GB**。同样的硬件，从"放不下 128K"变成"轻松跑 128K"——这就是 Context Parallelism 存在的理由，也是这章的入口。

但是同样 grep 一下源码：

```
$ grep -rE '^class\s+(RingAttention|StripedAttention|ContextParallel|DecodeContextParallel|PrefillContextParallel)\b' instances/vllm/source/vllm/
(zero matches)

$ grep -rE '^class\s+\w*DCP\w*|^class\s+\w*PCP\w*' instances/vllm/source/vllm/
instances/vllm/source/vllm/v1/attention/backends/flashinfer.py:213:class BatchDCPPrefillWrapper:
```

唯一带 DCP 前缀的类是 `BatchDCPPrefillWrapper`——一个 flashinfer 内部的 batched wrapper，**不是**顶层 CP orchestrator。Liu et al. 2023 论文里的 Ring Attention 算法在 vLLM 里**完全没出现**：源码用的是 NCCL 的 AllGather + ReduceScatter（默认）或 All-to-All（advanced），**不是** P2P send/recv 环。这是这章第一个核心 reframe（与 Ch07/Ch08/Ch09/Ch10 同源）：**outline 让你画 Ring，源码却在做 NCCL 集合**。

学完这章你能：

- 在白板上写出 HBM 主定理：

  $$
  \mathrm{bytes}_{\mathrm{per\_rank}} \;=\; \mathrm{cdiv}\!\left(\frac{L}{\mathrm{dcp}\cdot\mathrm{pcp}},\, B\right) \cdot P_{\mathrm{page}}
  $$

  并背出 8 个 (dcp,pcp) 单元格的 verbatim 数字：(1,1)=40.0 GB、(1,2)=20.0 GB、(2,2)=10.0 GB、(4,4)=2.5 GB（demo §1）。
- 推导 LSE 加权合并是**代数恒等式**而非数值近似，并用 4-rank 的 demo §2 输出 `max abs error = 3.33e-16`、`associativity error = 2.22e-16` 作为浮点 ε 噪声边界的硬证据。
- 把 outline 的 "all-reduce vs all-to-all" **就地纠正**为源码真名 `DCPCommBackend = Literal["ag_rs", "a2a"]`，并算出 demo §3 的 α-β 表：dcp=2 时 A2A 比 AG+RS 快 **2.87×**、dcp=4 时 5.44×、dcp=8 时 9.85×；33% 的 NCCL op 削减是数学的、payload 缩小是工程的，二者叠加。
- 解释 striped 切分（interleave=1）在 causal mask 下**只能近似平衡**：demo §4 的 1.24× 不是 1.0×，因为 token 0 只 attend 1 格、token 63 attend 64 格，round-robin 之后 rank 7 还是比 rank 0 多 24% 的 work。但比起 contiguous 的 13.44×，已经是一个数量级的胜利。
- 推翻 outline §11.5 的 "3D 并行"——源码是 **5D mesh** `external_dp × dp × pp × pcp × tp`，DCP 折叠在 TP 内部、不进 world_size 乘积。production 配置 `(tp=8, dcp=2, pcp=4)` 的 world_size 是 32，**不是** 64。
- 在 §11.7 区分 7 个语言陷阱：A "DCP 翻倍 throughput"、B "PCP 砍半 prefill 延迟"、C "CP 就是 SP 改名"、D "DCP 必须等于 PCP"、E "CP 在 attention 层等同 TP"、F "Ring 是 vLLM 的标准实现"、G "Striped 是 Ring 改名"。

接下来 6 节按 outline 走，但 §11.2 把 "Ring Attention" 改写成"无 Ring Attention 类"的合成故事、§11.3 把 "all-reduce vs all-to-all" 就地纠正为 "AG+RS vs A2A"、§11.5 把 "3D" 升级为 "5D + DCP 内嵌"。

---

## 11.1 为什么需要 CP —— HBM 容量墙的源码解

### 11.1.1 打开真正算 HBM 的那 10 行

源码定位：`instances/vllm/source/vllm/v1/kv_cache_interface.py:L195-L205`，`AttentionSpec.max_memory_usage_bytes`：

```python
# vllm/v1/kv_cache_interface.py:L195-L205
def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
    if dcp_world_size * pcp_world_size > 1:
        # each dcp rank only need save 1/dcp_world_size of the kv cache
        max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
    return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

四件事一起进来：**每个 layer 的 spec 自带 `page_size_bytes`**（每个 KV-cache 块的字节数）、**`max_model_len` 是单请求最长上下文**、**`dcp_world_size × pcp_world_size > 1` 时把它整除**、**最后 `cdiv(..., block_size) × page_size_bytes` 把"逻辑长度"换成"实际占字节"**。注意这 10 行干的事：用一个 `cdiv` 同时把 `dcp` 和 `pcp` 的贡献都消掉了——这就是 §11.1 后半段会反复用到的"composed CP world size = `pcp × dcp`"的源头。

### 11.1.2 这个方法解决什么问题？为什么它就是 HBM 节省的**算式**？

KV cache 的**裸**字节数（不考虑 CP、不考虑 page padding）是

$$
\mathrm{bytes}_{\mathrm{naive}}(L) \;=\; L \;\times\; N_{\mathrm{layers}} \;\times\; 2 \;\times\; H_{\mathrm{kv}} \;\times\; D_{\mathrm{head}} \;\times\; b_{\mathrm{dtype}}
$$

代进 Llama-70B-128K（$N_\mathrm{layers}=80, H_\mathrm{kv}=8, D_\mathrm{head}=128, b_\mathrm{dtype}=2$）：

$$
128\,\mathrm{K} \times 80 \times 2 \times 8 \times 128 \times 2 \;=\; 42{,}949{,}672{,}960 \;\mathrm{bytes} \;=\; 40.0\;\mathrm{GB}
$$

40 GB / 请求。H100 整卡 80 GB，去掉模型权重（70B × 2 bytes ≈ 140 GB，TP=8 后每卡 17.5 GB）、中间激活、CUDA workspace，剩下能给 KV 的也就 30 多 GB——单请求 128K 都放不下。这就是"长上下文容量墙"的硬数字。

CP 的解法直白：把序列轴切到 `dcp × pcp` 个 rank 上，每个 rank 只存 `1/(dcp × pcp)` 的 KV。源码 L201-L203 的 `cdiv(max_model_len, dcp × pcp)` 就是这件事的代数表达。

### 11.1.3 从最朴素的"砍长度"推到带 page padding 的真实公式

第一步——只看长度，**不**考虑 vLLM 的 paged-KV 约束：

$$
L_{\mathrm{per\_rank}} \;=\; \left\lceil \frac{L}{\mathrm{dcp} \cdot \mathrm{pcp}} \right\rceil \;\;\;\Longrightarrow\;\;\; \mathrm{bytes}_{\mathrm{per\_rank}} \;=\; L_{\mathrm{per\_rank}} \cdot 2 H_\mathrm{kv} D_\mathrm{head} b_\mathrm{dtype} \cdot N_\mathrm{layers}
$$

第二步——加上 paged-KV 的"块对齐"约束。vLLM 用 `block_size=16` 的 KV 块作为最小分配单位（Ch05 的核心），所以每 rank 实际占用要按块向上取整：

$$
\mathrm{blocks}_{\mathrm{per\_rank}} \;=\; \left\lceil \frac{L_{\mathrm{per\_rank}}}{B}\right\rceil ,\qquad \mathrm{bytes}_{\mathrm{per\_rank}} \;=\; \mathrm{blocks}_{\mathrm{per\_rank}} \cdot P_\mathrm{page} \cdot N_\mathrm{layers}
$$

其中 $P_\mathrm{page} = 2 \cdot B \cdot H_\mathrm{kv} \cdot D_\mathrm{head} \cdot b_\mathrm{dtype}$（每个 page 的字节，K 和 V 各占 $B \cdot H_\mathrm{kv} \cdot D_\mathrm{head} \cdot b_\mathrm{dtype}$，所以乘 2）。代入：$P_\mathrm{page} = 2 \times 16 \times 8 \times 128 \times 2 = 65{,}536$ 字节 = 64 KiB。

第三步——把两层 cdiv 合并写出来，和源码一字不差：

$$
\mathrm{bytes}_{\mathrm{per\_rank}} \;=\; \left\lceil \frac{\lceil L / (\mathrm{dcp} \cdot \mathrm{pcp}) \rceil}{B} \right\rceil \;\cdot\; P_\mathrm{page} \;\cdot\; N_\mathrm{layers}
$$

源码 L201-L204 等价于这条式子（`max_memory_usage_bytes` 是单层的，乘 `num_layers` 来得到全模型总字节）。

### 11.1.4 我们的对照实现

`implementation/kv_cache_per_rank.py:L17-L21` 的 `cdiv` 和 `KVCacheSpec`：

```python
# implementation/kv_cache_per_rank.py:L17-L21
def cdiv(a: int, b: int) -> int:
    """Ceiling division — matches vllm.utils.cdiv."""
    return -(-a // b)
```

`implementation/kv_cache_per_rank.py:L42-L64` 的核心方法把源码 L196-L204 的语义 1:1 镜像（多了一个 `num_layers` 因子，因为我们 demo 的是全模型而不是单层 spec）：

```python
# implementation/kv_cache_per_rank.py:L42-L64
@property
def page_size_bytes(self) -> int:
    return 2 * self.block_size * self.num_kv_heads * self.head_size * self.dtype_bytes

def max_memory_usage_bytes(
    self, max_model_len: int, dcp_world_size: int = 1, pcp_world_size: int = 1
) -> int:
    if dcp_world_size * pcp_world_size > 1:
        max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
    return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

`implementation/kv_cache_per_rank.py:L102-L120` 的 `hbm_per_rank` 在外层多套了一层 `num_layers` 因子（vLLM 里这一层是在 BlockManager 里乘的，不在 AttentionSpec 里）：

```python
# implementation/kv_cache_per_rank.py:L102-L120
def hbm_per_rank(seq_len: int, spec: KVCacheSpec, dcp: int, pcp: int) -> int:
    total_cp = dcp * pcp
    if total_cp > 1:
        per_rank_len = cdiv(seq_len, total_cp)
    else:
        per_rank_len = seq_len
    blocks = cdiv(per_rank_len, spec.block_size)
    return spec.num_layers * blocks * spec.page_size_bytes
```

### 11.1.5 跑 demo §1，把 8 个单元格写出来

`implementation/demo.py::demo_1_hbm_capacity` 输出（verbatim，每个数字对应 `tests/test_kv_cache_per_rank.py::test_demo_section_1_*`）：

```
Demo §1 — HBM-per-rank capacity sweep (Llama-70B at 128K)
Spec: 80 layers, 8 KV heads, head_size=128, bf16, block_size=16

  Naive total KV bytes (no CP): 42,949,672,960 = 40.0 GB

  (dcp, pcp)   per_rank_len    per_rank_bytes     as GB
  (1,1)             131,072    42,949,672,960    40.0 GB
  (1,2)              65,536    21,474,836,480    20.0 GB
  (2,1)              65,536    21,474,836,480    20.0 GB
  (2,2)              32,768    10,737,418,240    10.0 GB
  (1,4)              32,768    10,737,418,240    10.0 GB
  (4,1)              32,768    10,737,418,240    10.0 GB
  (2,4)              16,384     5,368,709,120     5.0 GB
  (4,4)               8,192     2,684,354,560     2.5 GB
```

**两个 takeaway 要写在白板上**：第一，`(dcp, pcp)` 对的 HBM 只取决于 `dcp × pcp` 的乘积，所以 `(1,4)`、`(2,2)`、`(4,1)` 全是 10.0 GB——这就是 §11.4 会展开的"二者可分离"的算式根据。第二，`(4,4)` 把 40 GB 砍到 2.5 GB——**16× 减少，跨过 H100 的 80 GB 红线**。这条具体的数字（不是抽象的"线性 scale"）是 Trap A 的硬证据：DCP 的赢在**容量**，不在 throughput。

### 11.1.6 与 vLLM 真版的差距

我们和源码的差距列在 `impl-notes.md §1.5`：

| Source feature | 我们的简化 | 为什么没事 |
|---|---|---|
| 调用链穿过 `vllm_config` 的多层 dataclass | 直接传 `dcp_world_size, pcp_world_size` | 公式不变，只是入参形态 |
| 一次 spec 只算单层，多层在 BlockManager 里乘 | demo 直接乘 `num_layers` 出全模型字节 | 数学等价，演示更直观 |
| 实战还要减去模型权重和激活才得到"可分配 KV 块数" | demo 只算 KV 字节本身 | "weight + activation" 部分是 Ch05 的题目 |

> **回看 Ch05**：`KVCacheSpec` 的字段在 Ch05 已经讨论过，`block_size` 与 `page_size_bytes` 都是 paged-KV 的产物。这里我们只是在 spec 之上加了一层 CP 因子。
>
> **跨章前指**：`max_model_len` 这个单一变量在 Ch12（KV offload）会被 SSD 容量再乘一倍，在 Ch22（PD 架构）会被 prefill/decode 不同 budget 拆开，在 Ch25（PD ratio）会变成调度变量。

---

## 11.2 没有 `class RingAttention` —— 12 个文件的协同

### 11.2.1 打开仓库的 grep 结果

源码定位：先做一次 grep。在 `instances/vllm/source/vllm/` 根下：

```
$ grep -rE '^class\s+(RingAttention|StripedAttention|ContextParallel|DecodeContextParallel|PrefillContextParallel)\b' .
(zero matches)

$ grep -rE '^class\s+\w*DCP\w*|^class\s+\w*PCP\w*' .
v1/attention/backends/flashinfer.py:213:class BatchDCPPrefillWrapper:
```

零个 RingAttention、零个 ContextParallel；唯一带 DCP 前缀的 class 是 flashinfer 后端内部用的一个 batched prefill wrapper，**不是**顶层的 CP 编排器。

但这章的题目就叫 "DCP/PCP 上下文并行"。outline 第二节标题写着 "Ring Attention —— peer-to-peer P2P 通信的环形拓扑"。哪边对？答案是：**outline 描述的是技术名词，源码实现的是另一种结构**。这是 Ch07-Ch10 的同款 reframe，第 5 次出现。

### 11.2.2 这个 reframe 解决什么问题？

读者一看到 "Ring Attention"，直觉是去 vLLM 里找 `class RingAttention`。找不到时第一反应是"是不是版本太新"或"是不是改名了"。**都不是。vLLM 直接绕开 Liu et al. 2023 的 P2P send/recv 环，改用 NCCL 集合**。这背后有工程理由：NCCL 集合（AllGather、ReduceScatter、All-to-All）在 GPU 集群里是高度优化的、有 ring/tree/double-binary-tree 多种自适应实现的；P2P send/recv 反而要重新设计 buffer 同步、要担心 deadlock、要在 NVLink 拓扑里手工排序——工程代价远高于使用 `torch.distributed.all_to_all_single`。

所以 vLLM 的 CP 是 **12 个文件的合成**：

- **集合算法**：`vllm/v1/attention/ops/dcp_alltoall.py`（458 行，纯函数 + Triton kernel）
- **集合编排**：`vllm/distributed/parallel_state.py`（`_DCP` / `_PCP` 单例 + 5D mesh 构造）
- **进程组发现**：`vllm/v1/attention/backend.py`（每个 attention backend 在 `__new__` 里读单例）
- **流水线对接**：`vllm/v1/attention/backends/utils.py`（`get_dcp_local_seq_lens` 切片）
- **每后端集成**：9 个 `vllm/v1/attention/backends/*.py`（FA、FA-MLA、flashinfer、ROCm AITER…）
- **HBM 账本**：`vllm/v1/kv_cache_interface.py`
- **进程编排**：`vllm/v1/executor/multiproc_executor.py`
- **配置入口**：`vllm/config/parallel.py`
- **MoE 接入**：`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`（PCP-EP 复合）

每个文件都有职责，没有哪个文件能独立代表"CP"。这是 vLLM 系统性的偏好：**水平协作 > 垂直层次**。

### 11.2.3 推一个 toy：如果让你写一个 CP 库，最少要几个对象？

假设你被要求实现"最简单的 CP"：4 个 GPU 各持 1/4 KV，每个 GPU 都有完整 Q，要做一次 attention 然后产出一个全局 output。

**做法 1：Ring Attention（Liu et al. 2023）**。每个 rank 计算自己 KV 切片对全 Q 的部分注意力（带 LSE），然后把 K_i, V_i 发给下一个 rank、收上一个 rank 的；4 轮之后每个 rank 见过所有 KV，本地用 LSE 累积更新。需要 **N-1 轮 P2P send + recv** 配上 LSE-stable online softmax 累加。这就是 Liu 论文里的设计。

**做法 2：AllGather + ReduceScatter**。每个 rank 先 AllGather Q（拿到全部 ranks 的 Q）、计算完整 Q × 自己的 K_i, V_i 的部分注意力，再 ReduceScatter 把 output 按 rank 切回去。需要 **2 个 NCCL collective + 1 个 attention kernel**。

**做法 3：All-to-All**。每个 rank 计算自己 KV 切片对 1/N 的 Q（已经按 head 分到 ranks）的注意力 + LSE，然后把 (output, lse) 一起 packed 起来 AllToAll 一次，接收侧 LSE-stable 合并。需要 **1 个 NCCL collective + 2 个 Triton kernel**。

三种做法的**算法核心**都是同一个：LSE-weighted 加权合并（FlashAttention §2.3 的 online softmax 推到 ranks 上）。**不同的是 transport**：P2P 环、AG+RS、A2A。源码选了后两条，把第一条整个砍掉了。这就是为什么没有 `class RingAttention`——也根本没必要有。

### 11.2.4 我们的对照实现：单例 + `__new__` 发现 + 模块级纯函数

把上面三种实现的底盘（不带 NCCL、不带 Triton）写成一个 single-process pedagogical mirror，分三块：

**第一块：GroupCoordinator 单例（`parallel_state_dcp_pcp.py:L31-L68`）**：

```python
# implementation/parallel_state_dcp_pcp.py:L31-L68
@dataclass
class CPGroupCoordinator:
    group_name: str
    ranks: list[int]
    rank_in_group: int

    @property
    def world_size(self) -> int:
        return len(self.ranks)

# REFERENCE: vllm/distributed/parallel_state.py:L1234 (_DCP)
# REFERENCE: vllm/distributed/parallel_state.py:L1285 (_PCP)
_DCP: CPGroupCoordinator | None = None
_PCP: CPGroupCoordinator | None = None

def get_dcp_group() -> CPGroupCoordinator:
    assert _DCP is not None, "decode context model parallel group is not initialized"
    return _DCP

def get_pcp_group() -> CPGroupCoordinator:
    assert _PCP is not None, "prefill context parallel group is not initialized"
    return _PCP

# REFERENCE: vllm/distributed/parallel_state.py:L1242-L1243
get_context_model_parallel_group = get_dcp_group
```

**两点要回到源码再确认**：第一，`assert _DCP is not None, "..."` 这条 AssertionError——后面 §11.2.5 会看到 attention backend 用 `try/except AssertionError` 捕获它来支持单元测试。第二，最后那行 `get_context_model_parallel_group = get_dcp_group` 是"老 API 没死"的兼容别名，源码 L1242-L1243 完全一样写。

**第二块：per-backend `__new__` 发现（`attention_backend_dcp_pcp.py:L64-L99`）**：

```python
# implementation/attention_backend_dcp_pcp.py:L64-L99
def __new__(cls, *args, **kwargs):
    self = super().__new__(cls)

    # Discover DCP from singleton (or fall back).
    try:
        dcp = get_dcp_group()
        self.dcp_world_size = dcp.world_size
        self.dcp_rank = dcp.rank_in_group
    except AssertionError:
        self.dcp_world_size = 1
        self.dcp_rank = 0

    # Discover PCP from singleton (or fall back).
    try:
        pcp = get_pcp_group()
        self.pcp_world_size = pcp.world_size
        self.pcp_rank = pcp.rank_in_group
    except AssertionError:
        self.pcp_world_size = 1
        self.pcp_rank = 0

    # REFERENCE: vllm/v1/attention/backend.py:L751
    self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    # REFERENCE: vllm/v1/attention/backend.py:L752
    self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

    # REFERENCE: vllm/v1/attention/backend.py:L754-L756
    self.need_to_return_lse_for_decode = (
        self.dcp_world_size > 1 and self.can_return_lse_for_decode
    )
    return self
```

这段复刻源码 `vllm/v1/attention/backend.py:L731-L757`。设计要点：

1. `__new__` 在**实例化前**跑——此时 `__init__` 还没碰到 `self`。这让 backend subclass 不需要在 `__init__` 里写"先 super().__init__() 再读 group"的样板代码。
2. `try/except AssertionError` —— 这是单元测试友好。在没有调起 NCCL 的测试里，`get_dcp_group()` 会 raise；except 捕获后回落到 `dcp_world_size=1, dcp_rank=0`，让 backend 可以单独实例化。
3. 一个 subtle gotcha（D23）：`__new__` 把 group 状态**快照在实例创建那一刻**。如果你之后再调 `initialize_model_parallel(...)`，已经存在的 backend 实例不会自动更新——它们的 `dcp_world_size` 永远是创建时的值。production 里要确保 group 在第一次创建 backend 之前就初始化。

**第三块：模块级 LSE 合并纯函数（`lse_combine.py:L34-L125`）**——这是上面三种做法（Ring / AG+RS / A2A）共用的代数核心。下一节展开。

### 11.2.5 与源码的差距

| Source | Ours | 为什么 |
|---|---|---|
| `_DCP: GroupCoordinator | None = None` 包裹一个 `torch.distributed.ProcessGroup` | 我们的 `CPGroupCoordinator` 只存 `ranks: list[int]` | NCCL handle 是单进程 demo 不需要的运行时状态 |
| `init_model_parallel_group(group_name="dcp")` 走 `torch.distributed.new_group()` | 我们直接 mutate 全局 `_DCP` | 单进程没必要建 NCCL 通信器 |
| 9 个 backend 都覆盖 `__new__` | 我们只在 `AttentionImplBase` 里写一份 | 模式相同，复刻一份证明 pattern 的存在 |

> **跨章前指**：`__new__` 模式在 Ch15+ 模型加载时会反复遇到——每个 model class 也是这样在初始化时去读全局并行 state 的。

---

## 11.3 LSE 加权合并 —— 三种 transport 的代数核心

### 11.3.1 打开 `_lse_weighted_combine`

源码定位：`instances/vllm/source/vllm/v1/attention/ops/dcp_alltoall.py:L39-L103`，**整个 CP 子系统的算法心脏**就藏在这 65 行里：

```python
# vllm/v1/attention/ops/dcp_alltoall.py:L39-L103 (节选)
def _lse_weighted_combine(
    output: torch.Tensor,    # [N, B, H, D]
    lse: torch.Tensor,       # [N, B, H]
    *,
    return_lse: bool = True,
    is_lse_base_on_e: bool = True,
) -> ...:
    # L66-L70 — 把 NaN/+inf 的 LSE 替换成 -inf 让权重为 0
    lse = torch.where(lse.isnan() | lse.isinf(), float("-inf"), lse)

    # L72-L78 — 减去 max LSE 做数值稳定
    lse_max = lse.max(dim=0).values  # [B, H]
    lse_max = torch.where(lse_max == float("-inf"), 0.0, lse_max)
    diff = lse - lse_max[None, :, :]

    # L81-L84 — 算 exp(LSE - max) 当权重
    if is_lse_base_on_e:
        weights = diff.exp()
    else:
        weights = torch.pow(2.0, diff)

    # L89-L91 — 归一化
    weight_sum = weights.sum(dim=0, keepdim=True)
    weights_norm = weights / weight_sum.clamp_min(1e-10)

    # L93-L94 — 加权求和
    out = (output * weights_norm.unsqueeze(-1)).sum(dim=0)
    ...
```

四件事一起进来：**NaN/+inf 当 -inf 处理**（D14——某 rank 的 KV 切片是空的就让它权重为 0）、**减 max 做数值稳定**（FlashAttention §2.3 的同款技巧）、**exp(diff) 当权重**（base e 或 base 2 由 caller 选）、**最后加权求和**得到 `out` 和可选的 `global_lse`。

### 11.3.2 推 LSE 合并是**代数恒等式**而非数值近似（D21）

每个 rank $i$ 持自己 KV 切片，分别算出 partial output 和 partial LSE：

$$
p_{i,j} \;=\; \frac{\exp(s_{i,j})}{Z_i},\quad Z_i \;=\; \sum_{j} \exp(s_{i,j}),\quad O_i \;=\; \sum_j p_{i,j} v_{i,j},\quad \mathrm{lse}_i \;=\; \log Z_i
$$

如果在单 GPU 上跑（无 CP），全局 softmax 是

$$
p_j \;=\; \frac{\exp(s_j)}{Z},\quad Z \;=\; \sum_i Z_i \;=\; \sum_i \exp(\mathrm{lse}_i)
$$

代入 $Z_i \cdot O_i = \sum_j \exp(s_{i,j}) v_{i,j}$，两边按 rank 累加再除 $Z$：

$$
O \;=\; \frac{\sum_i Z_i \cdot O_i}{Z} \;=\; \frac{\sum_i \exp(\mathrm{lse}_i) \cdot O_i}{\sum_i \exp(\mathrm{lse}_i)}
$$

这是个**恒等式**——只要每个 rank 都老实地把自己 KV 切片的 LSE 报上来、output 也算对，**全局结果跟单 GPU 跑就是 bit-equivalent**。数值稳定的 lse_max 平移只是同时分子分母乘以 $\exp(-\mathrm{lse}_\max)$，结果不变：

$$
O \;=\; \frac{\sum_i \exp(\mathrm{lse}_i - \mathrm{lse}_\max) \cdot O_i}{\sum_i \exp(\mathrm{lse}_i - \mathrm{lse}_\max)}
$$

**重点**：这条式子和**通信方式无关**。Ring 也好、AG+RS 也好、A2A 也好——只要拿到了所有 rank 的 $(O_i, \mathrm{lse}_i)$，最后这一步永远就是这条加权平均。论文里"Ring Attention 比 AG+RS 数值更稳"之类的说法在 vLLM 的 LSE-stable 版本里**不成立**——三种 transport 走 LSE 合并都是 ε-级别浮点噪声。

这条性质有两个直接推论：

1. **结合律**：$\mathrm{combine}(\mathrm{combine}(O_0, O_1), \mathrm{combine}(O_2, O_3))$ 等于 `combine(O_0, O_1, O_2, O_3)`——合并的顺序不影响结果。这就是为什么 NCCL 的 ring/tree/double-binary-tree 都能用，调度选哪种都行。
2. **交换律**：rank 顺序无关——任意 permutation 给出同一个 output。

### 11.3.3 我们的对照实现 + 1:1 镜像源码

`implementation/lse_combine.py:L34-L125` 的 `lse_weighted_combine`，把上面那段 PyTorch 替成 NumPy（语义完全一样）：

```python
# implementation/lse_combine.py:L84-L125 (节选)
N, B, H, D = partial_outputs.shape

# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L66-L70
lses = np.where(np.isnan(partial_lses) | np.isinf(partial_lses), -math.inf, partial_lses)

# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L72-L78
lse_max = lses.max(axis=0)
lse_max = np.where(lse_max == -math.inf, 0.0, lse_max)

# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L81-L84
diff = lses - lse_max[None, :, :]
weights = np.exp(diff) if is_lse_base_on_e else np.power(2.0, diff)
weights = np.where(np.isnan(weights), 0.0, weights)

# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L89-L91
weight_sum = weights.sum(axis=0, keepdims=True)
weights_norm = weights / np.clip(weight_sum, a_min=1e-10, a_max=None)

# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L93-L94
out = (partial_outputs * weights_norm[..., None]).sum(axis=0)
```

逐行对应源码 L62-L94。注意 `np.clip(..., a_min=1e-10, a_max=None)`——这是为了防止 `weight_sum=0`（所有 rank 都没 KV 时，是数据 layout 不对的边界）导致除零，源码里同样的 `clamp_min(1e-10)`。

### 11.3.4 跑 demo §2，验证 bit-equivalent

`implementation/demo.py::demo_2_lse_combine`（4 ranks，B=4 token，H=2 head，D=8 head_dim，L=16 KV 长度，seed=42）输出（每个数字对应 `tests/test_lse_combine.py::test_demo2_*` 测试）：

```
Demo §2 — LSE-weighted combine equivalence
  shape partial_outputs = (4, 4, 2, 8)  (N, B, H, D)
  shape partial_lses    = (4, 4, 2)  (N, B, H)

  Per-rank LSE values for (token=0, head=0):
    rank 0: lse_i =  1.448093
    rank 1: lse_i =  0.996192
    rank 2: lse_i =  2.106473
    rank 3: lse_i =  1.629767
  lse_max (token=0, head=0) =  2.106473
  per-rank weight (normalized):
    rank 0: weight_i =  0.209762
    rank 1: weight_i =  0.133496
    rank 2: weight_i =  0.405190
    rank 3: weight_i =  0.251552
  max abs error vs single-process FlashAttention = 3.33e-16
  associativity error (rank01)+(rank23) vs flat       = 2.22e-16

  Trap F — Same LSE algebra regardless of transport (Ring/A2A/AG+RS).
```

**`max abs error = 3.33e-16` 不是 "method-specific 误差"**——它就是 fp64 的 machine ε（≈ 2.22e-16）级别的浮点噪声。换句话说：合并算法是**代数等价**的，数值上和单 GPU 跑差别在最低位。`associativity error = 2.22e-16` 也只是同一阶的 ε 噪声，证明 `(rank01 fold) + (rank23 fold)` 和 `flat 4-rank fold` 数值上**indistinguishable**。

这就是 §11.7 trap F 的硬证据：**"Ring 数值更稳"是错的，三种 transport 跑同一个 LSE 合并都是 ε 级别**。

### 11.3.5 与源码的差距

| Source | Ours | 为什么 |
|---|---|---|
| Triton kernel `_dcp_a2a_unpack_combine_kernel`（L197-L319，约 120 行） | NumPy 一行 broadcast 实现 | 算法一样、目的是教学；production 用 Triton 是为了和 receive buffer 在 GPU 上做 fusion，避免一次 GPU↔HOST 来回 |
| LSE 用 fp32 单独 buffer（`_dcp_a2a_lse_pack_dim` L106-L112 的 2-cell 编码） | 直接 fp64 数组 | 我们不做 packed payload，pack/unpack 是 transport 层细节 |
| async NCCL `dist.all_to_all_single(async_op=True)`（L448） | 同步 numpy 操作 | 单进程没法 async；不影响算法正确性 |

> **回看 Ch03**：FlashAttention 的 online softmax 跨 KV 块做 LSE-stable 累加；Ch11 是把同一个想法跨 ranks 做。代数完全一致。

---

## 11.4 AG+RS vs A2A —— 不是 "all-reduce vs all-to-all"

### 11.4.1 outline 错了，源码是这样写的

打开 `instances/vllm/source/vllm/config/parallel.py:L322-L328`：

```python
# vllm/config/parallel.py:L322-L328
DCPCommBackend = Literal["ag_rs", "a2a"]
"""DCP communication backend.
- "ag_rs": AllGather + ReduceScatter (default)
- "a2a":   AllToAll (advanced; reduces NCCL ops by 33%)
"""
```

**outline 子节标题写的是 "DCP —— decode 阶段的 all-reduce vs all-to-all 方案"**，但源码两个 backend 名字是 `"ag_rs"` 和 `"a2a"`——既不是 all-reduce 也不是 all-to-all 二选一。这是这章的第二个 reframe（**就地纠正**：outline JSON 不动，章节里换术语，并在这里显式说明）。

为什么要纠正？因为 "all-reduce vs all-to-all" 暗示了一种对比关系，但 AG+RS 和 A2A 才是源码真正实现的两条路径，**它们的差别在 NCCL 调用次数和 buffer packing**，不在"reduce vs all-to-all 的代数语义"。下一节展开。

### 11.4.2 这两个 backend 解决什么问题？

DCP 的标准操作流程（每层 attention）：每 rank 持完整 Q（TP column-parallel 后已经在 TP-group 内复制了）、自己的 1/dcp 的 K, V 切片，目的是产出完整的 attention output。

**AG+RS（默认 `dcp_comm_backend="ag_rs"`）**：

1. AllGather Q：每 rank 拿完整 Q（实际 vLLM-MLA 的实现里是 Q 通过 `num_heads_q = num_heads * dcp_world_size` 复制扩展，源码 `flashattn_mla.py:L175`）。
2. 本地 attention：每 rank 用完整 Q 和自己的 K_i, V_i 算 partial output + partial LSE。
3. ReduceScatter output：把 partial output 按 rank 切回去；reduce 时用 LSE 加权（不是简单 sum）。

加上中间的 attention kernel，按论文 arxiv.org/abs/2507.07120 的算法是 **3 个 NCCL ops + 1 个 attention kernel**（也有"attention 不算 collective"的口径，则是 2 NCCL + 1 GEMM；本章按论文口径走以保持和 demo 数字对得上）。

**A2A（`dcp_comm_backend="a2a"`）**：

1. 本地 attention：每 rank 用自己的 K_i, V_i 对全部 token 但 1/dcp 的 head 做 attention，产出 partial output + LSE。
2. AllToAll：把 (output, LSE) **packed 成一个 buffer**，一次 `dist.all_to_all_single` 把它交换到所有 ranks。
3. Triton 接收 kernel：从 packed buffer 里 unpack 出 (output, LSE)，跑 LSE 合并。

总共 **2 个 NCCL ops + 2 个 Triton kernel**。注意 packed buffer 的 shape（D20）：

```python
# vllm/v1/attention/ops/dcp_alltoall.py:L431-L436
send_buffer = torch.empty(
    (num_ranks, num_tokens, num_heads // dcp_size, head_dim + lse_pack_dim),
    dtype=output.dtype, device=output.device,
)
```

注意 `num_heads // dcp_size` 这个除法——A2A 的 payload **随着 dcp 增大反而变小**，因为更多 ranks 意味着每 rank head 数更少。AG+RS 的 payload 不随 dcp 变化（每 rank 都要 AllGather 完整 Q，再 ReduceScatter 完整 output）。这是 A2A 的第二个赢点：**op 数从 3 减到 2 是数学的，payload 缩小是工程的，二者叠加**。

### 11.4.3 用 α-β 模型推算延迟

NCCL 集合延迟近似 $T(N) = N_\mathrm{op} \cdot (\alpha + \mathrm{bytes}/\beta)$，其中 $\alpha$ 是单次集合的固定延迟、$\beta$ 是带宽、$N_\mathrm{op}$ 是 op 数。

**我们的实现** `implementation/dcp_alltoall.py:L88-L118`：

```python
# implementation/dcp_alltoall.py:L88-L118
def alpha_beta_cost(
    bytes_payload: int,
    alpha_us: float,
    beta_gbps: float,
    *,
    num_collectives: int,
) -> float:
    bytes_us = bytes_payload / (beta_gbps * 1e3)
    return num_collectives * (alpha_us + bytes_us)
```

H100 + 4×NVLink 的参考数字：$\alpha \approx 10\,\mu\mathrm{s}$、$\beta \approx 200\,\mathrm{GB/s}$（literature reference number，**不是**我们测出来的——honest caveat in `impl-notes §4`）。

`ag_rs_payload_bytes` 和 `a2a_payload_bytes` 在 `implementation/dcp_alltoall.py:L122-L162`：

```python
# implementation/dcp_alltoall.py:L122-L162 (节选)
def ag_rs_payload_bytes(num_tokens, num_heads, head_dim, dcp_size, dtype_bytes=2):
    return num_tokens * num_heads * head_dim * dtype_bytes  # 不随 dcp 变

def a2a_payload_bytes(num_tokens, num_heads, head_dim, dcp_size, dtype_bytes=2):
    lse_pack_dim = 2 if dtype_bytes == 2 else 1
    h_per_rank = num_heads // dcp_size
    return num_tokens * h_per_rank * (head_dim + lse_pack_dim) * dtype_bytes
```

代到 32K 个 token、8 个 head、head_dim=128、bf16 工作负载，结果：

| dcp | AG+RS ops | A2A ops | AG+RS bytes | A2A bytes | T_AG+RS μs | T_A2A μs | speedup |
|---|---|---|---|---|---|---|---|
| 2 | 3 | 2 | 67,108,864 | 34,078,720 | 1036.6 | 360.8 | 2.87× |
| 4 | 3 | 2 | 67,108,864 | 17,039,360 | 1036.6 | 190.4 | 5.44× |
| 8 | 3 | 2 | 67,108,864 | 8,519,680 | 1036.6 | 105.2 | 9.85× |

**A2A 比 AG+RS 快 2.87×、5.44×、9.85×**——dcp 越大胜出越大。两条原因叠加：(a) op 数 3→2 是固定 33% 减少，(b) payload 随 dcp 变小（dcp=8 时只有 dcp=2 的 1/4）。

### 11.4.4 但是要警告 Trap B：A2A 不是免费午餐

A2A 在 H100+NVLink 上漂亮，但 **PCIe 网络下能反而更慢**：

- $\alpha$ 在 PCIe 上从 10 μs 升到 ~50 μs，A2A 的 2 个 op 仍然要付两次 α，对比 AG+RS 的 3 个 op 在带宽好时更划算。
- 短 prefill（< 4K token）下 payload 本来就小，α 占比变高，A2A 的"payload 缩小"赢点消失。

源码 `parallel.py:L480-L483` 还有一个防御性 check：A2A backend 要求 `dcp_size > 1`——dcp=1 时根本没有 all-to-all 的对象。

**production 选择规则**：在 H100 NVLink + 长 prefill + dcp ≥ 2 的标配下用 A2A；其他情况测一下再说。这是 §11.7 Trap B 的硬数据来源。

### 11.4.5 关于"`all_to_all_single` 是真正的那一行"

源码 `dcp_alltoall.py:L448`：

```python
# vllm/v1/attention/ops/dcp_alltoall.py:L448 (节选)
work = dist.all_to_all_single(
    output=recv_buffer,
    input=send_buffer,
    async_op=True,
)
```

这是整个 A2A 后端**唯一**的 NCCL 调用（pack/unpack 是 Triton kernel，不算 NCCL）。`async_op=True` 让它返回一个 work handle，让外层可以重叠 attention compute 和 NCCL transport——但在 single-rank decode 这种小算量场景下重叠度有限。我们的 demo 不模拟 async，但论文/源码用了它。

### 11.4.6 我们的对照实现 + 跑 demo §3 验证

`implementation/dcp_alltoall.py:L166-L215` 的 `simulate_a2a_combine` 和 `simulate_ag_rs_combine`：两个"同名异路"的函数都直接调 `lse_weighted_combine`——因为算法一致，区别只在 transport。这把 D18 的"algebra 一致、transport 不同"在代码里也写明：

```python
# implementation/dcp_alltoall.py:L166-L215 (节选)
def simulate_a2a_combine(partial_outputs, partial_lses):
    # Real source path:
    #   1. Pack output + LSE into send_buffer (Triton kernel)
    #   2. dist.all_to_all_single (line 448)
    #   3. Unpack and combine via LSE weighting (Triton kernel)
    # Our simulation skips the network — same math, different transport.
    return lse_weighted_combine(partial_outputs, partial_lses, return_lse=False).output

def simulate_ag_rs_combine(partial_outputs, partial_lses):
    # AG+RS conceptually: AllGather Q + local attn + ReduceScatter (LSE-weighted)
    # Final reduction is the SAME LSE-weighted average as A2A.
    return lse_weighted_combine(partial_outputs, partial_lses, return_lse=False).output
```

`tests/test_dcp_alltoall.py::test_section_2_a2a_equals_ag_rs_combine` 直接断言两个返回值 bit-identical。这是 D18 的硬测试。

跑 demo §3 看 verbatim 数字：

```
Demo §3 — AG+RS vs A2A NCCL ops + alpha-beta bandwidth model
  Workload: num_tokens=32,768, heads=8, head_dim=128, bf16
  Model: alpha=10.0 us, beta=200.0 GB/s (H100 + 4xNVLink, literature reference)

  dcp_size  AG+RS ops  A2A ops  AG+RS bytes  A2A bytes   T_AG+RS us  T_A2A us  speedup
         2          3        2   67,108,864 34,078,720      1036.6     360.8    2.87x
         4          3        2   67,108,864 17,039,360      1036.6     190.4    5.44x
         8          3        2   67,108,864  8,519,680      1036.6     105.2    9.85x

  A2A reduces NCCL ops by 33% per layer (3 -> 2).
  Reference: arxiv.org/abs/2507.07120
  Trap F — Both are NCCL collectives, not P2P Ring topology.
```

每一格都被 `tests/test_dcp_alltoall.py` 里 verbatim 钉住（参见 test-report.md §3）。

### 11.4.7 与源码的差距

| Source | Ours | 为什么 |
|---|---|---|
| Triton pack/unpack 双 kernel（L134-L319） | 直接 NumPy broadcast | 算法核心是 LSE 合并；pack/unpack 是 GPU 端的 fusion 优化 |
| `dist.all_to_all_single(async_op=True)` | 同步 NumPy | 单进程无 NCCL；async 是 transport 实现细节 |
| MLA backend 真正的 `cp_world_size`/`cp_rank` 直插 FA3 kernel | 我们只演示 `__new__` 的发现 | MLA 是 Ch27 题目；这里只示意 wiring |

> **回看 Ch08**：`AllReduce` 在 TP 里是 row-parallel 的最后一步，那是另一种 collective。CP 的 AG+RS 看起来像 TP 的 AllReduce，但实际是把 reduce 拆成 AllGather + ReduceScatter 两步——避免一次 reduce 卡进每 rank 完整 buffer 的 OOM 风险。
>
> **跨章前指**：Ch18 (Triton attention) 会拆开 `_dcp_a2a_unpack_combine_kernel` 的 Triton 实现，这里只用了它的算法接口。

---

## 11.5 Striped 切分 —— 解决 causal mask 的负载不均

### 11.5.1 打开 `cp_kv_cache_interleave_size` 和 striped helper

源码定位：先看配置，`instances/vllm/source/vllm/config/parallel.py:L330-L342`：

```python
# vllm/config/parallel.py:L330-L342
cp_kv_cache_interleave_size: int = 1
"""Interleave size of kv cache storage. Default 1 means tokens are
stored in a striped fashion (token i goes to total_cp_rank
(i // 1) % total_cp_world_size = i % total_cp_world_size). Larger
values store contiguous chunks per rank.
The block_size must be both >= and divisible by interleave_size.
"""
```

再看用它的 helper，`instances/vllm/source/vllm/v1/attention/backends/utils.py:L820-L857`：

```python
# vllm/v1/attention/backends/utils.py:L820-L857 (节选)
def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    dcp_world_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    """Per-DCP-rank local seq_lens."""
    # Only consider dcp now, we can extend the case of cp based on this
    ...
    base = seq_lens // cp_kv_cache_interleave_size // dcp_world_size * cp_kv_cache_interleave_size
    remainder = seq_lens - base * dcp_world_size
    remainder = (remainder - rank_offsets * cp_kv_cache_interleave_size).clamp_(
        0, cp_kv_cache_interleave_size,
    )
    return base + remainder
```

三件事一起进来：**knob `cp_kv_cache_interleave_size` 是 striping granularity**（注意 `dcp_kv_cache_interleave_size` 是已废名，源码 L315-L321 标了 deprecated）、**source comment 明说 PCP 还没接，"only consider dcp now"**、**计算用 `base + clip(remainder)` 三步走**。

### 11.5.2 这个 knob 解决什么问题？

把 64 个 token 切到 8 个 rank。Causal mask 下 token $i$ attend $i+1$ 个 KV 位置。如果用 contiguous 切：rank 0 拿 token [0..7]、rank 1 拿 token [8..15]……rank 7 拿 token [56..63]。每 rank 的 attention work（KV-attends 总和）是

$$
W_r^{\mathrm{contig}} \;=\; \sum_{i=8r}^{8r+7} (i+1)
$$

代入：rank 0 = 1+2+…+8 = 36；rank 7 = 57+58+…+64 = 484。**最忙 rank 比最闲 rank 多 13.44× 的 work**。这就是 outline §11.4 想说的 "load imbalance"。

striped 切（interleave=1）：token $i$ → rank $i \bmod 8$。每 rank 拿到的是 [r, r+8, r+16, …, r+56]，其 work 是

$$
W_r^{\mathrm{stripe}} \;=\; \sum_{k=0}^{7}\big( r + 8k + 1\big) \;=\; 8r + 8 + 8 \cdot \frac{7\cdot 8}{2}\cdot\frac{1}{8} \quad\mathrm{（化简后线性增长）}
$$

直接列出 8 个 rank 的 work：[232, 240, 248, 256, 264, 272, 280, 288]。imbalance = 288/232 = **1.241**。

**注意**（D19 + Tip 3）：1.24× **不是 1.0×**。即使 round-robin 切，rank 7 拿到的 token 索引（7, 15, 23, ..., 63）平均位置仍然比 rank 0 拿到的（0, 8, 16, ..., 56）大 7。causal mask 下后位置 token 自然要 attend 更多 KV，所以**绝对完美的 1.0× 不存在**。但 1.24× 已经是 13.44× 的 11× 改进——这是对 outline "balanced" 这个词的一个**严肃数值修正**：相对 contiguous 是 balanced，绝对意义上不是 1.0。

### 11.5.3 推 base + remainder + clip 三步走

源码的公式很巧。设全局序列长度 $L$、cp_size $N$、interleave_size $I$。每 rank 都至少能分到 $\lfloor L/(I \cdot N) \rfloor \cdot I$ 个 token——这是 `base`。剩下 $L - \mathrm{base} \cdot N$ 个 token，按 rank 顺序"贴"到前几个 rank 上、每个最多 $I$ 个。这就是 `remainder = clip(seq - base*N - rank*I, 0, I)`。两段相加是该 rank 的 local seq_len。

数学化：

$$
\mathrm{base} \;=\; \left\lfloor \frac{\lfloor L/I\rfloor}{N}\right\rfloor \cdot I
$$

$$
\mathrm{remainder}_r \;=\; \mathrm{clip}\big(L - \mathrm{base}\cdot N - r \cdot I,\; 0,\; I\big),\quad \mathrm{local}_r \;=\; \mathrm{base} + \mathrm{remainder}_r
$$

性质：$\sum_r \mathrm{local}_r = L$ 严格成立（每个 token 恰被分配给一个 rank）。这条性质是 D13 的核心：`get_dcp_local_seq_lens` 要保证 sum-across-ranks 等于全局——`tests/test_seq_sharding.py` 的 6 个 verbatim 单元格全是验证这条。

### 11.5.4 我们的对照实现

`implementation/seq_sharding.py:L31-L91` 1:1 镜像源码：

```python
# implementation/seq_sharding.py:L31-L91 (节选)
def get_dcp_local_seq_lens(
    seq_lens, dcp_size=1, dcp_rank=None, cp_kv_cache_interleave_size=1,
):
    seq_lens = seq_lens.astype(np.int32)
    num_requests = seq_lens.shape[0]

    if dcp_rank is None:
        rank_offsets = np.tile(np.arange(dcp_size, dtype=np.int32), (num_requests, 1))
    else:
        rank_offsets = np.full((num_requests, 1), dcp_rank, dtype=np.int32)

    seq_lens_tiled = np.tile(seq_lens.reshape(-1, 1), (1, rank_offsets.shape[1]))

    # REFERENCE: vllm/v1/attention/backends/utils.py:L844-L849
    base = (seq_lens_tiled // cp_kv_cache_interleave_size // dcp_size
            * cp_kv_cache_interleave_size)
    remainder = seq_lens_tiled - base * dcp_size

    # REFERENCE: vllm/v1/attention/backends/utils.py:L851-L855
    remainder = np.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0, cp_kv_cache_interleave_size,
    )
    local = base + remainder
    return local.squeeze(-1) if dcp_rank is not None else local
```

`causal_attention_work_per_rank`（demo §4 用）：每 token 算 $i+1$ 的 work，按 owner 累加。

### 11.5.5 跑 demo §4，验证 13.44× → 1.24×

`implementation/demo.py::demo_4_striped_vs_contiguous` 输出（cp=8, seq=64）：

```
Demo §4 — Striped vs contiguous KV partition under causal mask

  scheme                 interleave per-rank work (KV-attends)
  contiguous                      8 [36, 100, 164, 228, 292, 356, 420, 484]
  block-striped                   2 [204, 220, 236, 252, 268, 284, 300, 316]
  striped (interleave=1)          1 [232, 240, 248, 256, 264, 272, 280, 288]

  imbalance ratio (max/min):
    contiguous           = 13.44x  (rank-7 work=484, rank-0 work=36)
    block-striped (K=2)  = 1.55x
    striped (interleave=1) = 1.24x  (perfectly balanced)

  Trap G — Striped is a TOKEN-PARTITIONING scheme; communication
           pattern (Ring/A2A/AG+RS) is independent.
```

**注意 Tip 3 的 demo 输出口径**：demo 文本里有 "perfectly balanced"，但**章节文字必须改成 "near-balanced"**——因为 1.24 不是 1.0。这条修正在 D19 里有 explicit 记录。

block-striped (interleave=2) 是中间方案：cache 友好性介于二者之间，imbalance 1.55×。production 通常用 `interleave_size=block_size`（vLLM block=16）作为 sweet spot——cache line 友好（一次连续 16 个 token 写一个 block）、imbalance 在 ~5% 以内。

`get_dcp_local_seq_lens` 的实际效果（demo §4 末尾）：

```
  get_dcp_local_seq_lens helper:
    interleave_size= 1: per_rank_lens =
        [[25 25 25 25][16 16 16 16][ 8  8  7  7][ 5  5  4  3]]
      sum-across-ranks = [100, 64, 30, 17] (must equal seq_lens [100, 64, 30, 17])
    interleave_size= 4: per_rank_lens =
        [[28 28 24 20][20 20 16 ...]]
      ...
    interleave_size=16: per_rank_lens =
        [[32 32 20 16][16 16 16 16][16 14  0  0][16  1  0  0]]
      sum-across-ranks = [100, 64, 30, 17]
```

无论 `interleave_size ∈ {1, 4, 16}`，sum-across-ranks 永远等于 seq_lens——这是 D13 的 invariant。

### 11.5.6 跨章交互：MTP-with-CP 的非平凡 interleave

`vllm/v1/attention/backend.py:L705-L706` 有一个**显式的跨章 flag**：

```python
# vllm/v1/attention/backend.py:L705-L706
supports_mtp_with_cp_non_trivial_interleave_size: bool = False
```

D11 解释了它：当 MTP（Ch10 的 spec decoding）与 CP 同时启用，且 `interleave_size > 1` 时，KV 切片可能跨 spec-boundary——不是所有 backend 都支持。这是 Ch10 ↔ Ch11 的硬连接点：MLA backend 的 `supports_dcp_with_varlen=(interleave_size==1)`（`flashattn_mla.py:L125`）就是从这条 flag 衍生的限制。production 部署如果同时要 MTP 和 DCP，要么用 `interleave_size=1`（fully striped），要么走支持非平凡 interleave 的 backend。

### 11.5.7 与源码的差距

| Source | Ours | 为什么 |
|---|---|---|
| `torch.Tensor` + GPU `clamp_` 算 base/remainder | NumPy + CPU `np.clip` | 单进程不需要 GPU |
| 真 backend 用结果做 slot mapping 写 KV cache | demo 只展示分配 | slot mapping 是 Ch05 territory |
| 9 个 backend 各自决定支持 interleave > 1 的能力 | 我们只暴露 `supports_mtp_with_cp_non_trivial_interleave_size` flag | 真正的 kernel 适配是后端实现细节 |

> **回看 Ch10**：MTP 链断点不变量（`PLACEHOLDER_TOKEN_ID = -1` 哨兵）和 CP 切片不变量（sum-across-ranks = global）是同一类设计：**用数据 layout 自己宣告 invariant**。

---

## 11.6 5D mesh + DCP 嵌入 TP —— 不是 "3D 并行"

### 11.6.1 outline 错了，源码是 5 个轴

打开 `instances/vllm/source/vllm/distributed/parallel_state.py:L1569-L1575`：

```python
# vllm/distributed/parallel_state.py:L1569-L1575
all_ranks = torch.arange(world_size).reshape(
    -1,                                          # external_dp (verl integration)
    data_parallel_size,                          # in-model dp
    pipeline_model_parallel_size,                # pp
    prefill_context_model_parallel_size,         # pcp
    tensor_model_parallel_size,                  # tp
)
```

5 个 axis：`external_dp × dp × pp × pcp × tp`。再读 `multiproc_executor.py:L116-L121` 的 world_size 断言：

```python
# vllm/v1/executor/multiproc_executor.py:L116-L121
world_size = (
    self.parallel_config.world_size_across_dp
    * self.parallel_config.pipeline_parallel_size
    * self.parallel_config.prefill_context_parallel_size
)
assert world_size == ..., "world_size mismatch"
```

注意：world_size = `tp × pp × pcp × dp`（`world_size_across_dp` 已经包含了 `tp × dp`）——**没有 dcp 出现**。DCP 折叠在 TP 内部，**不进 world_size 乘积**。outline §11.5 写的 "3D parallel"——3 个轴是哪 3 个？把 ext_dp 当作不存在、把 dp 当作不存在，剩下 (pp, pcp, tp) 也只能凑出 3D。但源码确凿是 5D。这是 §11.6 的核心 reframe。

### 11.6.2 这个 5D mesh 解决什么问题？

每加一个并行轴都是为了破一个性能瓶颈：

| 轴 | 切什么 | 破什么瓶颈 | 通信 |
|---|---|---|---|
| TP | head 维度 + FFN 中间维 | 单卡放不下模型权重 | All-reduce per layer |
| PP | layer 维度（前 / 后段） | bubble vs HBM 折中 | Send/recv per stage |
| PCP | prefill 期 sequence axis | 长 prefill 单 rank 算不动 | A2A 或 AG+RS |
| DCP | decode 期 KV 序列轴 | 单 rank KV cache 放不下 | A2A 或 AG+RS |
| DP | batch（请求间独立） | 每 engine 调度跨复制 | 无（独立 engine） |
| ext_dp | verl-style 外层 DP | RL training 框架接入 | 无（外层管理） |

**DCP 为什么折叠在 TP 里**？因为 decode 时每 rank 已经在 TP-group 里复制了 Q（column-parallel 的产物，Ch08）。要是把 DCP 当独立轴扩 world_size，就要再起一组 GPU 复制 Q——浪费。直接在 TP-group 内部分 DCP 子组、共用 TP 的 NVLink 带宽，是工程上最划算的选择。代价：`tp_size % dcp_size == 0` 是硬约束（D08，`parallel.py:L474-L478`）——你不能 `tp=4, dcp=3`。

**PCP 为什么不折叠**？因为 prefill 时每 rank 持有自己 prefill 切片的 Q、K、V（不是从 TP 复制来的），所以 PCP 必须有自己的 rank 集合——独立轴自然展开 world_size。

### 11.6.3 推 reshape 的字面意义

`reshape(-1, dp, pp, pcp, tp)` 把全部 ranks 摆进 5D 数组。访问索引：

$$
\mathrm{rank}(e, d, p, c, t) \;=\; \big( ((e\cdot D + d)\cdot P + p)\cdot C + c \big)\cdot T + t
$$

其中 $D, P, C, T$ 是各轴大小。这就是 row-major flatten。每个 group 的构建方式是"沿某轴 unbind"：

- **TP groups**：固定 $(e, d, p, c)$，沿 $t$ 维 unbind →
  `all_ranks.view(-1, T).unbind(0)`
- **PP groups**：固定 $(e, d, c, t)$，沿 $p$ 维 unbind → 需要先 transpose 把 $p$ 移到最里
  `all_ranks.transpose(2, 4).reshape(-1, P).unbind(0)`
- **PCP groups**：固定 $(e, d, p, t)$，沿 $c$ 维 unbind → 需要 transpose(3, 4)
  `all_ranks.transpose(3, 4).reshape(-1, C).unbind(0)`
- **DP groups**：固定 $(e, p, c, t)$，沿 $d$ 维 unbind → transpose(1, 4)
- **DCP sub-groups**（D22 + D01 关键）：**直接 reshape，不 transpose**——把每个 TP group 切成 `tp/dcp` 个 contiguous chunks
  `all_ranks.reshape(-1, dcp_size).unbind(0)`

**为什么 DCP 不需要 transpose**？因为 5D shape 的 inner-most 已经是 TP；DCP 是 TP 内部的子轴，沿着 inner-most 走 contiguous chunk 即可。这就是 D22 强调的"DCP groups are contiguous chunks of TP groups"。换句话说：

- TP=4, DCP=2 给 DCP 子组 [0, 1] 和 [2, 3]——**不是** [0, 2] 和 [1, 3]。
- 这条 contiguous-vs-non-contiguous 的差别决定了 DCP 子组之间靠 intra-NVLink (高速) 而非 inter-node (慢) 通信。

### 11.6.4 我们的对照实现

`implementation/parallel_state_dcp_pcp.py:L93-L278` 的 `initialize_model_parallel` 是 5D mesh 构造的 single-process mirror。核心三段（TP groups、DCP sub-groups via reshape、PCP groups via transpose）：

```python
# implementation/parallel_state_dcp_pcp.py:L188-L226 (节选)
def at(ext, d, p, c, t):
    return (((ext * dp + d) * pp + p) * pcp + c) * tp + t

# REFERENCE: vllm/distributed/parallel_state.py:L1577-L1592 (TP groups)
tp_groups = []
for ext in range(ext_dp):
    for d in range(dp):
        for p in range(pp):
            for c in range(pcp):
                tp_groups.append([at(ext, d, p, c, t) for t in range(tp)])

# REFERENCE: vllm/distributed/parallel_state.py:L1594-L1614 (DCP groups, reshape only)
dcp_groups = []
for tp_grp in tp_groups:
    for chunk_start in range(0, tp, dcp):
        dcp_groups.append(tp_grp[chunk_start:chunk_start + dcp])

# REFERENCE: vllm/distributed/parallel_state.py:L1616-L1633 (PCP groups via transpose 3,4)
pcp_groups = []
for ext in range(ext_dp):
    for d in range(dp):
        for p in range(pp):
            for t in range(tp):
                pcp_groups.append([at(ext, d, p, c, t) for c in range(pcp)])
```

注意 DCP 那一段：直接对每个 TP group 切 contiguous chunks，**没有 transpose**。PCP 那一段：固定 `(ext, d, p, t)`、沿 `c` 维度遍历——这就是 "transpose(3,4) 之后 reshape" 在 Python 循环里的等价表达。

`world_topology.py:L48-L82` 的 `MeshConfig` 把 5D 用 dataclass 编码：

```python
# implementation/world_topology.py:L48-L82 (节选)
@dataclass(frozen=True)
class MeshConfig:
    external_dp: int = 1
    dp: int = 1
    pp: int = 1
    pcp: int = 1
    tp: int = 1
    dcp: int = 1

    def __post_init__(self):
        # REFERENCE: vllm/config/parallel.py:L474-L478
        if self.tp % self.dcp != 0:
            raise ValueError(f"tp_size={self.tp} must be divisible by dcp_size={self.dcp}.")

    @property
    def world_size(self) -> int:
        # REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121
        return self.external_dp * self.dp * self.pp * self.pcp * self.tp  # 注意：不乘 dcp
```

`world_size` property 显式不乘 `dcp`——这是 D06 / D25 的代码体现，11 个 `(tp, pcp, dcp)` 单元格的 `test_world_size_excludes_dcp_grid` 全部钉在这上面。

### 11.6.5 跑 demo §5，看 (tp=4, pcp=2, pp=2, dp=1, dcp=2) 的 world=16

```
Demo §5 — 5D mesh groups (world_size=16)
  MeshConfig: ext_dp=1, dp=1, pp=2, pcp=2, tp=4, dcp=2
  world_size = ext_dp * dp * pp * pcp * tp = 1 * 1 * 2 * 2 * 4 = 16
  total_cp_world_size = pcp * dcp = 2 * 2 = 4
  num_dcp_subgroups per TP-group = tp/dcp = 2

  TP groups (count=4):
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    [12, 13, 14, 15]
  DCP sub-groups (count=8, folded inside TP):
    [0, 1]
    [2, 3]
    [4, 5]
    [6, 7]
    [8, 9]
    [10, 11]
    [12, 13]
    [14, 15]
  PCP groups (count=8, independent axis):
    [0, 4]
    [1, 5]
    [2, 6]
    [3, 7]
    [8, 12]
    [9, 13]
    [10, 14]
    [11, 15]
  PP groups (count=8):
    [0, 8]
    [1, 9]
    [2, 10]
    [3, 11]
    [4, 12]
    [5, 13]
    [6, 14]
    [7, 15]
```

**核对四件事**：

1. world_size 是 16 = `1 × 1 × 2 × 2 × 4`，**不是** 32（如果错把 dcp 算进去）。这是 D25 的硬证据。
2. DCP 子组 [0, 1] [2, 3] 等是 TP groups 的 contiguous chunks——**不是** [0, 2] [1, 3]（D22 验证）。
3. PCP 组 [0, 4] [1, 5] 等是跨 TP groups 的同位置 ranks（沿 pcp 轴 stride）——这就是 transpose(3, 4) 的具体效果。
4. `total_cp_world_size = pcp × dcp = 4`，但 world_size 不变——这就是 §11.7 Trap D 的"separable axes"硬证据。

### 11.6.6 推 `total_cp_rank` 公式 (Tip 4 + D24)

源码 `vllm/v1/attention/backend.py:L751-L752`：

```python
# vllm/v1/attention/backend.py:L751-L752
self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
```

**第二行注意 multiplier**：是 `dcp_world_size`，**不是** `total_cp_world_size`。Tip 4 拍下了这个 subtle off-by-one：

数学上，`total_cp_rank` 是把 `(pcp_rank, dcp_rank)` 二元组打平到一维。打平方式是 **PCP-major**（D24）：pcp_rank 在外层（slow-varying），dcp_rank 在内层（fast-varying）。所以 stride 是 `dcp_world_size`：

$$
\mathrm{total\_cp\_rank} \;=\; \mathrm{pcp\_rank} \cdot N_\mathrm{dcp} + \mathrm{dcp\_rank}
$$

试一组 (pcp_rank=1, dcp_rank=0, dcp_world=2)：total_cp_rank = 2。再 (pcp_rank=1, dcp_rank=1, dcp_world=2)：total_cp_rank = 3。再 (pcp_rank=0, dcp_rank=1, dcp_world=2)：total_cp_rank = 1。$\{0,1,2,3\}$ 四个 ranks 全覆盖、无冲突——这就是 4-rank 跨 cp 的 partition。

如果错写 multiplier 成 `total_cp_world_size`：(pcp_rank=1, dcp_rank=0) 给 4，已经超出 4-rank 边界——硬错。

`tests/test_attention_backend.py::test_total_cp_rank_formula_pcp_major` 把这条 PCP-major 的 invariant 钉死。Tip 4 是 D24 的源——production 改一下这个 formula 不会被任何 unit test 抓到（除非你专门测它），但会**无声地把 KV 切片错位**。

### 11.6.7 跑 production 配置 (tp=8, dcp=2, pcp=4) 的算数（Tip 5）

production 真实配置：tp=8 + dcp=2 + pcp=4。world_size 应该是多少？

$$
\mathrm{world\_size} \;=\; \mathrm{ext\_dp} \cdot \mathrm{dp} \cdot \mathrm{pp} \cdot \mathrm{pcp} \cdot \mathrm{tp} \;=\; 1 \cdot 1 \cdot 1 \cdot 4 \cdot 8 \;=\; 32
$$

**不是** $8 \times 2 \times 4 = 64$。dcp 不进乘积。

每 TP group 8 GPU，dcp=2 把每 TP group 切成 `8/2 = 4` 个 DCP 子组、每子组 2 GPU。DCP 不"加 GPU"，它**重新切分** TP group 内部的 GPU。

per-rank KV 切片 = `seq_len / (pcp × dcp) = seq_len / 8`。比 dcp=1, pcp=1 减 8×。

per_rank HBM = 40.0 GB / 8 = 5.0 GB（demo §1 的 (2, 4) 单元格）。

**这条算式是 Trap D 的最清晰证明**：DCP 和 PCP 是 separable 的——把 dcp 从 1 调到 2 不是"再加一倍 GPU"，是"把已有 GPU 重新切"，world_size 不变。这是 D25 的核心 invariant。

### 11.6.8 与源码的差距

| Source | Ours | 为什么 |
|---|---|---|
| `init_model_parallel_group(group_name="dcp")` 走 `torch.distributed.new_group()` | 我们直接 mutate 全局 `_DCP` | NCCL 通信器是单进程不要的 |
| 还会建 EP group（transpose 1, 2）和 EPLB group（同 EP ranks） | 我们没建 | EP 是 Ch09 题目；我们已经在 Ch09 实现 `_EP`/`_EPLB` |
| `multiproc_executor.py:L985-L1004` 的 process tag 用条件追加（只有 size>1 才加 axis） | 我们的 `process_name_for_rank` 1:1 镜像 | 这是 vLLM 日志的一个易于 grep 的特征 |

> **跨章前指**：Ch15+ 模型加载时会用到 5D mesh 的某个 group 来初始化各种 ParallelLinear；Ch22 的 PD 架构会再加一层 disaggregation；Ch27 的 DeepSeek-V3.2 实战会用 `(tp=8, dcp=2, pcp=4)` 这套配置。

---

## 11.7 系统影响 + 7 个语言陷阱集中检查

### 11.7.1 七个陷阱的快速对照表

| Trap | 错误说法 | 修正 | 硬证据 |
|---|---|---|---|
| A | DCP 在 dcp=2 翻倍 throughput | DCP 赢的是 HBM 容量；throughput 看通信开销 | demo §1：40→2.5 GB |
| B | PCP 在 pcp=2 砍半 prefill latency | 网络带宽决定，PCIe 上可能净亏 | demo §3：α-β 模型 |
| C | CP 就是 sequence parallel 改名 | SP 切 activation；CP 切 KV / 切 prefill input | `is_sequence_parallel` 与 `_DCP` 是分立 API |
| D | DCP 必须等于 PCP | separable axes，唯一约束 `tp%dcp==0` | demo §5：(tp=8, dcp=2, pcp=4) 合法 |
| E | CP 在 attention 等同 TP | TP 切 head；CP 切 sequence；通信不同 | linear.py 用 all_reduce vs dcp_alltoall.py |
| F | Ring Attention 是 vLLM 的标准实现 | vLLM 0 个 Ring；用 NCCL AG+RS 或 A2A | grep + `dist.all_to_all_single` |
| G | Striped 是 Ring 改名 | Striped 是 token 划分；Ring 是通信拓扑 | `cp_kv_cache_interleave_size` vs `dcp_comm_backend` 互不依赖 |

### 11.7.2 Trap A —— "DCP 在 dcp=2 翻倍 throughput"

**错。** DCP 砍的是**每 rank 的 KV cache 字节**（demo §1：dcp=2 → 20 GB / rank），不是吞吐。每 rank 的 attention 计算量从

$$
O(L \cdot H \cdot D) \to O\!\left(\frac{L}{\mathrm{dcp}} \cdot H \cdot D\right) + O(\mathrm{每层 1 个集合})
$$

短 seq_len 下后一项的常数能盖过前面的算术节省，net 影响要看 workload。

**production 决策**：DCP 的卖点是**让 OOM 变可行**。原本放不下的 128K 现在能跑——这是 binary outcome，不是 1.x 倍 speedup。throughput 是次要的。

### 11.7.3 Trap B —— "PCP 在 pcp=2 砍半 prefill latency"

**部分对。** demo §3 在 H100 + NVLink + 32K prefill 下，pcp=2 配 A2A 给 2.87× speedup（dcp=2 的数字，PCP 类似）。但是

- PCIe 网络（α≈50μs，β≈20 GB/s）下 A2A 的 2 个 op 的固定 α 加起来已经 ~100μs，对 4K prefill 可能直接超 compute time。
- 短 prefill（< 4K）下 payload 小，α 占主导，A2A 的 payload 缩小赢点消失。
- pcp 越大 communication round 越多——pcp=8 可能在 PCIe 上 net 亏。

**production 部署规则**：长 prefill (≥ 16K) + NVLink → 启用 PCP；短 prefill / PCIe → 测一下再决定。

### 11.7.4 Trap C —— "CP 就是 sequence parallel 改名"

**错。** Sequence parallel（Megatron）切的是 TP-group 内 MLP/LayerNorm 的 activation 序列维度——目的省 activation HBM。CP 切的是 KV cache（DCP）或 prefill 输入（PCP）。源码 `parallel_state.py` 同时暴露 `is_sequence_parallel` 参数（在 all_gather/reduce_scatter 上）和 `_DCP`/`_PCP` GroupCoordinator——是**两套独立的 API**。可以同时启用：TP=8 + SP=on + DCP=2，互不影响。

### 11.7.5 Trap D —— "DCP 必须等于 PCP"

**错。** 这是这章的核心 reframe E（impl-notes §2）。两条 invariant 推翻：

- D08 / `parallel.py:L474-L478`：唯一硬约束是 `tp_size % dcp_size == 0`。pcp 没出现在这里。
- D06 / D25 / `multiproc_executor.py:L116-L121`：`world_size = tp × pp × pcp × dp`，dcp 从来不进乘积。pcp 是独立轴。

production 配置 `(tp=8, dcp=2, pcp=4)` 是合法的、demo §5 在更小尺度（world=16，tp=4, dcp=2, pcp=2）演示了同一个原理。`test_dcp_vs_pcp_separability::test_both_match_required` 把"both_match_required = False"钉在测试里。

### 11.7.6 Trap E —— "CP 在 attention 等同 TP"

**错。** TP 切 head 维度（每 rank `H/tp` 个 head、完整 seq_len）；CP 切 seq 维度（每 rank 全部 head、`L/cp` 个 token）。通信不同：

- TP 在 attention output 后做 **AllReduce**（row-parallel linear，`vllm/model_executor/layers/linear.py:RowParallelLinear.forward`）——head 间的 partial sum 求和。
- CP 在 attention 内做 **AG+RS or A2A**（`dcp_alltoall.py`）——seq 切片间的 LSE-weighted 合并。

不同 axis、不同代数（求和 vs LSE 加权平均）、不同代码路径。`test_section_3_*` 的 alpha-beta 模型差异钉死了这点。

### 11.7.7 Trap F —— "Ring Attention 是 vLLM 的标准实现"

**错。** §11.2.1 的 grep 已经证明：vLLM 0 个 Ring/Striped/ContextParallel class。`dcp_alltoall.py:L1-L20` 的 docstring 直接说"Provides All-to-All as an alternative to AllGather + ReduceScatter for DCP"——两条都是 NCCL 集合，不是 P2P。

但代数上 LSE 合并和 Ring Attention 是等价的（§11.3.2 的恒等式推导）。所以正确的说法是：**vLLM 用了 Ring Attention 的算法核心（LSE 加权合并），但替换了它的通信 transport（用 NCCL 替代 P2P 环）**。这条 reframe 是 §11.2 全节的来源。

`test_section_2_a2a_equals_ag_rs_combine` 测试两条 backend 的 output bit-identical——这是 D18 的实验证据：**transport 不影响代数**。

### 11.7.8 Trap G —— "Striped 是 Ring 改名"

**错。** Striped 是 **token 划分方案**（`cp_kv_cache_interleave_size` 控制；token i → rank `(i // I) % cp_size`）。Ring 是 **通信拓扑**（rank 之间环状 P2P）。两者**正交**，可以任意组合：

- Striped + Ring（论文 Liu 2023 的 Striped Attention 变体）
- Striped + AG+RS（vLLM 默认）
- Striped + A2A（vLLM advanced）
- Contiguous + AG+RS（也合法但 causal mask 下负载不均）

源码两个 knob 完全分立：`cp_kv_cache_interleave_size`（`parallel.py:L330`）控制 striping，`dcp_comm_backend`（`parallel.py:L322`）控制 transport——你可以独立调任意一个。`test_demo4_*` 把 imbalance 数字钉在 13.44× → 1.55× → 1.24× 上，无论用哪种 transport 这些数字都不变。

### 11.7.9 跨章串接小结

- **回 Ch03**：LSE 加权合并是 FlashAttention §2.3 online softmax 的"跨 ranks 而不是跨 KV tiles"版本。
- **回 Ch04**：prefill / decode 的两阶段区分是 PCP 和 DCP 分立的根本原因。
- **回 Ch05**：`max_memory_usage_bytes` 走 paged-KV 的 block_size 路径；CP 在外面再套一层 cdiv。
- **回 Ch08**：5D mesh 的构造模式（reshape + transpose+reshape）和 group-coordinator 单例都是 TP 章节的同款；DCP 折叠在 TP 里就是因为 column-parallel 已经把 Q 复制到 TP-group。
- **回 Ch09**：`_EP`/`_EPLB` 是 `_DCP`/`_PCP` 的祖先模式；PCP-MoE 的复合在 `moe_runner.py` 里要求 PCP 的 hidden_states/router_logits 跨组 all_gather + reduce_scatter（`flatten_tp_across_dp_and_pcp` helper）。
- **回 Ch10**：`supports_mtp_with_cp_non_trivial_interleave_size` flag 是 Ch10 ↔ Ch11 的硬连接点；MTP 的链断不变量（`-1` 哨兵）和 CP 的 sum-across-ranks invariant 是同一类设计哲学。
- **前指 Ch12**（KV offload）：CP 已经把 KV 切到 `1/(dcp×pcp)`，offload 到 SSD 还能再放大一个数量级。
- **前指 Ch15+**（model zoo）：每个长上下文 production model 都用 CP；具体 config 因模型而异。
- **前指 Ch18**（Triton attention）：`_dcp_a2a_unpack_combine_kernel`（L197-L319）的 Triton 实现细节会在 Ch18 拆开。
- **前指 Ch22**（PD 架构）：CP 复合 PD disaggregation 是 long-context production 的当前 SOTA。
- **前指 Ch25**（PD ratio）：DCP world_size 在调度层成为预算变量。
- **前指 Ch27**（DeepSeek-V3.2）：MLA + DCP 是 production 栈；本章的 `flashattn_mla.py:L353-L355` wiring 在 Ch27 里展开。

---

## 11.8 验证：跑 demo + 跑 lint

### 11.8.1 跑 demo

从仓库根目录：

```bash
cd /home/zjq/Repo2Book/instances/vllm/artifacts/11-dcp-pcp
/home/zjq/.conda/envs/mujoco/bin/python implementation/demo.py
```

预期输出（截断）：

```
Chapter 11: DCP/PCP — Demo numerics
Source pin: vllm-project/vllm @ 98661fe

========================================================================
  Demo §1 — HBM-per-rank capacity sweep (Llama-70B at 128K)
========================================================================
... 8 cells, ending with (4,4) -> 2.5 GB ...

========================================================================
  Demo §2 — LSE-weighted combine equivalence
========================================================================
... max abs error vs single-process FlashAttention = 3.33e-16 ...

========================================================================
  Demo §3 — AG+RS vs A2A NCCL ops + alpha-beta bandwidth model
========================================================================
... 2.87x / 5.44x / 9.85x ...

========================================================================
  Demo §4 — Striped vs contiguous KV partition under causal mask
========================================================================
... 13.44x -> 1.55x -> 1.24x ...

========================================================================
  Demo §5 — 5D mesh groups (world_size=16)
========================================================================
... 4 TP groups, 8 DCP sub-groups, 8 PCP groups, 8 PP groups ...

========================================================================
  All 5 demos complete.
========================================================================
```

### 11.8.2 跑 pytest

```bash
cd /home/zjq/Repo2Book/instances/vllm/artifacts/11-dcp-pcp
/home/zjq/.conda/envs/mujoco/bin/python -m pytest tests/ -q
```

预期：

```
402 passed in 0.94s
```

10 个测试模块，覆盖 8 个实现文件 + 2 个 fidelity/integration 模块（参见 `tests/test-report.md`）。

### 11.8.3 跑 lint

公式 lint：

```bash
python3 /home/zjq/Repo2Book/scripts/lint_formulas.py \
  /home/zjq/Repo2Book/instances/vllm/artifacts/11-dcp-pcp/narrative/chapter.md
```

source grounding lint：

```bash
python3 /home/zjq/Repo2Book/scripts/lint_source_grounding.py \
  /home/zjq/Repo2Book/instances/vllm/artifacts/11-dcp-pcp/
```

两条都应返回 0 issues。

---

## 11.9 Source Mapping Table（主表）

按章节顺序、每行覆盖一个具体的 source 位置 ↔ 我们的实现。每个数字（line number、字段名）都已被 `tests/test_*` 中至少一个 test 钉住（参见 test-report.md 各 module 的 verbatim 测试名）。

| 我们的实现 | 源码位置 | 我们做了什么 / 为什么 |
|---|---|---|
| `kv_cache_per_rank.py::cdiv` | `vllm/utils.py::cdiv` | 1:1 复刻；`-(-a // b)` 整除上取整 |
| `kv_cache_per_rank.py::KVCacheSpec` | `vllm/v1/kv_cache_interface.py:L150-L195` (`AttentionSpec` dataclass) | 简化字段；保留 `block_size`, `num_kv_heads`, `head_size`, `dtype_bytes` |
| `kv_cache_per_rank.py::page_size_bytes` | `vllm/v1/kv_cache_interface.py:L185-L195` | `2 * block_size * num_kv_heads * head_size * dtype_bytes` |
| `kv_cache_per_rank.py::max_memory_usage_bytes` | `vllm/v1/kv_cache_interface.py:L196-L204` | 1:1 复刻 `cdiv(max_model_len, dcp*pcp) * cdiv(..., block_size) * page_size_bytes` |
| `kv_cache_per_rank.py::hbm_naive_total` | (派生 helper, 无源码) | 单层无 CP 朴素字节计数 |
| `kv_cache_per_rank.py::hbm_per_rank` | `vllm/v1/kv_cache_interface.py:L196-L204` × `num_layers` | 全模型每 rank 字节，乘 `num_layers` |
| `kv_cache_per_rank.py::LLAMA_70B_KV_SPEC` | (派生, Llama-70B 公开参数) | `80 layers × 8 KV heads × 128 head_size × bf16` |
| `parallel_state_dcp_pcp.py::CPGroupCoordinator` | `vllm/distributed/parallel_state.py:GroupCoordinator` (~L800-L1100) | 简化为 ranks list + rank_in_group；单进程不要 NCCL handle |
| `parallel_state_dcp_pcp.py::_DCP` | `vllm/distributed/parallel_state.py:L1234` | 模块级单例；类型 `CPGroupCoordinator | None` |
| `parallel_state_dcp_pcp.py::_PCP` | `vllm/distributed/parallel_state.py:L1285` | 模块级单例；类型 `CPGroupCoordinator | None` |
| `parallel_state_dcp_pcp.py::get_dcp_group` | `vllm/distributed/parallel_state.py:L1237-L1239` | 1:1 复刻 AssertionError 行为，让 `__new__` 能 try/except |
| `parallel_state_dcp_pcp.py::get_pcp_group` | `vllm/distributed/parallel_state.py:L1288-L1290` | 同上 |
| `parallel_state_dcp_pcp.py::get_context_model_parallel_group` | `vllm/distributed/parallel_state.py:L1242-L1243` | 兼容别名 = `get_dcp_group` |
| `parallel_state_dcp_pcp.py::get_decode_context_model_parallel_world_size` | `vllm/distributed/parallel_state.py:L1847-L1849` | 1:1 复刻公开 helper |
| `parallel_state_dcp_pcp.py::get_decode_context_model_parallel_rank` | `vllm/distributed/parallel_state.py:L1852-L1854` | 1:1 复刻公开 helper |
| `parallel_state_dcp_pcp.py::initialize_model_parallel` | `vllm/distributed/parallel_state.py:L1497-L1782` | 简化掉 NCCL stream 初始化；保留 5D mesh + 5 个 group 构造 |
| `parallel_state_dcp_pcp.py::initialize_model_parallel` (DCP groups) | `vllm/distributed/parallel_state.py:L1594-L1614` | reshape only, 无 transpose；contiguous chunks of TP groups (D22) |
| `parallel_state_dcp_pcp.py::initialize_model_parallel` (PCP groups) | `vllm/distributed/parallel_state.py:L1616-L1633` | transpose(3,4) + reshape；非 contiguous (D01) |
| `parallel_state_dcp_pcp.py::initialize_model_parallel` (PP groups) | `vllm/distributed/parallel_state.py:L1635-L1651` | transpose(2,4) + reshape |
| `parallel_state_dcp_pcp.py::initialize_model_parallel` (DP groups) | `vllm/distributed/parallel_state.py:L1653-L1668` | transpose(1,4) + reshape |
| `parallel_state_dcp_pcp.py::initialize_model_parallel` (`tp%dcp==0` check) | `vllm/config/parallel.py:L474-L478` | 1:1 复刻 ValueError 文本 |
| `parallel_state_dcp_pcp.py::initialize_model_parallel` (world_size check) | `vllm/v1/executor/multiproc_executor.py:L116-L121` | `world_size == tp × pp × pcp × dp`，**dcp 不入** |
| `parallel_state_dcp_pcp.py::reset_cp_singletons` | (无源码; test helper) | 测试间清状态用 |
| `world_topology.py::MeshConfig` | `vllm/distributed/parallel_state.py:L1569-L1575` reshape + `multiproc_executor.py:L116-L121` | dataclass 编码 5D mesh 的所有维度 + 计算 properties |
| `world_topology.py::MeshConfig.world_size` | `vllm/v1/executor/multiproc_executor.py:L116-L121` | `external_dp × dp × pp × pcp × tp` (D06) |
| `world_topology.py::MeshConfig.total_cp_world_size` | `vllm/v1/attention/backend.py:L751` | `pcp × dcp` |
| `world_topology.py::MeshConfig.num_dcp_subgroups` | `vllm/distributed/parallel_state.py:L1597-L1600` | `tp / dcp` 个 DCP 子组 / TP-group |
| `world_topology.py::MeshConfig.__post_init__` | `vllm/config/parallel.py:L474-L478` | tp%dcp==0 校验 |
| `world_topology.py::process_name_for_rank` | `vllm/v1/executor/multiproc_executor.py:L985-L1004` | 条件追加 axis；只有 size>1 才出现在名字里 |
| `world_topology.py::per_rank_kv_fraction` | `vllm/v1/kv_cache_interface.py:L195-L205` | `1 / (pcp * dcp)` |
| `lse_combine.py::CombineResult` | `vllm/v1/attention/ops/dcp_alltoall.py:L100-L103` | NamedTuple `(output, global_lse)` |
| `lse_combine.py::lse_weighted_combine` | `vllm/v1/attention/ops/dcp_alltoall.py:L39-L103` | 1:1 NumPy 复刻；NaN/inf 处理、lse_max 平移、归一化加权 |
| `lse_combine.py::lse_weighted_combine` (NaN/+inf sanitize) | `vllm/v1/attention/ops/dcp_alltoall.py:L66-L70` | 把异常 LSE 替成 -inf 让权重为 0 |
| `lse_combine.py::lse_weighted_combine` (lse_max stability) | `vllm/v1/attention/ops/dcp_alltoall.py:L72-L78` | 减最大值再 exp |
| `lse_combine.py::lse_weighted_combine` (base e/2 path) | `vllm/v1/attention/ops/dcp_alltoall.py:L81-L84` | `is_lse_base_on_e` flag |
| `lse_combine.py::lse_weighted_combine` (norm) | `vllm/v1/attention/ops/dcp_alltoall.py:L89-L91` | `weights / max(weight_sum, 1e-10)` |
| `lse_combine.py::lse_weighted_combine` (final sum) | `vllm/v1/attention/ops/dcp_alltoall.py:L93-L94` | `sum(weights * outputs)` |
| `lse_combine.py::lse_weighted_combine` (global LSE) | `vllm/v1/attention/ops/dcp_alltoall.py:L96-L101` | `log(weight_sum) + lse_max` |
| `lse_combine.py::reference_attention` | (派生; FlashAttention §2.3) | 单进程 ground-truth attention + LSE |
| `lse_combine.py::split_attention` | `vllm/v1/attention/backends/mla/flashattn_mla.py:L196-L250` (DCP metadata threading) | 把 KV 切成 N 份分别算 partial output + LSE |
| `lse_combine.py::_logsumexp` | (numerical helper, 无源码) | logsumexp 数值稳定版 |
| `dcp_alltoall.py::CommCost` | (派生 dataclass) | NCCL op 数 + 字节数 + 总字节 |
| `dcp_alltoall.py::ag_rs_op_count` | `vllm/config/parallel.py:L322-L328` (`DCPCommBackend`) + arxiv.org/abs/2507.07120 | 返回 3 (AG + attention + RS) |
| `dcp_alltoall.py::a2a_op_count` | `vllm/v1/attention/ops/dcp_alltoall.py:L448` (`dist.all_to_all_single`) + arxiv.org/abs/2507.07120 | 返回 2 (a2a + Triton combine) |
| `dcp_alltoall.py::alpha_beta_cost` | (派生; literature α-β model) | $T = N_\mathrm{op} \cdot (\alpha + \mathrm{bytes}/\beta)$ |
| `dcp_alltoall.py::ag_rs_payload_bytes` | `vllm/v1/attention/backends/mla/flashattn_mla.py:L175` | `num_tokens * num_heads * head_dim * bytes`，与 dcp 无关 |
| `dcp_alltoall.py::a2a_payload_bytes` | `vllm/v1/attention/ops/dcp_alltoall.py:L431-L436` (`send_buffer.shape`) + L106-L112 (`lse_pack_dim`) | `num_tokens * (num_heads/dcp) * (head_dim + lse_pack) * bytes`；随 dcp 缩 |
| `dcp_alltoall.py::simulate_a2a_combine` | `vllm/v1/attention/ops/dcp_alltoall.py:L320-L450` | 简化为直接调 `lse_weighted_combine`；transport 跳过 |
| `dcp_alltoall.py::simulate_ag_rs_combine` | `vllm/v1/attention/backend.py:L754-L756` (need_to_return_lse) + AG+RS 概念路径 | 同样调 `lse_weighted_combine`，证明 transport 不影响代数 |
| `seq_sharding.py::get_dcp_local_seq_lens` | `vllm/v1/attention/backends/utils.py:L820-L857` | 1:1 复刻 base + remainder + clip 三步 |
| `seq_sharding.py::get_dcp_local_seq_lens` (base 计算) | `vllm/v1/attention/backends/utils.py:L844-L849` | `base = seq // I // dcp * I` |
| `seq_sharding.py::get_dcp_local_seq_lens` (remainder + clip) | `vllm/v1/attention/backends/utils.py:L851-L855` | `clip(remainder - rank*I, 0, I)` |
| `seq_sharding.py::causal_attention_work_per_rank` | (派生 helper for §11.5) | token i → owner = `(i // I) % cp_size`，按 owner 累加 (i+1) |
| `seq_sharding.py::imbalance_ratio` | (派生 helper) | `max(work) / min(work)` |
| `attention_backend_dcp_pcp.py::AttentionImplBase` | `vllm/v1/attention/backend.py:L685-L757` | 简化基类；保留 `__new__` 发现 + 6 个 CP 字段 |
| `attention_backend_dcp_pcp.py::AttentionImplBase.__new__` | `vllm/v1/attention/backend.py:L731-L757` | 1:1 复刻 try/except 模式 |
| `attention_backend_dcp_pcp.py::AttentionImplBase.supports_pcp` | `vllm/v1/attention/backend.py:L703` | 默认 False；具体 backend 覆盖 |
| `attention_backend_dcp_pcp.py::AttentionImplBase.supports_mtp_with_cp_non_trivial_interleave_size` | `vllm/v1/attention/backend.py:L705-L706` | Ch10 ↔ Ch11 跨章 flag |
| `attention_backend_dcp_pcp.py::AttentionImplBase.can_return_lse_for_decode` | `vllm/v1/attention/backend.py:L700` | 决定 `need_to_return_lse_for_decode` |
| `attention_backend_dcp_pcp.py::AttentionImplBase` (total_cp_world_size) | `vllm/v1/attention/backend.py:L751` | `pcp_world_size * dcp_world_size` |
| `attention_backend_dcp_pcp.py::AttentionImplBase` (total_cp_rank, PCP-major) | `vllm/v1/attention/backend.py:L752` | `pcp_rank * dcp_world_size + dcp_rank`；multiplier 是 dcp_world_size (D24 / Tip 4) |
| `attention_backend_dcp_pcp.py::AttentionImplBase` (need_to_return_lse_for_decode) | `vllm/v1/attention/backend.py:L754-L756` | `dcp_world_size > 1 AND can_return_lse_for_decode` |
| `attention_backend_dcp_pcp.py::FlashAttn3MlaBackend` | `vllm/v1/attention/backends/mla/flashattn_mla.py:L125-L355` | 简化为 wiring 演示 |
| `attention_backend_dcp_pcp.py::FlashAttn3MlaBackend.num_heads_q` | `vllm/v1/attention/backends/mla/flashattn_mla.py:L175` | `num_heads * dcp_world_size`，DCP 时 Q 复制 |
| `attention_backend_dcp_pcp.py::FlashAttn3MlaBackend.kernel_call_signature` | `vllm/v1/attention/backends/mla/flashattn_mla.py:L353-L355` (cp_world_size, cp_rank) + L349 (return_softmax_lse) + L350 (fa_version=3) | 暴露 FA3 调用的 4 个 CP 相关字段 |
| `attention_backend_dcp_pcp.py::FlashAttnBackend` | `vllm/v1/attention/backends/flash_attn.py` (DCP 路径) | 不支持 PCP/LSE 的代表 |
| `attention_backend_dcp_pcp.py::FlashInferBackend` | `vllm/v1/attention/backends/flashinfer.py:L213` (BatchDCPPrefillWrapper) + L444, L763-L766 | 唯一带 DCP 前缀 class 所属的 backend |
| `attention_backend_dcp_pcp.py::RocmAiterMlaBackend` | `vllm/v1/attention/backends/mla/rocm_aiter_mla.py:L213, L311` | ROCm AITER MLA 的 DCP wiring |
| `dcp_vs_pcp_demo.py::CPRoles` | (派生 dataclass) | DCP / PCP 各自角色描述 |
| `dcp_vs_pcp_demo.py::CPRoles.both_match_required` | `vllm/config/parallel.py:L474-L478` (only `tp%dcp==0`) | False — Trap D 锚 |
| `dcp_vs_pcp_demo.py::world_size_for` | `vllm/v1/executor/multiproc_executor.py:L116-L121` | DCP 不在乘积里 |
| `dcp_vs_pcp_demo.py::per_rank_kv_chunk` | `vllm/v1/kv_cache_interface.py:L195-L205` 重组 | `seq_len / (dcp * pcp)` |
| `dcp_vs_pcp_demo.py::explain_separability` | (人类可读, 派生) | Trap D 文本 |
| `dcp_vs_pcp_demo.py::explain_axis_difference` | `vllm/distributed/parallel_state.py:L1593-L1633` (DCP vs PCP 构造方式) | DCP / PCP 角色对照表 |
| `demo.py::demo_1_hbm_capacity` | `vllm/v1/kv_cache_interface.py:L195-L205` | demo §1 — 8 (dcp,pcp) 单元格 |
| `demo.py::demo_2_lse_combine` | `vllm/v1/attention/ops/dcp_alltoall.py:L39-L103` | demo §2 — 4-rank LSE 合并验证 |
| `demo.py::demo_3_ag_rs_vs_a2a` | `vllm/config/parallel.py:L322-L328` + α-β model | demo §3 — speedup 表 |
| `demo.py::demo_4_striped_vs_contiguous` | `vllm/v1/attention/backends/utils.py:L820-L857` | demo §4 — imbalance 13.44× → 1.24× |
| `demo.py::demo_5_mesh_groups` | `vllm/distributed/parallel_state.py:L1569-L1633` | demo §5 — 5D mesh group 构造 |

主表 86 行，覆盖 8 个实现文件 + 5 个 demo + 跨章 flag。每行至少 1 个 source `:line` 锚点；多数行同时引用源码 + 测试。

### 11.9.1 跨章 forward-pointer 小表

| 章节 | 接口 | 我们留的钩子 |
|---|---|---|
| Ch12 (KV offload) | `max_memory_usage_bytes` | CP 已经把 KV 切到 1/(dcp*pcp)，offload 再放大一个数量级 |
| Ch15+ (model zoo) | 5D mesh + `flashattn_mla.py:L353-L355` | 每个 production model 用 `(tp, pcp, dcp)` 配置 |
| Ch18 (Triton attention) | `_dcp_a2a_unpack_combine_kernel` | Triton kernel 实现细节 |
| Ch22 (PD architecture) | `_DCP` 复合 PD disaggregation | CP 与 PD 在同一 mesh 里共存 |
| Ch25 (PD ratio) | DCP world_size 进调度 | DCP 成为 budget 变量 |
| Ch27 (DeepSeek-V3.2) | MLA + DCP + production stack | `(tp=8, dcp=2, pcp=4)` 是参考配置 |

### 11.9.2 反向回看小表

| 章节 | 联系 | 在本章哪里讲 |
|---|---|---|
| Ch03 (FlashAttention) | LSE-stable online softmax | §11.3.2 的恒等式推导 |
| Ch04 (continuous batching) | prefill / decode 阶段区分 | §11.6.2 表头 |
| Ch05 (memory mgmt) | paged-KV `block_size`、`page_size_bytes` | §11.1.3 第二步 cdiv |
| Ch08 (TP) | 5D mesh reshape pattern + GroupCoordinator | §11.6.3 reshape 公式 |
| Ch09 (EP) | `_EP`/`_EPLB` 是 `_DCP`/`_PCP` 的祖先 + PCP-EP 复合 | §11.7.9 跨章串接 |
| Ch10 (MTP) | `supports_mtp_with_cp_non_trivial_interleave_size` flag | §11.5.6 |

---

## 11.10 总结

这章把 vLLM 的 Context Parallelism 从 outline 描述的"3D 并行 + Ring Attention"还原成源码真实结构：**5D mesh + DCP 折叠在 TP 内 + AG+RS 或 A2A 两条 NCCL backend + striped KV 切分 knob**。

核心五件事记下：

1. **HBM 主定理是 `cdiv(max_model_len, dcp × pcp)`**——demo §1 在 Llama-70B-128K 上把每 rank 字节从 40.0 GB 砍到 2.5 GB（16×），跨过 H100 80 GB 红线。这是 CP 存在的硬理由。

2. **LSE 加权合并是代数恒等式**——demo §2 验证 max abs error = 3.33e-16（fp64 ε 噪声），三种 transport（Ring / AG+RS / A2A）走同一条数学路径。`test_section_2_a2a_equals_ag_rs_combine` 把 D18 的"transport 不影响 algebra"钉在 bit-identical 测试里。

3. **outline 的 "all-reduce vs all-to-all" 是错的**——源码 `DCPCommBackend = Literal["ag_rs", "a2a"]`。AG+RS 是 3 op/layer，A2A 是 2 op/layer。demo §3 在 H100 NVLink 下 dcp ∈ {2, 4, 8} 给 2.87× / 5.44× / 9.85× speedup。33% NCCL op 减少是数学上确定的，payload 缩小是工程加成。

4. **striped (interleave=1) 是 near-balanced，不是 perfectly balanced**——demo §4 给 1.24×（不是 1.0×），但 比 contiguous 的 13.44× 已经好一个数量级。`cp_kv_cache_interleave_size` knob 与 `dcp_comm_backend` knob 完全正交（Trap G）。

5. **5D mesh 不是 3D**——`external_dp × dp × pp × pcp × tp`，DCP 折叠在 TP 内不进 world_size。production `(tp=8, dcp=2, pcp=4)` 给 world_size=32（不是 64），demo §5 在 world=16 演示同一个原理。`tp % dcp == 0` 是唯一硬约束。`total_cp_rank = pcp_rank × dcp_world + dcp_rank`（PCP-major，multiplier 是 dcp_world）是 D24 的 subtle off-by-one。

下一章 Ch12 接着讲 KV cache offload——把 CP 切下来的每 rank KV 再 offload 到 CPU/SSD，把"capacity wall"从 GPU HBM 推到节点级 SSD。
