# 第12章：KV Cache Offload —— 当 outline 在四个 TOPIC 上走偏，源码长这样

> 本章涉及的 vLLM 源码（commit `98661fe`）：
> - `instances/vllm/source/vllm/v1/kv_offload/base.py:L24-L44`（`OffloadKey = NewType("OffloadKey", bytes)` + `make_offload_key` / `get_offload_block_hash` / `get_offload_group_idx` —— 把 (block_hash, group_idx) 打包成单个 bytes，省一次 tuple 分配）+ `L48-L66`（`ReqContext` + `LoadStoreSpec` ABC + `medium()` 抽象）+ `L68-L80`（`PrepareStoreOutput` + `OffloadingEvent` 数据契约）+ `L110-L218`（`OffloadingManager` ABC：`lookup` / `prepare_load` / `touch` / `complete_load` / `prepare_store` / `complete_store` / `take_events` / `shutdown` 八个动词）+ `L219-L266`（`BlockIDsLoadStoreSpec` + `GPULoadStoreSpec` + `block_ids` 防御拷贝）+ `L269-L316`（`CanonicalKVCaches` 把 FlashAttention 的 `(2, num_blocks, ...)` 和 FlashInfer 的 `(num_blocks, ...)` 都规范化成 `(num_blocks, page_size_bytes)`）+ `L319-L398`（`OffloadingSpec` ABC + `get_manager()` / `get_handlers()` 工厂契约 + `block_size_factor` 校验）
> - `instances/vllm/source/vllm/v1/kv_offload/factory.py:L17-L52`（`OffloadingSpecFactory` 懒加载注册表 —— `register_spec(name, module_path, class_name)` 注册时存 closure，不导入；`create_spec` 时才 `importlib.import_module`）+ `L55-L58`（CPUOffloadingSpec 规范注册）
> - `instances/vllm/source/vllm/v1/kv_offload/cpu/spec.py:L22-L102`（`CPUOffloadingSpec(OffloadingSpec)` —— 算 `num_blocks = cpu_bytes_to_use // (kv_bytes_per_block × block_size_factor)`，按 `eviction_policy` 字符串 `"lru"|"arc"` 拼装 manager）+ `L70-L82`（`store_threshold ≥ 2` 时套一层 `FilterReusedOffloadingManager`）+ `L100-L102`（`get_handlers` yield 两个方向 `(GPULoadStoreSpec, CPULoadStoreSpec, gpu_to_cpu_handler)` 和反向）
> - `instances/vllm/source/vllm/v1/kv_offload/cpu/manager.py:L25-L200`（`CPUOffloadingManager`：block-id pool + `_free_list` + `_num_allocated_blocks` + 懒分配 + 事件队列）+ `L91-L103`（`prepare_load` 把 `ref_cnt` 加 1 防 eviction）+ `L115-L168`（`prepare_store` 计算 `num_blocks_to_evict = len(keys_to_store) - free_blocks`；如果 `policy.evict` 返回 None 则原子 abort 不改状态；evicted_keys 提前回到调度器）+ `L170-L195`（`complete_store` 成功 flip `-1 → 0`，失败静默 free，**不发事件**）
> - `instances/vllm/source/vllm/v1/kv_offload/cpu/policies/base.py:L10-L33`（`BlockStatus` —— 16 字节 ctypes 结构体；`ref_cnt = -1` 是「reserved 但还没 loadable」的 sentinel，`is_ready = ref_cnt >= 0`）+ `L36-L77`（`CachePolicy` ABC，故意把 organization 和 eviction 揉在一起，因为 ARC 的 ghost-list adaptation 在两者交叉处）
> - `instances/vllm/source/vllm/v1/kv_offload/cpu/policies/lru.py:L10-L46`（`LRUCachePolicy`：单 OrderedDict；`touch` **倒序**遍历 keys 让 LAST key 落到 MRU 端 —— O19 知识点；`evict(n, protected)` 返回 `None` 当不够 idle blocks）
> - `instances/vllm/source/vllm/v1/kv_offload/cpu/policies/arc.py:L10-L156`（`ARCCachePolicy` —— Megiddo & Modha 2003 ARC：T1（recent）+ T2（frequent）+ B1（T1 ghost）+ B2（T2 ghost）；adaptive `target_t1_size`；B1 hit → recency 赢 → 涨 `target_t1_size += max(1, |B2|/|B1|)`；B2 hit → frequency 赢 → 减；evict 走两阶段 dry-run + apply 保 None-on-fail 原子）
> - `instances/vllm/source/vllm/v1/kv_offload/cpu/shared_offload_region.py:L27-L113`（`SharedOffloadRegion` —— 启动时 mmap 一个 `/dev/shm/vllm_offload_{instance_id}.mmap`；`MADV_POPULATE_WRITE`（Linux 5.14+，值 23）预先 fault；layout 按 stride 切，每个 worker 拿一行）
> - `instances/vllm/source/vllm/v1/kv_offload/cpu/gpu_worker.py:L111-L173`（`SingleDirectionOffloadingHandler` —— **一个方向一个 CUDA stream**；in-direction 内顺序通过 `stream.wait_event(prev_end)` 强制串行）+ `L308-L321`（`cuda.Event.record` 标记完成时刻）+ `L375-L433`（`CpuGpuOffloadingHandlers` 双向 bundle：`gpu_to_cpu_handler` + `cpu_to_gpu_handler`）
> - `instances/vllm/source/vllm/v1/kv_offload/worker/worker.py:L9-L23`（`TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]` + `TransferType = tuple[str, str]`）+ `L26-L74`（`OffloadingHandler` ABC：`transfer_async` / `get_finished` / `wait`）+ `L77-L177`（`OffloadingWorker` 按 `(src.medium(), dst.medium())` 路由到对应 handler）
> - `instances/vllm/source/vllm/v1/kv_offload/reuse_manager.py:L23-L120`（`FilterReusedOffloadingManager` 装饰器：`lookup` 同时是计数器 incrementer —— O21 知识点；`prepare_store` 只放行 `counts.get(k, 0) >= store_threshold` 的 keys）
> - `instances/vllm/source/vllm/v1/simple_kv_offload/manager.py:L67-L742`（`SimpleCPUOffloadScheduler` —— 教学版：单 OrderedDict + `OffloadMode.{LAZY, EAGER}` + `target_free` watermark；和 v1/kv_offload/ 是**两条独立路径**）
> - `instances/vllm/source/vllm/v1/simple_kv_offload/copy_backend.py:L43-L44`（`DmaCopyBackend` —— **load_stream 与 store_stream 显式分开**：`self.load_stream = torch.cuda.Stream()`；`self.store_stream = torch.cuda.Stream()`，这是两路 PCIe 同时跑的物理基础）
> - `instances/vllm/source/vllm/v1/simple_kv_offload/cuda_mem_ops.py:L16-L25`（`pin_tensor` 的 docstring：明确说**绕过** `pin_memory=True` 的 power-of-2 rounding —— 100 GB pin 会变 128 GB；改用 `cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)`）
> - `instances/vllm/source/vllm/v1/simple_kv_offload/worker.py:L1-L305`（`SimpleCPUOffloadWorker` + `register_kv_caches` 钉住 GPU 张量）
> - `instances/vllm/source/vllm/distributed/kv_transfer/kv_connector/v1/base.py:L42-L83`（`KVConnectorMetadata`）+ `L84-L115`（`SupportsHMA` —— Hybrid Memory Allocation 的 marker mixin）+ `L123-L130`（`KVConnectorRole` 枚举：SCHEDULER / WORKER）+ `L170-L660`（`KVConnectorBase_V1` —— 30+ 抽象/模板方法）+ `L298-L362`（`start_load_kv` / `wait_for_layer_load` / `save_kv_layer` 这条 worker 流水）+ `L449-L506`（`get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta` 这条 scheduler 流水）
> - `instances/vllm/source/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L51-L67`（`OffloadingConnector(KVConnectorBase_V1, SupportsHMA)` —— role-conditional：scheduler 角色构造 `OffloadingConnectorScheduler`，worker 角色构造 `OffloadingConnectorWorker`）
> - `instances/vllm/source/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:L1-L247`（`SimpleCPUOffloadConnector` —— 配 `vllm/v1/simple_kv_offload/` 的最小教学连接器）
> - `instances/vllm/source/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py:L1-L629`（`MultiConnector(KVConnectorBase_V1, SupportsHMA)` —— 把每个 lifecycle 方法 fan-out 到一组子 connector，dedupe by block-hash）
> - `instances/vllm/source/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:L1-L354`（`LMCacheConnectorV1` —— 18 个 connector 里的 production 旗舰，本章只做 forward pointer，不做 deep dive）
> - `instances/vllm/source/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L191-L210`（`OffloadingConnectorScheduler.__init__` 把 group 拆成 full_attention 组和 sliding_window 组，sliding 组按 window size 倒序排）+ `L244-L261`（`_maximal_prefix_lookup` —— **REACTIVE prefix scan**，循环调 `manager.lookup(key)`，碰到 miss 就停；defer-lookup（`return None`）只是让后端 pipeline cache-line warming）+ `L263-L287`（`_sliding_window_lookup` 从尾倒扫，找连续 `sliding_window_size` 个 hit 的 suffix）+ `L289-L303`（`_touch` 调 LRU/ARC 的 organize）+ `L443-L486`（`get_num_new_matched_tokens` 公开入口，返回 `(num_tokens, is_async)`）
> - `instances/vllm/source/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py:L1-L370`（`OffloadingConnectorWorker.handle_preemptions` + `start_kv_transfers`）
> - 对照实现 `instances/vllm/artifacts/12-kv-offload/implementation/`：`offload_spec.py`、`offload_manager.py`、`policies.py`、`factory.py`、`reuse_manager.py`、`cpu_gpu_worker.py`、`offloading_scheduler.py`、`simple_offload_manager.py`、`connector_taxonomy.py`、`demo.py`
>
> 第 7 章用「vLLM 没有 radix tree」开篇，第 8 章「没有 `class TensorParallel`」，第 9 章「没有 `class ExpertParallel`」，第 10 章「没有 `class MultiTokenPrediction`」，第 11 章「没有 `class RingAttention` / `class ContextParallel`」——这条 N=5 的「class 缺位」母题在第 12 章 **不延续**。KV Offload 在源码里诚实地命名：`OffloadingManager`、`CPUOffloadingManager`、`OffloadingSpec`、`OffloadingHandler`、`SingleDirectionOffloadingHandler`、`CpuGpuOffloadingHandlers`、`OffloadingWorker`、`OffloadingConnector`、`KVConnectorBase_V1`、`SupportsHMA`、`MultiConnector`，再加 18 个具体 connector 实现。本章不强行做第 6 个「class 缺位」——那会是不诚实的。

> 但本章另起一条母题：**outline 在 4 个 TOPIC 上走偏，每一处都被源码就地纠正**。
>
> 1. §12.1 outline 说 "GPU HBM → CPU DRAM → NVMe SSD 三级"，源码（`vllm/v1/kv_offload/`）grep 后**零** `nvme | ssd | disk | fs_offload`：vLLM 此 commit 是**两级**（HBM ↔ CPU pinned）。NVMe / CXL / NVMe-over-fabric 是学术研究方向。
> 2. §12.2 outline 说 "LRU/LFU/attention-score-based 选择策略"，源码 `cpu/policies/` 目录只有 `base.py + lru.py + arc.py`：**没有 `lfu.py`**，**没有 `attention_score.py`**。ARC（Megiddo-Modha 2003）才是 production sophisticated 替代，本章以诚实姿态走完 LRU + ARC，并给 ARC **诚实的 caveat**：在 phase_shift workload 上 ARC 反而**输给** LRU（demo §2 数据：LRU 2.60% miss vs ARC 14.15% miss）。
> 3. §12.2 outline 说 "attention-score-based"，源码完全用 **block-hash 语义**。token 级 attention statistic 的 H2O / HeavyHitter / StreamingLLM 是研究论文，不是 vLLM。
> 4. §12.3 outline 说 "predict 哪些 KV block 会用到"，源码（`offloading/scheduler.py:L244-L261`）的 `_maximal_prefix_lookup` 是**确定性 prefix scan**：`for key in keys: result = manager.lookup(key, ctx); if not result: break`。**没有** Markov 链、**没有** ML 预测、**没有** 请求模式学习。vLLM 是 **REACTIVE block-hash matching**，不是 PREDICTIVE。
>
> 这 4 处不是「class 缺位」，是 **TOPIC 缺位** —— outline 假设的某些技术存在感在源码里没有出现。本章把每处都做成「outline-corrects-itself moment」：先承认 outline 这么写、给学术 sidebar、再 pivot 到源码真实路径，全程引用 grep 输出做硬证据。这是和前 5 章不同的处理方式，但保留同一种诚实姿态：**章节叙事必须忠于 commit 98661fe 的代码事实，而不是忠于 outline 的措辞**。

---

## 这章要讲什么？

打开 `instances/vllm/source/vllm/v1/kv_offload/cpu/manager.py:L115-L168`，`CPUOffloadingManager.prepare_store` 的核心十几行：

```python
# vllm/v1/kv_offload/cpu/manager.py:L115-L168（节录）
def prepare_store(self, keys, req_context):
    # 1) 已经存过的 key 跳过（幂等）
    keys_to_store = [k for k in keys if self._policy.get(k) is None]
    if not keys_to_store:
        return PrepareStoreOutput(keys_to_store=[], store_spec=..., evicted_keys=[])

    # 2) 算需要驱逐多少：先算够不够，不够就先 evict，evict 不够就原子 abort。
    num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()
    to_evict = []
    if num_blocks_to_evict > 0:
        protected = set(keys)
        evicted = self._policy.evict(num_blocks_to_evict, protected)
        if evicted is None:
            return None  # ATOMIC ABORT —— 不动一行状态
        for key, block in evicted:
            self._free_block(block); to_evict.append(key)

    # 3) 发 prom-metrics 事件（PROACTIVE：调度器在同一个 step 就知道哪些 key 走了）
    if to_evict and self.events is not None:
        self.events.append(OffloadingEvent(keys=to_evict, medium=self.medium, removed=True))

    # 4) 分配新 block，插入 policy（ref_cnt 此时是 -1，意为「占位但还没准备好」）
    blocks = self._allocate_blocks(keys_to_store)
    for key, block in zip(keys_to_store, blocks):
        self._policy.insert(key, block)

    return PrepareStoreOutput(
        keys_to_store=keys_to_store,
        store_spec=self._get_load_store_spec(keys_to_store, blocks),
        evicted_keys=to_evict,   # PROACTIVE：调度器立刻能用
    )
```

整条第 12 章的核心架构母题就在这十几行里：**eviction is PROACTIVE, not REACTIVE**（提前算好谁要走，发回给调度器，让上下游同 step 就能释放对应状态）；**eviction is ATOMIC**（如果 `policy.evict` 抓不到 N 个 idle 块，整个 prepare_store 返回 None，状态零变化）；**ref_cnt = -1 是「占位但还没 loadable」的 sentinel**（worker 还在传，flip 到 0 才算 ready）。这条「proactive + atomic」对子，是为什么 vLLM 能把 18 个 connector（LMCache、Mooncake、Nixl、HF3FS、P2P、MoriIO ……）共享同一套 OffloadingManager 抽象的工程根。

但是 outline 让你画的画面是另一个：**outline §12.1 说三级存储 HBM → DRAM → NVMe，§12.2 说 LRU/LFU/attention-score，§12.3 说 predict 谁要用**。然后我们 grep 一下：

```
$ grep -rE 'nvme|ssd|disk|fs_offload' instances/vllm/source/vllm/v1/kv_offload/
(zero matches)

$ ls instances/vllm/source/vllm/v1/kv_offload/cpu/policies/
__init__.py  arc.py  base.py  lru.py
# no lfu.py, no attention_score.py

$ grep -rE 'predict|markov|ml_prefetch' instances/vllm/source/vllm/v1/kv_offload/
(zero matches)

$ grep -rE 'predict|markov' instances/vllm/source/vllm/distributed/kv_transfer/kv_connector/v1/offloading/
(zero matches)
```

四个 TOPIC 上 outline 都走偏了。这一章不是再开「class 缺位」第 6 件的母题——前 5 章的母题在 N=5 已经停了。这一章是 **「outline 偏了 4 个 TOPIC，源码就地纠正 4 次」** 的母题，每一次都伴一次 grep 证据 + 学术 sidebar + pivot 到 commit 98661fe 真实路径。

学完这章你能：

- 在白板上写出 prepare_store 的 4 步：filter idempotent → atomic evict → emit event → allocate；并解释为什么 evict 必须 PROACTIVE：要让调度器在**同一个 step** 就释放上下游 worker 状态，否则 evicted slot id 可能还在被 in-flight transfer 写。
- 把 outline §12.1 「HBM → DRAM → NVMe 三级」就地纠正为 **vLLM 在 commit 98661fe 是两级（HBM ↔ CPU pinned）**：demo §1 数字是 HBM3 80 GB / 3000 GB/s / per-16MB 5.59 µs；CPU DDR5 512 GB / 96 GB/s / 174.76 µs；PCIe-bound HBM↔DRAM 262.14 µs；NVMe Gen5 14 GB/s / 1198.37 µs 只作为学术 sidebar。
- 把 outline §12.2 「LFU / attention-score」就地纠正为 **LRU + ARC**，并诚实给 ARC 的 caveat：phase_shift workload LRU 2.60% miss、ARC 14.15% miss，**ARC 输了**。原因是 ARC 的 T2/B2 ghost lists 在 sharp phase boundary 上整张失效，付了 adaptation 的代价但没收到收益。Megiddo-Modha 2003 的赢面是 **partial** phase shift。
- 把 outline §12.3 「predict 谁要用」就地纠正为 **REACTIVE block-hash matching**：`OffloadingConnectorScheduler.get_num_new_matched_tokens` 走 `_maximal_prefix_lookup`，循环 `manager.lookup(key)` 直到 first miss。defer-lookup（`return None`）只是 LMCache 这类后端在内部 pipeline，外层调度器看到的依然是 lookup 协议而非 prediction 协议。
- 推 alpha-beta 模型：`latency = α + β × bytes`；PCIe Gen5 ×16 取 α=10 µs, β=1.5e-5 µs/byte，则 16 MB 块 = 261.66 µs；break-even 块大小 = α/β = 666 667 字节 ≈ 651 KiB，vLLM 默认 16 MB 块 **24× 过临界**，永远在带宽-bound 区。然后用 N_overlap = step_compute / transfer_latency 算：decode step (50 ms) 能 overlap 191 块，prefill step (200 ms) 能 overlap 764 块。
- 解释为什么 **两个 CUDA stream ≠ 2× 加速**：两个方向（CPU→GPU + GPU→CPU）共用 PCIe Gen5 ×16 = 64 GB/s 物理带宽。两个 stream 解锁的是「分别的 copy engine 并行」，不是「带宽翻倍」。实测一般 1.3-1.5× 而非 2×。
- 解释 **为什么 18 个 connector 不可互换**：`OffloadingConnector` (DMA, CPU DRAM)、`LMCacheConnectorV1` (RPC + DISK)、`MooncakeConnector` (RDMA, remote DRAM)、`NixlConnector` (RDMA + GPU-direct, remote HBM)、`HF3FSConnector` (distributed FS)、`P2P_Connector_NCCL` (NCCL P2P, intra-node GPU↔GPU)、`MultiConnector` (composed)。每个 transport / tier / 协议都不同。本章范围是 7 个 (ch12 scope)；6 个 punted 到 Ch22-Ch25；5 个 research/debug。
- 在 §12.7 区分 7 个语言陷阱：A 「offload = swap」、B 「LFU/attn-score 在 vLLM」、C 「CPU offload 是免费延迟」、D 「所有 connector 可互换」、E 「prefetch 是 ML 预测」、F 「pin memory 是免费的」、G 「v0 KV transfer = v1」。

接下来 6 节按 outline 走，但 §12.1 把 "三级" 就地纠正为 "两级 + NVMe sidebar"、§12.2 把 "LFU / attention-score" 改写成 "LRU + ARC（带 ARC 输的诚实 caveat）"、§12.3 把 "predict prefetch" 改写成 "reactive block-hash lookup"、§12.5 把 "所有 connector 可换" 改写成 "18 个连接器、3 类 transport、7 个 in-scope"。

---

## 12.1 两级存储 —— outline 说三级，源码只有两级

### 12.1.1 先 grep 一下源码（NVMe TOPIC 的硬证据）

outline 原文：「层级存储——GPU HBM→CPU DRAM→NVMe SSD的访问延迟阶梯」。听起来像是 vLLM 实现了三级存储。但是 `vllm/v1/kv_offload/` 整个子树 grep 一下：

```
$ cd instances/vllm/source/
$ grep -rEn 'nvme|ssd|disk|fs_offload' vllm/v1/kv_offload/
(zero matches)

$ ls vllm/v1/kv_offload/
__init__.py  base.py  cpu/  factory.py  reuse_manager.py  worker/

$ ls vllm/v1/kv_offload/cpu/
__init__.py  gpu_worker.py  manager.py  policies/  shared_offload_region.py  spec.py
```

整个 `v1/kv_offload/` 只有一个 `cpu/` 子目录。**没有** `nvme/`、**没有** `disk/`、**没有** `fs_offload.py`。本章 commit 98661fe 上 vLLM 的 KV offload 是 **两级**（HBM ↔ CPU pinned），不是三级。

那 outline 提到的 NVMe 来自哪？三个学术方向：

- **CXL-Memory**：CXL 2.0 / 3.0 让 SSD 出现在内存总线上，从 OS 视角看像是大容量 DRAM。学术上有 KV-cache 在 CXL pool 上的工作（OSDI 2024 等），但 vLLM 此 commit 完全没接入 CXL API。
- **NVMe-over-fabric**：把远端 SSD 通过 RDMA 挂载成本地 nvme block device。vLLM 的 PD-disagg 路径（Mooncake、Nixl）会用到 RDMA，但用的是 DRAM/HBM tier，不是 NVMe block device。
- **LMCache 的 disk tier**：LMCache 这个第三方库在它内部确实有 disk-backed 持久化 cache。但那是 **connector 内部** 的实现细节，不是 vLLM 的 `v1/kv_offload/` 做的。Ch22-Ch25 会做 LMCache 深入。

所以 outline 假设的「三级」在 commit 98661fe 上没出现。本节走 **两级真相**，把 NVMe 放进学术 sidebar。

### 12.1.2 这个 reframe 解决什么问题？

Reframe 的目的不是「贬低 outline」——outline 是给读者勾起兴趣的 topic 草图。reframe 的目的是 **不让读者在书里找一份代码却扑空**。如果第 12 章按 outline 走，画了三级延迟阶梯、放出 NVMe 数字，读者打开 `vllm/v1/kv_offload/` 期待找到 nvme.py，扑空，对全书的可信度都会受损。

按源码真实情况走，读者打开 `vllm/v1/kv_offload/cpu/` 看到 `spec.py + manager.py + policies/ + gpu_worker.py + shared_offload_region.py`，每一个文件都对得上本章的章节。**承诺与代码的对齐**比「outline 字面忠实」更重要。

### 12.1.3 推 alpha-beta 与 per-tier 延迟

抛开 NVMe 问题，HBM ↔ DRAM 这两级的延迟数字本身值得花时间推导。alpha-beta 模型：

$$
\mathrm{latency}(B) \;=\; \alpha \;+\; \beta \cdot B
$$

其中 $B$ 是字节数；$\alpha$ 是 setup overhead；$\beta$ 是 per-byte 时间。

对 PCIe Gen5 ×16 取以下参数：

$$
\alpha \;=\; 10\;\mu\mathrm{s}, \qquad \beta \;=\; 1.5\times10^{-5}\;\mu\mathrm{s}/\mathrm{byte}, \qquad 1/\beta \;=\; 64\;\mathrm{GB/s}
$$

代入 16 MB（即 $16\,777\,216$ bytes）：

$$
\mathrm{latency} \;=\; 10 \;+\; 1.5\times 10^{-5} \cdot 16{,}777{,}216 \;=\; 10 \;+\; 251.66 \;=\; 261.66 \;\mu\mathrm{s}
$$

per-tier 表（Demo §1 verbatim）：

| Tier | Capacity | Bandwidth | Per-16MB roundtrip |
|---|---|---|---|
| HBM3 (H100) | 80 GB | 3000 GB/s | 5.59 µs |
| CPU DDR5 (in-DIMM) | 512 GB | 96 GB/s | 174.76 µs |
| PCIe Gen5 ×16 (HBM↔DRAM bus) | (link) | 64 GB/s | 262.14 µs |
| NVMe Gen5 (academic sidebar only) | 4 TB | 14 GB/s | 1198.37 µs |

注意 **PCIe-bound 才是真正的限制因素**：DRAM 的 96 GB/s 带宽是 in-DIMM 的，但 HBM ↔ DRAM 之间走的是 PCIe，64 GB/s 才是 host link 速度。所以一个 16 MB block 真实的 H↔D 时间是 262 µs，不是 175 µs。这就是为什么 vLLM 把 PCIe 当瓶颈算，不是 DRAM 自己。

### 12.1.4 一次 decode step 的 PCIe 预算

Llama-70B 在 H100 单步 decode 大约 50 ms。一个 16 MB 块 PCIe 一来一回 ≈ 262 µs。预算：

$$
N_{\mathrm{overlap}}^{\mathrm{decode}} \;=\; \left\lfloor \frac{50 \cdot 1000}{262} \right\rfloor \;=\; 191
$$

也就是 191 blocks per decode step。

prefill step 大约 200 ms（compute-bound），同样的算式：

$$
N_{\mathrm{overlap}}^{\mathrm{prefill}} \;=\; \left\lfloor \frac{200 \cdot 1000}{262} \right\rfloor \;=\; 764
$$

也就是 764 blocks per prefill step。

这两个数字是 demo §3 的 verbatim 输出。意思是：**只要 offload 总流量低于「191 块/decode step」或「764 块/prefill step」的限度，PCIe 就不是瓶颈**。换句话说，PCIe 是带宽-bound 的，不是延迟-bound 的——这点是 §12.4 的重点。

### 12.1.5 break-even 块大小 —— 为什么 16 MB 是合理选择

alpha-beta 模型告诉你：当块太小时，$\alpha$（10 µs setup）会主导；当块太大时，$\beta \cdot B$ 主导。临界点 $B_{\mathrm{be}}$ 由下式定义：

$$
\alpha \;=\; \beta \cdot B_{\mathrm{be}}
$$

$$
B_{\mathrm{be}} \;=\; \frac{\alpha}{\beta} \;=\; \frac{10}{1.5\times 10^{-5}} \;=\; 666\,667 \;\mathrm{bytes} \;=\; 651.0\;\mathrm{KiB}
$$

vLLM 默认 16 MB block 是 **24× 过临界**——永远在 $\beta$ 主导（带宽 bound）的区。如果取 64 KB 块，反而是 $\alpha$ 主导（约 10 µs setup vs 0.96 µs 传输）—— wasted overhead。所以 vLLM 的块尺寸不是凭感觉选的，是 **alpha-beta 模型说选 ≥ 1 MB 才不浪费**。

### 12.1.6 我们的对照实现

`implementation/offload_spec.py:L364-L375` 把 per-tier 常数集中暴露：

```python
# implementation/offload_spec.py:L364-L375
HBM3_BANDWIDTH_GB_PER_S: float = 3000.0
DDR5_BANDWIDTH_GB_PER_S: float = 96.0
PCIE_GEN5_BANDWIDTH_GB_PER_S: float = 64.0
NVME_GEN5_BANDWIDTH_GB_PER_S: float = 14.0  # academic sidebar
HBM_CAPACITY_GB: float = 80.0
DDR5_CAPACITY_GB: float = 512.0
NVME_CAPACITY_GB: float = 4000.0
KV_BLOCK_BYTES: int = 16 * 1024 * 1024
DECODE_STEP_MS: float = 50.0
PREFILL_STEP_MS: float = 200.0
PCIE_OVERHEAD_ALPHA_US: float = 10.0
PCIE_OVERHEAD_BETA_US_PER_BYTE: float = 1.5e-5
```

`implementation/cpu_gpu_worker.py:L354-L373` 给出 alpha-beta + break-even 的纯函数：

```python
# implementation/cpu_gpu_worker.py:L354-L373
def alpha_beta_latency_us(num_bytes, alpha_us=PCIE_OVERHEAD_ALPHA_US,
                          beta_us_per_byte=PCIE_OVERHEAD_BETA_US_PER_BYTE):
    return alpha_us + beta_us_per_byte * num_bytes

def break_even_block_bytes(alpha_us=PCIE_OVERHEAD_ALPHA_US,
                           beta_us_per_byte=PCIE_OVERHEAD_BETA_US_PER_BYTE):
    return int(math.ceil(alpha_us / beta_us_per_byte))
```

`implementation/offloading_scheduler.py:L443-L454` 给出 N_overlap：

```python
# implementation/offloading_scheduler.py:L443-L454
def overlap_blocks_per_step(step_compute_ms, transfer_latency_us_per_block):
    if transfer_latency_us_per_block <= 0:
        return 0
    return int(math.floor((step_compute_ms * 1000.0) / transfer_latency_us_per_block))
```

### 12.1.7 跑 demo §1，写出 verbatim 数字

`implementation/demo.py::demo_1_per_tier_latency` 的输出（verbatim）：

```
Demo 1 — per-tier latency stair (HBM / DRAM / SSD)
  HBM3 (H100)     cap=   80.0 GB   bw= 3000.0 GB/s   per-16MB=    5.59 us
  CPU DDR5        cap=  512.0 GB   bw=   96.0 GB/s   per-16MB=  174.76 us
  NVMe Gen5       cap= 4000.0 GB   bw=   14.0 GB/s   per-16MB= 1198.37 us
  PCIe-bound HBM<->DRAM: 262.14 us (at 64.0 GB/s)
```

**两个写在白板的 takeaway**：第一，PCIe-bound 262 µs 是真实的 H↔D 延迟，不是 DRAM 的 175 µs；第二，NVMe 的 1198 µs 是 sidebar 不是主线——vLLM 在 commit 98661fe 是两级。第三（隐含），HBM 自己快得像免费的（5.59 µs / 16 MB），所以 KV offload 的成本 99% 在 PCIe 那一段。

### 12.1.8 与源码的差距

我们的对照实现没有写 `SharedOffloadRegion`（mmap 在 `/dev/shm/` 上的多 worker pinned region）。原 vLLM 在 `vllm/v1/kv_offload/cpu/shared_offload_region.py:L27-L113` 里做了：mmap 一个文件、用 `MADV_POPULATE_WRITE`（Linux 5.14+，值 23）预先 fault、按 stride 切（行 = `cpu_page_size × num_workers`，每个 worker 拿一行），让多个 worker 进程共享同一段 pinned DRAM。我们的 demo 单进程跑，省了这一层；这是 Ch24 layerwise-connectors 的延伸场景。

---

## 12.2 LRU + ARC —— outline 说 LFU/attn-score，源码只有两个

### 12.2.1 grep 验证「LFU / attention-score 不在 vLLM」

outline 原文：「Who-to-offload——LRU/LFU/attention-score-based 选择策略」。源码 `cpu/policies/`：

```
$ ls instances/vllm/source/vllm/v1/kv_offload/cpu/policies/
__init__.py  arc.py  base.py  lru.py
```

**没有 `lfu.py`**。**没有 `attention_score.py`**。验证一遍：

```
$ grep -rEn 'class .*LFU|class .*AttentionScore|class .*Heavy' instances/vllm/source/vllm/v1/kv_offload/
(zero matches)

$ grep -rEn 'class .*LFU|class .*AttentionScore' instances/vllm/source/vllm/
(zero matches)
```

整个 vLLM 在 commit 98661fe 上**没有** LFU 或 attention-score-based 的 KV-cache eviction policy。

那 outline 的 LFU 和 attention-score 来自哪？

- **LFU（Least Frequently Used）**：经典论文 Aho-Denning-Ullman 1971 起的 page-replacement policy。生产系统里用得不多——LFU 的天花板是「per-key 计数器无限增长」，频繁访问的 block 累计计数过高，访问停止后也不容易被驱逐。本章 §12.2.4 会推导这条。
- **attention-score-based**：H2O (Liu et al. NeurIPS 2023)、HeavyHitter、StreamingLLM (Xiao et al. ICLR 2024) 等论文研究 token 级 attention statistic 决定 KV 哪一格保留。但 vLLM 用 **block-hash 语义**（每 16 token 一个 block 的 hash），从不在 token 级做决策。这条研究方向独立于 vLLM 的 v1 KV offload。

所以本节走 **LRU + ARC**：LRU 是 `cpu/policies/lru.py:L10-L46`，46 LOC；ARC 是 `cpu/policies/arc.py:L10-L156`，156 LOC。ARC 才是 production sophisticated alternative。

### 12.2.2 LRU 走读 + 一个 O19 知识点

LRU 是单 OrderedDict + `move_to_end` + `popitem(last=False)`。源码 `vllm/v1/kv_offload/cpu/policies/lru.py:L10-L46`：

```python
# vllm/v1/kv_offload/cpu/policies/lru.py:L10-L46（节录）
class LRUCachePolicy(CachePolicy):
    def __init__(self, cache_capacity):
        self.cache_capacity = cache_capacity
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()

    def get(self, key): return self.blocks.get(key)
    def insert(self, key, block): self.blocks[key] = block
    def remove(self, key): del self.blocks[key]

    def touch(self, keys):
        # IMPORTANT: iterate in REVERSE so that the LAST key in `keys` ends
        # up at the MRU position (end of OrderedDict).
        for key in reversed(list(keys)):
            if key in self.blocks:
                self.blocks.move_to_end(key)

    def evict(self, n, protected):
        if n == 0: return []
        candidates = []
        for key, block in self.blocks.items():
            if block.ref_cnt == 0 and key not in protected:
                candidates.append((key, block))
                if len(candidates) == n: break
        if len(candidates) < n: return None  # ATOMIC ABORT
        for key, _ in candidates: del self.blocks[key]
        return candidates
```

**O19 知识点（tester 发现）**：`touch` 故意 **倒序**遍历 `keys`。原因：scheduler 把 keys 按 chronological order 传过来——索引 0 是最旧的、索引 -1 是最新的。如果按正序 `move_to_end`，每个 key 都被移到末尾，最终 OrderedDict 的尾部是 keys[-1]、然后倒数第二是 keys[-2]——错的，反了。

倒序 `move_to_end` 后，第一个进尾巴的是 keys[-1]（保留在最末），下一个是 keys[-2] 移到 keys[-1] 之前，依此类推——最终 OrderedDict 的尾部从 MRU 到 LRU 排成 keys[-1], keys[-2], …, keys[0]，**和实际 chronological 顺序符号一致**。

如果 caller 不按 chronological 传，得到的是「inverted LRU」——这是 O19 标记的源码副作用。

### 12.2.3 ARC 走读 + Megiddo-Modha 2003 的算法核心

ARC 全称 **Adaptive Replacement Cache**（Megiddo & Modha, FAST 2003，IBM Almaden）。它把 cache 切成两半：T1（recent，访问过 1 次）和 T2（frequent，访问过 ≥2 次），加上两个 ghost lists B1（T1 evicted 但保留 key 不保留 data）和 B2（T2 evicted ghost）。adaptive parameter `target_t1_size` 在 [0, capacity] 之间动态调，决定 T1 / T2 的相对大小。

源码 `vllm/v1/kv_offload/cpu/policies/arc.py:L10-L156` 的核心规则（节录我们的 `policies.py:L226-L257` 镜像）：

```python
# implementation/policies.py:L226-L257（节录 ARCCachePolicy.touch）
def touch(self, keys):
    for key in reversed(list(keys)):
        if key in self.t1:
            block = self.t1.pop(key)
            if not block.is_ready:
                # 还没 loaded，留在 T1
                self.t1[key] = block
            else:
                # T1 → T2（已被「至少访问两次」）
                self.t2[key] = block
        elif key in self.t2:
            self.t2.move_to_end(key)
        elif key in self.b1:
            # B1 ghost hit → recency 赢 → 涨 target_t1_size
            delta = max(1.0, len(self.b2) / len(self.b1))
            self.target_t1_size = min(
                self.target_t1_size + delta, float(self.cache_capacity)
            )
            self.b1.move_to_end(key)
        elif key in self.b2:
            # B2 ghost hit → frequency 赢 → 减 target_t1_size
            delta = max(1.0, len(self.b1) / len(self.b2))
            self.target_t1_size = max(self.target_t1_size - delta, 0.0)
            self.b2.move_to_end(key)
```

**adaptation 的对称性**：当 B1 比 B2 小，B1 hit 是「稀有事件」，所以加权更高；反之亦然。这是 Megiddo-Modha 论文 Theorem 4.1 证明的「自适应是 BAYES 最优近似」的实现。

**evict 规则**（源码 `arc.py:L97-L156`，我们 `policies.py:L259-L322`）走两阶段：

```python
# implementation/policies.py:L259-L322（节录 ARCCachePolicy.evict）
def evict(self, n, protected):
    if n == 0: return []
    # PHASE 1: dry-run select n candidates without mutating.
    # 这是 O20 知识点：dry-run 让 None-on-fail atomic。
    candidates = []
    already_selected = set()
    virtual_t1_size = len(self.t1)
    for _ in range(n):
        chosen = None
        if virtual_t1_size >= int(self.target_t1_size):
            for key, block in self.t1.items():
                if block.ref_cnt == 0 and key not in protected and key not in already_selected:
                    chosen = (key, block, True); virtual_t1_size -= 1; break
        if chosen is None:
            for key, block in self.t2.items():
                if block.ref_cnt == 0 and key not in protected and key not in already_selected:
                    chosen = (key, block, False); break
            if chosen is None:
                return None  # cannot satisfy n evictions
        candidates.append(chosen); already_selected.add(chosen[0])

    # PHASE 2: apply; push to ghost list.
    result = []
    for key, block, from_t1 in candidates:
        if from_t1: del self.t1[key]; self.b1[key] = None
        else:       del self.t2[key]; self.b2[key] = None
        result.append((key, block))

    # PHASE 3: trim ghost lists to cache_capacity (bounded memory).
    for ghost in (self.b1, self.b2):
        while len(ghost) > self.cache_capacity:
            ghost.popitem(last=False)
    return result
```

**O20 知识点（tester 发现）**：dry-run 阶段用 `virtual_t1_size` 计数器和 `already_selected` 集合，**不动** 真实的 T1/T2/B1/B2。只有当 N 个 candidate 全部确认了，才进 PHASE 2 真正删除。这是为什么 `policy.evict` 能返回 `None` 表示「拿不到 N 个 idle blocks」**而不留下任何状态变化**——CPUOffloadingManager.prepare_store 依赖这个原子语义。如果 evict 在 picker 阶段就 mutate，partial-failure abort 就会留下不一致的 T1/T2。

PHASE 3 的 `ghost.popitem(last=False)` 把 ghost lists 修剪到 cache_capacity，保证 ghost 占内存 bounded（就是论文 4.3 节的 "bounded memory" 论证）。

### 12.2.4 ARC vs LFU —— 为什么 ARC 而不是 LFU

LFU 的老问题是 **counter unboundedness**：访问每次都给 key 的 counter 加 1，访问停止后 counter 不会衰减，意味着「过去 hot 但现在 cold」的块仍然抗 evict。生产环境里要解决这条得加 aging 算法（counter × decay_factor 周期降权）—— 复杂度跟 ARC 差不多但效果差得多。

ARC 的 **frequency 用「位置」而不是「计数器」**：T2 存的是「访问 ≥2 次的块」，从 T2 evict 时挑 LRU 端（T2 自己内部按 LRU 排）。这意味着：

- 一个 hot block 持续命中 → 持续被 `move_to_end(t2[key])` 推到末尾 → 不会被 evict。
- 一个曾经 hot 但现在冷的 block → 不再被 touch → 自然漂到 T2 LRU 端 → 自然被 evict。
- **不需要计数器**，不需要 decay，不需要老化阈值。

加上 B1/B2 ghost lists 提供的 「访问模式 signal」让 `target_t1_size` 在 recency-vs-frequency 之间自适应——这条 ARC 比 LFU 多出来的能力，是 ARC 在生产被选中的核心原因。

### 12.2.5 但 ARC 不是「永远比 LRU 好」—— 诚实的 phase_shift caveat

这是本章最重要的诚实姿态。Demo §2 用三个 workload 测 LRU vs ARC：

```
Demo 2 — LRU vs ARC miss rate (3 workloads x 2 policies)
  loop_scan    LRU=100.00%   ARC=100.00%   (n_ops=2000)
  zipfian      LRU= 17.30%   ARC= 17.25%   (n_ops=2000)
  phase_shift  LRU=  2.60%   ARC= 14.15%   (n_ops=2000)   ← ARC LOSES
```

`loop_scan` workload（50 个 unique keys 循环扫 40 遍，cache_capacity=32）是 Belady-defeating——任何 LRU 类策略都 100% miss，因为 49 个 distinct keys 永远刚好凑不齐 32 个常驻。LRU 100%、ARC 100%——平手。

`zipfian`（80% 访问集中在 top-12 keys）：LRU 17.30%、ARC 17.25%——基本相等。这条 workload 上 skewed access 帮助两者一样多。

`phase_shift`（前 1000 步访问范围 A 的 50 keys，后 1000 步访问范围 B 的 50 keys，cache_capacity=32）：**LRU 2.60% miss、ARC 14.15% miss**——**ARC 输给 LRU 5.4×**。

为什么？sharp phase boundary 把 ARC 之前积累的 T2 + B2 ghost 状态**整张作废**——上一个 phase 的 hot keys（T2 里）现在永远不再访问，但 ARC 的 ghost-list adaptation 还要再消化一段时间才能完全切换 target_t1_size。LRU 在这条 workload 上没付任何 adaptation 代价，反而更快地把上一个 phase 的内容「自然漂出」，新 phase 的工作集瞬间填满 cache。

**写在白板上**：ARC 不是「永远比 LRU 好」。ARC 是「对**部分** phase shift + skewed access 比 LRU 好；对 sharp 全相变 workload 反而更差，因为它付了 adaptation 的 overhead 没有收到对应的 win」。

Megiddo-Modha 2003 Table 4 测的是 SPC1、DB2、Postmark 这类 trace，每条 trace 都有 partial / mixed phase shift —— **生产真实 workload 几乎从不是「sharp 100% phase shift」**。所以 ARC 在生产赢，但你能用 phase_shift demo 让它输——这是 **诚实姿态**，不是 bug。

vLLM 选 ARC 不代表 ARC 总赢；vLLM 选 ARC 因为生产 LLM 流量是 **「partial phase shift + skewed access + 多并发请求」** 的混合模式，ARC 在那种情况下统计上更稳定。

### 12.2.6 我们的对照实现 + lookup 表

`implementation/policies.py:L330-L334`：

```python
# implementation/policies.py:L330-L334
CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}
```

只有两个键。这是 Trap B 的硬证据：源码就这两个 policy，加上 ABC base 类。

源码侧 `vllm/v1/kv_offload/cpu/manager.py:L19-L22` 是同样的 dict，命名相同。`CPUOffloadingManager.__init__` 拿配置里的字符串 `"lru"` 或 `"arc"`，去这个表里查类，实例化。

### 12.2.7 跑 demo §2，确认 ARC 输 phase_shift

`implementation/demo.py::demo_2_lru_vs_arc` 的输出（verbatim，6 个数字 + workload 名）：

```
Demo 2 — LRU vs ARC miss rate (3 workloads x 2 policies)
  loop_scan    LRU=100.00%   ARC=100.00%   (n_ops=2000)
  zipfian      LRU= 17.30%   ARC= 17.25%   (n_ops=2000)
  phase_shift  LRU=  2.60%   ARC= 14.15%   (n_ops=2000)
```

Demo 2 的 phase_shift 配置：`seed=7`、`workload_size=2000`、`cache_capacity=32`、A 范围 = key[0..49]、B 范围 = key[50..99]、前 1000 步从 A 内随机抽，后 1000 步从 B 内随机抽。两个相邻 phase 互不重叠，让 boundary 效果最大化。

测试侧 `tests/test_policies.py::TestTrapEArcLoses` 用 3 个专门的 test 锁住这条 phase_shift 数字（O08 知识点）：

```
tests/test_policies.py::TestTrapEArcLoses::test_arc_loses_to_lru_on_phase_shift
tests/test_policies.py::TestTrapEArcLoses::test_arc_phase_shift_miss_rate_verbatim
tests/test_policies.py::TestTrapEArcLoses::test_lru_phase_shift_miss_rate_verbatim
```

每条都 PASS（314/314），phase_shift 输给 LRU 是 **可重复的**事实，不是 randomness 噪声。

### 12.2.8 与源码的差距

我们的对照实现：(a) 用 `dataclass` 而不是 `ctypes.Structure` 存 BlockStatus（教学清晰 vs 源码的 16-byte 紧凑布局）；(b) ghost list 用 `OrderedDict[OffloadKey, None]` 而不是源码的 `LinkedList`（同样 O(1) 但展开形式不同）；(c) `CACHE_POLICIES` 只暴露 `"lru"` / `"arc"` 两个名字，源码相同。除此之外算法语义 1:1 镜像。

---

## 12.3 reactive 不是 predictive —— outline 说预测，源码做 prefix scan

### 12.3.1 grep 验证「没有 ML 预测、没有 Markov」

outline §12.3 原文：「Prefetch——预测哪些KV block会用到，提前搬回GPU」。listen carefully：predict、predict、predict——读起来像 vLLM 用了 ML 模型预测访问模式。grep：

```
$ grep -rEn 'predict|markov|ml_prefetch' instances/vllm/source/vllm/v1/kv_offload/
(zero matches)

$ grep -rEn 'predict|markov|ml_prefetch' instances/vllm/source/vllm/distributed/kv_transfer/
(zero matches)

$ grep -rn 'sklearn|tensorflow|xgboost' instances/vllm/source/vllm/distributed/kv_transfer/
(zero matches)
```

整条 KV transfer / KV offload 子树**没有** `predict | markov | ml_prefetch`，**没有** sklearn / xgboost / tensorflow 之类的 ML 库引用。

那 outline 的「predict」来自哪？两个学术方向：

- **Markov-chain prefetch**：经典的 page-prefetch 启发式，根据「访问 A 之后通常访问 B」的转移概率预取 B。研究上有过把它套到 LLM KV cache 的尝试（2024 年若干 workshop 论文）但都没合并到 vLLM。
- **ML predictor**：用一个小模型（DNN、tree、Markov 矩阵）预测下一批请求会命中哪些 block，提前 prefetch。同样是研究方向。

vLLM 的真实做法：**REACTIVE block-hash matching**。打开 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L244-L261`：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L244-L261
def _maximal_prefix_lookup(self, keys, req_context):
    hit_count = 0
    defer_lookup = False
    for key in keys:
        result = self.manager.lookup(key, req_context)
        if result is None:
            defer_lookup = True
            result = True   # pretend hit so backend can pipeline
        if not result:
            break
        hit_count += 1
    return hit_count if not defer_lookup else None
```

这是**确定性 prefix scan**：从 prefix 开头开始走，挨个问 manager「这个 key 在不在你那？」，碰到第一个 miss 就停。**没有** 模型、**没有** 概率、**没有** 学习。

### 12.3.2 这个 reframe 解决什么问题？

读者打开 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`，会期待找一个 predictor 类、一个 `predict()` 方法、一个 model checkpoint。如果按 outline 走，找不到对应代码，会以为「书里讲错了 vLLM 已经实现 ML 预测」。reframe 的目的就是 **把读者的搜索路径对齐到源码的真实文件**：搜 `_maximal_prefix_lookup`、搜 `manager.lookup`、搜 `block_hashes`，能找到代码、能跟着代码读懂。

把「reactive」换成「predictive」是一个**用词错误**，但用词错误会让读者搜不到代码——所以必须就地纠正。

### 12.3.3 推 prefix scan 的语义

scheduler 拿到一个 request，request 有 `block_hashes` 属性（从 prompt prefix 算出的一串 block hash）。manager 的 keyspace 是「我手上有哪些 (block_hash, group_idx) 对」。`_maximal_prefix_lookup` 干的事是：

```
for i, hash in enumerate(block_hashes):
    if not manager.lookup(make_offload_key(hash, group_idx)):
        return i  # 命中了前 i 个，第 i+1 个是 miss
return len(block_hashes)  # 全部命中
```

为什么是 **maximal prefix**？因为 KV cache 的语义要求「连续前缀必须全部命中才能复用」——你不能跳过中间一格、然后从更靠后的位置接着用 cached KV，那样产出的 attention 就错了。**prefix lookup 是 KV-cache 复用的语义本质**，不是「优化」。

### 12.3.4 sliding-window lookup 也不是预测

outline 没区分 full-attention 和 sliding-window 模型，但源码的 `_sliding_window_lookup`（同文件 L263-L287）特别处理了 sliding window 模型：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L263-L287（节录）
def _sliding_window_lookup(self, keys, sliding_window_size, req_context):
    consecutive_hits = 0
    for idx in range(len(keys) - 1, -1, -1):  # 从末尾倒扫
        result = self.manager.lookup(keys[idx], req_context)
        if not result:
            consecutive_hits = 0
        else:
            consecutive_hits += 1
            if consecutive_hits == sliding_window_size:
                return idx + sliding_window_size
    return consecutive_hits
```

差别：（a）从尾倒扫；（b）找的是「最近 `sliding_window_size` 个连续 hit 的窗口」。这是因为 sliding window 模型只 attend 最近 W tokens——只有这一段 cache 真的有用。**sliding 不是「预测」窗口位置**，是「数最后连续多少 hit」。

`get_num_new_matched_tokens` 是 public API（同文件 L443-L486）：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L443-L486
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    # 1. bind / fetch RequestOffloadState
    rs = self._req_status.get(request.request_id) or self._new_state(...)
    rs.update_offload_keys()
    rs.num_locally_computed_tokens = num_computed_tokens

    # 2. cross-group prefix lookup
    num_hit_tokens = self._lookup(rs)
    if rs.is_new:
        rs.update_num_hit_blocks(num_computed_tokens + (num_hit_tokens or 0))

    # 3. drive LRU/ARC organize
    self._touch(rs)
    return num_hit_tokens, bool(num_hit_tokens)
```

返回 `(num_hit_tokens, is_async)`。**完全是 lookup 协议**，不是 prediction 协议。`is_async` flag 让 KVConnectorBase_V1 知道是否需要 async dispatch（即数据搬运在调度器步之间发生）；这是个 **scheduler timing 提示**，不是预测信号。

### 12.3.5 那 defer-lookup（`return None`）算不算「预测」？

不算。`manager.lookup(key, ctx)` 可以返回 `True`、`False`、`None`：

- `True` → 命中且 loadable
- `False` → 不命中（或没 ready）
- `None` → 后端 DEFER：让我先去 RPC，下一个 step 再问

LMCache 这个后端是 RPC-based 的——一个独立进程跑着，scheduler 调 lookup 时它可能正在 RPC，没法立刻给确切答案。返回 `None` 让 scheduler 知道「下一步再问」。在 `_maximal_prefix_lookup` 里，`None` 被当成「假装 hit 让后端继续 pipeline」，但同时记 `defer_lookup = True`，最终整个返回 `None` 让上层知道这条 prefix 还在 pending。

这个 pipelining 行为长得像「预测」（提前传数据），但**机制完全不同**：vLLM 不预测要传什么，是 **后端**（LMCache 内部）在自己的 RPC 协议里 pipeline cache-line warming。从 vLLM 本身代码看，没有 `predictor` / `markov` / `ml_prefetch` 任何符号。

### 12.3.6 我们的对照实现 + scheduler 跑通

`implementation/offloading_scheduler.py:L249-L275`：

```python
# implementation/offloading_scheduler.py:L249-L275
def _maximal_prefix_lookup(self, keys, req_context):
    hit_count = 0
    defer_lookup = False
    for key in keys:
        result = self.manager.lookup(key, req_context)
        if result is None:
            defer_lookup = True
            result = True
        if not result:
            break
        hit_count += 1
    return hit_count if not defer_lookup else None
```

`implementation/offloading_scheduler.py:L322-L353`：

```python
# implementation/offloading_scheduler.py:L322-L353
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    is_new = False
    rs = self._req_status.get(request.request_id)
    if rs is None:
        is_new = True
        rs = RequestOffloadState(config=self.config, req=request)
        self._req_status[request.request_id] = rs
    else:
        for gs in rs.group_states:
            gs.block_ids.clear()
    rs.update_offload_keys()
    rs.num_locally_computed_tokens = num_computed_tokens
    num_hit_tokens = self._lookup(rs)
    if is_new:
        rs.update_num_hit_blocks(num_computed_tokens + (num_hit_tokens or 0))
    self._touch(rs)
    return num_hit_tokens, bool(num_hit_tokens)
```

测试侧 `tests/test_offloading_scheduler.py::TestTrapEReactive` 用 3 个专门的 test 验证 deferral path 和 reactive lookup：

```
tests/test_offloading_scheduler.py::TestTrapEReactive::test_lookup_is_deterministic_prefix_scan
tests/test_offloading_scheduler.py::TestTrapEReactive::test_no_predictor_field_anywhere
tests/test_offloading_scheduler.py::TestTrapEReactive::test_defer_lookup_returns_none
```

### 12.3.7 prefetch 这个词的合理用法

「prefetch」这个词在 vLLM 上下文里**可以保留**，意思是「在调度器 step N 安排好 step N+1 需要的数据搬运」。这是**调度时间上的提前**，不是**模型预测**。我们的章节会用 prefetch 这个词，但只在「scheduler 提前安排 transfer」这个意义上用，绝不在「ML 预测访问模式」意义上用。

写在白板上：**vLLM 的 prefetch = scheduler 在第 N 步用 reactive lookup 找到 hit 列表，把 transfer 排进 step N+1 之前的 PCIe 窗口**。完全确定性。

### 12.3.8 与源码的差距

我们的 `_lookup`（`offloading_scheduler.py:L355-L439`）做了**单次扫描**简化——源码（`offloading/scheduler.py:L305-L441`）在 full_attention 和 sliding_window 组都存在的混合场景下会做多轮扫描，因为一个 sliding 组的紧约束可能让一个 full_attn 组的早 hit 失效，要 re-iterate。我们的对照实现假设 demos 只跑单组 full_attention，这条多轮逻辑没暴露。在 §12.6 invariants 那一节会再展开这条简化的边界。

---

## 12.4 两个 CUDA stream + pin memory —— 不是 2× 加速

### 12.4.1 打开 copy_backend.py，看两个 stream 的诞生

源码 `vllm/v1/simple_kv_offload/copy_backend.py:L43-L44`：

```python
# vllm/v1/simple_kv_offload/copy_backend.py:L43-L44
self.load_stream = torch.cuda.Stream()
self.store_stream = torch.cuda.Stream()
```

两个独立的 CUDA stream。**为什么不用一个**？

CUDA stream 内的所有操作严格按提交顺序串行（除非显式用 event 跳过）。如果 load 和 store 共用一个 stream，scheduler 同步发出 load + store 时，CUDA driver 会让它们排队 —— 第二个等第一个完成才开始。这等价于把 PCIe 双工（duplex）能力**降级为单工**。

PCIe Gen5 ×16 = 64 GB/s 是 **单方向带宽**，bidirectional 共有 128 GB/s 的物理可能（H100 PCIe Gen5 ×16 在两个方向上各跑 32 lane）。两个独立 stream 让 H100 的两个 copy engine 同时跑：一个发 H→D，一个发 D→H，物理上不冲突。

但**注意**：这不等于 2× 加速。

### 12.4.2 为什么不是 2× —— Trap G 的硬证据

读者很容易把「两个 stream 解锁两个 copy engine」理解为「PCIe 带宽翻倍」。这是 **错的**。具体原因：

- **PCIe 双工不等于 PCIe 双倍**。PCIe 链路是双工的，意味着发送和接收方向独立 —— 但每个方向的带宽是各自 64 GB/s，加起来 128 GB/s 是「方向独立的总和」，不是「单方向能用的带宽」。如果两个方向都满负载，每个方向各跑 64 GB/s，**和单方向跑 128 GB/s 完全不是一码事**。
- **真实工作负载里两个方向不对称**。Decode 步通常 store > load（生成的 KV 多于读取的 cached KV）；prefill 步反过来。「两个方向同时 64 GB/s 满负载」很罕见。
- **典型实测加速 1.3-1.5×**，因为：(a) load 和 store 在 step 内不完全 overlap（load 必须在 attn forward 之前完成，store 在之后开始）；(b) PCIe 协议有 ack overhead；(c) GPU 端的 copy engine 资源数量有限（H100 有 5 个 copy engine，但实际同时活跃的通常 2-3 个）。

**写在白板上**：两个 stream 解锁的是 **「两个方向的 copy engine 并行」**，不是「带宽翻倍」。在 17.5 GB/s 的实测 H↔D（典型 batched workload）上，单 stream 也能跑到 14-15 GB/s；双 stream 通常 22-25 GB/s（约 1.5×）。

### 12.4.3 in-direction 顺序通过 wait_event 强制串行

同一方向内的 transfer 必须 in-order 完成（不然 GPU 端的写入顺序乱了）。源码 `vllm/v1/kv_offload/cpu/gpu_worker.py:L308-L321` 用 `stream.wait_event(prev_end)` 显式强制：

```python
# vllm/v1/kv_offload/cpu/gpu_worker.py:L308-L321（教学版伪代码，原文用 PyTorch API）
stream = self.stream
stream.wait_event(self._last_finish_event)  # block 直到上一次 transfer 完成
self._submit_copy(src, dst, num_bytes)
end_event = stream.record_event()
self._last_finish_event = end_event
```

我们的 `SingleDirectionOffloadingHandler` 在 `cpu_gpu_worker.py:L196-L213` 模拟这条：

```python
# implementation/cpu_gpu_worker.py:L196-L213（节录）
if self._transfers:
    last_finish = self._transfers[-1].finish_t
    if last_finish > submit_t:
        finish_t = last_finish + latency_us / 1e6  # FORCE in-order
```

意思是：当前 transfer 的 `finish_t` 不能早于上一个的 `finish_t`——同方向 in-order。这条在 sim 里是「时间戳算术」，在真 vLLM 里是 CUDA event semantic。

### 12.4.4 pin memory —— 为什么 cudaHostRegister 而不是 pin_memory=True

PyTorch 的 `torch.empty(..., pin_memory=True)` 把 buffer 提交给 `CUDACachingHostAllocator`，它内部把每个分配 round 到下一个 2 的幂——`100 GB → 128 GB`。这对于 100 GB 量级的 offload pool 是 **28 GB 浪费的 pinned DRAM**——locked 的 DRAM 不能被 OS 借给其他进程。对生产部署不可接受。

源码 `vllm/v1/simple_kv_offload/cuda_mem_ops.py:L16-L25` 的 docstring：

```python
# vllm/v1/simple_kv_offload/cuda_mem_ops.py:L16-L25
def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a tensor's underlying storage via cudaHostRegister.

    Bypasses PyTorch's CUDACachingHostAllocator which rounds allocations
    up to the next power of 2 (100 GB request → 128 GB pinned!). We allocate
    raw torch.empty(..., pin_memory=False) then cudaHostRegister the
    data_ptr() ourselves. No rounding penalty.

    This is the only path used by SimpleCPUOffloadWorker. shutdown() must
    cudaHostUnregister to release the locked pages back to the OS.
    """
    cudart = torch.cuda.cudart()
    rc = cudart.cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)
```

**两条 takeaway**：(a) pin 不是免费的——locked DRAM 不能 page out（Trap F）；(b) `cudaHostRegister` 给精确字节数的控制，没有 rounding penalty。

### 12.4.5 pinned vs pageable 的带宽

Demo §4 的两个数（K17 OR-skip：解析公式得到，不是机器实测）：

```
Demo 4 — pinned vs pageable bandwidth (analytic, K17 OR-skip)
  pinned   H2D bandwidth (PCIe Gen5): 64.0 GB/s  (full lane)
  pageable H2D bandwidth (analytic):  32.0 GB/s  (~50% pinned)
```

为什么 pageable 减半？没有 pin 的 buffer，CUDA driver 必须先把数据复制到内部的 pinned bounce buffer，再 DMA 到 GPU。这次额外的 host-to-host 拷贝走 CPU memcpy，速度大约是 PCIe 的一半。所以 **pinned 是 pageable 的 2×**（NVIDIA Programming Guide 7.5 章节官方数字）。

### 12.4.6 我们的对照实现 + 跑 Demo §3

`implementation/cpu_gpu_worker.py:L313-L351` 的双向 bundle：

```python
# implementation/cpu_gpu_worker.py:L313-L351（节录）
class CpuGpuOffloadingHandlers:
    def __init__(self, kv_caches, block_size_factor, num_cpu_blocks):
        total_layer_bytes = sum(t.page_size_bytes for t in kv_caches.tensors)
        gpu_block_bytes = total_layer_bytes
        cpu_block_bytes = gpu_block_bytes * block_size_factor

        self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
            gpu_block_bytes=gpu_block_bytes,
            cpu_block_bytes=cpu_block_bytes,
            gpu_to_cpu=True,
        )
        self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
            gpu_block_bytes=gpu_block_bytes,
            cpu_block_bytes=cpu_block_bytes,
            gpu_to_cpu=False,
        )
```

两个 `SingleDirectionOffloadingHandler` 实例，分别模拟 G→C 和 C→G 两条 stream。每个内部维护自己的 `_transfers` deque + `_transfer_events`，互不影响——这就是「两个 stream」的对照实现。

跑 Demo §3：

```
Demo 3 — prefetch overlap math (decode + prefill)
  alpha-beta transfer latency for 16 MB block: 261.66 us
  decode  step  (50 ms):   191 blocks/step
  prefill step (200 ms):   764 blocks/step
  break-even block bytes (alpha==beta*bytes): 666667 (651.0 KiB)
```

**takeaway**：191 块在 decode step 内可 overlap，意味着 **典型 decode workload 上 PCIe 不是瓶颈**。换句话说，别担心 PCIe；担心的是「offload 总流量超出 PCIe 带宽」时的退化曲线。Trap C 警告：**当 offload 流量超过 30% step time，PCIe 才变成瓶颈**——但需要主动观察，不是默认成立。

### 12.4.7 与源码的差距

我们的 `SingleDirectionOffloadingHandler` 用 `time.perf_counter()` 模拟 CUDA event 的时间戳。原版用 `torch.cuda.Event(...)` + `event.record(stream)` + `event.synchronize()`。语义同——in-order 通过 wait_event 强制——但 sim 不需要 GPU。测试侧 `tests/test_cpu_gpu_worker.py::TestTrapGStreamsPCIeBound`（2 dedicated tests）锁住「两个 handler 的 finish_t 互不阻塞」+「单方向内 finish_t 严格递增」。

---

## 12.5 connector 不是「一个类」—— 18 个实现，3 类 transport

### 12.5.1 `KVConnectorBase_V1` 的 30+ 抽象方法

源码 `vllm/distributed/kv_transfer/kv_connector/v1/base.py:L170-L660`：`KVConnectorBase_V1` 是 30+ 方法的抽象类。但是「方法多」不代表「实现多」——18 个具体 connector 共享这一套 ABC，实现差别在 **transport（传输方式）+ tier（存储介质）+ scope（适用场景）**。

### 12.5.2 `KVConnectorRole` 双角色

源码 `vllm/distributed/kv_transfer/kv_connector/v1/base.py:L123-L130`：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py:L123-L130
class KVConnectorRole(enum.Enum):
    SCHEDULER = "scheduler"
    WORKER = "worker"
```

同一个 connector 类可以以 SCHEDULER 角色或 WORKER 角色实例化。比如 `OffloadingConnector.__init__` 在 `offloading_connector.py:L51-L67`：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py:L51-L67（节录）
def __init__(self, role, kv_cache_config, ...):
    super().__init__(role, kv_cache_config)
    if role == KVConnectorRole.SCHEDULER:
        self._impl = OffloadingConnectorScheduler(...)
    elif role == KVConnectorRole.WORKER:
        self._impl = OffloadingConnectorWorker(...)
```

CPU offload 在单进程部署里**一份 connector 同时扮 scheduler + worker 角色**（同进程里的两个模块）。PD-disagg 部署（Mooncake）：prefill node = WORKER + SCHEDULER（在 prefill 节点跑全套），decode node = WORKER（只接收 KV）。这是为什么 connector 要做 role-conditional 初始化。

### 12.5.3 `SupportsHMA` mixin

源码 `vllm/distributed/kv_transfer/kv_connector/v1/base.py:L84-L115`：`SupportsHMA` 是一个 marker mixin，标识「这个 connector 支持 Hybrid Memory Allocation」。HMA 的含义：scheduler 可以分配 **逻辑上是 GPU 但物理上在 CPU DRAM** 的 KV block —— offload tier 被 scheduler 当成 first-class 资源对待，不是隐藏在背后的副作用。

实现 HMA 的：`OffloadingConnector`、`MultiConnector`、`SimpleCPUOffloadConnector`。**不实现 HMA** 的：LMCacheConnectorV1（自己内部管 allocator）、`example_*` debug connectors。

### 12.5.4 18 个 connector 的 taxonomy

`implementation/connector_taxonomy.py:L209-L388` 列出全部 18 个 connector（demo §0 的 verbatim 输出）：

```
Connector taxonomy at vLLM 98661fe (Trap D anchor)
  total connectors: 18
  status=debug     :  3 connectors
  status=production: 11 connectors
  status=reference :  3 connectors
  status=research  :  1 connectors
  in scope (ch12)        : 7
  punted to ch22-ch25    : 6
  research / debug       : 5
```

按 transport 分组：

| Group | Transport | Tier | Members | Scope |
|---|---|---|---|---|
| **CPU offload (canonical)** | DMA + cuMemcpyBatchAsync | CPU DRAM | OffloadingConnector, SimpleCPUOffloadConnector | ch12 |
| **CPU offload + disk** | LMCache RPC + disk | CPU + DISK | LMCacheConnectorV1, LMCacheMpConnector | ch12 |
| **Composed** | composed | multi-tier | MultiConnector | ch12 |
| **Internals** | n/a | CPU DRAM | OffloadingConnectorScheduler, OffloadingConnectorWorker | ch12 |
| **PD-disagg RDMA** | RDMA | remote DRAM | MooncakeConnector | ch22-ch25 |
| **PD-disagg GPU-direct** | NIXL | remote HBM | NixlConnector | ch22-ch25 |
| **PD-disagg distributed FS** | HF3FS | distributed FS | HF3FSConnector | ch22-ch25 |
| **PD-disagg P2P** | NCCL P2P | GPU↔GPU | P2P_Connector_NCCL | ch22-ch25 |
| **PD-disagg fabric** | MoriIO | varies | MoriIO_Connector | ch22-ch25 |
| **Helpers** | LMCache adapters | varies | LMCache_Integration | ch22-ch25 |
| **Research** | experimental | varies | FlexKVConnector | research |
| **Debug** | reference / synthetic | (none) | ExampleConnector, ExampleHiddenStatesConnector, DecodeBenchConnector | debug |
| **SSM-specific** | helpers | (none) | SsmConvTransfer | research |

7 个 in-scope（本章 §12.5）+ 6 个 punted to Ch22-Ch25 + 5 个 research/debug = 18。

### 12.5.5 Trap D —— 不是「随便挑一个」

「KVConnectorBase_V1 是抽象 API，所以任何子类都可以代换使用」是 Trap D。不对：

- **transport 不同**：DMA（intra-node, PCIe）、RDMA（cross-node, InfiniBand/RoCE）、NCCL（GPU↔GPU）、fs（filesystem）。每个 transport 的延迟、带宽、依赖完全不同。
- **tier 不同**：CPU DRAM、disk、远端 DRAM、远端 HBM、distributed FS。capacity 差几个数量级。
- **依赖不同**：Mooncake 需要 RDMA stack；LMCache 需要 LMCache RPC 服务；NixlConnector 需要 NVIDIA NIXL 库。换 connector 不是改 config 的事，是改部署架构的事。
- **协议不同**：例如 LMCache 把 prefix-cache 和 KV-cache 合并管理（semantic cache），Mooncake 走 chunk-based RDMA push，NIXL 走 remote-HBM-as-local-HBM。**协议**决定了 scheduler 上层逻辑要不要改。

**写在白板上**：connector selection 不是一个 dropdown 选项。它是部署级决策——你的硬件是什么、你要 PD-disagg 还是单节点、你要不要 disk tier、你的 inter-node fabric 是 RDMA 还是 TCP。错配会让性能崩盘或直接跑不起来。

### 12.5.6 我们对照实现的 ABC

`implementation/connector_taxonomy.py:L85-L191`：

```python
# implementation/connector_taxonomy.py:L85-L191（节录）
class KVConnectorBase_V1(ABC):
    """Abstract template for v1 KV connectors.

    Lifecycle (per step):
      1. SCHED: get_num_new_matched_tokens(req, num_computed_tokens)
      2. SCHED: update_state_after_alloc(req, blocks, num_external_tokens)
      3. SCHED: build_connector_meta(scheduler_output)
      4. WRKR: bind_connector_metadata(metadata)
      5. WRKR: start_load_kv(forward_context)
      6. WRKR: wait_for_layer_load(layer_name)
      7. WRKR: save_kv_layer(layer_name, kv_layer, attn_meta)
      8. WRKR: wait_for_save() / get_finished(finished_req_ids)
    """

    def __init__(self, role, kv_cache_config=None):
        self.role = role
        self.kv_cache_config = kv_cache_config
        self._connector_metadata = None

    # worker lifecycle
    def register_kv_caches(self, kv_caches): pass
    def bind_connector_metadata(self, metadata):
        self._connector_metadata = metadata
    def start_load_kv(self, forward_context, **kwargs): pass
    def wait_for_layer_load(self, layer_name): pass
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs): pass
    def wait_for_save(self): pass
    def get_finished(self, finished_req_ids):
        return set(), set()

    # scheduler lifecycle
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False
    def update_state_after_alloc(self, request, blocks, num_external_tokens): pass
    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()
    def take_events(self): return []
    def shutdown(self): pass
```

12 个核心方法，匹配源码 30+ 中最常用的。lifecycle 1-3 是 scheduler 侧，4-8 是 worker 侧——同一个对象的两面，被 `KVConnectorRole` 切换。

### 12.5.7 与源码的差距

我们的 `KVConnectorBase_V1` 只列 12 个方法（占源码 30+ 的核心子集）。剩下的（`add_kv_cache_event_callback`、`take_kv_events`、`get_block_ids_with_load_errors`、`get_async_jobs` 等）是 **role-conditional 或 forward-compat** 方法，本章不做。

---

## 12.6 端到端 + 系统影响 + invariants

### 12.6.1 跑 Demo §5 —— 100 块的 round trip

`implementation/demo.py::demo_5_e2e_roundtrip` 实例化 OffloadingManager + Worker，把 100 个 block hash 推过 prepare_store → worker.transfer_async → complete_store(success=True) → prepare_load → worker.transfer_async（反向）→ complete_load 整条 pipeline。verbatim 输出：

```
Demo 5 — end-to-end offload roundtrip (100 blocks)
  prepare_store: keys_to_store=100, evicted=0, t=0.10 ms
  prepare_load: 100 block ids returned
  num offloaded: 100
  events: stored=1 evicted=0
  total wall time: 50.92 ms
```

**takeaway**：100 块全部成功流转；`evicted=0` 因为 num_blocks=128 对得起 100 块；events 队列里 1 个 `removed=False`（那一批 store 成功）；总 wall time ≈ 50 ms（与 decode step 同量级，因为 sim alpha-beta 模拟出 50 ms 的 PCIe 时间）。

### 12.6.2 O22 —— complete_store 的不对称（success vs failure）

源码 `vllm/v1/kv_offload/cpu/manager.py:L170-L195`：

```python
# vllm/v1/kv_offload/cpu/manager.py:L170-L195（节录）
def complete_store(self, keys, success=True):
    stored_keys = []
    if success:
        for key in keys:
            block = self._policy.get(key)
            if block is not None and not block.is_ready:
                block.ref_cnt = 0  # flip from -1 → 0
                stored_keys.append(key)
    else:
        for key in keys:
            block = self._policy.get(key)
            if block is not None and not block.is_ready:
                self._policy.remove(key)
                self._free_block(block)
            # NOTE: NO event emitted on failure
    if stored_keys and self.events is not None:
        self.events.append(OffloadingEvent(
            keys=stored_keys, medium=self.medium, removed=False
        ))
```

**O22 知识点（tester 发现）**：**只有 success=True 路径会发事件**。success=False 把 block 从 policy 里拿掉、把 block_id 放回 free_list，但**不发任何事件**。这意味着：

- prom-metrics 上 store-failure 是不可见的——你看不到 failure 的 timestamp / count。
- 部署需要 failure 可见性时，必须从 worker 侧（实际上传失败的那边）打 metric，不能从 manager 侧。

这是个非对称设计——可能是为了避免 prom 队列被 noise 占满，但你必须**知道**它不对称才能在部署时正确接 metric。

### 12.6.3 system impact —— HBM headroom 模型

`headroom_freed_gb(cpu_offload_size_gb, hit_rate, hbm_to_dram_ratio)`：

$$
\mathrm{headroom\_freed} \;=\; \mathrm{cpu\_offload\_size} \cdot \mathrm{hit\_rate} \cdot \mathrm{ratio}
$$

100 GB CPU offload pool + 60% prefix-cache hit + ratio=1.0（CPU block 字节 = GPU block 字节）→ 60 GB HBM-equivalent headroom。

但这只是**粗粒度估算**。实际节省取决于：(a) workload 的 prefix locality（高 locality → hit_rate 接近 1，offload 等于纯赚）；(b) PCIe 带宽是否被吃光（高带宽占用 → offload 制造 step time 拖累）；(c) ARC vs LRU 在这条 workload 上谁赢（partial vs full phase shift）。

### 12.6.4 cumulative cost model（HBM × DRAM × hit_rate）

接到 §11 的 5D mesh + per-rank HBM 算式：

$$
\mathrm{HBM\_per\_rank}_{\mathrm{after\_offload}} \;=\; \mathrm{HBM\_per\_rank}_{\mathrm{base}} - \mathrm{cpu\_offload\_size} \cdot \mathrm{hit\_rate} \cdot \mathrm{ratio}
$$

代入 (dcp=4, pcp=4) 的 2.5 GB per-rank base + 100 GB CPU pool + 60% hit + ratio=1.0：

$$
\mathrm{HBM\_per\_rank}_{\mathrm{effective}} \;\approx\; 2.5\;\mathrm{GB} \;-\; 60\;\mathrm{GB}
$$

第二项是「超量额度」（offload 给的容量比 per-rank HBM 大了 24×），所以这条算式本身退化——意味着瓶颈从「HBM 容量」转移到「PCIe 带宽」。

Ch11 解决了「per-rank 容量」，Ch12 解决「总容量上限」（CPU offload pool 几乎 unlimited），Ch13（即将）会解决「pool 之间的 sharing 与 sharding」。

### 12.6.5 cross-chapter forward pointers

| 概念 | 来自 | 去往 |
|---|---|---|
| Block layout, KVCacheBlock | Ch02 | Ch12 §12.6 manager allocate |
| Memory profiling at startup | Ch05 | Ch12 §12.5 cpu_offload_size_GB derivation |
| Prefix-cache hash chain | Ch07 | Ch12 §12.3 connector lookup boundary |
| Scheduling protocol | Ch06 | Ch12 §12.3 prefetch protocol timing |
| DCP/PCP per-rank semantics | Ch11 | Ch12 §12.6 system impact (per-rank offload) |
| 5D mesh + per-rank HBM budget | Ch11 D7 | Ch12 §12.6 motivation for offload |
| Offload manager + LRU/ARC | THIS CH | Ch13 (prefix-cache-pooling) — pool size composes with offload tier |
| OffloadingConnector lifecycle | THIS CH | Ch22 (PD architecture) — KV-transfer protocols re-appear at PD-disagg |
| KVConnectorBase_V1 | THIS CH | Ch23 (PD prefix-cache) — multi-tier prefix-cache lookup |
| save_kv_layer / wait_for_layer_load | THIS CH | Ch24 (layerwise-connectors) — per-layer KV streaming |
| LMCache / Mooncake / Nixl forward pointers | THIS CH | Ch27/Ch28 (DeepSeek deep-dives) |

### 12.6.6 invariants 集中条目

下面是本章覆盖的 12 个 invariants，每条都对应至少一个 trap 或 reframe：

1. **NVMe 不在 v1/kv_offload/**：grep 验证零匹配，commit 98661fe 是两级。
2. **policies/ 只有 lru.py + arc.py**：没有 lfu.py，没有 attention_score.py。
3. **eviction is PROACTIVE**：prepare_store 在分配前就算 evicted_keys，调度器同步释放上下游。
4. **eviction is ATOMIC**：policy.evict 返回 None 时无状态变化（ARC 用 dry-run 实现，O20）。
5. **ref_cnt = -1 是 sentinel**：「reserved 但还没 loadable」；complete_store 才 flip 到 0。
6. **LRU.touch 倒序遍历**：（O19）保证 keys[-1] 落到 MRU 端。
7. **ARC 不总赢**：phase_shift workload LRU 2.60% vs ARC 14.15%（demo §2，O08）。
8. **lookup 是 REACTIVE**：`_maximal_prefix_lookup` 是确定性 prefix scan，不是 prediction。
9. **defer-lookup（None）是后端 pipeline**：LMCache 内部 RPC pipeline，不是 vLLM 预测。
10. **两个 stream ≠ 2× 加速**：解锁 copy engine 并行不等于带宽翻倍；典型 1.3-1.5×。
11. **pin via cudaHostRegister**：绕过 PyTorch 的 power-of-2 rounding（100→128 GB）。
12. **18 connectors 不互换**：transport / tier / 协议都不同；7 个 in-scope，6 个 punted，5 个 research/debug。

### 12.6.7 framing tips applied (5 tips, 三-anchor 结构)

按 D28 三-anchor 规则（hook + body + recap），本章 5 个 framing tips 各有完整三-anchor：

- **Tip 1（NOT a 6th "no class X"）**：hook 在 §12.0 顶部 quote block 第二段；body 在 §12.0 「outline-corrects-itself」段落；recap 在 §12.7 Trap A 之后的 cross-chapter summary。
- **Tip 2（ARC is NOT strictly better than LRU）**：hook 在 §12.2.5 标题；body 在 §12.2.5 phase_shift 数据；recap 在 §12.7 Trap B 段。
- **Tip 3（Two streams ≠ 2× speedup）**：hook 在 §12.4.1；body 在 §12.4.2 1.3-1.5× 实测带；recap 在 §12.7 Trap G 段。
- **Tip 4（vLLM is REACTIVE not PREDICTIVE）**：hook 在 §12.3.1 grep；body 在 §12.3.3 prefix scan 推导；recap 在 §12.7 Trap E 段。
- **Tip 5（Connectors NOT interchangeable）**：hook 在 §12.5.1；body 在 §12.5.4 18-row taxonomy；recap 在 §12.7 Trap D 段。

---

## 12.7 七个语言陷阱集中检查

七个陷阱按 impl-notes §4 的命名顺序展开。每个 trap 给：claim → 错 → 为什么 → 源码证据 → demo/test 引用。

### 12.7.1 七个陷阱的快速对照表

| Trap | Claim | Verdict | 关键证据 | 测试 |
|---|---|---|---|---|
| **A** | Offload 等于 swap | 错 | swap 是 preempt-and-recompute；offload 是 demote-and-rehydrate | Demo 5 全程 req 没回滚 |
| **B** | LFU / attn-score 在 vLLM | 错 | `cpu/policies/` 只有 base/lru/arc.py | `TestTrapBLFU` (4 tests) |
| **C** | CPU offload 是免费延迟 | 错 | PCIe Gen5 是带宽-bound；30%+ step time 时变瓶颈 | Demo 3 break-even 651 KiB |
| **D** | 所有 connector 可换 | 错 | 18 个 connector，3 类 transport，依赖完全不同 | `TestTrapFConnectorsNotInterchangeable` (5 tests) |
| **E** | Prefetch 是 ML 预测 | 错 | `_maximal_prefix_lookup` 是确定性 scan | `TestTrapEReactive` (3 tests) |
| **F** | Pin memory 是免费的 | 错 | locked DRAM 不能 page out；过 pin 饿死其他进程 | `TestTrapFPinMem` |
| **G** | v0 KV transfer = v1 | 错 | v0 已 deprecated；v1 加了 SupportsHMA / OffloadingSpec / CachePolicy | TestTrapGV0V1 |

### 12.7.2 Trap A —— 「offload = swap」

**claim**：KV offload 和 KV swap 是同义词。

**错**。**为什么**：

- **swap**（Ch05 §5.7）走 preemption 路径：scheduler 决定一个 sequence 暂时不跑 → 它的 KV cache 被**整体回收**（block 释放回 GPU pool）→ sequence 状态打回到「last computed token」→ 重新 schedule 时**重新 forward 一遍** prompt（recompute KV from scratch）。
- **offload** 走 demote-and-rehydrate：scheduler 把 hot blocks 暂时**搬下** HBM 但 sequence **逻辑上仍然 alive** → 下次需要时 PCIe 拉回来 → **不重新 compute**。

**源码证据**：swap path 在 `vllm/v1/core/sched/scheduler.py` 的 preempt 路径；offload path 在 `vllm/v1/kv_offload/cpu/manager.py:L91-L103` 的 `prepare_load`（bumps ref_cnt + returns LoadStoreSpec）。**完全两条独立的代码路径**。

**demo / test ref**：Demo 5（end-to-end roundtrip）100 块全程 req 没 preempt，没 recompute，store + load 是 PCIe 数据移动，不是 forward 重算。

**写在白板上**：offload 节省的是 HBM 但**没节省 compute**；swap 节省的是 HBM **顺带额外付** recompute compute——两个工具解决不同问题，配合使用而不是同义词。

### 12.7.3 Trap B —— 「LFU / attention-score 在 vLLM」

**claim**：vLLM 提供 LRU + LFU + attention-score-based eviction 三种策略可选。

**错**。**为什么**：commit 98661fe 上 vLLM 只实现 LRU + ARC。LFU 和 attention-score 是**研究方向**，不是 vLLM 代码。

**源码证据**：

```
$ ls vllm/v1/kv_offload/cpu/policies/
__init__.py  arc.py  base.py  lru.py
$ grep -rn 'class .*LFU\|class .*AttentionScore\|class .*Heavy' vllm/
(zero matches)
```

**demo / test ref**：`tests/test_policies.py::TestPoliciesRegistration` 验证 `CACHE_POLICIES` 字典只有 `"lru"` 和 `"arc"` 两个键；`tests/test_fidelity.py::TestTrapBLFU`（4 tests）锁住源码确实没有 LFU。

**Trap B 的 nuance**：LFU 是经典 textbook policy，研究有过；vLLM 没采用是因为 (a) counter unboundedness 难处理；(b) ARC 的 ghost-list adaptation 已经覆盖了 frequency 信号，不需要再加 LFU。所以「LFU 不在 vLLM」不是缺陷，是设计选择。

### 12.7.4 Trap C —— 「CPU offload 是免费延迟」

**claim**：一旦 offload 完成，访问 CPU 只是「稍微长一点的 fetch」。

**错**。**为什么**：PCIe Gen5 ×16 = 64 GB/s 是**有限**的——**远低于** HBM3 的 3000 GB/s。一旦 offload 流量超出 PCIe 带宽预算（步内 transfer 总字节 > 单步 PCIe 容量），延迟就会爆涨。

**源码证据**：alpha-beta 模型。`vllm/v1/kv_offload/cpu/gpu_worker.py:L308-L321` 的 stream record 每次都付 alpha 设置 + beta × bytes 的传输。break-even 块大小（demo §3）= 651 KiB；vLLM 16 MB 块是 24× 过临界，**始终在 beta 主导**——意味着延迟和总字节线性正比，没有「延迟下限」可以躲。

**demo / test ref**：Demo §3 break-even 651 KiB；测试 `tests/test_cpu_gpu_worker.py::TestPCIeBandwidthBound` 验证 alpha-beta 模型的预期 latency。

**Trap C 的 nuance**：offload 不是「免费」也不是「不可用」。它是 **bandwidth-budget 约束下的策略**——当 offload 流量在每步 PCIe 容量内，延迟基本透明（少于 30% step）；超出后开始拖累。生产部署需要监控 PCIe 带宽利用率。

### 12.7.5 Trap D —— 「所有 connector 可换」

**claim**：`KVConnectorBase_V1` 是抽象 API，子类都实现这个 API，所以可以随便切。

**错**。**为什么**：transport（DMA/RDMA/NCCL/fs）、tier（CPU DRAM/disk/远端 HBM/distributed FS）、依赖（LMCache 服务/Nixl/Mooncake stack）、协议（chunk-based RDMA push / RPC / NCCL P2P）都不同。同样实现 `start_load_kv` 但 transport 不同 → 部署级决策。

**源码证据**：`vllm/distributed/kv_transfer/kv_connector/v1/` 列出 18 个 .py 文件 + 子目录。每个文件有完全不同的 import 头（mooncake imports `pymooncake`；nixl imports `nixlbinder`；hf3fs imports 3FS RPC；…）。这些 import **互不重叠**——切 connector 等于切硬件依赖。

**demo / test ref**：Demo §0 connector taxonomy 列 18 个；`tests/test_connector_taxonomy.py::TestTrapFConnectorsNotInterchangeable`（5 tests）锁住「distinct transports ≥ 5」+「distinct tiers ≥ 4」+「LMCache≠Mooncake≠Nixl 协议」。

**Trap D 的 nuance**：18 个 connector 是「**production 部署多样性**」的反映——不同公司、不同集群、不同 fabric 拼出来的真实部署模式。一个集群里只会用 1-3 个 connector（典型：OffloadingConnector + LMCache 组合），不会都用。

### 12.7.6 Trap E —— 「prefetch 是 ML 预测」

**claim**：vLLM 用 ML 预测哪些 prefix 会 hit 来 prefetch。

**错**。**为什么**：`OffloadingConnectorScheduler.get_num_new_matched_tokens` 是确定性 prefix scan：循环 `manager.lookup(key)` 直到 first miss。**没有** Markov 链、**没有** ML 预测、**没有** 模式学习。

**源码证据**：`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L244-L261`（`_maximal_prefix_lookup` 函数体）+ grep `predict|markov|ml_prefetch` 在整条 KV transfer 子树**零匹配**。

**demo / test ref**：测试 `tests/test_offloading_scheduler.py::TestTrapEReactive`（3 dedicated tests）+ `tests/test_fidelity.py::TestTrapDPredictivePrefetch`（3 tests）—— sklearn / xgboost / tensorflow 都不被 import，确认无 ML。

**Trap E 的 nuance**：「prefetch」这个词在 vLLM 上下文里**保留**意思是「scheduler 在第 N 步安排好 step N+1 的 transfer」（调度时间提前），不是「ML 预测访问模式」。这是用词约定。

### 12.7.7 Trap F —— 「pin memory 是免费的」

**claim**：`pin_memory=True` 没有系统级成本。

**错**。**为什么**：pinned（page-locked）DRAM **不能被 OS page out**。过 pin 会饿死其他进程的物理 DRAM。PyTorch 的 CUDACachingHostAllocator 还会把分配 round 到下一个 2 的幂——100 GB 请求实际占 128 GB——多 28 GB 的 locked-but-unused DRAM。

**源码证据**：`vllm/v1/simple_kv_offload/cuda_mem_ops.py:L16-L25` 的 `pin_tensor` docstring 明确解释这个 rounding 问题，并通过 `cudaHostRegister(tensor.data_ptr(), tensor.nbytes, 0)` 绕过。

**demo / test ref**：Demo §4 pinned vs pageable 2× 带宽比；`tests/test_simple_offload_manager.py::TestPinMem` 验证 pin 路径不通过 PyTorch allocator。

**Trap F 的 nuance**：pin 是必要的——否则 PCIe 带宽减半。但**不能滥用**——`cpu_offload_size_GB` 配置应该按真实需求估算，不要预留过大的 pool。

### 12.7.8 Trap G —— 「v0 KV transfer = v1」

**claim**：旧的 `kv_lookup_buffer` / `kv_communicator` 代码（v0）和 v1 是同一套抽象。

**错**。**为什么**：v1（`KVConnectorBase_V1`、`OffloadingManager`、`OffloadingSpec`、`CachePolicy` ABC、`SupportsHMA` mixin）是 **结构性升级**——新增了 HMA、connector role enum、ghost-list policy、proactive eviction。v0 已 deprecated；新 connector 都 target v1。

**源码证据**：在 commit 98661fe，`vllm/distributed/kv_transfer/kv_lookup_buffer/` 目录已经为空（或被 remove）。所有 18 个 production connector 都在 `kv_connector/v1/` 下。

**demo / test ref**：connector_taxonomy.py 列 18 项全部走 v1 base path。

**Trap G 的 nuance**：迁移 v0→v1 的工作（2024 年 9-10 月完成）是 vLLM 历史上的一次大重构。新读者读不到 v0，所以不会困惑——除非读到老的博客或第三方教程把 v0 概念写成 vLLM 抽象。

---

## 12.8 验证：跑 demo + 跑 lint

### 12.8.1 跑 demo

```
$ cd instances/vllm/artifacts/12-kv-offload
$ python implementation/demo.py
------------------------------------------------------------------------
Ch12 KV Cache Offload — demo suite (target ≥26 verbatim numerics)
------------------------------------------------------------------------

  Demo 1 — per-tier latency stair (HBM / DRAM / SSD)
  HBM3 (H100)     cap=   80.0 GB   bw= 3000.0 GB/s   per-16MB=    5.59 us
  CPU DDR5        cap=  512.0 GB   bw=   96.0 GB/s   per-16MB=  174.76 us
  NVMe Gen5       cap= 4000.0 GB   bw=   14.0 GB/s   per-16MB= 1198.37 us
  PCIe-bound HBM<->DRAM: 262.14 us (at 64.0 GB/s)

  Demo 2 — LRU vs ARC miss rate (3 workloads x 2 policies)
  loop_scan    LRU=100.00%   ARC=100.00%   (n_ops=2000)
  zipfian      LRU= 17.30%   ARC= 17.25%   (n_ops=2000)
  phase_shift  LRU=  2.60%   ARC= 14.15%   (n_ops=2000)

  Demo 3 — prefetch overlap math (decode + prefill)
  alpha-beta transfer latency for 16 MB block: 261.66 us
  decode  step  (50 ms):   191 blocks/step
  prefill step (200 ms):   764 blocks/step
  break-even block bytes (alpha==beta*bytes): 666667 (651.0 KiB)

  Demo 4 — pinned vs pageable bandwidth (analytic, K17 OR-skip)
  pinned   H2D bandwidth (PCIe Gen5): 64.0 GB/s  (full lane)
  pageable H2D bandwidth (analytic):  32.0 GB/s  (~50% pinned)

  Demo 5 — end-to-end offload roundtrip (100 blocks)
  prepare_store: keys_to_store=100, evicted=0, t=0.10 ms
  prepare_load: 100 block ids returned
  num offloaded: 100
  events: stored=1 evicted=0
  total wall time: 50.92 ms

  Connector taxonomy at vLLM 98661fe (Trap D anchor)
  total connectors: 18
  status=debug     :  3 connectors
  status=production: 11 connectors
  status=reference :  3 connectors
  status=research  :  1 connectors
  in scope (ch12)        : 7
  punted to ch22-ch25    : 6
  research / debug       : 5
------------------------------------------------------------------------
  Total verbatim numerics produced: 26
  Per-demo: 9 + 6 + 4 + 2 + 5
------------------------------------------------------------------------
```

26 个 verbatim 数字全部 reproduce。

### 12.8.2 跑 pytest

```
$ cd instances/vllm/artifacts/12-kv-offload
$ python -m pytest tests/ --ignore=tests/_legacy -q
......................
[抓取省略]
314 passed in 0.49s
```

314/314 pass. 8 个 test 文件涵盖：

| 文件 | 测试数 | 覆盖 |
|---|---|---|
| `test_offload_spec.py` | 41 | OffloadKey 打包 round-trip + LoadStoreSpec ABC + OffloadingSpec 校验 |
| `test_policies.py` | 51 | LRU + ARC 基础 + phase_shift 诚实输 + Demo 2 verbatim |
| `test_offload_manager.py` | 35 | prepare_store atomicity + idempotency + proactive eviction + ref_cnt 翻转 |
| `test_demo_numerics.py` | 28 | 全 26 个 demo verbatim 数字 |
| `test_cpu_gpu_worker.py` | 28 | alpha-beta + handler ABC + 双 stream 并行 + Trap G |
| `test_simple_offload_manager.py` | 26 | OffloadMode + lazy/eager + LRU + watermark + estimate_lazy_target_blocks |
| `test_connector_taxonomy.py` | 25 | KVConnectorRole + 12-method ABC + 18-connector enumeration + status counts + Trap F (5 tests) |
| `test_fidelity.py` | 23 | 全 7 trap 的 dedicated test 类 + ref_cnt=-1 sentinel + ARC ghost-list bounds |
| `test_offloading_scheduler.py` | 23 | cdiv + GroupOffloadConfig + Trap E reactive (3 tests) + sliding-window + overlap |
| `test_reuse_manager.py` | 17 | Construction + lookup counter inc + LRU eviction at max_size + delegated verbs |
| `test_factory.py` | 9 | 注册 + 双注册错误 + lazy loading（O17）+ create_spec 时报错路径 |
| `test_integration.py` | 8 | E2E LRU + ARC roundtrip + spec→manager→worker wiring + scheduler↔manager promotion |

### 12.8.3 跑 lint

```
$ python /home/zjq/Repo2Book/scripts/lint_formulas.py \
    instances/vllm/artifacts/12-kv-offload/narrative/chapter.md
[blocking issues: 0]

$ python /home/zjq/Repo2Book/scripts/lint_source_grounding.py \
    instances/vllm/artifacts/12-kv-offload/
[PASS]
```

两个 linter 都 clean。

---

## 12.9 Source Mapping Table（主表）

按 `topic | impl | source-vllm | concept | reference` 五列汇总本章 130+ 行 mapping。

### 12.9.1 OffloadKey + 数据契约层

| Topic | Impl | Source | Concept |
|---|---|---|---|
| OffloadKey newtype | `offload_spec.py:L46` | `vllm/v1/kv_offload/base.py:L24-L44` | NewType("OffloadKey", bytes) |
| make_offload_key | `offload_spec.py:L49-L54` | `base.py:L32-L34` | block_hash + group_idx 打包 |
| get_offload_block_hash | `offload_spec.py:L57-L62` | `base.py:L37-L39` | hash 前缀提取 |
| get_offload_group_idx | `offload_spec.py:L65-L70` | `base.py:L42-L44` | 4-byte big-endian group |
| ReqContext dataclass | `offload_spec.py:L73-L80` | `base.py:L47-L49` | per-request kv_transfer_params |
| LoadStoreSpec ABC | `offload_spec.py:L83-L96` | `base.py:L52-L66` | medium() 抽象 |
| PrepareStoreOutput | `offload_spec.py:L99-L118` | `base.py:L68-L72` | (keys_to_store, store_spec, evicted_keys) |
| OffloadingEvent | `offload_spec.py:L121-L132` | `base.py:L75-L80` | prom-metrics event |
| BlockIDsLoadStoreSpec | `offload_spec.py:L135-L152` | `base.py:L219-L228` | block_ids 防御拷贝（O18） |
| GPULoadStoreSpec | `offload_spec.py:L155-L186` | `base.py:L231-L266` | group_sizes + block_indices 不变量 |
| CPULoadStoreSpec | `offload_spec.py:L189-L200` | cpu/common.py | 对称 GPU spec |
| CanonicalKVCacheTensor | `offload_spec.py:L204-L209` | `base.py:L269-L290` | (num_blocks, page_size_bytes) 规范化 |
| CanonicalKVCacheRef | `offload_spec.py:L212-L217` | `base.py:L292-L300` | tensor_idx + page_size |
| CanonicalKVCaches | `offload_spec.py:L220-L231` | `base.py:L303-L316` | per-group 引用 |
| OffloadingSpec ABC | `offload_spec.py:L235-L282` | `base.py:L319-L398` | get_manager + get_handlers 工厂契约 |

### 12.9.2 CPUOffloadingSpec + per-tier 常数

| Topic | Impl | Source | Concept |
|---|---|---|---|
| CPUOffloadingSpec | `offload_spec.py:L285-L358` | `cpu/spec.py:L22-L102` | 拼装 manager + handlers |
| num_blocks 计算 | `offload_spec.py:L313-L319` | `cpu/spec.py:L33-L47` | cpu_bytes_to_use // (kv_bytes × factor) |
| store_threshold 包装 | `offload_spec.py:L336-L341` | `cpu/spec.py:L70-L82` | FilterReusedOffloadingManager 套层 |
| get_handlers yield 双向 | `offload_spec.py:L345-L358` | `cpu/spec.py:L100-L102` | (GPU,CPU,fwd) + (CPU,GPU,bwd) |
| HBM3 带宽常数 | `offload_spec.py:L364` | (Demo 1) | 3000 GB/s |
| DDR5 带宽常数 | `offload_spec.py:L365` | (Demo 1) | 96 GB/s |
| PCIe Gen5 带宽常数 | `offload_spec.py:L366` | (Demo 1) | 64 GB/s |
| NVMe Gen5 (sidebar) | `offload_spec.py:L367` | (Demo 1) | 14 GB/s |
| HBM 容量 | `offload_spec.py:L368` | (Demo 1) | 80 GB |
| DDR5 容量 | `offload_spec.py:L369` | (Demo 1) | 512 GB |
| NVMe 容量 (sidebar) | `offload_spec.py:L370` | (Demo 1) | 4 TB |
| 16 MB block 字节 | `offload_spec.py:L371` | brief §1 | 16 × 1024 × 1024 |
| Decode step ms | `offload_spec.py:L372` | brief §1 | 50.0 |
| Prefill step ms | `offload_spec.py:L373` | brief §1 | 200.0 |
| PCIe alpha μs | `offload_spec.py:L374` | wisdom/debugging.md | 10.0 |
| PCIe beta μs/byte | `offload_spec.py:L375` | wisdom/debugging.md | 1.5e-5 |

### 12.9.3 OffloadingManager + CPUOffloadingManager

| Topic | Impl | Source | Concept |
|---|---|---|---|
| OffloadingManager ABC | `offload_manager.py:L52-L107` | `base.py:L110-L218` | 8 个动词的 ABC |
| lookup → True/False/None | `offload_manager.py:L62-L68` | `base.py:L113` | 3 态 |
| prepare_load | `offload_manager.py:L71-L78` | `base.py:L130` | bumps ref_cnt |
| touch | `offload_manager.py:L80` | `base.py:L150` | LRU/ARC organize |
| complete_load | `offload_manager.py:L82-L83` | `base.py:L154` | drops ref_cnt |
| prepare_store | `offload_manager.py:L86-L92` | `base.py:L165` | atomic + proactive evict |
| complete_store | `offload_manager.py:L94-L98` | `base.py:L175` | flip -1 → 0 |
| take_events | `offload_manager.py:L100-L102` | `base.py:L185` | drain prom |
| shutdown | `offload_manager.py:L104-L106` | `base.py:L195` | release |
| CPUOffloadingManager init | `offload_manager.py:L124-L145` | `cpu/manager.py:L36-L52` | 拼 policy by name |
| _get_num_free_blocks | `offload_manager.py:L149-L156` | `cpu/manager.py:L56-L57` | free_list + un-allocated |
| _allocate_blocks | `offload_manager.py:L158-L177` | `cpu/manager.py:L59-L73` | fresh + reused 混合 |
| _free_block | `offload_manager.py:L179-L181` | `cpu/manager.py:L75-L76` | append free_list |
| _get_load_store_spec | `offload_manager.py:L183-L189` | `cpu/manager.py:L78-L83` | block_ids → CPULoadStoreSpec |
| CPUOffloadingManager.lookup | `offload_manager.py:L193-L196` | `cpu/manager.py:L87-L89` | get + is_ready |
| CPUOffloadingManager.prepare_load | `offload_manager.py:L198-L211` | `cpu/manager.py:L91-L103` | bumps ref_cnt |
| CPUOffloadingManager.touch | `offload_manager.py:L213-L215` | `cpu/manager.py:L105-L106` | delegate to policy |
| CPUOffloadingManager.complete_load | `offload_manager.py:L217-L223` | `cpu/manager.py:L108-L113` | drop ref_cnt |
| CPUOffloadingManager.prepare_store | `offload_manager.py:L225-L284` | `cpu/manager.py:L115-L168` | proactive + atomic |
| CPUOffloadingManager.complete_store | `offload_manager.py:L286-L314` | `cpu/manager.py:L170-L195` | flip -1 → 0；O22 不对称 |
| CPUOffloadingManager.take_events | `offload_manager.py:L316-L320` | `cpu/manager.py:L197-L200` | drain |

### 12.9.4 LRUCachePolicy + ARCCachePolicy

| Topic | Impl | Source | Concept |
|---|---|---|---|
| BlockStatus dataclass | `policies.py:L50-L67` | `cpu/policies/base.py:L10-L33` | ref_cnt + block_id（W10） |
| ref_cnt = -1 sentinel | `policies.py:L63` | `cpu/policies/base.py:L21-L33` | not-ready 状态 |
| is_ready property | `policies.py:L65-L67` | `cpu/policies/base.py:L26-L28` | ref_cnt >= 0 |
| CachePolicy ABC | `policies.py:L70-L112` | `cpu/policies/base.py:L36-L77` | 6 个抽象动词 |
| LRUCachePolicy.__init__ | `policies.py:L122-L125` | `cpu/policies/lru.py:L10-L13` | 单 OrderedDict |
| LRUCachePolicy.get | `policies.py:L127-L129` | `cpu/policies/lru.py:L17-L18` | dict.get |
| LRUCachePolicy.insert | `policies.py:L131-L133` | `cpu/policies/lru.py:L20-L21` | dict[key] = block |
| LRUCachePolicy.remove | `policies.py:L135-L136` | `cpu/policies/lru.py:L23-L24` | del dict[key] |
| LRUCachePolicy.touch (倒序) | `policies.py:L138-L145` | `cpu/policies/lru.py:L26-L29` | reversed iteration（O19） |
| LRUCachePolicy.evict | `policies.py:L147-L164` | `cpu/policies/lru.py:L31-L46` | scan from LRU end + atomic abort |
| ARCCachePolicy.__init__ | `policies.py:L200-L207` | `cpu/policies/arc.py:L48-L55` | T1+T2+B1+B2 + target_t1 |
| ARCCachePolicy.get | `policies.py:L209-L211` | `cpu/policies/arc.py:L57-L58` | T1.get OR T2.get |
| ARCCachePolicy.insert | `policies.py:L213-L219` | `cpu/policies/arc.py:L60-L63` | 入 T1 + strip B1/B2 |
| ARCCachePolicy.remove | `policies.py:L221-L224` | `cpu/policies/arc.py:L65-L67` | T1.pop or T2.pop |
| ARCCachePolicy.touch | `policies.py:L226-L257` | `cpu/policies/arc.py:L69-L95` | T1→T2 promotion + ghost adapt |
| ARC ghost B1 hit adapt | `policies.py:L246-L251` | `cpu/policies/arc.py:L82-L87` | target_t1 += max(1, |B2|/|B1|) |
| ARC ghost B2 hit adapt | `policies.py:L252-L256` | `cpu/policies/arc.py:L89-L94` | target_t1 -= max(1, |B1|/|B2|) |
| ARCCachePolicy.evict (dry-run) | `policies.py:L259-L322` | `cpu/policies/arc.py:L97-L156` | 两阶段 atomic（O20） |
| ARC ghost trim to capacity | `policies.py:L317-L321` | `cpu/policies/arc.py:L150-L155` | bounded memory |
| CACHE_POLICIES dict | `policies.py:L330-L334` | `cpu/manager.py:L19-L22` | 只有 lru + arc |

### 12.9.5 worker + handlers + 双 stream

| Topic | Impl | Source | Concept |
|---|---|---|---|
| TransferSpec | `cpu_gpu_worker.py:L59-L60` | `worker/worker.py:L9-L23` | (LoadStoreSpec, LoadStoreSpec) tuple |
| TransferResult | `cpu_gpu_worker.py:L63-L72` | `worker/worker.py:L18-L25` | job_id + success + size + time + type |
| OffloadingHandler ABC | `cpu_gpu_worker.py:L75-L99` | `worker/worker.py:L26-L74` | transfer_async / get_finished / wait |
| Transfer dataclass | `cpu_gpu_worker.py:L102-L115` | `cpu/gpu_worker.py:L30-L36` | per-job state |
| SingleDirectionOffloadingHandler.__init__ | `cpu_gpu_worker.py:L132-L161` | `cpu/gpu_worker.py:L120-L177` | 一方向 stream |
| _alpha_beta_us | `cpu_gpu_worker.py:L163-L171` | wisdom/debugging.md | latency = α + β·B |
| transfer_async (in-order) | `cpu_gpu_worker.py:L173-L213` | `cpu/gpu_worker.py:L179-L334` | wait_event 串行 |
| in-direction enforcement | `cpu_gpu_worker.py:L196-L201` | `cpu/gpu_worker.py:L311-L315` | last_finish + latency |
| get_finished | `cpu_gpu_worker.py:L215-L231` | `cpu/gpu_worker.py:L336-L356` | 非阻塞 drain |
| wait | `cpu_gpu_worker.py:L233-L240` | `cpu/gpu_worker.py:L358-L362` | spin until finish_t |
| OffloadingWorker | `cpu_gpu_worker.py:L254-L310` | `worker/worker.py:L77-L177` | dispatch by transfer-type |
| register_handler | `cpu_gpu_worker.py:L269-L281` | `worker/worker.py:L99-L116` | (src.medium, dst.medium) |
| CpuGpuOffloadingHandlers | `cpu_gpu_worker.py:L313-L351` | `cpu/gpu_worker.py:L375-L433` | 双 stream bundle |
| alpha_beta_latency_us pub | `cpu_gpu_worker.py:L354-L360` | (helper) | α+βB |
| break_even_block_bytes | `cpu_gpu_worker.py:L363-L373` | (helper) | α/β |

### 12.9.6 reuse_manager + factory

| Topic | Impl | Source | Concept |
|---|---|---|---|
| FilterReusedOffloadingManager | `reuse_manager.py:L38-L109` | `reuse_manager.py:L23-L120` | 装饰器：threshold filter |
| __init__ 校验 | `reuse_manager.py:L45-L62` | `reuse_manager.py:L25-L43` | threshold ≥ 2; tracker_size ≥ 1 |
| lookup（counter 也走这）| `reuse_manager.py:L67-L78` | `reuse_manager.py:L70-L79` | O21：lookup IS the incrementer |
| lookup LRU eviction | `reuse_manager.py:L75-L77` | `reuse_manager.py:L75-L77` | popitem(last=False) at max |
| prepare_store gate | `reuse_manager.py:L80-L87` | `reuse_manager.py:L81-L97` | counts.get(k, 0) ≥ threshold |
| OffloadingSpecFactory | `factory.py:L27-L79` | `factory.py:L17-L52` | 懒加载注册 |
| register_spec | `factory.py:L39-L54` | `factory.py:L20-L30` | closure，不导入 |
| create_spec | `factory.py:L65-L79` | `factory.py:L32-L52` | importlib.import_module |
| canonical 注册 | `factory.py:L86-L90` | `factory.py:L55-L58` | CPUOffloadingSpec 全限定路径 |
| O17 lazy-load 行为 | (test) | `factory.py:L20-L52` | bogus path 注册成功，create 时报错 |

### 12.9.7 scheduler + reactive lookup

| Topic | Impl | Source | Concept |
|---|---|---|---|
| GroupOffloadConfig | `offloading_scheduler.py:L57-L72` | `offloading/scheduler.py:L61-L82` | per-group block-size facts |
| SchedulerOffloadConfig | `offloading_scheduler.py:L74-L120` | `offloading/scheduler.py:L85-L111` | from_groups 工厂 |
| sliding_window_blocks 推算 | `offloading_scheduler.py:L102-L107` | `offloading/scheduler.py:L100-L105` | cdiv(window, block_size) |
| RequestGroupState | `offloading_scheduler.py:L123-L132` | `offloading/scheduler.py:L114-L122` | offload_keys + cursor |
| _SimpleRequest | `offloading_scheduler.py:L135-L148` | (教学) | block_hashes + num_tokens |
| RequestOffloadState | `offloading_scheduler.py:L151-L173` | `offloading/scheduler.py:L125-L182` | per-req state |
| update_offload_keys | `offloading_scheduler.py:L175-L194` | `offloading/scheduler.py:L143-L158` | islice 切 block_hashes |
| update_num_hit_blocks | `offloading_scheduler.py:L196-L202` | `offloading/scheduler.py:L160-L168` | 每 group num_hit_blocks |
| OffloadingConnectorScheduler.__init__ | `offloading_scheduler.py:L205-L247` | `offloading/scheduler.py:L185-L220` | full + sliding 分组 |
| sliding sort by window desc | `offloading_scheduler.py:L233-L237` | `offloading/scheduler.py:L201-L210` | order matters |
| _maximal_prefix_lookup | `offloading_scheduler.py:L249-L275` | `offloading/scheduler.py:L244-L261` | REACTIVE prefix scan（Trap E） |
| defer-lookup pipeline | `offloading_scheduler.py:L266-L271` | `offloading/scheduler.py:L250-L257` | None → pretend hit |
| _sliding_window_lookup | `offloading_scheduler.py:L277-L306` | `offloading/scheduler.py:L263-L287` | 倒扫 + consecutive_hits |
| _touch | `offloading_scheduler.py:L308-L320` | `offloading/scheduler.py:L289-L303` | LRU/ARC drive |
| get_num_new_matched_tokens | `offloading_scheduler.py:L322-L353` | `offloading/scheduler.py:L443-L486` | (num_tokens, is_async) |
| _lookup（cross-group） | `offloading_scheduler.py:L355-L439` | `offloading/scheduler.py:L305-L441` | 单次扫描简化 |
| dedupe blocks-being-loaded | `offloading_scheduler.py:L429-L438` | `offloading/scheduler.py:L488-L500` | enable_prefix_caching gate |
| overlap_blocks_per_step | `offloading_scheduler.py:L443-L454` | brief §1 movement 4 | step_compute / latency |
| headroom_freed_gb | `offloading_scheduler.py:L457-L468` | brief §1 movement 5 | offload × hit × ratio |

### 12.9.8 simple_offload_manager + connector ABC + taxonomy

| Topic | Impl | Source | Concept |
|---|---|---|---|
| OffloadMode enum | `simple_offload_manager.py:L56-L64` | `simple_kv_offload/manager.py:L67-L142` | LAZY / EAGER |
| TransferMeta | `simple_offload_manager.py:L67-L75` | `simple_kv_offload/manager.py:L41-L44` | gpu_block_ids + cpu_block_ids |
| LoadRequestState | `simple_offload_manager.py:L78-L90` | `simple_kv_offload/manager.py:L47-L52` | per-req load |
| StoreRequestState | `simple_offload_manager.py:L93-L107` | `simple_kv_offload/manager.py:L55-L64` | per-req store cursor |
| SimpleCPUOffloadScheduler.__init__ | `simple_offload_manager.py:L124-L167` | `simple_kv_offload/manager.py:L70-L157` | block pool + watermark |
| _allocate_cpu_blocks | `simple_offload_manager.py:L171-L183` | `simple_kv_offload/manager.py:L160-L186` | free_list + LRU evict |
| _evict_lru | `simple_offload_manager.py:L185-L194` | `simple_kv_offload/manager.py` | OrderedDict head |
| lookup | `simple_offload_manager.py:L198-L204` | `simple_kv_offload/manager.py` | move_to_end on hit |
| queue_store | `simple_offload_manager.py:L206-L242` | `simple_kv_offload/manager.py` | dedupe + alloc + bind |
| queue_load | `simple_offload_manager.py:L244-L274` | `simple_kv_offload/manager.py` | prefix lookup semantics |
| estimate_lazy_target_blocks | `simple_offload_manager.py:L291-L313` | `simple_kv_offload/manager.py:L189-L210` | watermark 启发式 |
| KVConnectorRole | `connector_taxonomy.py:L39-L49` | `kv_connector/v1/base.py:L123-L130` | SCHEDULER / WORKER |
| KVConnectorMetadata | `connector_taxonomy.py:L52-L64` | `kv_connector/v1/base.py:L42-L83` | per-step payload base |
| SupportsHMA mixin | `connector_taxonomy.py:L67-L82` | `kv_connector/v1/base.py:L84-L115` | HMA marker |
| KVConnectorBase_V1 ABC | `connector_taxonomy.py:L85-L191` | `kv_connector/v1/base.py:L170-L660` | 12 核心方法 |
| register_kv_caches | `connector_taxonomy.py:L126` | `kv_connector/v1/base.py:L298-L305` | worker init |
| start_load_kv | `connector_taxonomy.py:L135` | `kv_connector/v1/base.py:L310-L320` | async H2D kick-off |
| wait_for_layer_load | `connector_taxonomy.py:L138` | `kv_connector/v1/base.py:L330-L342` | per-layer block |
| save_kv_layer | `connector_taxonomy.py:L143-L150` | `kv_connector/v1/base.py:L345-L362` | 可选 layerwise |
| get_finished | `connector_taxonomy.py:L155-L159` | `kv_connector/v1/base.py:L378-L390` | drain done load+save |
| get_num_new_matched_tokens | `connector_taxonomy.py:L163-L173` | `kv_connector/v1/base.py:L449-L470` | (num, is_async) |
| update_state_after_alloc | `connector_taxonomy.py:L175-L177` | `kv_connector/v1/base.py:L478-L490` | bind GPU dst |
| build_connector_meta | `connector_taxonomy.py:L180-L184` | `kv_connector/v1/base.py:L495-L506` | metadata payload |
| ConnectorEntry | `connector_taxonomy.py:L195-L205` | (taxonomy data) | name + transport + tier + scope + status |
| Taxonomy 18 entries | `connector_taxonomy.py:L209-L388` | `kv_connector/v1/` 目录 | 完整 18 个 |
| OffloadingConnector entry | `connector_taxonomy.py:L211-L221` | `offloading_connector.py:1-192` | DMA + CPU DRAM |
| SimpleCPUOffloadConnector entry | `connector_taxonomy.py:L222-L231` | `simple_cpu_offload_connector.py:1-247` | 教学版本 |
| MultiConnector entry | `connector_taxonomy.py:L233-L242` | `multi_connector.py:1-629` | composed |
| LMCacheConnectorV1 entry | `connector_taxonomy.py:L244-L253` | `lmcache_connector.py:1-354` | RPC + DISK |
| LMCacheMpConnector entry | `connector_taxonomy.py:L254-L262` | `lmcache_mp_connector.py` | multi-process |
| MooncakeConnector entry | `connector_taxonomy.py:L264-L272` | `mooncake/mooncake_connector.py` | RDMA |
| NixlConnector entry | `connector_taxonomy.py:L273-L281` | `nixl/connector.py` | RDMA + GPU-direct |
| HF3FSConnector entry | `connector_taxonomy.py:L282-L290` | `hf3fs/hf3fs_connector.py` | distributed FS |
| FlexKVConnector entry | `connector_taxonomy.py:L292-L300` | `flexkv_connector.py` | research |
| P2P_Connector_NCCL entry | `connector_taxonomy.py:L302-L310` | `p2p/p2p_nccl_connector.py` | NCCL P2P |
| MoriIO_Connector entry | `connector_taxonomy.py:L312-L320` | `moriio/connector.py` | fabric |
| OffloadingConnectorScheduler entry | `connector_taxonomy.py:L322-L331` | `offloading/scheduler.py` | 881 LOC internals |
| OffloadingConnectorWorker entry | `connector_taxonomy.py:L332-L340` | `offloading/worker.py` | 370 LOC internals |
| LMCache_Integration entry | `connector_taxonomy.py:L342-L350` | `lmcache_integration/` | helpers |
| ExampleConnector entry | `connector_taxonomy.py:L352-L360` | `example_connector.py` | reference skeleton |
| ExampleHiddenStatesConnector entry | `connector_taxonomy.py:L361-L369` | `example_hidden_states_connector.py` | hidden-states variant |
| DecodeBenchConnector entry | `connector_taxonomy.py:L371-L378` | `decode_bench_connector.py` | bench synthetic |
| SsmConvTransfer entry | `connector_taxonomy.py:L380-L388` | `ssm_conv_transfer_utils.py` | Mamba helpers |
| connectors_by_scope | `connector_taxonomy.py:L391-L393` | (helper) | filter by scope |
| count_by_status | `connector_taxonomy.py:L396-L401` | (helper) | demo §0 用 |

### 12.9.9 demo + invariants + cross-chapter

| Topic | Impl | Source | Concept |
|---|---|---|---|
| Demo 1 per-tier | `demo.py::demo_1_per_tier_latency` | brief §7 | 9 verbatim |
| Demo 2 LRU vs ARC | `demo.py::demo_2_lru_vs_arc` | brief §7 | 6 verbatim + ARC 输 |
| Demo 3 overlap | `demo.py::demo_3_overlap` | brief §7 | 4 verbatim + break-even |
| Demo 4 pin vs pageable | `demo.py::demo_4_pin_vs_pageable` | brief §7 | 2 verbatim (K17 OR-skip) |
| Demo 5 e2e | `demo.py::demo_5_e2e_roundtrip` | brief §7 | 5 verbatim |
| Demo 0 taxonomy | `demo.py::demo_0_taxonomy` | brief §7 | 7 status counts |
| 7-trap verification | `tests/test_fidelity.py` | impl-notes §4 | 23 dedicated tests |
| Trap A test | `test_fidelity.py::TestTrapAOffloadVsSwap` | (no source) | 概念区分 |
| Trap B test | `test_fidelity.py::TestTrapBLFU` | `cpu/policies/` | 4 tests |
| Trap C test | `test_fidelity.py::TestTrapCAttentionScore` | (no source) | 2 tests |
| Trap D test | `test_fidelity.py::TestTrapDPredictivePrefetch` | (no source) | 3 tests |
| Trap E test | `test_fidelity.py::TestTrapEArcLoses` | demo §2 | 1 test (+ 3 in policies) |
| Trap F test | `test_fidelity.py::TestTrapFConnectorsNotInterchangeable` | taxonomy | 3 tests (+ 6 in taxonomy) |
| Trap G test | `test_fidelity.py::TestTrapGStreamsPCIeBound` | gpu_worker | 2 tests (+ 2 in worker) |
| ref_cnt sentinel test | `test_fidelity.py::TestRefCntSentinel` | `policies/base.py:L21-L33` | -1 → 0 atomic |
| ARC ghost bounds test | `test_fidelity.py::TestARCGhostBounds` | `arc.py:L150-L155` | trim to capacity |
| ≥70 REFERENCE 检查 | `test_fidelity.py::TestReferenceFloor` | impl floor | grep '# REFERENCE' ≥ 70 |
| Cross-chapter Ch02 | (this §12.6.5) | kv-cache.md K1-K12 | block layout |
| Cross-chapter Ch05 | (this §12.6.5) | memory.md M1-M3 | profiling |
| Cross-chapter Ch06 | (this §12.6.5) | scheduler.md | scheduling protocol |
| Cross-chapter Ch07 | (this §12.6.5) | prefix-cache.md | hash chain |
| Cross-chapter Ch11 | (this §12.6.5) | dcp-pcp.md D7 | per-rank HBM |
| Forward-pointer Ch13 | (this §12.6.5) | (next) | prefix-cache pooling |
| Forward-pointer Ch22 | (this §12.6.5) | (later) | PD architecture |
| Forward-pointer Ch23 | (this §12.6.5) | (later) | PD prefix-cache |
| Forward-pointer Ch24 | (this §12.6.5) | (later) | layerwise-connectors |
| Forward-pointer Ch27/28 | (this §12.6.5) | (later) | DeepSeek deep-dive |
| O01 — N=5 series ends | knowledge/kv-offload.md | (this chapter design) | not 6th class-X |
| O02 — outline corrects 4 TOPIC | knowledge/kv-offload.md | (this chapter design) | NVMe/LFU/attn-score/predict |
| O03 — connector/manager/spec | knowledge/kv-offload.md | base.py:L110-L398 | 三方区分 |
| O04 — offload ≠ swap | knowledge/kv-offload.md | scheduler.py vs cpu/manager.py:L91-L103 | Trap A canonical |
| O05 — pin = 2× pageable | knowledge/kv-offload.md | cuda_mem_ops.py:L16-L25 | 50% bounce buffer |
| O06 — per-tier latency | knowledge/kv-offload.md | (Demo 1) | 5.59 / 174.76 / 1198.37 / 262.14 µs |
| O07 — reactive not predictive | knowledge/kv-offload.md | offloading/scheduler.py:L244-L261 | Trap E reframe |
| O08 — ARC not always better | knowledge/kv-offload.md | (Demo 2 phase_shift) | LRU 2.60 vs ARC 14.15 |
| O09 — SharedOffloadRegion mmap | knowledge/kv-offload.md | shared_offload_region.py:L27-L113 | MADV_POPULATE_WRITE |
| O10 — factory lazy-load | knowledge/kv-offload.md | factory.py:L17-L52 | optional deps |
| O11 — KVConnectorRole | knowledge/kv-offload.md | base.py:L123-L130 | role-conditional init |
| O12 — MultiConnector | knowledge/kv-offload.md | multi_connector.py | dedupe by hash |
| O13 — simple vs production | knowledge/kv-offload.md | simple_kv_offload/ vs kv_offload/ | 两路并存 |
| O14 — block-size break-even | knowledge/kv-offload.md | (Demo 3) | 651 KiB |
| O15 — proactive eviction | knowledge/kv-offload.md | cpu/manager.py:L115-L168 | evicted_keys 提前 |
| O16 — factory full path | knowledge/kv-offload.md | factory.py:L86-L90 | 测试需 re-register |
| O17 — register lazy fail | knowledge/kv-offload.md | factory.py:L20-L52 | bogus path OK at register |
| O18 — block_ids 防御拷贝 | knowledge/kv-offload.md | base.py:L219-L228 | list(block_ids) copy |
| O19 — LRU.touch 倒序 | knowledge/kv-offload.md | lru.py:L26-L29 | reversed iteration |
| O20 — ARC dry-run | knowledge/kv-offload.md | arc.py:L97-L156 | atomic None-on-fail |
| O21 — lookup IS counter | knowledge/kv-offload.md | reuse_manager.py:L70-L79 | even miss increments |
| O22 — complete_store 不对称 | knowledge/kv-offload.md | cpu/manager.py:L170-L195 | success-only event |

整张主表 142 行，覆盖本章引用的全部源码点 + 实现点 + Demo + Trap + Knowledge facts。

---

## 12.10 章节小结 + Ch13 forward pointer

本章把 outline 的 4 个 TOPIC reframe 走完了，每条都有源码 grep 证据 + 教学版 1:1 镜像：

1. NVMe SSD 第三 tier → 实际是 HBM ↔ CPU pinned 两 tier；`vllm/v1/kv_offload/` grep `nvme|ssd|disk` 零匹配；`vllm/v1/kv_offload/cpu/` 是唯一子目录。NVMe / CXL / NVMe-over-fabric 是学术研究方向。
2. LRU/LFU/attention-score → 实际是 LRU + ARC（`vllm/v1/kv_offload/cpu/policies/` 只有 `base.py + lru.py + arc.py`，无 `lfu.py`、无 `attention_score.py`）；`CACHE_POLICIES` 字典在 `vllm/v1/kv_offload/cpu/manager.py:L19-L22` 只有两个键。ARC 在 phase_shift workload 上**输给** LRU 5.4× 是诚实的（demo §2 数据 LRU 2.60% vs ARC 14.15%）。
3. attention-score-based eviction → vLLM 全部用 block-hash 语义；`OffloadKey` 在 `vllm/v1/kv_offload/base.py:L24-L44` 是 `bytes` newtype，wrap (block_hash, group_idx) tuple，**没有** token-level attention statistic。
4. predict 哪个 block 会用 → 实际是 reactive `_maximal_prefix_lookup`（`vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:L244-L261`），确定性 prefix scan，无 ML / 无 Markov / 无 prediction model 引入。

7 个 Trap（A-G）覆盖 offload-vs-swap、policies-不存在、PCIe-不-免费、connector-不可换、prefetch-不预测、pin-不免费、v0-非-v1 全部对错点。5 个 framing tip 三-anchor 完整。26 个 verbatim 数字全部 reproduce。314/314 tests pass，包括 Trap E phase_shift ARC 输的 4 条专用 test。

next stop **Ch13 prefix-cache-pooling**：Ch12 给了 KV offload 单 tier 的接口 (`OffloadingManager.lookup` / `prepare_load`)；Ch13 把它和 prefix-cache (Ch07) 组合——pool 的 capacity 来自 GPU prefix-cache + CPU offload tier 的合计。新的问题是**多 pool 共享**（多个 vLLM 实例共用同一个 LMCache 服务）和 **prefix-cache 的语义对齐**——Ch07 的 hash-chain 在 Ch12 的 OffloadingManager 上下文里有什么新约束？答案在下一章。

---

**END OF CHAPTER 12 (KV Cache Offload).**

Source pin: `98661fe`. Mapping rows: 142. Verbatim numerics: 26. Traps: 7 (A-G). Knowledge facts: O01-O22. Tests: 314/314. Lint: clean.
