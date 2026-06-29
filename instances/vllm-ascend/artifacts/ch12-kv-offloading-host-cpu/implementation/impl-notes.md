# ch12 KV 卸载 — 精简版实现笔记（subtract-only）

把昇腾 KV 卸载的两条路径（标准路径 v1/kv_offload + 极简路径 v1/simple_kv_offload）砍成
host 可跑、可断点的精简版。**只做减法**：与 `vllm_ascend/...` 真实源码同名/同结构/同控制流。

## host 运行边界（dossier 明示）

昇腾代码 host 无 NPU/CANN 不可跑。`runtime_stub.py`（**NOT subtract-only**，每个替身标 `# SOURCE`）
把 `torch.npu.Stream/Event/stream/current_stream/set_device` 与 `torch.ops._C_ascend.swap_blocks_batch`
补丁到真 torch 上（前者 no-op，后者把调用参数记进 `SWAP_CALLS`、不真搬字节），并接住基座
`vllm.v1.kv_offload.*` / `vllm.v1.simple_kv_offload.*` 抽象。**可跑可验**的是：分层搬运节拍
（deque + wait_stream/wait_event 编排）、block 视图重建（`set_()` 按 stride/shape 裁视图）、
DMA 拷贝调度（队列+后台线程+Event 轮询）、指针算术（`base + block_id * bytes_per_block` 的 numpy 广播）。
**不真跑**的是实际 device↔host 字节搬运（aclrtMemcpyBatchAsync）。

## 1:1 Source Map

| 精简版符号 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `npu.py:NPUOffloadingSpec.get_manager` | `vllm_ascend/kv_offload/npu.py:L31` | 删 `kv_events_config`/`enable_events` 读取，固定 `enable_events=False` | 事件开关只透传给基座 Manager 做 KV-event 上报，与 device↔host 搬运无关（delete 批准项） |
| `npu.py:NPUOffloadingSpec.get_handlers` | `vllm_ascend/kv_offload/npu.py:L45` | 删向 Handler 透传 `attn_backends` | Handler 内从未读取该形参，死参（delete 批准项） |
| `cpu_npu.py:CpuNpuOffloadingHandler.__init__` | `vllm_ascend/kv_offload/cpu_npu.py:L54` | 删 `attn_backends` 形参 + 2 处 `logger.info/debug` | 死参 + 纯日志旁路（delete 批准项） |
| `cpu_npu.py:transfer_async` | `vllm_ascend/kv_offload/cpu_npu.py:L142` | 逐字保留（含 wait_stream/wait_event/deque/swap_blocks_batch） | must_keep：分层搬运节拍全在此 |
| `cpu_npu.py:get_finished` / `wait` | `vllm_ascend/kv_offload/cpu_npu.py:L232/L254` | 逐字保留 | must_keep：`query()` 非阻塞轮询 vs `synchronize()` 阻塞对照 |
| `cpu_npu.py:expand_block_ids` | `vllm_ascend/kv_offload/cpu_npu.py:L22` | 逐字保留 | must_keep：粗/细粒度换算 + skip_count 对齐 |
| `worker.py:register_kv_caches` | `vllm_ascend/simple_kv_offload/worker.py:L75` | 删 3 处 `logger.warning/info` | 纯日志旁路（delete 批准项） |
| `worker.py:_build_block_views` | `vllm_ascend/simple_kv_offload/worker.py:L160` | 逐字保留（单段/多段两布局、`set_()` 按 stride/shape 取尺寸） | must_keep：为什么卸载要重建 block view |
| `worker.py:_flatten_kv_value` | `vllm_ascend/simple_kv_offload/worker.py:L39` | 逐字保留 | must_keep：K/V 分开分配，少了它漏 V cache |
| `copy_backend.py:NPUDmaCopyBackend`（init/launch_copy/_copy_loop/shutdown） | `vllm_ascend/simple_kv_offload/copy_backend.py:L24+` | 逐字保留 | must_keep：FIFO 队列 + 后台线程 + Event 轮询的 DMA 调度节拍 |
| `npu_mem_ops.py:build_params` / `copy_blocks` / `BatchMemcpyParams` / `DIRECTION_*` | `vllm_ascend/simple_kv_offload/npu_mem_ops.py:L17+` | 逐字保留（含 `_ordered_tensors`，未在 delete 批准列内故不内联） | must_keep：指针布局 + 收口 swap_blocks_batch |

## 保留但 delete 计划允许删的项（出于保真选择保留）

- **`_event_pool` 复用**（`_get_event`/`_recycle_event`）：delete 标为"可简化为每次新建"，但 dossier 自注
  "若要点明复用避免分配开销建议保留" → **保留**。
- **`Transfer.num_bytes` / `TransferResult.transfer_size`/`transfer_time` 统计链路**：delete 批准可删，
  但保留使在途搬运记录与轮询节拍完整、`elapsed_time` 由 Event 桩平凡支持 → **保留**（保真优先，删除非强制）。
- **`copy_backend.shutdown`**：dossier embed 仅 elide 不展开，未在 delete 列 → **保留**（亦便于测试清线程）。

## 验证

- `python3 -m pytest`（host）：22 passed —— 覆盖 expand/方向判定/指针广播/deque 节拍/query 轮询/
  视图重建单段·多段/去重不漏 V/后台线程 DMA 调度/收口 swap_blocks_batch 方向码。
- `python3 scripts/lint_fidelity.py <chapter_dir>`：✓ 全通过（must_keep 28 符号全在、SOURCE 全覆盖、无杜撰）。
