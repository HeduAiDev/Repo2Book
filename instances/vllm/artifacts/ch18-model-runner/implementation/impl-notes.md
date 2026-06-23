# ch18 reduced companion — implementation notes

Subtract-only companion for ch18 (持久化批次与输入准备). Faithful subset of the real
vLLM v1 model runner: same names, same structure, same control flow; only the
`subtraction_plan.delete` items are removed (each marked `# SUBTRACTED:`). Source
pin `f3fef123`.

## Files

- `block_table.py` ← `vllm/v1/worker/block_table.py` — `BlockTable` / `MultiGroupBlockTable` / `_compute_slot_mapping_kernel`
  (+ the real `CpuGpuBuffer` helper). The CPU/GPU双镜像 and the Triton position→slot kernel.
- `gpu_input_batch.py` ← `vllm/v1/worker/gpu_input_batch.py` — `CachedRequestState` + the persistent `InputBatch`
  (add/remove/condense/swap, slot reuse, sampling-param packing).
- `logits_processor_state.py` ← `vllm/v1/sample/logits_processor/state.py` — `BatchUpdateBuilder` (removed-降序 / pop_removed /
  peek_removed): the actual slot-reuse + condense bookkeeper.
- `gpu_model_runner.py` ← `vllm/v1/worker/gpu_model_runner.py` — slimmed `GPUModelRunner` with `_update_states`,
  `_prepare_inputs`, `_build_attention_metadata`, `_get_cumsum_and_arange`,
  `_may_reorder_batch` + faithful `SchedulerOutput`/`CachedRequestData`/`NewRequestData`.
- `_support.py` — minimal faithful stand-ins (`SamplingType`, `SamplingParams` field
  subset, `length_from_prompt_token_ids_or_embeds`) so the batch runs host-side.

## Source Map (精简版 ↔ vllm/...:Lxxx ↔ 改动 ↔ 原因)

| 精简版符号 | vllm 源 | 改动 | 原因 |
|---|---|---|---|
| `InputBatch._register_add_request` | `gpu_input_batch.py:L310` | 原样 | slot 复用入口：`pop_removed()` 取最小空 slot，否则 append 末尾 |
| `InputBatch.add_request` | `gpu_input_batch.py:L336` | 删 prompt_embeds/pooling/LoRA 分支 | 采样参数装填保留；杂项特性按 delete 批准删 |
| `InputBatch.remove_request` | `gpu_input_batch.py:L510` | 删 LoRA/pooling/thinking-budget 清理 | 只保留打洞标记 + 对称采样态清理 |
| `InputBatch.condense` | `gpu_input_batch.py:L684` | 删 pooling 早退/prompt_embeds 搬移 | 核心压实循环保留：尾部活请求滑入最小空洞，只拷 `:num_tokens` |
| `InputBatch.refresh_metadata` / `_make_sampling_metadata` | `gpu_input_batch.py:L812`/`L832` | metadata 装配缩为快照 | SamplingMetadata 全量装配属采样章；本章只需"批变→重建"信号 |
| `BatchUpdateBuilder` | `state.py:L18` | `get_and_reset` 的 BatchUpdate 缩为元组 | logits-processor 接线在本章主线外；保留 removed 降序/pop/peek 语义 |
| `BlockTable.append_row`/`add_row`/`move_row` | `block_table.py:L102`/`L120`/`L130` | 删 hybrid `map_to_kernel_blocks` 调用 | `kernel_block_size==block_size` 时该路径本就跳过 |
| `BlockTable.compute_slot_mapping` / `commit_block_table` | `block_table.py:L141`/`L166` | 原样 | CPU→GPU 拷贝 + 启动 Triton kernel |
| `_compute_slot_mapping_kernel` | `block_table.py:L318` | 删 CP `is_local`/`local_block_offsets` | 单 rank `TOTAL_CP_WORLD_SIZE==1`：`is_local` 恒真、offset=pos%bs |
| `GPUModelRunner._update_states` | `gpu_model_runner.py:L1065` | 删 PP/async-spec/ngram/pooling/LoRA/mrope 分支 | token-only 末 rank 主线：增删/重排调和持久批次 |
| `GPUModelRunner._prepare_inputs` | `gpu_model_runner.py:L1787` | 删 mrope/prompt_embeds/async-spec/LoRA 分支 | 保留 `np.repeat`→`req_indices`、扁平 `index_select`、positions、slot_mapping |
| `GPUModelRunner._get_cumsum_and_arange` | `gpu_model_runner.py:L1572` | 原样 | `[2,5,3]`→cu`[2,7,10]` + 请求内 arange |
| `GPUModelRunner._build_attention_metadata` | `gpu_model_runner.py:L2098` | 缩为 dict；删 padding/cudagraph/DCP/routed-experts | 取 block_table GPU 镜像+slot_mapping+seq_lens 收束成 attn 输入 |
| `GPUModelRunner._may_reorder_batch` | `gpu_model_runner.py:L1003` | 删 kv-group 守卫；阈值 None 时 no-op | attention backend 重排钩子 |

## 验收判据

把真实 vLLM 删掉所有 `# SUBTRACTED:` 分支（M-RoPE/XD-RoPE、prompt_embeds、async spec
decode、pooling、LoRA、PP/KV-connector、CP/DCP、hybrid block、thinking-budget 等），
应 ≈ 得到本精简版。

## 怎么跑

- 持久批次 + gather 索引算术（纯 CPU/numpy）：`python3 -m pytest tests/test_persistent_batch.py`
- Triton slot-mapping kernel + 端到端 `_update_states`+`_prepare_inputs`（需 CUDA）：
  `python3 -m pytest tests/test_prepare_inputs_gpu.py`（host 无 CUDA 时自动 skip；
  容器内 `scripts/vllm_docker.sh -m pytest /work/.../tests/`）。
