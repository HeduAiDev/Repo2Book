# v0.21.0 更新摘要 — F 组：GPU model runner + worker + input batch（ch17/ch18/ch19）

基线 `f3fef1235` → 标签 `v0.21.0`。文件组：
`vllm/v1/worker/gpu_model_runner.py`（+133/-78）、`gpu_input_batch.py`（+3/-4）、
`block_table.py`（+7）、`gpu_worker.py`（+107/-33）。

只抽取「面向读者、可教学」的变更（新执行路径 / dummy-profile / CUDA graph 捕获行为 /
worker 初始化与显存步骤 / API 签名 / 行为变化）。纯重构、移动、格式、类型标注一律 SKIP。

---

## 1. Worker 显式权重热更新三段式：`start_weight_update` / `update_weights` / `finish_weight_update`

- **class**: API-CHANGE（兼 NEW-FEATURE）
- **anchor**: `vllm/v1/worker/gpu_worker.py:Worker.start_weight_update` /
  `Worker.update_weights` / `Worker.finish_weight_update`；新增状态位
  `Worker._weight_update_active`、`Worker._is_checkpoint_format`；新私有断言
  `Worker._check_weight_transfer_engine`
- **target**: ch17（Worker 生命周期）
- **改了什么**：基线里权重热更新是单方法 `update_weights()` 一把梭——内部自己做
  `initialize_layerwise_reload` → `receive_weights` → `finalize_layerwise_reload`。
  v0.21.0 拆成显式三段式：`start_weight_update(is_checkpoint_format)` 先开局
  （checkpoint 格式时调 `initialize_layerwise_reload` 建逐层重载状态、置
  `_weight_update_active=True`）；`update_weights()` 此后可被调一次或多次接收权重分块
  （checkpoint 格式走 `receive_weights`，kernel 格式走就地 `param.copy_`）；
  `finish_weight_update()` 收尾跑 `finalize_layerwise_reload` 并复位状态。
  这对应控制平面新增的 `/start_weight_update`、`/finish_weight_update` HTTP 端点（#39212），
  让 RLHF 训练侧能分块流式推权重而非整批阻塞。
- **集成（书声）**：第17章 Worker 段落讲到 collective_rpc 暴露的引擎指令时，可补一句：
  「v0.21.0 起，`vllm/v1/worker/gpu_worker.py` 的权重热更新从单步 `update_weights`
  演进为 `start_weight_update → update_weights*（可多次）→ finish_weight_update` 的显式三段式——
  开局建逐层重载状态、中段按分块接收、收尾做后处理并复位 `_weight_update_active` 守卫，
  以支持训练端分块流式推送权重。」纯增量，不与现有正文冲突。
- **diagram impact**：无。ch17 现有图不涉及权重热更新；如未来补 RLHF 权重热更新时序图再考虑，本轮不需出图。

## 2. CUDA graph 显存估计的平台门控：ROCm 排除 → 仅 CUDA 准入（XPU 也被排除）

- **class**: BEHAVIOR-CHANGE
- **anchor**: `vllm/v1/worker/gpu_worker.py:Worker`（`determine_available_memory` 路径内
  `cudagraph_memory_estimate` 的守卫条件，v0.21.0 约 L400-407）
- **target**: ch17（Worker 显存探测 / profile run），亦与 ch19 §19.6 CUDA graph 分派呼应
- **改了什么**：基线守卫是 `not current_platform.is_rocm()`（即「非 ROCm 才估计 graph 显存」）；
  v0.21.0 收紧为 `current_platform.is_cuda()`（「只有 CUDA 平台才估计」）。语义差别在于
  XPU 等非 CUDA/非 ROCm 平台现在也一并跳过 graph 显存估计（#41344，XPU 平台禁用该估计）。
  对 NVIDIA CUDA 主线行为不变，是平台覆盖面的收口。
- **集成（书声）**：若 ch17 讲到 `determine_available_memory` 里为 CUDA graph 预留显存的估计时，
  可点一句平台门控由「排除 ROCm」收紧为「仅 CUDA 准入」，使 XPU 等平台同样跳过该项估计。属可选脚注，非必须。
- **diagram impact**：无。

## 3. warmup 收尾启动 Triton JIT 编译监视器

- **class**: NEW-FEATURE
- **anchor**: `vllm/v1/worker/gpu_worker.py`（`compile_or_warm_up_model` 收尾处，
  `set_random_seed` 之后调 `vllm.triton_utils.jit_monitor.activate`）
- **target**: ch17（Worker 出生/服役阶段的编译 warmup）
- **改了什么**：所有 warmup 完成、随机种子设好之后，新增激活 Triton JIT 编译监视器
  （#40137）。意图是：warmup 之后推理热路径上若再触发 Triton kernel 的 JIT 即时编译，
  会造成延迟尖刺；监视器在 warmup 后开启，用于侦测这类「不该出现的运行时编译」。
- **集成（书声）**：ch17 讲 worker 编译 warmup 收尾时，可补一句：v0.21.0 在 warmup 结束后
  激活 Triton JIT 编译监视器，把「warmup 之后还触发即时编译」视作延迟尖刺信号加以侦测。
  一句话脚注即可。
- **diagram impact**：无。

## 4. PP 模式写回修复：写回起点改用 `num_tokens_no_spec`，支持 chunked prefill 与异步 PP 推进

- **class**: BEHAVIOR-CHANGE（重要，正中 ch19 §19.5 教学代码块）
- **anchor**: `vllm/v1/worker/gpu_model_runner.py:GPUModelRunner._update_states`
  （`is_last_rank` 为否时把 `new_token_ids` 并入 `token_ids_cpu` 的那段，基线约 L1357）；
  配套 `gpu_input_batch.py:InputBatch.update_req_spec_token_ids` 新增
  `is_token_ids[...] = True`
- **target**: ch19（§19.5 写回闭环），波及 ch18（§18.4 在途请求就地更新）
- **改了什么**：修复 PP（流水线并行）模式下的 token 丢失导致精度下降（#41133）。基线里非末位 rank
  并入新 token 时，起点直接取 `start_token_index = num_computed_tokens`、终点
  `num_computed_tokens + len(new_token_ids)` 无条件覆盖写。v0.21.0 改为：
  起点取 `start_token_index = num_tokens_no_spec[req_index]`（已落位计数），
  终点取 `max(start_token_index, num_computed_tokens + len(new_token_ids))`——
  原因有二：(a) chunked prefill 下 `num_computed_tokens` 可能**小于** `num_tokens_no_spec`；
  (b) 异步调度的 PP 下可能**没有** `new_token_ids`，此时需仅按 `num_computed_tokens`
  推进计数。仅当 `end > start` 才写：有 `new_token_ids` 时只取尾部
  `new_token_ids[-num_new_tokens:]` 追加（避免重复覆盖），并同步置
  `is_token_ids[req_index, start:end] = True`，再推进 `num_tokens_no_spec`。
- **集成（书声）**：ch19 §19.5「写回侧」与「读回侧」当前以**末位 rank / 非 PP** 视角讲
  「写指针 `num_tokens_no_spec`、读指针 `num_computed_tokens` 一拍拍咬在同一格」的不变式
  （正文已正确以 `num_tokens_no_spec[req_idx]` 为写回起点）。可在 §19.5「异步调度下的一个变奏」
  之后补一小段「PP 非末位 rank 的变奏」：非末位 rank 用 `_update_states` 把上游传来的
  `new_token_ids` 并入 `token_ids_cpu`，v0.21.0 起其写回起点同样改用 `num_tokens_no_spec`、
  终点取 `max(start, num_computed_tokens + len(new_token_ids))`，以同时容纳
  chunked prefill（`num_computed_tokens < num_tokens_no_spec`）与异步 PP（无 new token、
  仅推进计数）两种边界——否则会丢 token、拉低精度。这与正文「同一计数撑闭环」的不变式同源、
  是其在 PP 路径上的严格化，不改变主线叙事。
- **diagram impact**：可选。§19.5 的 `f13-writeback.png` 图仍正确（讲的是单 rank 同格闭环）；
  如要覆盖 PP 非末位 rank 的 `max(start, ...)` 推进可加一个小注，但**不强制改图**。

## 5. prompt logprobs 在途累积从批次级 dict 迁到请求级字段

- **class**: BEHAVIOR-CHANGE / API-CHANGE（数据归属变更）
- **anchor**: `vllm/v1/worker/gpu_input_batch.py:CachedRequestState.in_progress_prompt_logprobs_cpu`
  （新增字段）；删去 `InputBatch.in_progress_prompt_logprobs_cpu` 这个
  `dict[str, LogprobsTensors]`；`gpu_model_runner.py` 内
  `_get_prompt_logprobs_dict` 改读写 `request.in_progress_prompt_logprobs_cpu`
- **target**: ch18（持久批次字段归属）；与 ch10（logprobs 装配）远程相关
- **改了什么**：修复 chunked prefill 下请求被驱逐（eviction）时 prompt logprobs 丢失（#41411）。
  基线把「跨 prefill 分块累积的 prompt logprobs 张量」存在 `InputBatch` 的一个
  `req_id → LogprobsTensors` 字典里，随 `remove_request` pop；问题是请求被驱逐再恢复时
  这份在途累积就丢了。v0.21.0 把它挪进 `CachedRequestState`（即 `self.requests[req_id]`
  这一层、跨 slot 进出存活的请求快照）的 `in_progress_prompt_logprobs_cpu` 字段，
  生命周期跟着请求走而非跟着 slot 行走，从而在驱逐/恢复中不丢累积。
- **集成（书声）**：ch18 在讲 `CachedRequestState`（请求快照）与 `InputBatch`（slot 行视图）
  的字段分工时，可补一句：v0.21.0 起，prompt logprobs 的「跨分块在途累积」归属于请求快照
  `CachedRequestState.in_progress_prompt_logprobs_cpu`，而非批次级字典——这样它随请求存活、
  在 chunked prefill 的驱逐与恢复中不丢。一句话呼应「快照随请求、slot 行随批次」的归属原则。
- **diagram impact**：无。

## 6. spec 草稿写回也置 `is_token_ids`

- **class**: BEHAVIOR-CHANGE（小）
- **anchor**: `vllm/v1/worker/gpu_input_batch.py:InputBatch.update_req_spec_token_ids`
  （`token_ids_cpu[...] = spec_token_ids` 之后新增 `is_token_ids[...] = True`）
- **target**: ch18 / ch19（写回侧），可选
- **改了什么**：投机草稿 token 写入 slot 行后，现在也把对应区间的 `is_token_ids` 掩码置真，
  与正常采样 token 写回（ch19 §19.5 已展示的 `is_token_ids[...] = True`）保持一致。
  保证「该格是 token 而非 embedding」的掩码对 spec token 也成立。
- **集成（书声）**：若 ch18/ch19 提到 `is_token_ids` 掩码用于区分 token 与 prompt_embeds，
  可顺带一句 spec 草稿写回也维护该掩码。可选脚注，多数情况下可并入第 4 条不单列。
- **diagram impact**：无。

## 7. block_table 行数对齐到 128/block_size 的整数倍（TRTLLM MLA 边角）

- **class**: BEHAVIOR-CHANGE
- **anchor**: `vllm/v1/worker/block_table.py:MultiGroupBlockTable.__init__`
  （`max_num_blocks` 对齐计算：`cdiv(n, 128//bs) * (128//bs) if bs <= 128 else n`）
- **target**: ch18（§18.8 block_table 双镜像），可选
- **改了什么**：构造 `MultiGroupBlockTable` 时，对每个 KV cache group 的 `max_num_blocks`
  按 `128 / block_size` 的整数倍向上对齐（#39324）。原因是部分 attention 后端（如 TRTLLM MLA）
  对 block table 列数有 128 元素对齐的边角要求。对常规 block_size（≤128）会略微抬高表的列数。
- **集成（书声）**：ch18 §18.8 讲 block_table 的 CPU/GPU 双镜像、形状由 `max_num_blocks` 决定时，
  可补一句：v0.21.0 起每组 `max_num_blocks` 会向上对齐到 `128/block_size` 的整数倍，
  以满足 TRTLLM MLA 等后端对 block table 列数的对齐约束。一句话脚注即可。
- **diagram impact**：无。

---

## SKIP（仅记录，不入正文）

- **Gemma4 MTP 投机解码 Proposer**（#41745）：`gpu_model_runner.py` 大量
  `Gemma4Proposer` isinstance 分支扩充、`use_gemma4_mtp()` 分派、`set_per_group_block_table`。
  这是一个**新模型专属的投机解码器接入**，属投机解码章（非本 F 组三章 ch17-19 的主题，
  ch17-19 不深入具体 proposer 实现）。如全书有投机解码专章可单列；对 ch17/18/19 = SKIP。
- **routed-experts capturer 大重构**（#39917、#42148）：`RoutedExpertsCapturer` 类 →
  一组自由函数（`get_global_experts_capturer` / `extract_routed_experts_for_current_batch`
  / `issue_routing_d2h_copy` / `init_routed_experts_capturer_with_shared_cache` /
  `free_routing_buffers`），改用 device cache + 异步 D2H 流水线、CUDA graph 捕获前先建
  capturer（buffer 地址烤进 graph）、新增 `_positions_cpu` pinned 缓冲、
  `ModelRunnerOutput.routed_experts_dict` 输出。这是 MoE 专家路由捕获子系统的内部重写，
  ch17-19 现有正文不涉及 routed-experts capturer，无对应教学点 = SKIP（属 MoE/EPLB 相关章范畴）。
- **`sync_and_slice_intermediate_tensors` → `sync_and_gather_intermediate_tensors`
  + SP residual all-gather**（#33322、#41133）：方法改名 + SP 模式下 residual 改为
  `get_tp_group().all_gather` 全收集（下游 QKV+Attention 需完整 residual）。这是
  序列并行（SP）+ 流水线并行（PP）的 intermediate tensor 同步细节，ch17-19 正文未覆盖
  SP/PP 的 intermediate tensor 传递路径 = SKIP（属 TP/SP/PP 并行章范畴）。
- **末位 rank 守卫 `world_size>1` → `not get_pp_group().is_last_rank`**：异步调度下
  PP 接收上拍采样 token 的条件收紧，纯边界正确性、不改主线叙事，且 ch19 正文未展开
  PP 接收路径 = SKIP（可并入第 4 条的 PP 语境理解，无需单独入正文）。
- **routed-experts 热路径短路**（#42148，`free_routing_buffers` 在 `_update_states`
  尾部按 finished/preempted 释放）：随 routed-experts 重构，SKIP 同上。
