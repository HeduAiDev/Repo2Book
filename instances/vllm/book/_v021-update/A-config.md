# v0.21.0 更新摘要 — A 组：config & wiring（ch01 / ch03）

基线 `f3fef1235` → 标签 `v0.21.0`。文件组：`vllm/config/vllm.py`、`vllm/engine/arg_utils.py`、`vllm/v1/engine/core.py`、`vllm/v1/engine/llm_engine.py`。

`vllm/v1/engine/llm_engine.py` 在区间内**无 diff**，无需改动。

教学性变更共 **6 项**（NEW-FEATURE 3 / API-CHANGE 1 / BEHAVIOR-CHANGE 2），外加若干 SKIP。下面按变更分组，每项均锚定真实 diff 行。

---

## 1. `VllmConfig._verify_kv_transfer_compat`：拒绝 NixlConnector + expandable_segments

- **class**: NEW-FEATURE（新校验分支）
- **来源 commit**: `60851b1d2`（#41237）
- **v0.21.0 锚点**: `vllm/config/vllm.py` 新增方法 `VllmConfig._verify_kv_transfer_compat`，在 `__post_init__` 尾段 `_post_init_kv_transfer_config()` 之后被调用（`self._verify_kv_transfer_compat()`）。
- **目标章节**: ch03（`__post_init__` 校验中枢，3.4 节）
- **要点**: 当配置了任意 KV connector，且 `PYTORCH_CUDA_ALLOC_CONF` 含 `expandable_segments:True`、又未开 `enable_sleep_mode` 时，直接 `raise ValueError`。原因写在源码注释里：PyTorch 的 CUDA VMM 分配器会把 KV cache 的虚拟地址重映射到不同物理页，使 NIXL / Mooncake 通过 `ibv_reg_mr` 注册（pin）的 KV 内存失效，触发 `IBV_WC_REM_ACCESS_ERR` / `NIXL_ERR_REMOTE_DISCONNECT`。sleep mode 例外，因为 `CuMemAllocator` 的内存池会在其作用域内自动关掉 expandable_segments。
- **integration suggestion**:
  > v0.21.0 在 `__post_init__` 的 KV transfer 收尾处补了一道 `_verify_kv_transfer_compat` 闸门：只要配了 KV connector，又把 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 打开、且没开 sleep mode，就直接抛错。这是「跨子配置校验集中在 `__post_init__`」这一模式的又一个典型——单看 `KVTransferConfig` 或环境变量都合法，凑在一起才知道会让 NIXL 注册的内存指向被重映射后的废弃物理页。
- **diagram impact**: 可选。3.4 节配图若列出 `__post_init__` 内的校验项，可在校验流里补一个 `_verify_kv_transfer_compat` 方块；非必需。

---

## 2. `VllmConfig._validate_return_routed_experts`：路由专家返回的并行度白名单

- **class**: NEW-FEATURE（新 config 字段 + 新校验分支）
- **来源 commit**: `8189a1591`（#39917）
- **v0.21.0 锚点**:
  - 新字段 `vllm/config/model.py` `ModelConfig.enable_return_routed_experts: bool = False`（CLI `--enable-return-routed-experts`，`vllm/engine/arg_utils.py:416`）。
  - 新方法 `vllm/config/vllm.py` `VllmConfig._validate_return_routed_experts`，在 `__post_init__` 中条件触发（`if self.model_config is not None and self.model_config.enable_return_routed_experts:`）。
- **目标章节**: ch03（3.4 节校验中枢）
- **要点**: 开启该旗标后，校验拒绝尚未验证的并行组合：`pipeline_parallel_size > 1`、`prefill_context_parallel_size > 1`、`decode_context_parallel_size > 1`、以及 `async_scheduling`（任一命中即 `raise ValueError`）。已验证范围（注释）：TP/EP/DP、单/多节点、prefix caching、投机解码。
- **integration suggestion**:
  > v0.21.0 引入 `--enable-return-routed-experts`（落在 `ModelConfig.enable_return_routed_experts`）。它带来的不是一段执行逻辑，而是 `__post_init__` 里一组「能力边界」校验：一旦开启，PP>1、PCP/DCP>1 和 async scheduling 这些尚未端到端验证的组合会被直接拒绝。这正好印证本章的观点——`VllmConfig.__post_init__` 是把「这个特性目前支持到哪」这类跨配置约束兜底落地的地方。
- **diagram impact**: 无需。

---

## 3. safetensors 预取可调（两个新 LoadConfig 旋钮）

- **class**: NEW-FEATURE（新 config 字段 + 新 CLI 参数）
- **来源 commit**: `21943d4c2`（#41499）
- **v0.21.0 锚点**:
  - `vllm/config/load.py`：`LoadConfig.safetensors_prefetch_num_threads`、`LoadConfig.safetensors_prefetch_block_size`（均 `Field(..., ge=1)`，注释说明分别是「预取 worker 线程数」与「每文件预取读取块字节数」）。
  - `vllm/engine/arg_utils.py`：`EngineArgs` 新增同名两字段（指向 `LoadConfig` 默认值），新增 CLI `--safetensors-prefetch-num-threads` / `--safetensors-prefetch-block-size`，并在 `create_load_config`（约 L1598）回填；`safetensors_prefetch_block_size` 加入接受 `human_readable_int`（如 `16M`）的字段名单。
- **目标章节**: ch03（3.2/3.3 节，扁平 `EngineArgs` → CLI → `create_load_config`）
- **要点**: 纯加法旋钮，不改主控制流，用来调把权重预取进 OS page cache 的并发度与读块大小。
- **integration suggestion**:
  > v0.21.0 给权重加载又添了两个旋钮：`--safetensors-prefetch-num-threads` 和 `--safetensors-prefetch-block-size`（落在 `LoadConfig`，由 `create_load_config` 回填）。后者跟 `max_num_batched_tokens` 一样支持人类可读写法（如 `16M`），是「`EngineArgs` 这个扁平参数袋子持续长字段、CLI 自动派生」模式的又一例证，无需展开讲，点到为止即可。
- **diagram impact**: 无需。

---

## 4. `get_kwargs` 统一处理 bool-or-str 可选参数（`--hf-token` 不再特判）

- **class**: API-CHANGE（CLI 解析行为的通用化 + 去除手写特例）
- **来源 commit**: `0d382ecde`（#40951）
- **v0.21.0 锚点**: `vllm/engine/arg_utils.py` `_compute_kwargs`（即 `get_kwargs` 内核）新增分支 `elif type_hints == {bool, str, type(None)}:` → 设 `type=str, nargs="?", const=True`（裸旗标→`True`，带值→`str`）。同时 `--hf-token` 的手写 `add_argument(... nargs="?", const=True ...)` 被删，改回 `model_group.add_argument("--hf-token", **model_kwargs["hf_token"])`。
- **目标章节**: ch03（3.2 节讲 `get_kwargs` / `BooleanOptionalAction` 自动派生 CLI 的地方）
- **要点**: 之前 `bool | str | None` 类型只能靠手工特判（源码里那条 `# This one is a special case ... TODO: Handle this in get_kwargs` 注释正是此事）；现在 `get_kwargs` 内建识别这种「裸开关或带字符串值」的可选参数，特例消失。
- **integration suggestion**:
  > 如果本章引用了 `--hf-token` 那段「这是个特例，TODO 以后并进 `get_kwargs`」的注释——v0.21.0 已经把这个 TODO 兑现了：`_compute_kwargs` 新增了对 `{bool, str, None}` 类型的识别（`nargs="?"`、`const=True`），`--hf-token` 回归普通的 `**kwargs` 展开。这是「字段类型驱动 CLI 形态」这条规则更完整的体现：bool→`--no-x/--x` 开关、bool|str|None→可裸可带值的可选旗标。
- **diagram impact**: 无需。

---

## 5. 非 MoE 模型禁用外部 DP 负载均衡（external LB 提前报错）

- **class**: BEHAVIOR-CHANGE（新增硬约束 + 帮助文本变化）
- **来源 commit**: `577b9623e`（#40839）
- **v0.21.0 锚点**: `vllm/engine/arg_utils.py`，在 `data_parallel_external_lb` 解析后新增：
  `if self.data_parallel_size > 1 and data_parallel_external_lb and not model_config.is_moe: raise ValueError(...)`（提示改为启动独立 vLLM 实例）。同时 `--data-parallel-rank` 帮助文本更新为「仅支持 MoE 数据并行部署；非 MoE 请改用独立实例」。
- **目标章节**: ch03（3.6/3.7 节附近讲 DP / external LB 推导，对应正文 L545 `data_parallel_external_lb` 那条线）
- **要点**: 以前非 MoE 模型也能进 external LB 路径（行为未定义/会出问题）；现在显式拒绝，引导用户起多个独立实例。
- **integration suggestion**:
  > v0.21.0 收紧了外部数据并行负载均衡的适用范围：当 `data_parallel_size > 1` 且走 external LB、而模型不是 MoE 时，`create_engine_config` 阶段就直接报错，并提示「非 MoE 请改起独立 vLLM 实例」。本章若展开 `data_parallel_external_lb` 那条推导线，可补一句——external LB 现在是 MoE 专属路径。
- **diagram impact**: 无需（除非该章 DP 配图细到画了 external LB 分支，可在其上标注「MoE only」）。

---

## 6. `EngineCore.add_request`：`abort_immediately` 立即中止以释放预准入 KV 资源

- **class**: BEHAVIOR-CHANGE（新分支）
- **来源 commit**: `3f5bd482f`（#41269）
- **v0.21.0 锚点**: `vllm/v1/engine/core.py` `EngineCore.add_request` 尾部新增：
  `if request.abort_immediately: self.abort_requests([request.request_id])`（注释：让 connector 的 `request_finished` 钩子运行以释放预准入的 KV-transfer 资源）。配套字段 `Request.abort_immediately`（`vllm/v1/request.py`、`vllm/v1/engine/__init__.py:129`），由 `vllm/v1/engine/async_llm.py:746` 在发预准入探测请求时置 `True`。
- **目标章节**: ch03（3.11 节 `EngineCore.__init__` / add_request 落地处）——属于次要细节，建议轻量处理；与 KV connector 主题深度相关，主战场在 disagg/KV-transfer 章节，本组仅记录。
- **要点**: 这是 NIXL 预准入拒绝路径的一环：请求加进 scheduler 后若标了 `abort_immediately`，立刻 abort，从而触发 connector 钩子归还被「搁浅」的 KV blocks。
- **integration suggestion**:
  > v0.21.0 给 `EngineCore.add_request` 末尾加了一个小分支：`request.abort_immediately` 为真时，请求一进 scheduler 就被立即 abort——目的是让 KV connector 的 `request_finished` 钩子跑起来，回收预准入阶段（pre-admission）申请、但因拒绝而搁浅的 KV blocks。这属于 disaggregated/KV-transfer 的边角控制流，本章点名即可，细节留给 KV 连接器专章。
- **diagram impact**: 无需。

---

## SKIP（纯重构 / 行为不变 / 仅注释）— 不更新章节

- **`enable_allreduce_rms_fusion` / `enable_norm_pad_fusion` 条件调整**（commit `5737770c6`、`d58c42e19`）：`vllm/config/vllm.py` 内部 fusion 开关判据改写（去掉 DP/PP 限制、改读 `kernel_config.ir_op_priority`）。属硬件后端融合策略内部细节，不在 ch01/ch03 教学范围。**SKIP**。
- **`OPTIMIZATION_LEVEL_01/02` 的 `enable_flashinfer_autotune: True → False`**（commit `c51df4300`）：因 flashinfer 正确性问题临时关闭，注释明说是临时项。3.9 节讲 O0–O3 时**不建议**写死这个布尔值（易过期）；如正文恰好逐字引用了该旗标的 `True`，可顺带改为 `False` 并一句带过「该项因上游正确性问题暂时关闭」。否则 **SKIP**。
- **TurboQuant 边界跳层逻辑搬迁**（commit `4f2af1a7c`）：`arg_utils.py` 里把 `get_boundary_skip_layers(num_layers)` 改为 `get_boundary_skip_layers(model_config)`、删掉 hybrid 模型的 `NotImplementedError` 和一条 info 日志。签名变化与 hybrid 支持都属 TurboQuant 量化内部，非本组主题。**SKIP（move/refactor）**。
- **`max_size is not None` → 增加 `and self.model_config is not None`**（commit `3e1ad4435`）、mamba `align` 模式对 V2 model runner 的 `assert`：均为防御性 None 检查 / 内部断言，无 reader-facing 行为。**SKIP**。

---

## ch01 评估

ch01 是 meta 概览章（`skip_impl` 轻流程），不讲具体 config 字段或控制流。上述 6 项均落在 ch03。**ch01 无需更新**。
