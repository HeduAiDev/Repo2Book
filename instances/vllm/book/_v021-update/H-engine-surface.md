# v0.21.0 更新摘要 — H 组：引擎表面 + 去 token 化 + 注意力后端 + XPU 采样

基线 `f3fef1235` → tag `v0.21.0`。行号锚点均以 `v0.21.0` 为准。仅列对读者可教学的变化。

---

## ch02 / ch04 / ch21 — `vllm/v1/engine/async_llm.py`

### [NEW-FEATURE] AsyncLLM 显式权重更新两段式 API：`start_weight_update` / `finish_weight_update`
- **anchor**：`vllm/v1/engine/async_llm.py` — `AsyncLLM.start_weight_update`（L1079）、`AsyncLLM.finish_weight_update`（L1105）
- **target**：ch04（AsyncLLM 公共方法面）、ch21（RL 训练侧权重热更新链路）
- **integration（书声线）**：在原有 `update_weights` 批量更新之外，`AsyncLLM` 现以 `start_weight_update`/`finish_weight_update` 把一次权重热更新显式括成事务：前者经 `collective_rpc("start_weight_update", kwargs={"is_checkpoint_format": ...})` 在各 worker 开启更新窗口（`is_checkpoint_format=True` 表示来料为 checkpoint 格式、需逐层处理，否则为可直拷的 kernel 格式），后者以 `collective_rpc("finish_weight_update")` 收尾。三者构成 RL 训练中"开窗—批量灌权重—闭窗"的标准时序。
- **diagram impact**：无强制；若 ch21 已有权重更新时序图，可补 start/finish 括号两端。

### [BEHAVIOR-CHANGE] KV 传输预准入拒绝的资源回收钩子：`notify_kv_transfer_request_rejected`
- **anchor**：`vllm/v1/engine/async_llm.py` — `AsyncLLM.notify_kv_transfer_request_rejected`（L723）
- **target**：ch21（disaggregated/NIXL KV 传输；若本书 KV connector 落在别处则随之）
- **integration（书声线）**：当 P 节点在预准入阶段拒绝一个请求时，已为其 prefill 预钉的 KV 块会成为孤儿。该方法构造一个 `abort_immediately=True` 的 `EngineCoreRequest`（携带 `kv_transfer_params`），提交后立即触发 connector 的 `request_finished` 钩子，从而释放 NIXL 等预准入资源——属边角运维路径。
- **diagram impact**：无。
- **ch02 备注**：ch02 为 meta 概览，此处两项一般无需落到 ch02 正文；若 ch02 列了 AsyncLLM 方法清单可顺带一提，否则 SKIP。

---

## ch02 / ch31 — `vllm/entrypoints/llm.py`

### [NEW-FEATURE] LLM 同步侧权重更新两段式 API：`start_weight_update` / `finish_weight_update`
- **anchor**：`vllm/entrypoints/llm.py` — `LLM.start_weight_update`（L1911）、`LLM.finish_weight_update`（L1940）
- **target**：ch31（`LLM` 离线入口的公共方法面）
- **integration（书声线）**：与 `AsyncLLM` 对称，离线 `LLM` 也新增 `start_weight_update(is_checkpoint_format=True)` 与 `finish_weight_update()`，分别经 `self.llm_engine.collective_rpc` 广播 `start_weight_update`/`finish_weight_update`，把一轮权重热更新显式括成事务，服务于训练—推理同进程的权重回灌。
- **diagram impact**：无（属方法清单增补）。
- **ch02 备注**：meta 章，通常 SKIP。

---

## ch08 / ch09 / ch10 — `vllm/v1/engine/output_processor.py`

### [NEW-FEATURE] 路由专家（routed_experts）按 prompt/generation 拆分到不同输出对象
- **anchor**：`vllm/v1/engine/output_processor.py` — `RequestState`（路由拆分逻辑 L323–L334；`split_routed_experts` 导入 L15；`_new_request_output(..., prompt_routed_experts)` 形参 L362、回填 `RequestOutput(prompt_routed_experts=...)` L398）
- **target**：ch08（输出装配 `RequestOutput`/`CompletionOutput` 结构）；ch10 若讲 logprobs/附加输出字段亦相关
- **integration（书声线）**：当请求开启 MoE 路由捕获时，`RequestState` 现调用 `split_routed_experts(routed_experts, prompt_len, num_gen)` 将路由数据切成两段——prompt 段挂到请求级 `RequestOutput.prompt_routed_experts`（n>1 时被多个 completion 共享），generation 段挂到每个 `CompletionOutput`。`prompt_len` 取自 `self.prompt_token_ids`，`num_gen` 取自 `self.detokenizer.num_output_tokens()`。这把"共享的提示路由"与"各自的生成路由"在数据结构上分离。
- **diagram impact**：若 ch08 有 `RequestOutput`/`CompletionOutput` 字段图，可加 `prompt_routed_experts`（请求级）与 completion 级路由两端；否则无。
- **ch09 备注**：本变更不触及去 token 化逻辑本身（仅借 `detokenizer.num_output_tokens()` 取计数），ch09 多半 SKIP。

---

## ch09 — `vllm/v1/engine/detokenizer.py`

### [BEHAVIOR-CHANGE] FastIncrementalDetokenizer 改为按模块属性查 `DecodeStream`（fastokens 可热替换）
- **anchor**：`vllm/v1/engine/detokenizer.py` — `FastIncrementalDetokenizer`：`self.stream = tokenizers.decoders.DecodeStream(...)`（构造 L183、重置 L243）；导入由 `from tokenizers.decoders import DecodeStream` 改为 `import tokenizers.decoders`
- **target**：ch09（增量去 token 化、`DecodeStream` 流式解码）
- **integration（书声线）**：原先 `DecodeStream` 在导入期就被绑定为局部名，现改为在使用点经 `tokenizers.decoders.DecodeStream` 按模块属性解析。这样像 fastokens 这类后端在运行期替换 `tokenizers.decoders.DecodeStream` 的 shim 才会被尊重，而不受 import 顺序影响——`FastIncrementalDetokenizer` 的流式语义不变，仅令底层解码流实现可被热替换。
- **diagram impact**：无（实现细节，不改数据流图）。

---

## ch24 — `vllm/v1/attention/selector.py` + `backends/registry.py` + `backends/flash_attn.py`

### [BEHAVIOR-CHANGE] Mamba 后端选择改为枚举驱动：移除 `MAMBA_TYPE_TO_BACKEND_MAP` 字符串映射
- **anchor**：`vllm/v1/attention/backends/registry.py`（删除 `MAMBA_TYPE_TO_BACKEND_MAP`）；`vllm/v1/attention/selector.py` — `get_mamba_attn_backend(mamba_type: MambaAttentionBackendEnum)` 与 `_cached_get_mamba_attn_backend`，内部由查表改为 `mamba_type.get_class()`
- **target**：ch24（注意力后端选择/registry）
- **integration（书声线）**：Mamba 后端选择不再经字符串字面量（`"mamba1"`/`"mamba2"`/...）查 `MAMBA_TYPE_TO_BACKEND_MAP`，而是直接以 `MambaAttentionBackendEnum` 枚举传入并调用 `mamba_type.get_class()` 惰性导入后端。`_cached_get_mamba_attn_backend` 的入参断言也从 `isinstance(str)` 收紧为 `isinstance(MambaAttentionBackendEnum)`——把"合法类型"从运行期字符串校验前移为类型级约束。
- **diagram impact**：若 ch24 后端选择图标了 `mamba_type: str → MAMBA_TYPE_TO_BACKEND_MAP → Enum` 这一跳，应简化为枚举直达 `get_class()`。

### [API-CHANGE] AttentionBackendEnum 增 `TOKENSPEED_MLA`、删 `TREE_ATTN`
- **anchor**：`vllm/v1/attention/backends/registry.py` — `AttentionBackendEnum.TOKENSPEED_MLA`（L66，指向 `mla.tokenspeed_mla.TokenspeedMLABackend`）；删除 `TREE_ATTN`（原 `tree_attn.TreeAttentionBackend`）
- **target**：ch24（后端枚举注册表）
- **integration（书声线）**：MLA 后端家族新增 `TOKENSPEED_MLA`（面向 DSR1/Kimi 在 Blackwell 上的 prefill+decode 合一），同时 tree attention 后端 `TREE_ATTN` 被整体移除。若 ch24 列举了 `AttentionBackendEnum` 成员，需补 `TOKENSPEED_MLA`、删 `TREE_ATTN`。
- **diagram impact**：ch24 后端枚举/家族图按上述增删一条目。

### [BEHAVIOR-CHANGE] FlashAttention：FA3→FA4 升级迁出 + KV 缓存退化 stride 规整
- **anchor**：`vllm/v1/attention/backends/flash_attn.py` —（1）`head_size>256` 在 SM90+ 强制 FA3→FA4 的就地逻辑被删除，统一迁入 `get_flash_attn_version()`（调用点 L637 等）；（2）`FlashAttentionImpl.forward` 中对 `key_cache`/`value_cache` 调 `canonicalize_singleton_dim_strides(...)`（L748–L749）
- **target**：ch24（FlashAttention 后端实现/版本选择）
- **integration（书声线）**：两处实现层修正。其一，`head_size>256` 时在 SM90+ 由 FA3 升 FA4 的判断不再散落在 `FlashAttentionImpl` 构造里，而统一收口到 `get_flash_attn_version()`——版本决策单一来源。其二，`forward` 在从 `kv_cache.unbind(0)` 取出 K/V 缓存后，对其调用 `canonicalize_singleton_dim_strides`：当 `num_kv_heads=1` 配合 TP 时 size-1 维会产生退化 stride，而 H100+ 上 FA3/FA4 走 TMA 需 ≥16 字节 stride 对齐，否则触发 `cudaErrorIllegalInstruction`；写缓存路径（scatter）不走 TMA 故无需规整。
- **diagram impact**：若 ch24 有 FA 版本选择图，把 FA3→FA4 升级节点归到 `get_flash_attn_version()` 内部；KV stride 规整属实现注脚，通常无需上图。

---

## ch27 — `vllm/v1/sample/ops/topk_topp_sampler.py`

### [NEW-FEATURE] TopKTopPSampler 新增 XPU 采样 kernel 后端分支 `forward_xpu`
- **anchor**：`vllm/v1/sample/ops/topk_topp_sampler.py` — `TopKTopPSampler.__init__` 派发新增 `elif current_platform.is_xpu():`（L85），受 `envs.VLLM_XPU_USE_SAMPLER_KERNEL` 开关，置 `self.forward = self.forward_xpu`，否则回退 `forward_native`；实现 `forward_xpu`（L251）
- **target**：ch27（采样后端派发 / top-k·top-p 采样）
- **integration（书声线）**：`TopKTopPSampler.__init__` 的平台派发链在 CUDA/CPU 之外新增 XPU 分支：当 `VLLM_XPU_USE_SAMPLER_KERNEL` 开启时绑定 `forward_xpu`，否则回退 `forward_native`。`forward_xpu` 经 `torch.ops.vllm.xpu_topk_topp_sampler` 调用 XPU 原生 top-k/top-p 采样 kernel，并从 `torch.xpu.default_generators` 取出 `(seed, offset)` 传入以复现随机性；它不支持 per-request generators（有则告警回退 native），且因 batch 侧 `top_k` 存为 int32 而 kernel 要 int64，会在调用前 `k.to(torch.int64)`。
- **diagram impact**：**ch27 图 `02-backend-dispatch` 需新增 XPU 分支**——在 CUDA(FlashInfer)/CPU/ROCm(aiter)/native 之外加一条 `is_xpu() + VLLM_XPU_USE_SAMPLER_KERNEL → forward_xpu`（关闭则 → forward_native）。

---

## 全 SKIP 文件
无。本组 8 文件均含可教学变化。
