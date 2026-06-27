# v0.21.0 更新摘要 — DeepSeek-V4 / MoE / MLA（ch25, ch26）

baseline: `f3fef1235` → tag: `v0.21.0`
文件组：`deepseek_v4.py`(+172/-53)、`fused_moe/layer.py`、`deepseek_v4_mtp.py`、`mla.py`

> 范围说明：`vllm/model_executor/layers/mla.py` 在本区间 **无任何 commit / 无 diff**，章节无需据它更新。

---

## 1. DeepSeek-V4 接入流水线并行（PP）

- **class**: NEW-FEATURE
- **anchor**: `vllm/model_executor/models/deepseek_v4.py` — `class DeepseekV4ForCausalLM(nn.Module, SupportsPP)`；`DeepseekV4Model.make_empty_intermediate_tensors`；`embed_tokens / norm / lm_head / _mtp_hidden_buffer` 改为 `get_pp_group().is_first_rank / is_last_rank` 分支挂 `PPMissingLayer()`；权重装载新增 `is_pp_missing_parameter(...)` 跳过。
- **target**: ch25（主），ch26（次，模块树/数据流读法）
- **集成（书声）**：基线版 V4 把 `embed_tokens`、`norm`、`lm_head` 无条件实例化、整模型驻留单卡；v0.21.0 让 `DeepseekV4ForCausalLM` 实现 `SupportsPP`，按 `get_pp_group().is_first_rank / is_last_rank` 把首尾层替换成 `PPMissingLayer()`，并由 `make_empty_intermediate_tensors` 在 PP 边界传递形状 `(num_tokens, hc_mult, hidden_size)` 的多流隐状态——也就是说，hc 多流残差不仅活在层内，还要跨 rank 沿 PP 管道传递。`forward` 因此分叉：首 rank 走 `embed_input_ids` 再 `unsqueeze(-2).repeat(1, hc_mult, 1)` 展开成 `hc_mult` 条流，非首 rank 直接取 `intermediate_tensors["hidden_states"]`；非末 rank 提前 `return IntermediateTensors(...)`，把 `hc_head`、`_mtp_hidden_buffer` 暂存与 `lm_head` 全部下放到末 rank。
- **diagram 影响**：ch25 的「V4 = Llama + delta」叠层图与 ch26 的模块树/数据流图若标注了 `embed_tokens→layers→norm→lm_head` 的单卡贯通，可补一句 PP 切分点；最小改动是数据流图在首/末 rank 处加一个 `IntermediateTensors` 传递的虚线边（多流 `hc_mult` 形状）。非阻塞，可作脚注。

---

## 2. hc 多流残差换成融合算子（mhc_fused_post_pre + hc_head_fused_kernel）

- **class**: BEHAVIOR-CHANGE
- **anchor**: `vllm/model_executor/models/deepseek_v4.py` — `DeepseekV4DecoderLayer.forward`（签名新增 `post_mix / res_mix / residual`，返回 `tuple[Tensor, Tensor, Tensor, Tensor]`）调 `torch.ops.vllm.mhc_fused_post_pre(...)`；`hc_head(...)` 内部由 `F.linear/sigmoid` 的 eager 实现换成 `torch.ops.vllm.hc_head_fused_kernel(...)`。
- **target**: ch25
- **集成（书声）**：基线把每个 `DeepseekV4DecoderLayer` 写成「`hc_pre` → attn → `hc_post` → `hc_pre` → ffn → `hc_post`」四段独立调用，且 `hc_head` 用 `F.linear` + `sigmoid` 的纯 PyTorch 算子拼出来。v0.21.0 做了算子融合：层内把上一层的 `hc_post` 与本层的 `hc_pre` 合并成一个自定义算子 `torch.ops.vllm.mhc_fused_post_pre`，于是 `forward` 改为在层间流水 `(residual, post_mix, res_mix)` 三元组——只有第一层单独跑 `hc_pre`，最后一层的 `hc_post` 被提到 `DeepseekV4Model.forward` 末尾的 `for...else` 里统一收口；`hc_head` 则整体落进 `torch.ops.vllm.hc_head_fused_kernel`。语义不变（仍是同一套 RMSNorm + sigmoid 门控的多流合并），但读者看到的不再是逐行可读的张量算子，而是一次 kernel 调用——讲解时需明确「融合算子 = 原四步的等价合并」，并保留对融合前数学语义的推导。
- **diagram 影响**：ch25 若画了 hc 多流残差的 `pre/post` 时序小图，需把跨层的 `post_mix/res_mix` 流水标出来（每层吐三元组、首层例外、末层 `hc_post` 收口）。建议新增/更新一张「逐层 hc 三元组流水」示意。

---

## 3. MegaMoE 开关解绑 EP 并强校验

- **class**: BEHAVIOR-CHANGE
- **anchor**: `vllm/model_executor/models/deepseek_v4.py` — `DeepseekV4MoE.__init__` 与 `DeepseekV4Model.__init__` 中 `self.use_mega_moe = (kernel_config.moe_backend == "deep_gemm_mega_moe")`，随后 `if self.use_mega_moe and not enable_expert_parallel: raise NotImplementedError(...)`。
- **target**: ch25
- **集成（书声）**：基线把 `use_mega_moe` 与专家并行（EP）耦合在一起——只有开了 `enable_expert_parallel` 才去看 `moe_backend == "deep_gemm_mega_moe"`，否则静默置 `False`。v0.21.0 把这两件事拆开：`use_mega_moe` 只由后端名决定，但若选了 MegaMoE 却没开 EP，则直接 `raise NotImplementedError`，提示 `--enable-expert-parallel`。读者侧的含义是：MegaMoE 后端目前**强制要求纯 TP 之外配合 EP**，配置组合不再被悄悄降级，而是显式报错。
- **diagram 影响**：无（配置/控制流变化）。若 ch25 的 MoE 双后端选择图标了「EP→MegaMoE」的隐含依赖，可改成「MegaMoE 需 EP，否则 raise」。

---

## 4. FusedMoE 新增 shared_expert_gate 与 moe_layer_id（路由重放改设备缓存）

- **class**: API-CHANGE
- **anchor**: `vllm/model_executor/layers/fused_moe/layer.py` — `FusedMoE.__init__` 新增形参 `shared_expert_gate: torch.nn.Module | None`，赋 `self.shared_expert_gate` 并透传到内部 experts；新增类级 `_next_moe_layer_id` + 实例 `self.moe_layer_id`（路由重放缓冲区绑定）；`is_act_and_mul=False` 放开到 XPU。
- **target**: ch25（次要，MoE 层接口）；多数为后端/平台细节，可不入正文
- **集成（书声）**：`FusedMoE` 的构造接口扩了一个 `shared_expert_gate`，用于在融合共享专家（FSE）路径上接共享专家的门控；同时每个 MoE 层在构造时领一个自增的 `moe_layer_id`，配合「路由重放改成设备端缓存 + 异步 D2H 流水」的新机制做缓冲区绑定。这些主要服务 ROCm/AITER 与 Qwen3-Next 等路径，对 DeepSeek-V4 主线讲解非必需；若 ch25 列了 `FusedMoE` 的构造签名，补一行 `shared_expert_gate` 即可，其余按平台细节略过。
- **diagram 影响**：无。

---

## 5. DeepSeek-V4 ROCm/AMD 支持（aux_stream 在 ROCm 上禁用）

- **class**: BEHAVIOR-CHANGE
- **anchor**: `vllm/model_executor/models/deepseek_v4.py` 与 `vllm/model_executor/models/deepseek_v4_mtp.py` — `aux_stream_list = None if current_platform.is_rocm() else [torch.cuda.Stream() for _ in range(3)]`。
- **target**: ch25
- **集成（书声）**：基线无条件分配三条辅助 CUDA 流，让 MLA 压缩器的 `kv_score`、`indexer.weights_proj` 等子算子与 `fused_wqa_wkv` 并行执行。v0.21.0 在 ROCm 上把 `aux_stream_list` 置 `None`（因 hang 问题串行回退），其余平台不变。讲 MLA 多流并行执行时可加一句平台前提：三辅助流是 CUDA 路径的优化，ROCm 上当前按串行跑。
- **diagram 影响**：无（若 ch25 画了 MLA 三辅助流并行图，加「CUDA only」标注即可）。

---

## 6. MTP draft 适配新的 hc 三元组接口

- **class**: BEHAVIOR-CHANGE（伴随 #1 的连带改动 + HC 状态修复）
- **anchor**: `vllm/model_executor/models/deepseek_v4_mtp.py` — `DeepSeekV4MultiTokenPredictorLayer.forward` 改为 `hidden_states, residual, post_mix, res_mix = self.mtp_block(...)`，再显式 `self.mtp_block.hc_post(hidden_states, residual, post_mix, res_mix)` 收口。
- **target**: ch25（MTP 小节）
- **集成（书声）**：因为 `DeepseekV4DecoderLayer.forward` 现在返回 `(hidden_states, residual, post_mix, res_mix)` 四元组、不再层内自闭 `hc_post`，MTP draft 的 `mtp_block` 调用也随之改成接住四元组、再手动调一次 `hc_post` 收尾。语义与基线一致（MTP 仍复用主干 decoder 层、`hc_head` 延后到 `compute_logits`），只是为对齐 #2 的融合算子接口做了连带改写。讲 MTP 复用主干层时点明这一接口对齐即可。
- **diagram 影响**：无。

---

## SKIP（不入正文）

- `fused_moe/layer.py` 中 `init_aiter_topK_meta_data` 的 import 来源从 `rocm_aiter_fused_moe` 移到 `experts/rocm_aiter_moe`（模块搬迁，#41979）— 纯 move。
- `humming` mxfp4 后端的 `weight_schema.quant_method` 解包分支被删（#41083 回退/重构）— 量化后端内部细节，非 DeepSeek 主线。
- `is_act_and_mul=False` 报错文案 CUDA/ROCm→CUDA/XPU 措辞调整 — 仅平台支持矩阵，已并入条目 4。
