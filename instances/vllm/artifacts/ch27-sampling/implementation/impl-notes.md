# ch27 精简版实现说明（只做减法 / pin f3fef123）

精简版忠实镜像 vLLM v1 token 采样层的 9 步流水线。与真实 vLLM **同名、同结构、同控制流**，
只删不增。所有删除点都带 `# SUBTRACTED:` 注释并标注原 `vllm/...:Lxxx`。

## 验收判据

把真实 vLLM 删掉所有 `# SUBTRACTED:` 标注的分支，应 ≈ 得到本精简版。删除项严格限于
`dossier.subtraction_plan.delete` 批准范围（投机解码 / thinking budget / generative-scoring
旁路 / Triton Qrita 内核 / aiter & CPU 后端 / persistent-batch 状态机）。`must_keep` 的 27 个
符号全部原样保留。

## 可运行性

纯 PyTorch（CPU），**不 import vllm**。host 无 CUDA → `TopKTopPSampler.__init__` 落到 `else`
分支绑定 `forward_native`；`HAS_TRITON=False` 使 `apply_top_k_top_p` 走 pytorch sort 主路径
（与真实 batch<8 路径同构）；重复惩罚走真实 `apply_repetition_penalties` 派发器里的 torch 实现。
flashinfer / triton kernel 仅保留外部契约（签名 + 算法说明），不在 CPU 测试路径上触发。

## 1:1 Source Map

| 精简版符号 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `Sampler` / `forward` | `vllm/v1/sample/sampler.py:L21-143` | 删 logprob_token_ids 旁路、num_logprobs==-1 全量分支 | generative_scoring 专用，正交于 9 步主线 |
| `Sampler.sample` | `vllm/v1/sample/sampler.py:L232-288` | 原样保留 | step7 全貌（greedy 早退/温度/min_p/截断/torch.where） |
| `Sampler.apply_logits_processors` | `vllm/v1/sample/sampler.py:L357-406` | 删 predict_bonus_token / thinking holder 组合分支 | 投机解码 + thinking budget，非主路径不触发 |
| `Sampler.apply_temperature/greedy_sample/compute_logprobs/gather_logprobs` | `sampler.py:L216-342` | 原样保留 | step7a-b / step8 核心 |
| `Sampler.apply_penalties` | `sampler.py:L408-425` | 原样保留 | step6 入口 |
| `SamplingMetadata` | `vllm/v1/sample/metadata.py:L14-55` | 删 logprob_token_ids/spec_token_ids/thinking_budget_state_holder 三旁路字段 | 不在 9 步主流水线，subtraction_plan 批准 |
| `apply_all_penalties` / `apply_penalties` / `get_token_bin_counts_and_mask` | `ops/penalties.py:L11-57` + `model_executor/layers/utils.py:L34-89` | 内联 make_tensor_with_pad / is_pin_memory_available | 自包含，避免 import vllm |
| `apply_repetition_penalties(_torch)` | `vllm/_custom_ops.py:L471-519` | 删 CUDA 融合内核分支 | 与 torch 版数值等价，host 无 CUDA |
| `apply_bad_words` / `_apply_bad_words_single_batch` | `vllm/v1/sample/ops/bad_words.py:L9-35` | 删 apply_bad_words_with_drafts | spec-decode 变体 |
| `TopKTopPSampler` + `forward_native/forward_cuda` | `ops/topk_topp_sampler.py:L22-152` | 删 init 分发措辞、forward_cpu/forward_hip/aiter | 保留 native+cuda 两个后端代表 |
| `apply_top_k_top_p` / `apply_top_k_top_p_pytorch` / `random_sample` / `flashinfer_sample` | `ops/topk_topp_sampler.py:L257-403` | 删 apply_top_k_only 免排序快路、flashinfer 版本校验 | 主 sort 路径已覆盖 top-k 语义 |
| `apply_top_k_top_p_triton` | `ops/topk_topp_triton.py:L965-1051` (+ kernel L70-962) | 减成 wrapper 桩 | 920 行 Qrita 内核超出"不开源码也能懂主流程"目标，只讲外部契约 |
| `LogitsProcessors` | `logits_processor/state.py:L148-165` | 原样保留 | argmax-invariance 分类容器 |
| `MinP/MinTokens/LogitBiasLogitsProcessor` | `logits_processor/builtin.py:L22-238` | 删 __init__ 双张量预分配 + update_state；构造改为接收就绪张量状态 | persistent batch 维护属另一子系统；保留 is_argmax_invariant + apply |
| `LogitsProcessor` (ABC) | `logits_processor/interface.py:L60-92` | 删 validate_params / update_state 抽象方法 | persistent batch 契约 |
| `batched_count_greater_than` | `ops/logprobs.py:L10-29` | 删 @torch.compile 装饰器 | 仅选编译后端，数值不变 |
| `LogprobsTensors` / `SamplerOutput` | `vllm/v1/outputs.py:L51-124` | 删 tolists/filter/empty_cpu 等辅助 + cu_num_generated_tokens | 不在采样主路径 |

## 测试

`tests/test_sampling_pipeline.py`（23 个，CPU，host 直跑）覆盖：bad words 前缀匹配屏蔽、
freq/presence/repetition 惩罚 OpenAI 算式、processor argmax-invariance 分类与各 apply、
top-k/top-p/温度截断、Gumbel random_sample 确定性、Sampler.forward 整链（all_greedy 早退、
混合批 torch.where 合并、gather_logprobs 形状与 rank、raw logprobs 在惩罚前抽取、后端绑定与回退）。
