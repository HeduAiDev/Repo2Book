# ch28 投机解码 — 精简版实现说明（只做减法）

源码 pin：`f3fef123504db07b3ac83ad4ef677915b53e8386`。
精简版与真实 vLLM **同名、同结构、同控制流，只删不增**。每个 def/class 标
`# SOURCE: vllm/...:Lxxx`，每处删除标 `# SUBTRACTED:`。

## 运行环境

- `metadata.py`（含 `calc_spec_decode_metadata`）、`ngram_proposer.py`：纯 numpy/Python，
  host CPU 即可运行（numba 仅作性能加速，已按 subtraction_plan 删除 `@njit/@jit` 装饰器，
  函数体逐字保留）。
- `rejection_sampler.py`：三个 triton kernel 需 **CUDA + triton**。本工作机自带可用
  GPU（NVIDIA RTX PRO 6000 Blackwell），测试已实跑通过；无 GPU 时相关测试自动 skip。
- `llm_base_proposer.py`：`SpecDecodeBaseProposer` 依赖完整 vLLM 模型/注意力栈，**无法
  脱离 vLLM 单独运行**，保留为对外**契约骨架**（EAGLE 主路径：单步早退 + 自回归多步
  链式）。其内部模型（EAGLE 头 / MTP）见 ch25。

## 1:1 Source Map

| 精简版 | 真实 vLLM | 改动 | 原因 |
|---|---|---|---|
| `metadata.py::SpecDecodeMetadata` | `vllm/v1/spec_decode/metadata.py:L9-L66` | 逐字保留（三组 index + 两累积和 + `__post_init__` + `make_dummy`） | must_keep 核心数据结构，零删减 |
| `metadata.py::calc_spec_decode_metadata` / `_get_cumsum_and_arange` | `vllm/v1/worker/gpu_model_runner.py:L2596-L2674` / `L1572-L1596` | GpuModelRunner 方法提为自由函数；`input_ids` 显式传入；省略 CPU→GPU `non_blocking` 拷贝；scratch buffer 改临时数组 | index 间接的真相源；让本章可脱离 model_runner 单测，索引算术逐行一致 |
| `ngram_proposer.py::NgramProposer` / `_find_longest_matched_ngram_and_propose_tokens` | `vllm/v1/spec_decode/ngram_proposer.py:L12-L166` / `L198-L285` | 删 numba 线程自适应、JIT 预热、`@njit/@jit`、`VllmConfig` 依赖；`__init__` 收标量 | 纯性能调优；算法体（KMP/LPS）逐字保留，草稿结果等价 |
| `rejection_sampler.py::rejection_sample` | `vllm/v1/sample/rejection_sampler.py:L392-L503` | 删 `synthetic_mode`/`synthetic_conditional_rates` 形参与透传 | subtraction_plan.delete 批准；准则退化为标准 greedy/random |
| `rejection_sampler.py::rejection_greedy_sample_kernel` | `vllm/v1/sample/rejection_sampler.py:L706-L757` | 删 `SYNTHETIC_MODE` 分支与对应形参 | synthetic 删除项；标准 greedy 准则（draft==argmax）逐字保留 |
| `rejection_sampler.py::rejection_random_sample_kernel` | `vllm/v1/sample/rejection_sampler.py:L760-L826` | 删 `SYNTHETIC_MODE` 分支与形参 | synthetic 删除项；`target_prob/draft_prob >= uniform` 接受准则（算法心脏）逐字保留，含 `NO_DRAFT_PROBS` 分支 |
| `rejection_sampler.py::sample_recovered_tokens(_kernel)` | `vllm/v1/sample/rejection_sampler.py:L659-L703` / `L853-L921` | 无功能删减 | 残差分布 `(p_target-p_draft)_+` + Gumbel-max（`score=prob*inv_q`）逐字保留，含 ngram 屏蔽分支 |
| `rejection_sampler.py::RejectionSampler.forward` | `vllm/v1/sample/rejection_sampler.py:L87-L195` | 删 logprobs 计算、penalties/bad_words/thinking、logprobs_mode 旁路；bonus 采样简化为 `self.sampler(...)` | logprobs 归 ch10/ch27；主线是 token 接受/拒绝 |
| `rejection_sampler.py::RejectionSampler.parse_output` | `vllm/v1/sample/rejection_sampler.py:L246-L281` | 删 logprobs filter / `discard_req_indices` 保留 | 摊平→变长还原闭环；`PLACEHOLDER_TOKEN_ID(-1)` + 越界过滤逐字保留 |
| `topk_topp.py::apply_top_k_top_p(_pytorch)` | `vllm/v1/sample/ops/topk_topp_sampler.py:L257-L311` | 删 triton 二级分流与 `apply_top_k_only` 免排序快路 | 纯性能；PyTorch sort 路径对相同输入产生相同 mask |
| `llm_base_proposer.py::SpecDecodeBaseProposer.propose` | `vllm/v1/spec_decode/llm_base_proposer.py:L413-L655` | 删 tree attention / M-RoPE·xdrope / 多模态 / cudagraph padding / DFlash extra-slots | subtraction_plan.delete 批准；保留 EAGLE 主路径（单步早退 + 自回归多步） |

## must_keep 核对

dossier `subtraction_plan.must_keep` 的 23 个符号全部保留（`lint_fidelity` 校验通过）：
`SpecDecodeMetadata` / `cu_num_draft_tokens` / `cu_num_sampled_tokens` /
`target_logits_indices` / `bonus_logits_indices` / `logits_indices` / `max_spec_len` /
`rejection_sample` / `rejection_greedy_sample_kernel` / `rejection_random_sample_kernel` /
`sample_recovered_tokens(_kernel)` / `bonus_token_ids` / `recovered_token_ids` /
`PLACEHOLDER_TOKEN_ID` / `parse_output` / `NO_DRAFT_PROBS` / `RejectionSampler` /
`NgramProposer` / `_find_longest_matched_ngram_and_propose_tokens` /
`SpecDecodeBaseProposer` / `propose` / `generate_uniform_probs`。

## 测试锚定的真实行为

`tests/test_spec_decode.py`（10 项，全通过）：
1. `calc_spec_decode_metadata` 的 `cu_num_draft/sampled_tokens`、`logits_indices`、
   `target_logits_indices`、`bonus_logits_indices`、`draft_token_ids` 与
   `gpu_model_runner.py:L2601-L2664` 注释里的具体数字逐位一致（含草稿数为 0 的请求只占 bonus 位）。
2. ngram KMP/LPS 最长匹配后缀 + 复制后续 k token；无匹配返回空。
3. `generate_uniform_probs` 用 float64、范围 [0,1)。
4. greedy 路径：全接受补 bonus；首拒截断后续为 `-1`（recovered=贪心目标 token），不补 bonus。
5. `parse_output` 过滤 `PLACEHOLDER_TOKEN_ID` 还原变长 list。
6. random 路径（`NO_DRAFT_PROBS`）：草稿 token 的经验接受率 ≈ `p_target(x)`（2 万样本，误差 <3%），
   验证 `min(1, p_target/p_draft)` 准则与分布等价性。
