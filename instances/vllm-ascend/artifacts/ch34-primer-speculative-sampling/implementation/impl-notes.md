# ch33 参考实现笔记 — 投机采样 / MTP（primer 论文忠实实现，非 subtract-only）

本章豁免「只做减法」：`implementation/` 是**论文忠实的小型参考实现**（NumPy/纯 CPU torch），
不是 vllm-ascend 源码的精简版。每个 def/class 标 `# PAPER: §x Eq.y`，对标 `# SOURCE:` 的角色。
目的是让读者用调试器逐行跑通 Algorithm 1 与 Eq.21-23，并用蒙特卡洛数值验证论文的定理性断言
（保分布、接受率、期望长度、加速比），而不是重建昇腾生产代码的向量化/TP/量化细节。

## 1:1 Paper Map

| 参考实现 | 论文出处 | 对应真实落地代码（锚点，非本文件对标对象） |
|---|---|---|
| `speculative_sampling.py::residual_distribution` | paper.md §2.3（p'=norm(max(0,p-q)))、§A.1（归一化常数 1-β） | `vllm_ascend/sample/rejection_sampler.py:L1238-L1260` `sample_recovered_tokens_pytorch` 的 `maximum(target-draft,0)/q` |
| `speculative_sampling.py::propose_and_check` | paper.md §2.3（accept-reject，min(1,p/q)） | `vllm_ascend/sample/rejection_sampler.py:L1035-L1037` `target_token_probs/draft_token_probs >= uniform_token_probs` |
| `speculative_sampling.py::speculative_sampling_step` | paper.md §2.3 + §A.1（单 token 完整流程 + 保分布证明） | 同上两处的组合 |
| `speculative_sampling.py::speculative_decoding_step` | paper.md §2.3 Algorithm 1（γ 步版本） | `vllm_ascend/sample/rejection_sampler.py:L919-L1060` `rejection_random_sample_pytorch`（向量化 batch 版，本文件是逐请求版） |
| `speculative_sampling.py::acceptance_rate` / `lukaszyk_karmowski_divergence` | paper.md §3.2 Lemma 3.3 / Theorem 3.5 / Corollary 3.6 | 无直接落地代码（分析量，供数值推演） |
| `speculative_sampling.py::expected_generated_tokens` | paper.md §3.1 Eq.1 | 无直接落地代码 |
| `speculative_sampling.py::walltime_improvement_factor` / `optimal_gamma` | paper.md §3.3 Theorem 3.8 / Corollary 3.9、§3.5 | 无直接落地代码 |
| `mtp_module.py::RMSNorm` | paper-mtp.md §2.2 Eq.21（RMSNorm(·)） | `vllm_ascend/models/deepseek_v4_mtp.py` 里的 `RMSNorm`（vLLM `layernorm.RMSNorm`，同名不同实现） |
| `mtp_module.py::MTPModule` | paper-mtp.md §2.2 Eq.21-23（单深度 MTP 模块） | `vllm_ascend/models/deepseek_v4_mtp.py:L56-L128` `DeepSeekMultiTokenPredictorLayer`（含 enorm/hnorm/e_proj/h_proj/mtp_block/SharedHead） |
| `mtp_module.py::DeepSeekMTPPredictor` | paper-mtp.md §2.2（D 个串行模块、保持因果链） | `vllm_ascend/models/deepseek_v4_mtp.py:L140-L197` `DeepSeekMultiTokenPredictor`（真实代码按 `spec_step_idx % num_mtp_layers` **只跑当前一深度**，本参考实现为教学清晰选择**一次跑全部 D 个深度**并显式返回每深度中间量） |

**刻意省略**（论文 Eq.21-23 之外、纯工程/生产细节，不属于本章 primer 推导范围）：
TP/EP 切分、量化（quant_config）、`hc_head`（V4 head-combine 变体）、`spec_step_idx` 模取路由、
`load_weights` 的 checkpoint 名字重映射、真实 `TRM_k`（完整 DeepSeek-V2 decoder block，含 MoE/MLA
注意力）——参考实现的 `TRM_k` 用单层 `nn.TransformerEncoderLayer` 作为「a Transformer block」的
具体小样本代替，因为论文本身把 TRM_k 留作架构无关（不点名具体结构）。

`rejection_random_sample_pytorch`/`sample_recovered_tokens_pytorch` 里的 `ENTROPY_VERIFY`（熵阈值
放宽）、`IS_NGRAM`（ngram 无 draft_probs 分支）、`enable_reduce_sampling`（压缩词表索引）等是
MagicMTP/工程加速项，非论文标准判定，参考实现里没有对应物——这些差异在正文「落地」一节会点出，
不在本参考实现里出现（避免把工程变体误当作论文机制讲）。

## 测试映射（`tests/`）

| 测试类 | 验证的论文断言 |
|---|---|
| `TestAcceptReject` | accept 概率恰为 min(1,p/q)（§2.3） |
| `TestResidualDistribution` | p'=norm(max(0,p-q))，归一化常数 = 1-β（§2.3, §A.1） |
| `TestDistributionPreserving` | 蒙特卡洛验证输出边际分布 = p（含 q 某处为 0 的边界情形）（§A.1 完整证明） |
| `TestAcceptanceRateAlpha` | α=Σmin(p,q)=1-D_LK，且与蒙特卡洛实测接受频率吻合（§3.2） |
| `TestExpectedGeneratedTokens` | E[#tokens] 与截断几何级数直接求和吻合（§3.1 Eq.1） |
| `TestWalltimeSpeedup` | 复现论文 Table 1 数值（α=0.8,γ=5→3.69X；α=0.9,γ=10→6.86X）、Corollary 3.9 下界、`optimal_gamma` 与暴力搜索一致（§3.3） |
| `TestSpeculativeDecodingStep` | Algorithm 1 全 γ 步版本：q=p 时 n 恒为 γ；返回 token 数 ∈[1,γ+1]；首 token 边际分布 = p_1（位置级保分布）（§2.3 Algorithm 1, §A.1） |
| `TestRMSNorm` | RMSNorm 输出（weight=1 时）单位 RMS |
| `TestMTPModuleShapes` | 单深度模块前向形状正确 |
| `TestSharedParameters` | 各深度共享同一 Emb/OutHead 对象（非各自拷贝）（paper-mtp.md §2.2） |
| `TestCausalChainAndSequentialDepths` | 有效窗口逐深度收缩 1；`t_{i+k}` 位移正确（扰动 token 影响对应深度输出）；扰动主模型隐状态传播到深度 2（证明串行因果链而非独立并行头）（paper-mtp.md §2.2） |

运行：`cd implementation && PYTHONPATH=. python3 -m pytest ../tests/ -q`（或从章目录
`PYTHONPATH=implementation python3 -m pytest tests/ -q`）——host 纯 CPU numpy/torch 即可跑，
无需容器；32 passed。

## 收工前自检
`python3 scripts/lint_paper_grounding.py <chapter_dir> --expect-primer` → 0 BLOCKING
（仅剩两条非阻断 WARNING：`paper_ref` 因 dossier 引用的是 paper-mtp.md 而 linter 固定只核对
paper.md，属 linter 已知限制而非本实现的问题；`narrative/chapter.md` 尚不存在，写作前属正常）。
