# ch25《读一个完整大模型：DeepSeek-V4 是 Llama 身上叠的一摞 delta》交付 APPROVED — Part VI 模型层 capstone

- **Type**: delivery
- **Chapter**: ch25
- **Date**: 2026-06-24
- **Timestamp**: 2026-06-24T00:53:09Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch25, deepseek-v4, MLA, MoE, MTP, delta, capstone, f16-payoff, Part-VI

## What happened

capstone 模型解读: 把 DeepSeek-V4 当作对 Llama 基线(ch22)的一叠 delta 端到端拼起来。(1) 骨架 DeepseekV4ForCausalLM/Model/DecoderLayer 与 Llama 同构, delta=MLA+MoE+MTP+混合残差; (2) MLA 注意力(DeepseekV4Attention: q_lora/kv_lora 低秩 KV 压缩 + 解耦 RoPE)对标准 MHA 的 delta; (3) MoE(DeepseekV4MoE/MegaMoEExperts: 路由 top-k 专家 + 共享专家 + FusedMoE)对 dense MLP 的 delta; (4) MTP(DeepSeekV4MTP 多 token 预测头)与混合残差 hc_head(多流正系数线性组合)。测试 19/19 host 通过(不 import vllm), 4/4 linter 全过。reviewer verdict=APPROVED, 7 条 issue 全 non-blocking+negotiable(2 条同源: lint_source_grounding 软检查 vllm_files_listed 针对 impl-notes.md 非正文, 非阻断; §25.4.2 hc_head '凸性加权和'宜改为'正系数线性组合(不归一)'数学精度润色; 公式连排/交通隐喻一致性/补一处数值-形状追踪等增强项)。

## Why it matters

Part VI 模型层 capstone, 回收 f16(Llama 刻意缺的 MoE/MLA/量化/混合残差作为对 Llama 的 delta) — 验证'基线 + delta'对照笔法能把一个完整大模型讲透。本章为全书最复杂模型章节, 是模型层叙事的收束。

## What to remember

capstone 模型解读: 把 DeepSeek-V4 当作对 Llama 基线(ch22)的一叠 delta 端到端拼起来。(1) 骨架 DeepseekV4ForCausalLM/Model/DecoderLayer 与 Llama 同构, delta=MLA+MoE+MTP+混合残差; (2) MLA 注意力(DeepseekV4Attention: q_lora/kv_lora 低秩 K...
