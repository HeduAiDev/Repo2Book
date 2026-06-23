# ch28《投机解码》交付 APPROVED

- **Type**: delivery
- **Chapter**: ch28
- **Date**: 2026-06-23
- **Timestamp**: 2026-06-23T20:39:19Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: spec-decode, rejection-sampling, proposer, SpecDecodeMetadata, bonus-token, Part-VII

## What happened

ch28《投机解码》全流水线交付。内容：多种 proposer(NgramProposer CPU prompt-lookup/NgramGPUKernel、SpecDecodeBaseProposer 统一 EAGLE/EAGLE3/DFlash/MTP draft via DeepSeekV4MTP)产 k 个草稿；SpecDecodeMetadata 把变长草稿摊平进批(cu_num_draft_tokens + target/bonus/logits_indices 三组 index 间接)，calc_spec_decode_metadata 在 model_runner 构建；RejectionSampler.forward 切 bonus/target、采 bonus、调 rejection_sample；greedy(写 target_argmax)/random(残差分布采 recovered token, u>=p/q 判据)两路内核；accepted/recovered/bonus/output 四类 token 术语；正确性论证输出分布与无投机严格等价(20000 次接受率实测)。reviewer APPROVED：9 条 issue 全 non-blocking+negotiable(术语漂移 recovered vs target_argmax 注释 x2、docstring 中段无标记删减、几处直觉/数值例子可补、速度量化公式 E[L]≈(1-α^{k+1})/(1-α) 可补、序列级归纳证明可补、formula linter 6 处内联密度警告非违规)。承接 ch27 采样 + ch25 MTP。登记 6 个精简版接口。foreshadow due ch28 为空(无应埋/应回收)。

## Why it matters

Part VII 采样收官关键章——投机解码是 vLLM 吞吐核心特性，本章把 ch25(MTP draft) 与 ch27(采样) 串成完整 accept/reject 闭环，并给出分布等价性这一非平凡正确性论证。

## What to remember

ch28 已 APPROVED 归档。reviewer 9 条均 non-blocking+negotiable，未阻断。术语铁律：recovered 专指 random 路径残差采样，greedy 拒绝位写 target_argmax(注释 L476 有轻微术语串台但值正确)。已登记 6 接口(SpecDecodeMetadata/calc_spec_decode_metadata/NgramProposer.propose/SpecDecodeBaseProposer.propose/RejectionSampler.forward/rejection_sample)。无伏笔应埋应回收。
