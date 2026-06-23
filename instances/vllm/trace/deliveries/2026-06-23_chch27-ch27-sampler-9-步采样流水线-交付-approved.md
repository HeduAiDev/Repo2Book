# ch27 Sampler 9 步采样流水线 交付 APPROVED

- **Type**: delivery
- **Chapter**: ch27
- **Date**: 2026-06-23
- **Timestamp**: 2026-06-23T19:58:14Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: sampling, sampler, logits-processor, penalties, top-k-top-p, greedy, SamplingMetadata, argmax-invariant

## What happened

ch27《Sampler 的 9 步采样流水线》交付并归档 APPROVED。主线: forward 抽 raw_logprobs(default 模式岿然不动)→penalties(repetition 正除负乘/presence/frequency)→bad_words 屏蔽(-inf)→logits_processors(argmax-invariant 与否决定能否走 greedy 快路: MinP 非 invariant, LogitBias/MinTokens invariant)→温度缩放(单调正变换保 argmax)→top-k/top-p 截断→随机 vs 贪心(sample: all_greedy 整批早退取 argmax; 混合批 torch.where 逐行合并 greedy_sampled/random_sampled)。TopKTopPSampler 多后端分发 flashinfer/triton/torch-native(forward_native/forward_cuda)。SamplingMetadata 承载逐请求参数, 全程批量张量化。核心非平凡论证: greedy 快路 argmax 与全随机路 argmax 严格等价。

## Why it matters

Part VII 采样章; 验证'参数全张量化 + greedy 严格等价 + argmax-invariant 决定快路'三反直觉结论。4/4 linter 全过, pytest 23 passed in 1.76s(host 纯 PyTorch CPU 单元, 不 import vllm 不需 CUDA), torch 2.11.0+cu130。reviewer verdict=APPROVED, 8 条 issue 全 non-blocking+negotiable: 27.2 'raw_logprobs 岿然不动'对 processed_* 模式需一句限定; 27.1 长清单句拆分; 27.6 决策三括号拆句; 27.8 'Qrita' 专名弱化; 口语比喻略密集收一收; repetition penalty 补微型数值追踪达 2+ 轮; min_p 等价引理补 max_prob>=min_p*max_prob 半句; 公式可选提升 $$ 块。登记 8 个精简版接口(Sampler/sample/apply_temperature/TopKTopPSampler/LogitsProcessor/apply_all_penalties/apply_bad_words/SamplingMetadata)。无伏笔应埋/应回收(bible due ch27 空, arc-map 无 ch27 条目)。

## What to remember

ch27《Sampler 的 9 步采样流水线》交付并归档 APPROVED。主线: forward 抽 raw_logprobs(default 模式岿然不动)→penalties(repetition 正除负乘/presence/frequency)→bad_words 屏蔽(-inf)→logits_processors(argmax-invariant 与否决定能否走 greedy 快路:...
