# ch30 交付：模型/LoRA/netloader 注册——全书最后一个代码章收官

- **Type**: delivery
- **Chapter**: 30
- **Date**: 2026-07-01
- **Timestamp**: 2026-07-01T15:10:42Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch30, delivery, APPROVED, registration, lora, netloader, last-code-chapter

## What happened

reviewer 判定 APPROVED（8 条 issue 全 negotiable/non-blocking，无阻断）。本章收束『注册到 vLLM 扩展点』主线（呼应 ch02 平台/ch23 CustomOp/ch27 量化/ch29 proposer）。三变体：register_model 登记 DeepseekV4ForCausalLM+MTP；PunicaWrapperNPU 按 device/rank 二选一绑 lora ops + refresh_all_lora_classes 全局类替换 trick（4 个 Ascend*LinearWithLoRA 追加进 _all_lora_classes）；@register_model_loader('netloader') 的 ModelNetLoaderElastic 弹性网络拉权重、失败 revert_to_default。已登记 5 个精简版接口到 bible；全书 12 个伏笔全部 resolved。

## Why it matters

全书代码章至此收官——vLLM 处处留扩展点，昇腾即『往每个扩展点登记一个昇腾实现』，多靠注册+薄壳、少数靠全局类替换或整模型特化。

## What to remember

reviewer 判定 APPROVED（8 条 issue 全 negotiable/non-blocking，无阻断）。本章收束『注册到 vLLM 扩展点』主线（呼应 ch02 平台/ch23 CustomOp/ch27 量化/ch29 proposer）。三变体：register_model 登记 DeepseekV4ForCausalLM+MTP；PunicaWrapperNPU 按 de...
