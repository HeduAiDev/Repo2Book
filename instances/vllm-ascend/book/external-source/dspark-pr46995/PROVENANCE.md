# DSpark 外部来源溯源(前瞻 primer 专用)

本目录是**本书 pin 版(vllm-ascend v0.21.0rc1 / vLLM v0.21.0)之外**的上游代码快照,
仅供 DSpark 前瞻原理章解读。DSpark 尚未合入 ascend(RFC #11126/#11163),
但已合入 vLLM 主线。

- **来源仓**: vllm-project/vllm(主线)
- **PR**: #46995 "[Spec Decode] DSpark",MERGED 2026-07-01
- **merge commit**: f5a8d73377d0f0a4e00cba172f9fbd0d50471b07
- **拉取日**: 2026-07-11(gh api contents@ref)

目录结构镜像上游真实路径(如 `vllm/model_executor/models/qwen3_dspark.py`),可当正常 source_root 解读。
正文内嵌这些片段时**必须**标注「来自 vLLM 主线 PR #46995 @f5a8d73,尚未合入本书 pin 的 v0.21.0——前瞻解读」,
不得伪装成 pin 树真源码(否则违反 HARD RULE 3 零脚手架/HARD RULE 2 只删不增的 pin 基线)。
