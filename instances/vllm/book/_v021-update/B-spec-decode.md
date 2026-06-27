# v0.21.0 更新摘要 — 投机解码（ch28）

基线 `f3fef1235` → `v0.21.0`，文件组：`llm_base_proposer.py` / `rejection_sampler.py` / `metadata.py` / `utils.py`。

## 文件级 diff 概览

| 文件 | 变动 | 结论 |
|---|---|---|
| `vllm/v1/sample/rejection_sampler.py` | 区间内无提交、空 diff | 无更新 |
| `vllm/v1/spec_decode/metadata.py` | 空 diff | 无更新 |
| `vllm/v1/spec_decode/utils.py` | 空 diff | 无更新 |
| `vllm/v1/spec_decode/llm_base_proposer.py` | +89 / −271，4 提交 | 见下，含 1 个移除 + 1 个新路径 + 1 个行为变更 |
| `vllm/v1/spec_decode/gemma4.py` | **新文件 +335** | 新 proposer 类型（NEW-FEATURE） |
| `vllm/v1/spec_decode/dflash.py` | +3/−2 trivial | SKIP |

> 关键判定：`llm_base_proposer.py` 的 −271 **不是**纯抽取搬家。其中约 −180 行是**彻底删除**了 tree-attention 草稿路径（`propose_tree` + `tree_choices`/`cu_drafts_per_level` 预计算 + import），与之配套的提交 `b1728c1e6 [Attention][Cleanup] Remove tree attention`。另有约 −70 行（`_update_positions_dependent_metadata`）是**抽方法**（逻辑不变，搬进同类的新私有方法），属 SKIP。新增的 `constant_draft_positions` 分支与 `gemma4.py` 是真新逻辑。

---

## 教学性变更

### 1. 移除 tree attention 草稿路径（`propose_tree`）
- **class**: BEHAVIOR-CHANGE（删除整条代码路径）
- **anchor**: `vllm/v1/spec_decode/llm_base_proposer.py`，删除 `SpecDecodeBaseProposer.propose_tree`、`tree_choices` / `cu_drafts_per_level` / `child_drafts_per_level` / `tree_draft_pos_offsets` 预计算，以及 `propose` 中 `isinstance(md, TreeAttentionMetadata)` 的分发分支。
- **target**: ch28
- **整合建议**：ch28 的精简版减法清单第 31 行已把"tree attention"列为正交删除分支——这与 v0.21.0 的官方动向**正好一致**，无需为它正名为"真实存在但被我们删掉"。建议在该处加一句脚注式说明：自 v0.21.0 起 vLLM 已在主线删除整条树形草稿路径（`propose_tree`），`propose` 退化为纯链式（chain）草稿，本书"减去 tree attention"从此与上游一致。无新增正文负担。
- **diagram 影响**：无。本章未画 tree attention，链式草稿图无需改动。

### 2. 新增 Gemma4 MTP proposer（`Gemma4Proposer`）
- **class**: NEW-FEATURE
- **anchor**: `vllm/v1/spec_decode/gemma4.py` → `class Gemma4Proposer(SpecDecodeBaseProposer)`；配套基类新增字段 `SpecDecodeBaseProposer.constant_draft_positions`（`vllm/v1/spec_decode/llm_base_proposer.py`，默认 `False`）。
- **target**: ch28
- **整合建议**：ch28 正文（第 37、108、170 行）已枚举 proposer 家族（n-gram / EAGLE / EAGLE3 / DFlash / MTP）。建议在"它们对外都遵守同一份契约"之后补一句 v0.21.0 新成员：Gemma4 MTP draft 头通过 `Gemma4Proposer` 接入同一 `SpecDecodeBaseProposer.propose` 入口，其特殊性在于**所有草稿步共享同一个位置**——草稿模型与目标模型跨模型共享 KV cache，每步都从目标的最后一个位置预测，故新增 `constant_draft_positions=True` 让位置缓冲与 `seq_lens` 在草稿循环内**不再逐步 +1**，且 `common_attn_metadata` 一次构建、整轮复用（见 `propose` 中 `if not self.constant_draft_positions or token_index == 0` 的 metadata 重建守卫）。这正好反衬出 EAGLE 链式草稿"逐步推进位置 + 每步重建 metadata"的默认行为，是讲清 `propose` 循环不变量的好对照。
- **diagram 影响**：可选。若 ch28 有 proposer 家族表/草稿循环时序图，可加一行 Gemma4 MTP（标注"恒定位置、KV 共享、metadata 仅建一次"）；非必需。

### 3. 多模态从"硬报错"放宽为"告警后继续"
- **class**: BEHAVIOR-CHANGE
- **anchor**: `vllm/v1/spec_decode/llm_base_proposer.py`，`_raise_if_multimodal` 重命名为 `_warn_if_multimodal`；原 `raise NotImplementedError(...)` 改为 `logger.warning(...)` 并继续以纯文本投机解码运行；`propose` 中调用点同步改名（提交 `2d7d6cf76 [Spec Decode] Allow multimodal models with a warning`）。同提交还在多模态模型名单加入 `Cohere2VisionForConditionalGeneration`。
- **target**: ch28
- **整合建议**：若 ch28 任何位置叙述"草稿模型 / 并行起草不支持多模态会直接抛 `NotImplementedError`"，需更新为：自 v0.21.0 起改为打印告警并**降级为纯文本投机解码**继续运行（`_warn_if_multimodal`），不再硬失败。当前正文未点名该方法，故仅在涉及"多模态限制"处补一句即可；属精确小修，不动主线。
- **diagram 影响**：无。

---

## 其余提交（SKIP）
- `a2812becd Cohere Eagle + Cohere MoE`：仅在多模态模型名单加 `Cohere2VisionForConditionalGeneration`（已并入变更 3），无新代码路径。
- `_update_positions_dependent_metadata`（+62 新方法）：纯抽取——把原 `propose` 内联的 slot mapping / seq_len 递增逻辑搬入私有方法，逐字搬运、行为不变，仅为给 `constant_draft_positions` 让出分支点。SKIP（移动，不教学）。
- `dflash.py` +3/−2、import 清理、docstring 把 "eagle" 改 "drafter"：格式 / 措辞，SKIP。

## 三个无变更文件
`rejection_sampler.py`、`metadata.py`、`utils.py` 在 `f3fef1235..v0.21.0` 区间内**零提交、空 diff**。ch28 关于 rejection sampler（含 `draft_probs` 两类分支与 `constexpr` 兼容，正文第 720 行）与 spec-decode metadata/utils 的全部解读**无需任何更新**。
