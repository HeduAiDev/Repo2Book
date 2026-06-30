# ch29 实现笔记 —— 投机解码的 NPU 对位（subtract-only 精简版）

> 范式：**工厂分发 + 薄壳继承 + 少数重量级覆写**。本章只覆盖提议侧（提出 draft token）；
> 验证侧（接受/拒绝采样）由 ch28 的 `AscendRejectionSampler` 负责，本章仅回指。

## 文件清单（与 `vllm_ascend/spec_decode/` 同构）

| 精简版文件 | 真实源码 | 角色 | 改动 |
|---|---|---|---|
| `factory.py` | `vllm_ascend/spec_decode/__init__.py:L33` | 工厂入口 `get_spec_decode_method`，一处 if-elif 分发 8 个 proposer | 仅删 L1-19 许可证抬头；**改名 `__init__.py`→`factory.py`**（lint_fidelity 不扫 `__init__.py`，会漏掉 must_keep 的工厂函数），控制流逐字一致 |
| `ngram_proposer_npu.py` | `vllm_ascend/spec_decode/ngram_proposer_npu.py` | 薄壳之极致：`AscendNgramProposerNPU(NgramProposerGPU)`，三方法全 no-op | 逐字保留（整文件 35 行无可删项） |
| `ngram_proposer.py` | `vllm_ascend/spec_decode/ngram_proposer.py` | CPU n-gram 薄壳：`propose` 写回 `token_ids_cpu` 后调父类 `batch_propose` | 逐字保留 |
| `suffix_proposer.py` | `vllm_ascend/spec_decode/suffix_proposer.py` | 最薄薄壳：`propose` 一行转发父类 | 逐字保留 |
| `medusa_proposer.py` | `vllm_ascend/spec_decode/medusa_proposer.py` | 中等薄壳：覆写 `dummy_run`(set_ascend_forward_context) + `propose`(gather 末位 hidden state) | 逐字保留 |
| `extract_hidden_states_proposer.py` | `vllm_ascend/spec_decode/extract_hidden_states_proposer.py` | 中等薄壳：覆写 `dummy_run` + `prepare_next_token_ids_padded` | 仅删 L1-15 许可证抬头 |
| `draft_proposer.py` | `vllm_ascend/spec_decode/draft_proposer.py:L8` | 走重量级 base 的薄入口（多继承），`pass_hidden_states_to_model=False` + 校验 | 逐字保留 |
| `eagle_proposer.py` | `vllm_ascend/spec_decode/eagle_proposer.py:L10` | 走重量级 base 的薄入口（多继承），`pass_hidden_states_to_model=True` | 逐字保留 |
| `dflash_proposer.py` | `vllm_ascend/spec_decode/dflash_proposer.py:L15` | 继承链延伸：`AscendDflashProposer(AscendEagleProposer)`，建 DFlash 缓冲 | 保留 `__init__`；删 L63-267 的 set_inputs_first_pass/dummy_run/build_model_inputs_first_pass DFlash Triton kernel 细节 |
| `llm_base_proposer.py` | `vllm_ascend/spec_decode/llm_base_proposer.py:L111` | 重量级核心 `AscendSpecDecodeBaseProposer`（真文件 2043 行） | **重度减法**：只留 code_spine 四骨架节点，见下 |

## `llm_base_proposer.py` 骨架取舍（重度减法）

只保留 dossier `code_spine` 指名的四个节点，其余整段标 `# SUBTRACTED:`：

| 精简版 | 真实源码 | 保留要点 | 删除内容（已批准） |
|---|---|---|---|
| `__init__` | `llm_base_proposer.py:L114` | 持久缓冲、昇腾并行组 `patch_tensor_parallel_group`、`use_cuda_graph=runner._use_aclgraph()`、`decode_threshold=1+num_speculative_tokens`、默认 `_runnable=_run_merged_draft` | `logger.debug`（L120-131）；mrope/xdrope/positions 三选一分支（L207-218） |
| `load_model` | `llm_base_proposer.py:L246` | 尾段 FULL graph 时 `_runnable = ACLGraphWrapper(_run_merged_draft, ...)` | 主体 L247-423（建 draft 模型/共享权重/mtp 特判）+ `logger.info` |
| `_propose` | `llm_base_proposer.py:L621` | 骨架 `combine_hidden_states→set_inputs_first_pass`，并以注释钉死后续 `dispatch→_runnable→return draft_token_ids` | 主体 L669-953（多步 draft 循环/采样）+ 主线外可选参数；末尾 `raise NotImplementedError`（不伪造 draft token） |
| `prepare_inputs` | `llm_base_proposer.py:L1701` | 按拒绝数重算 `query_start_loc`/`seq_lens`、`np.repeat`+`token_arange` 构造 `token_indices`、装箱 `AscendCommonAttentionMetadata` 返回 | docstring 内 ASCII 算法示例（L1713-1727） |

注：`_propose` 主体被减后无法真产出 draft token，故以 `raise NotImplementedError` + 骨架注释明示「本章范围外、重 NPU/Triton/MLA 不真跑」，**不伪造返回值**（遵守 no-invention）。

## must_keep 全保留（lint_fidelity 校验通过）

`get_spec_decode_method` / 8 个 `Ascend*Proposer` / `NgramProposerGPU` / `SpecDecodeBaseProposer` /
`pass_hidden_states_to_model` / `prepare_inputs` / `_propose` / `ACLGraphWrapper` / `_runnable` /
`num_speculative_tokens` / `patch_tensor_parallel_group` 均原样在精简版中。

## 测试（host 可跑，`tests/conftest.py` 桩掉 NPU/vLLM 依赖）

`python3 -m pytest tests/ -q` → **10 passed**。覆盖：
1. 工厂 if-elif 的 method→类映射（含非同名：`ngram_gpu`→NPU、`draft_model`→DraftModel、`eagle/eagle3/mtp` 共用 Eagle）+ 每 proposer 构造签名差异 + 未知 method 抛 `ValueError`；
2. CPU n-gram `propose`：跳过空/不支持/超长请求、写回 `token_ids_cpu`、交父类 `batch_propose`；
3. `ngram_gpu` no-op 薄壳：`propose` 返回 `None`（不复用父类 GPU kernel）；
4. suffix 一行转发父类（补 `runner.input_batch`）；
5. medusa `propose` 的 `offsets/indices` gather 计算（含「无 draft token」直通分支）；
6. eagle/draft 薄入口转调 base 时 `pass_hidden_states_to_model` 分别为 True/False；
7. `prepare_inputs` 按拒绝数收缩 + `token_indices` 构造（worked example `q=[2,4,3]`、`rejected=[0,2,0]` → `token_indices=[0,1,2,3,6,7,8]`、`new_qsl=[0,2,4,7]`）。

重 NPU 路径（ACLGraph 捕获 / 昇腾 Triton spec_decode kernel / MLA）需 NPU/CANN，不在精简版内真跑。
