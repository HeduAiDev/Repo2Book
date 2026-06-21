# ch05 Stage 1 输入处理 —— 精简版实现说明（只做减法）

精简版与真实 vLLM **同名、同结构、同控制流**，可 host 直接跑（不 import vllm）。
所有删除项均为 dossier `subtraction_plan.delete` 批准项；`must_keep` 全部保留。
验收判据：把真实 vLLM 删掉所有标注的 `# SUBTRACTED` 分支，应当 ≈ 得到本精简版。

## 文件分工

- `input_processor.py` —— 本章主角 `InputProcessor`（process_inputs / assign_request_id /
  各 _validate_* / _get_mm_identifier）。真实代码原样，只删 delete 批准项。
- `parallel_sampling.py` —— `ParentRequest`（n>1 fan-out + 子参数派生 + 输出聚合）。
- `async_llm.py` —— `AsyncLLM.add_request` 的『输入处理 → assign_request_id → n 路由』片段。
- `messages.py` —— 主线**依赖**的数据结构忠实最小替身（SamplingParams/PoolingParams/
  EngineCoreRequest/MultiModalFeatureSpec/PlaceholderRange/argsort_mm_positions/
  split_enc_dec_input/random_uuid/length_from_prompt_token_ids_or_embeds/json_iter_leaves）。
- `config.py` —— VllmConfig/ModelConfig/LoRAConfig/Renderer/current_platform 等环境替身。
- `preprocess.py` —— deprecated raw-prompt 兜底路径 `InputPreprocessor`（控制流骨架）。

## 1:1 Source Map（精简版 ↔ 真实 vllm ↔ 改动 ↔ 原因）

| 精简版符号 | 真实出处 (pin f3fef123) | 改动 | 原因 |
|---|---|---|---|
| `InputProcessor.process_inputs` | `vllm/v1/engine/input_processor.py:L234-L377` | 删 tokenization_kwargs/raw-prompt 警告文案、mm_hashes 错误文案简写 | dossier delete 批准；纯日志/文案不改数据流 |
| `InputProcessor._validate_params` | `input_processor.py:L81-L136` | 删 thinking_token_budget 校验、pooling task 默认补全 | dossier delete 批准；次要旁支，保留 verify 骨架 |
| `InputProcessor._validate_lora` | `input_processor.py:L138-L155` | 删 per-LoRA tokenizer 警告文案 | dossier delete 批准；纯日志 |
| `InputProcessor.assign_request_id` | `input_processor.py:L214-L232` | 删 disable-randomization 警告文案 | dossier delete 批准；保留 external_req_id + 8 字符后缀逻辑 |
| `InputProcessor._validate_prompt_len` | `input_processor.py:L379-L424` | 删各 suggestion 文案构造 | dossier delete 批准；保留空/超长/等长三类 raise |
| `InputProcessor._validate_model_input` | `input_processor.py:L426-L476` | 删编码器缓存超限文案、Qwen3 vocab 长注释 | dossier delete 批准；保留 `max(tokenizer.max_token_id, vocab-1)` 判定 |
| `InputProcessor._get_mm_identifier` | `input_processor.py:L157-L173` | 原样 | must_keep；多模态缓存键（LoRA 前缀） |
| `inject_into_mm_cache` | `input_processor.py:L175-L212` | 整体删除 | dossier delete 批准；旁路优化不在主控制流 |
| `ParentRequest` (__init__/_get_child_sampling_params/get_child_info/get_outputs) | `parallel_sampling.py:L36-L126` | 原样 | must_keep；n>1 fan-out + 聚合主线 |
| `ParentRequest.observe_*` | `parallel_sampling.py:L128-L151` | 删除 | dossier delete 批准；metrics 统计旁路 |
| `AsyncLLM.add_request` (n 路由片段) | `vllm/v1/engine/async_llm.py:L349-L398` | 仅保留输入处理+id+fan-out 片段 | 完整 add_request 属 Stage 2/3 |
| `EngineCoreRequest` | `vllm/v1/engine/__init__.py:L80-L131` | msgspec.Struct → dataclass，删跨进程/推理字段 | 替身；保留本章全部字段 + .params |
| `SamplingParams.update_from_generation_config/update_from_tokenizer/clone/verify` | `vllm/sampling_params.py:L543-L664` | verify 体内 6 个 _validate_* 删为 pass | must_keep 入口保留；子校验非本章主线 |
| `argsort_mm_positions` / `split_enc_dec_input` | `vllm/multimodal/utils.py:L112`, `vllm/inputs/engine.py:L365` | 原样 | must_keep |

## 测试

`tests/test_input_processing.py`（37 用例，纯单元，host 跑）覆盖：rendered/raw/embeds/多模态
四类 prompt 路径、max_tokens 默认补全、clone 不污染、cache_salt 透传、参数/LoRA/dp_rank 校验、
空/超长/等长/越界四类模型输入校验、eos/bad_words 补全、多模态按 offset 排序展平 + LoRA 前缀、
assign_request_id 唯一性、ParentRequest fan-out（child id / n=1 / seed 分支 / 流式 vs FINAL_ONLY 聚合）、
AsyncLLM n 路由（n=1 / n>1 / pooling）。
