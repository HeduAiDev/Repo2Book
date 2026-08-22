# ch08 精简版 impl-notes — 输出的另一个维度：logprobs（Part II：上行泳道 logprobs 支路）

- **Pin**：vLLM v0.27.1（`6e448d0ea`）。全部 `# SOURCE:` 行号对当前 pin 现核（收工前跑了
  148 个唯一引用区间的机械核对脚本：引用区间的真实源码首行逐条比对，可疑项逐一人工回查
  真文件修正——见 §收工审计；未照抄 v2 资产的 v0.21.0 旧行号）。
- **产物**：`implementation/logprobs_lane.py`（单模块精简版 ~2850 行，170 个 `# SOURCE` 锚 +
  83 个 `# SUBTRACTED` 标记）+ `implementation/_msgspec_seam.py`（msgspec 宿主替身，ch05/
  ch06/ch07 同款 seam + 本章新增 NamedTuple 解码面——见 §Seam 清单）。host 可跑：真
  tokenizers 0.22.2 Rust tokenizer（**byte-fallback 词表**——`<0xXX>` 逐字节 token、多字节
  汉字拆三 token、单 token 解码出 U+FFFD 是真 Rust 行为，不是模拟）、真 msgpack 字节过线、
  真 torch（CPU）；无 vllm 包、无 msgspec 包、host 无 CUDA。
- **跑法**：`cd instances/vllm/artifacts-v3/ch08-logprobs && python -m pytest tests/ -q`
  → **90 passed**（~3.4s，纯 host 单元测试，无平台分支 → 无需进容器）。
  `python scripts/lint_fidelity.py <本章目录>` → **全部通过（无 BLOCKING、无警告）**；
  must_keep 50 个符号经 linter `over_subtraction` 项全数核在。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本模块（HOST SEAM 例外见
  §Seam 清单——每个 seam 行内标注并在此登记）。

## 本章主题 = 全真部分（与 ch05/ch06/ch07 的分工）

- **`vllm/logprobs.py` 整文件**（m9/m10/m12 容器侧）：`Logprob` dataclass（OpenAI 输出序列化
  暂不用 msgspec 的 TODO 注释逐字）、`LogprobsOnePosition` 类型别名、`FlatLogprobs` 六条平行
  原生列表 + start/end_indices 区间索引（`append`/`append_fast` 免中间 dict/`__len__`/
  `__getitem__` int 现造 dict + slice 重建/三个 TypeError 占位——按 delete 项 8 删
  `extend`/`__iter__` 覆写）、`PromptLogprobs`/`SampleLogprobs` 联合、`create_prompt_logprobs`
  （首位 `append(None)` 占位——首 token 无条件概率）/`create_sample_logprobs`（flat/nested
  选择）、**`append_logprobs_for_next_position`** rank 链 `chain((rank,), range(1,k+1))` +
  dict 键去重——『第 0 个=被采样 token』不变式的源头。
- **`vllm/v1/engine/logprobs.py` 整类逐字**（本章主角，m6-m8/m12/m13/m18/m20）：
  `LogprobsProcessor` 八方法全真——`from_new_request`（三 None/0.0 分支=哪些容器被开）、
  `update_from_output`（唯一入口、sample/prompt 双路分派）、`_update_sample_logprobs`
  （逐列 tolist → 非增量 `convert_ids_list_to_tokens` → `cumulative += logprobs[0]` →
  append）、`_update_prompt_logprobs`（torch 恢复 [num_tok,k] 形状、扁平一次性去 token、
  逐位置切片）、`pop_prompt_logprobs`（DELTA 一次性取走清空）、`_get_sampled_context_ids`
  （flat 直索 token_ids/nested `next(iter(entry))` 双路径、max_context=4）、
  **`_correct_decoded_token`**（1..4 上下文 token 拼接重解码、剥干净前缀、tokenizer
  normalization 最长公共前缀兜底——本章技术核心）、`_verify_tokens`（横向候选 vs 纵向
  上下文，仅修以 U+FFFD **结尾**者）。
- **采样器 logprobs 两步**（m1/m2/m16/m17/m18，`vllm/v1/sample/sampler.py` 的本章域）：
  `forward` 骨架顺序逐字（raw 留底 NOTE(woosuk) 段 → fp32 → processors 占位 → sample →
  sparse gather → `-1` 全词表直通 → `gather_logprobs` → 稀疏优先覆盖 → int32 →
  SamplerOutput）；`compute_logprobs`（`log_softmax` fp32 一行——raw 语义数学落点）；
  `gather_logprobs`（topk 值+下标 + 被采样 token gather + `batched_count_greater_than`
  计数 rank + cat 成 [num_tok,k+1]、被采样恒第 0 列）；`sample` 的 greedy 快路径（m16：
  `processed_logits`/`processed_logprobs` 唯一物化点——随机路径按 delete 项 1 删、保
  `NotImplementedError` 失败面）；`gather_specific_token_logprobs`（m17 稀疏路：padding+
  valid_mask+gather 三步核心，pinned 构造循环按 delete 项 10 删——列 0=sampled、无效位
  -inf、mask 语义原样）；`ops/logprobs.py` 的 `batched_count_greater_than` 整函数
  （`torch._check`×2 + `(x >= values).sum(-1)`）。
- **批登记与形态载体**（m3/m4/m5）：`InputBatch` 本章域（`num_logprobs[req_id]` 登记 -1→
  vocab_size、`logprob_token_ids` 登记与 req_index 键换算、`max_num_logprobs = max(全部)`
  属性、finish 弹出）；`SamplingMetadata` logprobs 字段面；`LogprobsTensors`/`LogprobsLists`
  NamedTuple（`to_cpu_nonblocking`/`tolists`/`empty_cpu`/`slice_request` 四接口）+
  `SamplerOutput`/`ModelRunnerOutput`；`AsyncGPUModelRunnerOutput`（独立 copy stream 上
  non_blocking 拷贝 + event 同步 + `get_output` 里 tolists——HOST SEAM 面：CPU host 上
  stream/event 退化为 no-op 上下文，CUDA host 上委托真 torch.cuda 对象）。
- **prompt 支路引擎侧**（m11，站 12）：`_get_prompt_logprobs_dict` 主线逐字（『rare feature
  优先简单可维护』注释、`empty_cpu` 整 prompt 预分配、chunk/末块判定（`==` 情形 defer 注释
  原话）、`hidden_states[offset:offset+num_logits]` → `compute_logits`（WC2 两方法契约）、
  目标=prompt[i+1]、logprobs_mode 四态在 prompt 路退化为 raw 的 NOTE、分块 non_blocking
  copy_、末块交付+注销+`in_progress_prompt_logprobs_cpu=None`）；按 delete 项 6 留
  `num_tokens is None` 抢占分支一个作代表，删 prompt_embeds/`num_logits<=0` 两防御分支。
- **调度切行与过线**（m3/m5，站 6/7）：`Scheduler.update_from_output` 本章行——
  `logprobs.slice_request(req_index, len(new_token_ids))`（三条件守卫逐字）+
  `prompt_logprobs_dict.get(req_id)` + 两个 logprobs 字段装 `EngineCoreOutput`；线载体
  `EngineCoreOutput`（logprobs 两字段在 L193-194 原位）、`EngineCoreOutputs`（行式布局
  NOTE(Nick) 逐字——WC4『班车』注脚）；`MsgpackEncoder`/`MsgpackDecoder` 的 ndarray/tensor
  钩子链（`enc_hook`/`_encode_ndarray`/`_encode_tensor`/`dec_hook`/`_decode_ndarray`/
  `_decode_tensor`/**`ext_hook`——RAW_VIEW 解包为裸 memoryview**，pickle 回退按 delete 项 4
  删、未知类型硬错误面保留）。
- **到港装配与出口装车**（m6/m14，站 8/13）：`OutputProcessor.process_outputs` 单循环骨架
  逐字（NOTE FOR DEVELOPERS 注释原话；主泳道步骤 1/2/4 按 delete 项 5 以 SUBTRACTED 占位，
  第 3 步 `update_from_output` 调用与第 4 步 queue/list 分叉保留）；`RequestState` 本章域
  （`from_new_request` 里 detokenize=False → tokenizer=None 的 m20 源头行、
  `_new_completion_output` 的 DELTA 切尾 `logprobs[-len(token_ids):]` + `cumulative_logprob`
  进 CompletionOutput、`_new_request_output` 的 DELTA 走 pop/其余直读）。
- **入口与出口**（m15/m17/m18/m19，站 1/14）：`ChatCompletionRequest` logprobs 参数面 +
  `to_sampling_params` 的 logprobs 三参数分叉（`logprob_token_ids` 优先设它则
  logprobs=None）、echo 默认 prompt_logprobs=top_logprobs、stream→DELTA/FINAL_ONLY；
  `SamplingParams` logprobs 四参数（docstring 逐字）+ `logprobs=True→1` 归一 +
  `skip_reading_prefix_cache = prompt_logprobs is not None`（m19 互斥一行）+
  `num_logprobs` property（把 `len(logprob_token_ids)` 并进账）；`LogprobsMode` 四态
  Literal；OpenAI 三件套 `ChatCompletionLogProb(token, logprob, bytes)`/`Content`/
  `LogProbs`、`_get_top_logprobs`（bytes=token.encode('utf-8')、钳底 max(·,-9999.0)、
  top_logprobs 截断 + return_all）、`_create_chat_logprobs`（逐位造 content、None step
  回退 decode、bytes None 分支）、`_get_decoded_token`/`format_token_id_placeholder`。
- **非增量去 token**（m7 helper）：`convert_ids_list_to_tokens` + `_get_leading_space_marker`
  + `_restore_leading_spaces` 全真（Metaspace 判据/缓存/SentencePiece add_dummy_prefix 逆
  ——与 ch7 增量法的对照面）。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因；核心行）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `Logprob`/`LogprobsOnePosition` | vllm/logprobs.py:L9-L24、L27 | **逐字**（含 dataclass TODO 注释） | must_keep×2；nested 叶子 |
| `FlatLogprobs`（append/append_fast/__len__/__getitem__/三占位） | vllm/logprobs.py:L30-L144 | 逐字 minus `extend`（L95-98）/`__iter__`（L146-152）覆写（SUBTRACTED 标注） | must_keep×6；m10（delete 项 8——Sequence 基类继承面等效） |
| `PromptLogprobs`/`SampleLogprobs`/`create_*`/`append_logprobs_for_next_position` | vllm/logprobs.py:L155-L206 | **逐字** | must_keep×4；m9/m12 rank 链 |
| `LogprobsProcessor` 八方法 | vllm/v1/engine/logprobs.py:L29-L352（逐方法 L43/L69/L121/L189/L209/L249/L312/L348 起） | **整类逐字**（无删除） | must_keep×10；本章主角 |
| `convert_ids_list_to_tokens` + marker 两助手 | vllm/tokenizers/detokenizer_utils.py:L62-L170 | **逐字** | must_keep；m7 非增量去 token |
| `Sampler.forward` 骨架 | vllm/v1/sample/sampler.py:L72-L149 | 顺序逐字；spec 分支/采样实现占位（SUBTRACTED 标注） | m1/m16/m18（delete 项 1） |
| `compute_logprobs`/`gather_logprobs` | vllm/v1/sample/sampler.py:L304-L306、L308-L356 | **逐字**（mark_unbacked 注释原话） | must_keep×2；m1/m2 |
| `batched_count_greater_than` | vllm/v1/sample/ops/logprobs.py:L10-L27 | **逐字**（@torch.compile 装饰行原样；backend 值为 seam，见 §Seam） | must_keep；计数 rank kernel |
| `greedy_sample`/`sample`（greedy 路径） | vllm/v1/sample/sampler.py:L239-L241、L243-L302 | greedy 快路径逐字（processed_* 物化点）；随机路径删+NotImplementedError | m16（delete 项 1——随机域归 Part VII ch29-33） |
| `gather_specific_token_logprobs` | vllm/v1/sample/sampler.py:L151-L225 | docstring/gather/mask/rank 原样；pinned 逐位填充循环 SUBTRACTED 标注（构造结果同契约） | m17 轻讲（delete 项 10） |
| `apply_logits_processors`/`apply_temperature`/`_combine_*` | vllm/v1/sample/sampler.py:L371-L417、L228-L237、L359-L369 | 调用点保留；实现体删/占位（空处理器批真码同样原样返回） | delete 项 1（Part VII 域） |
| `SamplingMetadata` | vllm/v1/sample/metadata.py:L14-L57 | logprobs 三字段+贪心面保留；采样域字段 SUBTRACTED | 站 2-4 触达面（delete 项 1） |
| `CachedRequestState` | vllm/v1/worker/gpu_input_batch.py:L34-L57 | logprobs 域字段（prompt_token_ids/num_computed_tokens/in_progress_prompt_logprobs_cpu）+采样参数面；其余 SUBTRACTED | m11 挂账字段（must_keep） |
| `InputBatch` | vllm/v1/worker/gpu_input_batch.py:L269-L273 字段、L435-L444 登记、L530/L573-574 弹出、L1149-L1151 max 属性 | logprobs 登记逐字；批的 persistent 机制/`_make_sampling_metadata` 准备段精简（恒贪心面 HOST 注） | must_keep `max_num_logprobs`；m3（delete 项 3） |
| `LogprobsLists`/`LogprobsTensors` | vllm/v1/outputs.py:L28-L137 | **逐字**（四接口全真；filter/cat 的 spec 域 SUBTRACTED——ch31） | must_keep×4；m4 切行/搬运 |
| `SamplerOutput`/`ModelRunnerOutput` | vllm/v1/outputs.py:L212-L219、L260-L308 | logprobs 字段面保留；pooler/nans/cudagraph/routed SUBTRACTED | 载体（delete 项 2/3） |
| `AsyncGPUModelRunnerOutput` | vllm/v1/worker/gpu_model_runner.py:L258-L344 | copy stream 段+get_output 逐字（routed/EP fault/parse_output 分支 SUBTRACTED——多 token 批按同一 tolists 面处理） | must_keep×2；m4（delete 项 2） |
| `async_tensor_h2d`/`tensor_data` | vllm/utils/torch_utils.py:L573-L584、vllm/v1/utils.py:L777-L787 | **逐字** | m11 触达的真实小件 |
| `GPUModelRunner._get_prompt_logprobs_dict` | vllm/v1/worker/gpu_model_runner.py:L5620-L5727 | 主线逐字；三防御分支留一删二、`_sync_device` 尾删（SUBTRACTED 标注） | must_keep×3；m11（delete 项 6） |
| `Scheduler.update_from_output` | vllm/v1/core/sched/scheduler.py:L1670-L2059（logprobs 行 L1909-1941 逐字） | 仅保留 logprobs 切行+装车行与循环骨架；停止判定/状态机/收尾全 SUBTRACTED | 站 6（delete 项 3——ch9 全章域） |
| `FinishReason`/三个线 Struct | vllm/v1/engine/__init__.py:L29-L65、L97-L146、L184-L215、L230-L258 | 字段名/顺序/默认保留（schema 契约）；他章域字段行 SUBTRACTED/类型放宽 Any | must_keep×2；m5 班车 |
| `MsgpackEncoder`/`MsgpackDecoder`/`ext_hook` | vllm/v1/serial_utils.py:L136-L178、L191-L197、L237-L273、L313-L348、L340-L348、L350-L365、L389-L425、L473-L484 | ndarray/tensor 钩子链逐字（ext_hook RAW_VIEW→裸 memoryview 逐字）；slice/mm/utility/pickle 回退 SUBTRACTED | must_keep；m5 过线机制（delete 项 4） |
| `RequestState`（from_new_request/_new_*） | vllm/v1/engine/output_processor.py:L129-L423 | logprobs 段逐字（L224-225/L366-371/L404-420）；主泳道字段/三道闸/pooling SUBTRACTED 占位 | must_keep×2；m14（delete 项 5——ch7 域） |
| `OutputProcessor.process_outputs` | vllm/v1/engine/output_processor.py:L589-L711 | 循环骨架+第 3 步+第 4 步分叉逐字；stats/detokenize/清理段 SUBTRACTED 占位 | 站 8（delete 项 5） |
| `RequestOutputKind` | vllm/sampling_params.py:L182-L188 | **逐字**（Enum 直名与真码一致） | 三态契约（WC5） |
| `SamplingParams` 本章域 | vllm/sampling_params.py:L267/L275/L278/L284/L293 字段、L486-L490 归一、L509-L513 互斥、L738-L746 property | logprobs 四参数 docstring 逐字+两段归一；校验体/其余字段 SUBTRACTED | must_keep×4（delete 项 9） |
| `LogprobsMode`/`PROCESSED_LOGPROBS_MODES` | vllm/config/model.py:L99-L105 | **逐字** | m16 四态契约 |
| `ChatCompletionLogProb` 三类 | vllm/entrypoints/openai/chat_completion/protocol.py:L81-L95 | **逐字** | must_keep×3；m15 |
| `ChatCompletionRequest`+`to_sampling_params` | protocol.py:L212、L219-L220、L285-L296、L303-L310、L398-L403、L646-L734 | logprobs 参数面+to_sampling_params logprobs 段逐字；SSE/工具域 SUBTRACTED | 站 1（delete 项 7/9） |
| `GenerateBaseServing._get_decoded_token`/placeholder | vllm/entrypoints/generate/base/serving.py:L113、L118-L133、L252-L274 | **逐字**（return_as_token_id 一行分支保留） | delete 项 7 指定保留面 |
| `OpenAIServingChat._get_top_logprobs`/`_create_chat_logprobs` | vllm/entrypoints/openai/chat_completion/serving.py:L110-L138、L1140-L1165、L1167-L1231 | **逐字**（SSE 生成器全貌 SUBTRACTED——ch2/ch7 域） | must_keep×2；m15（delete 项 7） |

## 删除台账 — dossier subtraction_plan 10 项 delete（全部执行）

1. **Sampler 采样实现体** ✓ — 随机路径（apply_temperature/TopKTopPSampler/torch.where 合并）、
   processors 实现（allowed/bad_words/min_tokens/logit_bias/penalties/thinking budget）、
   spec decode（predict_bonus_token/_combine_outputs_with_spec_tokens/forward spec 分支）。
   保留骨架顺序与 greedy 快路径（m16 物化点），随机路径 `raise NotImplementedError`
   保住可观察失败面。
2. **D2H 无关载荷** ✓ — routed_experts 拷贝与拼接、EP fault 查询、num_nans、
   RejectionSampler.parse_output 分支、LogprobsTensors.filter/cat。
3. **scheduler.update_from_output 非本章域** ✓ — 停止判定/_handle_stopped_request/KV 释放、
   prefill_stats、num_nans 登记、EngineCoreOutput 的 pooling/events/kv/ec/trace/
   prefill_stats/routed_experts/num_nans 字段行、错误请求/KV 收尾/stats 尾部
   （L1679-1682/L1684-1726/L1695-1732/L1734-1759/L1766-1795/L1800-1843/L1845-1907/
   L1917-1918/L1946-2011/L2019-2033 各段 SUBTRACTED 标注）。
4. **serial_utils 旁支钩子** ✓ — slice 钩子、多模态 _encode_mm_*/UtilityResult/pickle 回退、
   encode_into 多帧零拷贝、OOB tensor provider、VLLM_ALLOW_INSECURE_SERIALIZATION 告警。
5. **output_processor 主泳道段** ✓ — RequestOutputCollector 不入、detokenizer.update 调用与
   stop-string 判定占位、make_request_output 三道闸/stream_interval/n>1、abort_requests、
   _update_stats_*、streaming_input、_finish_request/清理尾段（L685-702）。
6. **_get_prompt_logprobs_dict 防御分支** ✓ — prompt_embeds 不兼容 continue、`num_logits<=0`
   continue、`_sync_device` 尾部；保留 `num_tokens is None` 抢占分支一个作代表。
7. **serving/SSE 全貌** ✓ — chat_completion_stream_generator 主体/DeltaMessage/finish_reason
   组装不入（ch2/ch7 域）；`_get_decoded_token` 的 return_as_token_id 一行分支按计划保留。
8. **FlatLogprobs 不可变占位** ✓ — `__setitem__`/`__delitem__`/`insert` 三行原文保留
   （MutableSequence ABC 缺它们无法实例化——三行即全部原文，无可删体）；`extend`/`__iter__`
   覆写删（Sequence 基类继承面等效：逐位置 append / getitem 整数递增迭代，行为一致）。
9. **SamplingParams 校验体与无关字段** ✓ — _validate_*/verify/clone/temperature 归一/stop
   缓冲/logit_bias 换算等全删；只留 logprobs 四参数+两段归一+互斥一行+num_logprobs 属性。
10. **稀疏路 pinned 构造细节** ✓ — token_ids_cpu/valid_mask_cpu 逐位填充循环 SUBTRACTED
    （保留 padding+mask+gather 三步核心与契约注释；构造结果与原循环同一矩阵）。

### 机械删除/替换（不在 delete 单——为可跑性与章边界所必需，**请 reviewer 逐条过目**）

| 位置 | 内容 | 理由 |
|---|---|---|
| gpu_input_batch `_make_sampling_metadata` | 恒贪心面（`greedy=True` HOST 注）；温度/top_p/top_k/generators 构造行 SUBTRACTED | 简化批未登记温度（真实判据 greedy_reqs 全覆盖在被删的 L935-941）；greedy 路径是本章可观察行为面 |
| gpu_model_runner 构造面 | GPUModelRunner 为字段面构造器（requests/num_prompt_logprobs/model/model_config/sampler/query_start_loc/input_batch 注入） | runner 其余机制归 ch4/ch9/ch13；num_prompt_logprobs 真实登记位在 _update_states L1312-1317（SUBTRACTED 标注注明） |
| scheduler 构造面 | `Scheduler(requests)` HOST 字段面（真实构造 L70 起 ch9 域） | 同上章边界先例（ch7 装配线） |
| EngineCoreOutput/EngineCoreRequest | 他章域字段类型放宽 Any / 尾部字段 SUBTRACTED（array_like 位置编码下省尾字段对解码无害——seam 已实测短数组回填默认） | 线 schema 契约保持字段名/顺序/默认 |
| `RequestState`/`OutputProcessor` | detokenizer/lora/stats/stream_interval 等 ch7 域字段 SUBTRACTED；detokenizer 由 HOST 注入面（get_next_output_text/output_token_ids） | ch7 精简版已含全量（互补不重复——delete 项 5 原文） |
| `process_outputs` 返回面 | `SimpleNamespace(request_outputs, reqs_to_abort)` HOST 注（真实 OutputProcessorOutput dataclass L708-711） | 类型面非本章域；两字段语义一致 |
| `make_request_output` | 走直线（三道闸 SUBTRACTED 占位、pooling 分支删） | ch7 精简版已实现三道闸全量 |
| PIN_MEMORY | `torch.cuda.is_available()` 替 `is_pin_memory_available()`（真值推导保留注释） | host 无 CUDA 时同源；torch_utils.py:L72 |
| `bytestr` | 去 `zmq.Frame` 联合项（SUBTRACTED 行内注） | 本章线载体无 zmq.Frame 面（ch5 域） |

## Seam 清单（HOST SEAM，全在行内标注）

| Seam | 位置 | 是什么 | 为什么仍忠实 |
|---|---|---|---|
| `_msgspec_seam` | 独立文件（ch05/ch07 同款 + 本章新增面） | msgspec API 子集的 msgpack-backed 实现；**新增 NamedTuple 解码**（`_from_builtin` 的 `_fields` 分支） | 线上字节是真 msgpack；真 msgspec 原生把 NamedTuple 按位置数组解码并应用字段注解（LogprobsLists/LogprobsTensors 过线正是这条——ndarray/tensor 字段随之进 dec_hook）；缺它 `new_logprobs` 会解成裸 list（修复过程见 §收工审计） |
| `_CurrentPlatform.simple_compile_backend` | §0 | 真值 "inductor"（interface.py:L165），seam 传 "eager" | host 无加速器工具链；dynamo 照常 trace、同一 eager 数学执行——数值与手 gather 全对齐（tests 断言） |
| `torch.cuda` Event/Stream 面 | §0 `_cuda_event`/`_CudaStreamCtx`/`_cuda_wait_stream`/`_cuda_current_stream` | CUDA-less host 上 event 退化为 record/synchronize no-op、stream 上下文直通；**CUDA host 上委托真 torch.cuda 对象**（测试机有 GPU 时走真路径） | `to_cpu_nonblocking` 在 CPU 张量上本就是 no-op（outputs.py:L73-75 原文）——搬运语义不变 |
| `TokenizerLike = Any` | 模块头 | 真文件是 transformers AnyTokenizer union 的 Protocol | 被保真代码只调 decode/convert_ids_to_tokens/.backend_tokenizer；测试给真 Rust tokenizer（byte-fallback + Metaspace 两面） |
| `OpenAIBaseModel` | §0 | pydantic BaseModel + extra=allow + field_names（真 config 逐字） | engine/protocol.py:L31-35 原样 |
| ch7 产品面 | `DetokFace`（测试侧）/detokenizer 注入 | get_next_output_text/output_token_ids 两方法面 | ch7 精简版已含全量；本章只消费 |
| `logger`/`init_logger` | §0 | NullHandler 面板 | 保真路径无日志输出 |

## 已知偏差（reviewer 重点）

1. **随机采样路径 NotImplementedError**：真实代码在混合批/随机批走 apply_temperature +
   TopKTopPSampler；本章域只有 greedy（`temperature=None` 面）。测试断言 greedy 行为与
   raw 留底/gather 数值；随机路径的可观察失败面保留（raise 而非静默错值）。
2. **`@torch.compile` backend 值**：真平台默认 "inductor"，seam "eager"（见 Seam 表）。
   `torch._check` 两条形状断言原样保留——测试曾靠它当场揪出一处 shape 不匹配的参考计算
   （这正是它存在的意义）。
3. **`omit_defaults=True` 在 array_like 上无线上效果**：seam 对 array_like Struct 编码全部
   字段（真 msgspec 0.19-0.21 容器实测同行为，seam 文档注明）；`test_no_logprobs_output_
   stays_small` 断言的是 None 字段解码回 None，不是字节变小。
4. **结构洞**：`InputBatch` 简化面无温度登记（恒贪心）、`Scheduler` 无停止判定（恒不
   finish）、`make_request_output` 无三道闸（FINAL_ONLY 中间输出照样构造）——与『删掉批准
   分支后的真码』严格一致（ch6/ch7 先例），测试不触达。
5. **msgpack 非 writable buffer UserWarning**：`torch.frombuffer` 对 decode 出的
   memoryview 发一次 warning（真 vLLM 同路径同 warning）；不抑制（真实行为）。
6. **异步测试无**：本章全同步面（process_outputs 单循环是同步函数），无 pytest-asyncio
   依赖。

## 收工审计（2026-08-22）

- **SOURCE 行号机械核对**：148 个唯一引用区间逐一比对真实源码区间首行；修正 12 处偏移
  （bytestr L49→L54、remove_request 补 def L530、process_outputs L592→L589、字段读取
  L632→L631、pooling_output L671→L670、尾段 L686-706→L685-702、RequestOutputKind
  L180→L182 + `enum.Enum`→`Enum`（真码直名）、metadata L11→L14、decode L336→L340、
  protocol L399→L398、GenerateBaseServing L120→L118 + 类锚 L113、scheduler 构造 L72→L70）
  并补齐 lint_fidelity 要求的 29 个 per-def 锚。
- **两处忠实性修复**（继续前次中断的运行时缺口）：① `MsgpackDecoder.ext_hook` 原实现
  返回 `Ext(code, data)`——真码（serial_utils.py:L473-476）把 RAW_VIEW 解包为**裸
  memoryview** 返回，已改逐字 + pickle 回退 SUBTRACTED；② seam `_from_builtin` 缺
  NamedTuple 解码面，`EngineCoreOutput.new_logprobs` 解码成裸 list——已按真 msgspec 语义
  补（位置数组→字段注解→dec_hook）。
- `python -m pytest tests/ -q` → **90 passed**；`python scripts/lint_fidelity.py 本章目录`
  → **全部通过**（含 must_keep 50 符号 over_subtraction 空账）。
- 测试面按机制覆盖 m1-m20 全部 20 个 mechanism + 14 站（见 tests/test_logprobs_lane.py
  头部机制图）。
- `logprobs_lane.py`/`_msgspec_seam.py` 均无 CRLF（`data.count(b'\r\n') == 0`）。

## 测试面（90 passed；断言的是 pin 可观察行为，非自洽）

- 入口：logprobs=True→top_logprobs、logprob_token_ids 优先分叉（logprobs=None + property
  并账 len）、echo 默认、stream→DELTA/FINAL_ONLY、True→1 归一、skip_reading_prefix_cache
  互斥、全关 None
- 批登记：num_logprobs 字典、-1→vocab_size、logprob_token_ids 登记、max 属性随弹出收缩、
  metadata 携带（req_id→req_index 键换算）
- 采样器：compute_logprobs==log_softmax fp32、raw 留底==变换前 logits 的 topk、None 全关、
  raw_logits/raw_logprobs 模式、processed_logprobs greedy 物化、-1 全词表直通、被采样恒
  第 0 列（含 k=1 落榜位）、计数 rank（并列取上界）、int64 断言、稀疏路三态（列 0 采样/
  padding -inf/空字典 None/稀疏优先覆盖）
- D2H：CPU no-op、tolists→numpy、slice_request 行切与 cu 偏移、empty_cpu 预分配形状、
  async 输出 get_output（含 invalid 行清空与 None 面）
- prompt 支路：单块计算与交付（对照手 gather）、分块累积延迟交付（num_computed 推进 +
  in_progress 行缓存）、抢占跳过、raw_logits 分数
- 调度：按请求切行路由 client、无 logprobs 请求 None、prompt 张量归属
- 过线：msgpack round-trip（LogprobsLists numpy/LogprobsTensors torch 双形态）、钩子原生
  元组形状、None 字段小消息
- 到港：双路分派（sample+prompt 同一 output）、process_outputs 调用与 queue=None 列表面、
  幽灵 id 幂等
- 装配：tolist/去 token/累计（真 byte-fallback tokenizer）、逐步累计、批均一 k+1 按自己 k
  截断、tokenizer=None 全 NONES、dict 键去重（被采样赢 rank）
- U+FFFD：三字节汉字修复（中间 ""/完成位 "中"）、横向候选独立修（AD→中、E4→""）、
  clean prefix 剥离（vLLM 单测原例）、真不完整空串、normalization 最长公共前缀、
  中置 U+FFFD 不修、上下文 4 上界（decode 全 � 走满 1..4）
- 非增量去 token：明文路径、空列表、Metaspace 前导空格恢复（单/双 marker）、无 marker
  短路
- FlatLogprobs：append_fast 平行列表、getitem 现造 dict、None 空位、slice 重建平移、
  对象数恒 6（O(1) GC 账）、Sequence 迭代、setitem TypeError、rank 链（被采样 rank 链头/
  -1 全列/nested 首 None/flat vs nested 工厂）
- prompt 装配：形状恢复+扁平一次性去 token、首 None、无累计；pop 一次性（取走即空、
  再 pop 返 []、禁用返 None）
- 出口：DELTA 切尾+累计进 CompletionOutput、FINAL 全量、DELTA prompt pop 一次性/二次空、
  CUMULATIVE 承载
- OpenAI：bytes=token 的 UTF-8 字节（中→[228,184,173]）、-9999 钳底、top_logprobs 截断/
  return_all/-1 全返、content 逐位造（含 decoded/bytes/top）、缺步回退 decode+默认
  logprob、token_id 占位、decoded None→bytes None
- e2e：双请求共车（批均一 k+1）→ 采样 → D2H → 切行 → 过线 → process_outputs →
  CompletionOutput；被采样=argmax 且恒第一；top-2 对照手 gather
