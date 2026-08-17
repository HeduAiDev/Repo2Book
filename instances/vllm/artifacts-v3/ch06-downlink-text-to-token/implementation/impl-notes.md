# ch06 精简版 impl-notes — 下行：从文本到 token（Part II：API 进程下行泳道）

- **Pin**：vLLM v0.27.1（`6e448d0ea`）。全部 `# SOURCE:` 行号对当前 pin 现核（写作当日又机械审计
  49 个核心锚点：路径存在、区间内确含所指符号——零失配；未照抄 v2 资产的 v0.21.0 旧行号）。
- **产物**：`implementation/downlink.py`（单模块精简版 ~4850 行）+ `implementation/_msgspec_seam.py`
  （msgspec 宿主替身，从 ch05 同名 seam 复制 + 本章补 enum 编解码一枚补丁，见 §已知偏差 6）。
  host 可跑（真 torch / 真 msgpack / 真 ThreadPoolExecutor / 真 asyncio；无 vllm 包、无 msgspec 包
  ——host 不允许 pip 安装，CLAUDE.md 硬规则 6）。
- **跑法**：`cd instances/vllm/artifacts-v3/ch06-downlink-text-to-token && python -m pytest tests/ -q`
  → **57 passed**（~2s）。`python scripts/lint_fidelity.py <本章目录>` → 无 BLOCKING。
- **验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本模块（HOST SEAM 例外见 §Seam
  清单——每个 seam 行内标注并在此登记）。

## 本章主题 = 全真部分（与 ch04/ch05/ch07 的分工）

- **渲染层四步流水**（m1）：render_prompt/render_messages → tokenize → extras → process_for_engine，
  chat 与 completion 同构；async 版全链 `asyncio.gather`（base.py:L985-L1109 逐字 minus 批准删除项）。
- **双线程池**（m3）：`_executor = ThreadPoolExecutor(renderer_num_workers)`（config/model.py:L355
  默认 1）承担 tokenize/decode/embeds 加载；`_mm_executor = ThreadPoolExecutor(1)` 单工承担 mm
  预处理——注释原话 "must stay single-worker per #38418 (P0/P1 order)" 逐字（base.py:L82-L98）。
- **池卸载与分流**（m2）：`InputProcessor.__init__` 的
  `process_inputs_async = make_async(process_inputs, executor=renderer._executor)`（PR #49608，
  input_processor.py:L77-L82 含注释原话）；`AsyncLLM.add_request` 按「输入是否已渲染」分流：
  dict 带 `'type'` → 同步 `process_inputs`（"no blocking preprocessing needed"）；raw prompt →
  `await process_inputs_async`（"must not block the event loop"）——两段注释原话逐字
  （async_llm.py:L352-L380）。`make_async` 本体逐字（utils/async_utils.py:L28-L45，含
  "The code in this function needs to be thread safe" docstring）。
- **OpenAI 门面**（站 1-2）：OnlineRenderer.render_chat 的 tool_choice 可用性校验 + chat 模板信任
  （trust_request_chat_template）→ preprocess_chat → render_chat_async（online_renderer.py:L117-L218
  common case + L379-L477）。
- **InputProcessor 校验链与组装**（站 6-9）：`_validate_params`（GENERATION/POOLING 任务路由）→
  `_validate_lora` → data_parallel_rank 界检查 → 平台校验（current_platform 接口位）→
  split_enc_dec_input → 三段长度校验（空/超长/等长必炸）→ vocab 越界
  `max(tokenizer.max_token_id, model_vocab_size-1)` → 编码器缓存预算；`params.clone()` 补全
  （max_tokens = max_model_len − seq_len、update_from_generation_config 注入 eos、
  update_from_tokenizer 展开 bad_words——三方法自 sampling_params.py:L646-L715 逐字）；mm 展平
  （argsort_mm_positions 按 offset 排序 dict-of-list → list[MultiModalFeatureSpec]）；构造
  EngineCoreRequest（无 prompt 文本字段，#11963）。
- **双轨 id**（m5）：assign_request_id 逐字（input_processor.py:L231-249）——`external_req_id ←
  用户 id`，`request_id = f"{external}-{random_uuid():.8}"`（8 hex）；VLLM_DISABLE_REQUEST_ID_RANDOMIZATION
  逃生舱保留（警告一行化）。`random_uuid` 逐字（utils/__init__.py:L11-L12）。
- **出港**（m11）：`_add_request` 双登记（先本进程 OutputProcessor、后跨进程 EngineCore，
  async_llm.py:L420-L435 两行注释逐字）→ `AsyncMPClient.add_request_async` 三行逐字盖
  client_index + ADD 帧（core_client.py:L1145-L1148）。**本章止于 _send_input 之前**：seam 客户端
  只记录帧标签，不触 ZMQ——帧序/编码是 ch05 的产品。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因；核心行）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `BaseRenderer.__init__` | vllm/renderers/base.py:L73-L153 | 双池装配 L82-L98 **逐字**；maybe_init_mm_gpu_ipc_pool（L114-L122）与 readonly processor（L136-L146）SUBTRACTED | must_keep×2；m3 载体（delete 项 5） |
| `BaseRenderer._tokenize_prompt` / `_build_tokens_prompt` | base.py:L472-L487 / L451-L470 | 主体逐字；offsets 分支 SUBTRACTED | must_keep；m1/m2（delete 项 3） |
| `BaseRenderer._tokenize_singleton_prompt`(±async) | base.py:L502-L572 | 逐字（含三 overload） | tokenize 第 2 步本体 |
| `BaseRenderer.tokenize_prompt(s)(_async)` | base.py:L614-L648 | enc-dec 分支行 SUBTRACTED | must_keep×2（delete 项 7） |
| `BaseRenderer._apply_prompt_extras` | base.py:L650-L661 | **逐字** | 四步流水第 3 步 |
| `BaseRenderer._process_multimodal` | base.py:L729-L767 | 主体逐字；skip_mm_cache 分流（L739-742）与 mm uuid 调用行（L749-751）SUBTRACTED | must_keep；站 4（delete 项 4/5） |
| `BaseRenderer._process_tokens(±async)` / `_process_embeds` | base.py:L769-L866 / L805-L833 | 逐字；offsets 拷贝分支 SUBTRACTED；`.cpu()` 序列化注释逐字 | must_keep×2；m10 |
| `BaseRenderer.process_for_engine(±async)` | base.py:L945-L982 | enc-dec 分支 SUBTRACTED | must_keep×2；四步流水第 4 步 |
| `BaseRenderer.render_cmpl(±async)` / `render_chat(±async)` | base.py:L985-L1109 | **逐字**（arrival_time 首行打点保留） | must_keep×4；m1 |
| `OnlineRenderer.render_chat` | online_renderer.py:L117-L218 | mistral 序列化分支（L138-143）与 harmony 分支（L202-216）SUBTRACTED；tool_choice/模板信任校验与 common case 逐字 | must_keep；站 1-2（delete 项 1） |
| `OnlineRenderer.preprocess_chat` | online_renderer.py:L379-L477 | decode-reuse 分支（L415-424）SUBTRACTED；其余逐字 | 站 2（delete 项 2） |
| `InputPreprocessor`（全类） | vllm/inputs/preprocess.py:L48-L291 | **逐字**（含 enc-dec 面） | must_keep；m4 raw 兜底 |
| `EngineCoreRequest` | vllm/v1/engine/__init__.py:L97-L154 | 字段表 minus reasoning_ended/reasoning_parser_kwargs/abort_immediately 三字段；client_index/external_req_id/current_wave/prompt_is_token_ids 注释逐字；`params` property 逐字 | must_keep；m4（delete 项 17） |
| `InputProcessor.__init__` | input_processor.py:L39-L82 | **逐字**（含 L77-82 注释原话） | must_keep×2；PR #49608 落点 |
| `InputProcessor._validate_params` | input_processor.py:L91-L153 | GENERATION 路由+verify、POOLING 不支持报错+verify 保留；thinking 分支（L111-126）与 task 默认（L134-146）SUBTRACTED | must_keep；m7（delete 项 9/10） |
| `InputProcessor.process_inputs` | input_processor.py:L251-L394 | 主体逐字；deprecation 文案一行化 | must_keep；站 6-9 |
| `InputProcessor.assign_request_id` | input_processor.py:L231-L249 | **逐字**（f-string 机制本体 L249 原样） | must_keep；m5 |
| `InputProcessor._validate_prompt_len` / `_validate_model_input(s)` | input_processor.py:L396-L505 | 三段 raise 语义全保；suggestion 文案与 Qwen3 长注 SUBTRACTED | must_keep×3；m7/m9 |
| `AsyncLLM.add_request` | async_llm.py:L283-L418 | 分流主干逐字（两段注释原话）；流式输入分支/kv_sharing 校验/reasoning 注入/n>1 fan-out SUBTRACTED；deprecation 文案一行化 | must_keep；m2 核心（delete 项 14/15/16） |
| `AsyncLLM._add_request` | async_llm.py:L420-L435 | **逐字** | must_keep；m11 双登记 |
| `make_async` / `random_uuid` / `length_from_prompt_token_ids_or_embeds` | utils/async_utils.py:L28-L45 / utils/__init__.py:L8-L36 | **逐字** | must_keep×3 |
| `argsort_mm_positions` / `PlaceholderRange` / `MultiModalFeatureSpec` | multimodal/utils.py:L145-L165 / multimodal/inputs.py:L121-L219 / L322-L365 | **逐字**（含 AAAA BBBB 教学例 docstring、is_embed/embeds_cumsum/get_num_embeds、gather_kwargs） | must_keep×3；m8/m9 |
| `SamplingParams.clone / update_from_generation_config / update_from_tokenizer` | sampling_params.py:L748-L753 / L646-L674 / L676-L715 | 三方法**逐字**（类本体为字段 seam） | must_keep；m6 |
| `split_enc_dec_input` / `build_enc_dec_input` / EngineInput 家族 | inputs/engine.py:L381-L387 / L287-L378 / L18-L284 | **逐字**（整文件 minus 无关字段） | must_keep×2；m10 |
| `ChatParams` / `TokenizeParams`（整文件） | renderers/params.py:L28-L496 | **逐字** | 渲染基础设施（未列删除项，全保） |
| renderers/inputs/{preparse,tokenize} | renderers/inputs/preprocess.py / tokenize.py | **逐字**（parse 族/prompt_to_seq/extract_prompt_components/PromptComponents） | 流水内部类型与解析 |
| `AtomicCounter` / `json_iter_leaves` / `is_list_of` / `set_default_torch_num_threads` / `is_mistral_tokenizer(±tool_parser)` | utils/counter.py:L21-L45 / utils/jsontree.py:L40-L45 / utils/collection_utils.py:L54-L64 / utils/torch_utils.py:L153-L177 / utils/mistral.py:L19-L38 | **逐字**（is_list_of 删 TypeIs 类型面） | 保留代码触到的真实小件 |

## 删除台账 — dossier subtraction_plan 17 项 delete（全部执行）

1. **OnlineRenderer mistral/harmony 特化** ✓ — mistral 序列化分支（L138-143）、GPT-OSS harmony
   分支（L202-216）、孤儿 helper `_make_request_with_harmony`（L220-267，仅被已删分支调用）。
2. **decode-side token reuse** ✓ — `_reused_prompt_token_ids`（L50-60）+ preprocess_chat 调用分支
   （L415-424）+ harmony 内调用（随项 1 整删）。
3. **offsets 机制** ✓ — `_can_produce_offsets`/`_wants_offsets`（L431-449）、`_tokenize_prompt` 的
   return_offsets_mapping 分支（L478-481）、`_build_tokens_prompt` offset 形参分支（L456-469）、
   `_process_tokens(±async)` 的 prompt_token_offsets 拷贝分支（L797-801/L860-864）、TokensPrompt/
   TokensInput 的 offsets 字段（inputs/llm.py:L115-L122、inputs/engine.py:L43-L45）。
4. **mm uuid 机制** ✓ — `_validate_mm_uuids`/`_process_mm_uuids`（L664-726）+
   `_process_multimodal` 调用行（L749-751）。`parse_mm_uuids`（parse.py:L740-746）保留——
   它不在批准区间内且 ProcessorInputs 签名需要它。
5. **readonly mm processor / skip_mm_cache / GPU IPC pool** ✓ — maybe_init_mm_gpu_ipc_pool 安装
   （L114-122）、readonly processor 创建（L136-146）、`_process_multimodal` 分流（L739-742）。
   `skip_mm_cache` 形参全链保留（签名保真），仅其消费分支删除。
6. **warmup** ✓ — `_warmup_mm_processor`+`warmup`（base.py:L214-283）。
7. **enc-dec 分支（base.py）** ✓ — `_tokenize_enc_dec_prompt(±async)`（L574-612）、
   `_process_enc_dec(±async)`（L890-943）、四处 `"encoder_prompt"` 分支行（L619-620/L636-637/
   L953-954/L970-974）。InputProcessor 侧 split_enc_dec_input 与 InputPreprocessor 的 enc-dec 面
   按批准保留。
8. **inject_into_mm_cache** ✓ — input_processor.py:L192-L229。
9. **thinking_token_budget/reasoning_config 校验** ✓ — L111-L126（含 use_v2_model_runner 子分支）。
10. **PoolingParams task 默认补全** ✓ — L134-L146（保留「不支持 pooling 报错 + params.verify」）。
11. **deprecation 警告文案** ✓ — input_processor L279-284/L291-295/L242-247、async_llm L338-350、
    _validate_lora L165-172：五处分支保留、文案一行化。
12. **suggestion 文案构造** ✓ — _validate_prompt_len L413-425/L432-436（三类 raise 全保）；mm_hashes
    错误文案压缩（L352-356，同项精神，校验语义保留）。
13. **Qwen3 vocab 长注释** ✓ — L481-L487（判定逻辑保留 + 一段短注）。
14. **流式输入分支** ✓ — add_request L319-334 + `_add_streaming_input_request`/
    `_validate_streaming_input_sampling_params`（L437-L538）+ InputStreamError 伴生。
15. **reasoning 注入 + kv_sharing_fast_prefill 校验** ✓ — L383-386 与 L308-317（签名形参保留）。
16. **n>1 fan-out** ✓ — L405-418（`if is_pooling or params.n == 1` 分支保留；n>1 落到函数尾部——
    **精简版 n>1 返回 None**，与「删掉 SUBTRACTED 分支后的真码」一致，ch07 讲扇出）。
17. **EngineCoreRequest 三字段** ✓ — reasoning_ended/reasoning_parser_kwargs（L139-140）+
    abort_immediately（L142-146）。

### 机械删除/替换（不在 delete 单——为可跑性与章边界所必需，**请 reviewer 逐条过目**）

| 位置 | 内容 | 理由 |
|---|---|---|
| OnlineRenderer.warmup（L108-115） | 随 base.warmup 删除的调用方 | 被删方法的唯一直接调用者，保留即悬空调用（项 6 的机械后果） |
| BaseRenderer.render_chat 两处 `list[list[...]]()` | 写作 `list()` | 纯类型注解差异，控制流不变 |
| hf.py（HfRenderer 整文件） | 未实现——render_messages 留抽象（raise NotImplementedError） | chat 模板引擎 = 本章显式黑盒（scope_note）；测试以 ChatRendererSeam 子类提供 step-1（契约同 HfRenderer.render_messages 返回 (conversation, DictPrompt)） |
| LLMEngine.add_request（llm_engine.py:L230-L278） | 未纳入（同步面） | 不在 must_keep/code_spine；同步泳道的主体 `process_inputs` 已实现，测试直接同步调用即复现「跑在调用方线程、无池」 |
| serving.py 交棒两行（chat_completion/serving.py:L217/L252） | 未纳入 | 站 1 的叙事锚点（writer 用 embed_excerpts）；实现以 OnlineRenderer.render_chat 直接进入 |
| `# type: ignore[...]` 注释 | 多数保留、个别省略 | 无 mypy 运行时语义 |

## Seam 清单（HOST SEAM，全在行内标注）

| Seam | 位置 | 是什么 | 为什么仍忠实 |
|---|---|---|---|
| `_msgspec_seam` | 独立文件（ch05 同款 + enum 补丁） | msgspec API 子集的 msgpack-backed 实现 | 线上字节是真 msgpack；array_like 全字段编码语义与真 msgspec 一致（ch05 容器实测三版） |
| config 族 | 模块头 | VllmConfig/ModelConfig/ParallelConfig/CacheConfig/LoRAConfig/MultimodalConfig 字段 seam（仅本章读到的字段） | ch03 装配线产物；`renderer_num_workers` 默认 1 与 config/model.py:L355 一致 |
| HF tokenizer + chat 模板 | 测试侧 `SeamWordTokenizer` + `ChatRendererSeam` | 确定性词级 tokenizer 与玩具模板 | tokenizer/模板引擎是外部依赖；被保真代码只依赖 TokenizerLike 鸭子面 |
| mm processor 黑盒 | impl 内 `BaseMultiModalProcessor` 等 | 确定性占位符展开（marker id → N 个占位 token）+ sha1 哈希 + processor cache 命中→(None, prompt_updates) | dossier scope：per-modality 处理黑盒，本章只展开 mm_features 形状与展平；cache.get_and_update_item 契约按 cache.py:L410-L422 形状实现（命中省 IPC 可被测试观察） |
| `AsyncMPClient` | impl 内 | ch05 产品的 seam：add_request_async 三行逐字 + `_send_input` 记录帧不触 ZMQ | 章边界：帧序/编码 ch05 已讲透，本章止于 _send_input 之前 |
| `OutputProcessor` / `RequestOutputCollector` / `ParentRequest` | impl 内 | ch04/ch07 产品的 seam（登记 RequestState、外→内映射、collector 按**内部 id** 建键） | 双登记顺序与 collector 键为可观察行为；上行回路归 ch07 |
| SamplingParams/PoolingParams | impl 内 | 字段 seam + clone/update_from_generation_config/update_from_tokenizer 三方法**逐字** | m6 must_keep 三方法本体保真；verify 的 7 个校验器是参数域非本章机制（no-op seam，已注明） |
| logger/envs/exceptions/platforms/parser/error_response | impl 内 | stdlib seam | ch04 同款；`warning_once` 去重注册表共享暴露（`_ONCE_SEEN`）供测试重置——真实 warning_once 就是进程级去重 |
| `renderer_from_config` | impl 内 | registry seam（默认 hf renderer、无 tokenizer） | 真实 tokenizer 装配是 registry 域；测试显式注入 renderer |

## 已知偏差（reviewer 重点）

1. **mm processor seam 的哈希/占位符是确定性的**：mm_hash = sha1(repr(item))[:16]（真 MultiModalHasher
   分块哈希）；占位 token 数由 `MultimodalConfig.seam_tokens_per_item` 配置（真实值 = 编码器特征
   尺寸，如 image 576）。**展平/排序/缓存命中语义与真码一致**，数值不同。
2. **n>1 返回 None**：fan-out 删除后 add_request 在 `params.n > 1` 时落函数尾（返回 None）——与
   「删掉批准分支后的真码」严格一致；ch07 讲扇出。测试不触碰 n>1。
3. **deprecation 文案一行化**：五处警告保留分支与「deprecated」关键词，完整原文在 pin 源码
   （delete 项 11 批准）。
4. **同步面 LLMEngine 未纳入**：process_inputs 同步调用即等价体验（测试覆盖）；LLMEngine.add_request
   的 fan-out/双登记同步版是 ch04 章域。
5. **SamplingParams.verify / PoolingParams.verify 为 no-op seam**：参数校验器族（logprobs/logit_bias/
   structured-output…）不是本章机制；任务路由校验（generation/pooling）在 _validate_params 里**全真**。
6. **_msgspec_seam 相对 ch05 版 +2 行**：plain `enum.Enum` 编解码分支（ch05 只处理 IntEnum；本章
   EngineCoreRequest 携带 RequestOutputKind）。msgspec 真行为：enum 按值编解码——一致。
7. **`test_mm_*` 依赖 seam marker 约定**：`<image>`→token 31、`<audio>`→32（SeamWordTokenizer 词表
   与 mm seam 的 `seam_marker_ids` 两侧约定）；这是测试装置耦合，不是实现耦合。

## 测试面（57 passed；断言的是 pin 可观察行为，非自洽）

- 四步流水顺序与产物（render→tokenize→extras→process；completion 同构；批量 gather 并行<串行）
- **tokenize 不占事件循环**（0.25s 阻塞 tokenizer 期间心跳协程持续 tick + 池线程 id ≠ loop 线程 id）
- 已渲染 EngineInput 快路径**零 tokenizer 调用**；raw prompt 走 renderer 池线程
- 双池分工（renderer_num_workers=4 → _executor 4 工；_mm_executor 恒 1 工）
- EngineCoreRequest 无 prompt 字段（AttributeError）+ 线字节含 token 数组不含文本 + array_like
  全字段数组 + round-trip
- 双轨 id：`^id-[0-9a-f]{8}$`、同外部 id 重试得不同内部 id、禁随机化开关保持两 id 相等（含警告）、
  预设 external_req_id 拒绝；collector 按内部 id 建键
- params 克隆隔离（caller 不被改）+ max_tokens 默认 = max_model_len−len(ids) + eos 注入 + bad_words
  展开 + pooling 克隆过线
- 校验链：空/超长/等长/词表越界（含 max(两侧) 判定）/dp_rank 越界/pooling-generation 互斥/
  params 类型/LoRA 未启用
- mm：交错 image+audio 按 offset 排序、同模态多 item 保序、**缓存命中 data=None**（第二次过线省
  载荷、哈希仍过线）、tower-connector LoRA 前缀 vs 普通 LoRA 裸哈希、编码器缓存预算前置拦截
- PlaceholderRange：AAAA/BBBB 教学例、is_embed 掩码 get_num_embeds、argsort 平铺排序
- embeds：(1,seq,h) 挤压为 (seq,h)、.cpu()、embeds 长度驱动 max_tokens、未启用报错；
  enc-dec：decoder_start_token 前插 + split 双侧校验（mm 注册口径同真实 enc-dec 模型）
- 出港：本地登记先于跨进程、client_index 盖章、ADD=b"\x00"、engine_dead → EngineDeadError
- arrival_time：渲染入口打点不被 process_inputs 时钟覆盖；raw prompt 无打点则用时钟
- OnlineRenderer 门面：auto/required tool_choice 拒绝文案、未信任模板拒绝、chat/completion 两面成功
