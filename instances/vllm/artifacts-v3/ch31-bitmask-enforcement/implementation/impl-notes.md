# impl-notes — v3 ch31《约束解码 II：bitmask 落地》(Part VII)

标准代码章（非 primer）：对 vLLM **v0.27.1（6e448d0ea）** 的只做减法精简版——
同名、同结构、同控制流；只删 dossier `subtraction_plan.delete` 批准的 8 项，
`must_keep` 53 个符号全保留；每 def/class 标 `# SOURCE: vllm/...:Lxxx`（行号
全部对 v0.27.1 现核，未沿用 v2 资产旧行号），删除处标 `# SUBTRACTED:`。

测试 host `python -m pytest`：**105 passed + 1 skip**（skip = 真 xgrammar 落地
kernel 测试，host 无 xgrammar）；GPU 容器（vllm/vllm-omni:latest，xgrammar
+ Triton 在场）：**106 passed 全绿**（含 V1 真 xgr.apply_token_bitmask_inplace
端到端与 V2 自写 Triton kernel 端到端）。`lint_fidelity` 无 BLOCKING。

**载体注意**：真实 StructuredOutputManager 住在 vllm/v1/structured_output/
__init__.py；镜像将类体放在同包 `manager.py`、`__init__.py` 退化为同款
re-export（fidelity lint 不扫 __init__.py；v2 ch31/ch32 的
structured_output_manager.py 同款处理）。类内全部 # SOURCE 锚点仍指向真实
__init__.py 的 v0.27.1 行号。

## 文件清单（implementation/ 下与真实仓库同构镜像）

**本章脊柱（subtract-only 主角，14 件）**

| 精简版文件 | 真实源 | 保留切面 |
|---|---|---|
| `vllm/v1/structured_output/manager.py`（__init__.py re-export） | structured_output/__init__.py | StructuredOutputManager 全部批装配面：L57-L68 缓冲+填充线程池、L99-L112 _get_reasoner、L194-L210 单行语义、L212-L359 grammar_bitmask 主函数（预算/并行/串行 spec 窗口/裁剪/.numpy()）、L361-L486 思考门控三件套、L488-L490 clear_backend |
| `vllm/v1/structured_output/utils.py` | structured_output/utils.py | L86-L175 apply_grammar_bitmask（V1 payoff：重排→pinned H2D→xgr） |
| `vllm/v1/structured_output/backend_types.py` | 同名 | **整文件逐字**（六方法契约，ch30 已立本章消费；仅插入 13 处 # SOURCE 标记：3 个类型/类 + 9 个方法 + 文件头） |
| `vllm/v1/core/sched/scheduler.py` | core/sched/scheduler.py | L1317-L1343 _update_after_schedule（置位门控）、L1646-L1668 get_grammar_bitmask、update_from_output 切面（L1761-L1791 spec 统计 + L1817-L1843 真推进逐字）、L2147-L2166 update_draft_token_ids、L2168-L2203 update_draft_token_ids_in_output、L2533-L2550 make_spec_decoding_stats |
| `vllm/v1/core/sched/async_scheduler.py` | 同名 | L13-L17 __init__ + L19-L49 _update_after_schedule（延后采样信号源） |
| `vllm/v1/core/sched/output.py` | 同名 | L286-L291 GrammarOutput 逐字 + SchedulerOutput 本章消费字段切面 |
| `vllm/v1/engine/core.py` | engine/core.py | L584-L614 step 四段、L616-L623 post_step、L625-L739 step_with_batch_queue（含 L719-L737 deferred 兑现链逐字）、L579-L582 throttle |
| `vllm/v1/worker/worker_base.py` | 同名 | L142-L157 两方法契约（docstring 自注技术债原文） |
| `vllm/v1/worker/gpu_model_runner.py` | 同名 | L437-L450 ExecuteModelState 逐字、L4165-L4175 状态防御、L4484-L4485 logits 产出、L4516-L4535 打包 return None、L4553-L4589 sample_tokens 第二幕、L3692-L3706 _sample 头段 |
| `vllm/v1/worker/gpu/structured_outputs.py` | 同名 | **整文件逐字**（V2 落地主体：copy_stream 双 H2D + cu_num_logits 映射 + Triton kernel；仅 InputBatch 导入 TYPE_CHECKING 化——纯注解消费） |
| `vllm/v1/worker/gpu/model_runner.py` | 同名 | L1143-L1175 sample() 调用点逐字（V2 先掩码后采样） |
| `vllm/v1/worker/gpu/spec_decode/utils.py` | 同名 | L11-L52 DraftTokensHandler（草稿 D2H 回传通道）+ L55-L70 get_parallel_drafting_token_id 逐字 |
| `vllm/v1/executor/uniproc_executor.py` | 同名 | L26-L42 AsyncOutputFuture、L85-L137 collective_rpc + 三方法转发 |
| `vllm/config/vllm.py` | config/vllm.py | L60-L100 架构名单逐字、L563-L658 num_speculative_tokens + use_v2_model_runner + 两判据 helper |

**HOST SEAM 支撑面（消费面最小承载，逐字或注明退化）**

- `vllm/logger.py`（init_logger/warning_once——once 方法 setattr 到 logger
  实例的机制逐字）
- `vllm/envs.py`（VLLM_USE_V2_MODEL_RUNNER 等三件，PEP 562 模块属性面）
- `vllm/platforms/__init__.py` + `vllm/utils/platform_utils.py`（PIN_MEMORY 派生链）
- `vllm/utils/torch_utils.py`（PIN_MEMORY + async_tensor_h2d 逐字——m11 搬运工）
- `vllm/utils/math_utils.py`（cdiv）/ `vllm/utils/import_utils.py`（LazyLoader 逐字）
- `vllm/triton_utils/__init__.py`（HAS_TRITON + tl/triton——ch29 同款 seam）
- `vllm/config/__init__.py`（六子配置消费字段 dataclass + VllmConfig re-export）
- `vllm/v1/utils.py`（record_function_or_nullcontext 逐字）
- `vllm/v1/outputs.py`（SamplerOutput/ModelRunnerOutput 消费切片/DraftTokenIds 逐字）
- `vllm/v1/request.py`（RequestStatus 逐字）/ `vllm/v1/serial_utils.py`（run_method）
- `vllm/v1/spec_decode/metrics.py`（SpecDecodingStats.new/observe_draft 逐字）
- `vllm/v1/worker/gpu/async_utils.py`（async_copy_to_np）/ `gpu/buffer_utils.py`（async_copy_to_gpu 逐字）

## 1:1 Source Map（脊校园地抽查行；全量以各文件就地标记为准）

| 精简版 | 真实源 | 改动 | 原因 |
|---|---|---|---|
| manager.grammar_bitmask L150-L297 | vllm/v1/structured_output/__init__.py:L212-L359 | 删 L253-L257/L278-L282 两处 TYPE_CHECKING 守卫断言 | delete[7] |
| utils.apply_grammar_bitmask L38-L135 | vllm/v1/structured_output/utils.py:L86-L175 | 删 L164-L175 CPU 兜底分支 | delete[4] |
| scheduler._update_after_schedule L57-L88 | vllm/v1/core/sched/scheduler.py:L1317-L1343 | 删 L1332-L1334 defer_block_free 栅栏、L1345-L1365 routed_experts | ch11/ch27（摘录 elide 注明 + delete[2] 同族） |
| core.step_with_batch_queue L137-L219 | vllm/v1/engine/core.py:L625-L739 | 删 is_ec_consumer/pooling/队列长度早退/abort/观测壳 | delete[2] |
| gpu_model_runner.execute_model L76-L121 | vllm/v1/worker/gpu_model_runner.py:L4165-L4535 | 删 L4177-L4483 执行臂 + L4528 kv_connector + L4530-L4533 纠偏回调 | delete[3]（前向归 ch18/19） |
| DraftTokensHandler.get_draft_tokens L75-L81 | vllm/v1/worker/gpu/spec_decode/utils.py:L45-L52 | 删 else 分支（[-1] 占位） | delete[5] |
| config.use_v2_model_runner L139-L186 | vllm/config/vllm.py:L577-L623 | 逐字（helper 裁剪见就地注记） | must_keep（m18） |

## 减法执行账（subtraction_plan.delete 八项逐一）

| delete | 内容 | 落点 |
|---|---|---|
| [0] 编译侧全部 | grammar_init/_create_grammar/四后端构造（L114-L192）、external_launcher 判定（L46-L55）、编译线程池+tokenizer+reasoning parser 装配（L70-L93） | structured_output/__init__.py；backend_xgrammar/backend_guidance 导入连带删。**注意 L95-L97 enable_in_reasoning 在 must_keep——delete 括注的『L70-L97』按 must_keep 优先保留 L95-L97** |
| [1] utils 编译侧 | compile_regex_with_timeout（L48-L83）+ Outlines 系（L178-L561） | structured_output/utils.py |
| [2] core.py 掩码无关分支 | EC consumer 记账、批队列长度调度（L682-L687 早退）、可观测性 with 块、abort 队列（含 _process_aborts_queue 本体） | engine/core.py；pooling 快路（L661）按同族『与掩码无关分支』并入 |
| [3] 两幕之外执行臂 | _prepare_inputs/注意力/cudagraph/前向/PP/pooling/EC、sample_tokens 尾部（L4591 起 drafter/bookkeeping） | gpu_model_runner.py；中段以 ENGINE SEAM 注入位承载（见下） |
| [4] CPU 后端兜底 | utils.py L164-L175（fp32 转换回写，#31901） | utils.py（删后 CPU 张量原样通过——本精简版固定演示 GPU 路径） |
| [5] DraftTokensHandler async 关闭分支 | get_draft_tokens 的 [-1] 占位 else | spec_decode/utils.py |
| [6] V2 runner 其余 | model_runner.py（V2）只留 sample() 调用点 | gpu/model_runner.py；StructuredOutputsWorker 整文件保留 |
| [7] 类型/观测样板 | TYPE_CHECKING 守卫断言（manager×3 处）、LazyLoader 样板（torch 改顶部 import；**xgr 的 LazyLoader 保留**——host 无 xgrammar，eager import 会炸 import 期，惰性语义与真实一致）、logger 调用（manager 侧无消费者） | 各文件就地 |

## ENGINE SEAM / 注入面（writer/explainer 须知）

1. **前向注入位**（gpu_model_runner）：真实 L4177-L4483 的前向产物
   `hidden_states` 以 `runner._seam_hidden_states` 注入（ch12 同款注入位）；
   十元组中 spec/PP/connector 系字段在无投机、单卡、无 connector 的演示部署下
   真实值即 None。`model/sampler/input_batch` 同为注入面（compute_logits/
   logits_indices/sampling_metadata 消费面）。
2. **grammar 注入**（dossier 口径『直接注入已构造的 grammar 对象』）：
   `manager.backend = FakeBackend(...)` 承载 allocate_token_bitmask；请求挂
   `structured_output_request.grammar`。编译链归 ch30。
3. **reasoner 注入**（delete[0] 连带）：reasoning parser 装配（L81-L93）删除后，
   思考门控（m14/m15）消费的 `manager.reasoner_cls` / `manager.tokenizer` 两属性
   按 grammar 同口径外部注入（`_get_reasoner` L104-L111 惰性构造位原样保留）。
   **deviation 说明**：delete[0] 把 L70-L97 整段记为编译侧装配，但其中
   L81-L93 的 reasoning parser 装配是思考门控的唯一开启器——本实现按
   『只删 L70-L93 中已列项、保留代码路径可经注入工作』执行，不保留装配行本身
   （理由：门控的运行语义完整保留在 _get_reasoner/should_*，装配只是生产者；
   与『grammar 直接注入』同一处理）。
4. **Executor/Scheduler 注入**（engine/core.py __init__）：model_executor/
   scheduler/structured_output_manager 由外部注入（真实装配 L74-L143 归 ch09）。
   测试以 Spy/Fake 承载（见 tests/conftest.py）。

## HOST SEAM 偏差清单（不影响控制流语义的承载性退化）

- `vllm/platforms`：is_pin_memory_available/is_rocm 以 torch 可用性承载（真实
  CUDA 平台还有 WSL 内核校验）。
- `config/__init__.py` 六子配置：消费字段子集 + dict 直建入口（真实装配归
  ch03）；`_get_v2_model_runner_unsupported_features` 只保 PCP/spec 方法两判据
  （compilation_config 深层判据删，就地注明）。
- `v1/outputs.py` ModelRunnerOutput：req_ids/req_id_to_index/sampled_token_ids
  三字段面（logprobs/pooler/connector 系删）；`with_kv_conn_output_only` 最小
  承载（早退分支已删，无消费者）。
- `vllm/tokenizers` / `vllm.reasoning` 包**不建**（编译侧装配删除后无消费者）。

## 测试布局（tests/，行为基准=真实 v0.27.1 可观察行为）

- `test_grammar_bitmask_assembly.py` — 批装配全貌（预算/并行 16 分块/串行 spec
  窗口/-1 哨兵时序/整行 -1/残留清理/diffusion bonus 跳过/裁剪）。
- `test_thinking_gates.py` — 门控三件套 + 窗口内思考结束检测（#42452/#43388/
  #44006 行为守护；flip 后容忍拒绝、bonus 双触发、reasoning_ended 不持久化）。
- `test_scheduler_bitmask.py` — 门控置位（prefill chunk 排除）/行序账本/草稿
  validate+(-1) 补齐+num_invalid 记账/真推进与 FINISHED_ERROR。
- `test_apply_grammar_bitmask.py` — V1 payoff：worked example 重排（批序
  [B,A]×spec 偏移→out_indices=[4,0,1,2,3]）/skip 快路径/meta 设备管线冒烟/
  CUDA 真 xgr 端到端（容器）。
- `test_two_act_window.py` — 两段式契约：状态防御/解包即清/先掩码后采样/
  logits 原地所有权/step 四段次序/UniProc 转发。
- `test_async_deferred_chain.py` — pending 置位/占位账/-1 占位数组/deferred
  兑现链因果序（take_draft→update_draft→bitmask→sample_tokens）。
- `test_v2_runner_path.py` — use_v2_model_runner 十路选择器（env/PCP/dspark/
  diffusion/稠密默认 V2/MoE 名单/pooling/无 Triton/ngram 回退）+ V2 worker
  Triton kernel 端到端（cu_num_logits 映射 + 词表尾谓词）+ V2 sample 次序。
- `test_draft_return_channel.py` — DraftTokensHandler 门控/D2H 往返/
  record_stream；AsyncOutputFuture 只在 result() 等待。

## 复核记录（关键语义与真实源逐字对照）

- xgrammar 位/索引语义以 **xgrammar 0.2.6 安装包源码**核（apply_token_bitmask_
  inplace_kernel_indices_torch：`bitmask[indices]` 的位=0 → 该行 -inf——bitmask
  与 logits 同形、indices 选行）：V1 路径 logits 形 sorted_bitmask + 子集
  indices 的用法与之吻合；V2 紧凑行版（kernel `bitmask_idx=program_id(0)` ↔
  `logits_indices[bitmask_idx]`）同一语义的行配对形态，其不变式由
  `assert num_masks == len(mapping)` 机器可查。测试 FakeXgr 按 0.2.6 torch
  kernel 逐字对齐（host 安装包仅在验证期间临时安装、已卸载——容器内为真路径）。
- v0.27.1 行号：所有 `# SOURCE:`/dossier 锚点在 instances/vllm/source（git
  6e448d0ea）逐行现核。
