# impl-notes — v3 ch34《投机解码 vLLM 落地》(Part VII)

标准代码章（非 primer）：对 vLLM **v0.27.1（6e448d0ea）** 的只做减法精简版——
同名、同结构、同控制流；只删 dossier `subtraction_plan.delete` 批准项，
`must_keep` 全保留；每 def/class 标 `# SOURCE: vllm/...:Lxxx`（行号对
v0.27.1 逐文件现核，v2 资产旧行号未沿用），删除处标 `# SUBTRACTED:`。
测试 host `python -m pytest` 全量可跑：**61 passed**（RTX PRO 6000 Blackwell
在场，三个 Triton kernel + numba @njit(parallel=True) 全部真跑、零 skip；
连跑 3 轮无抖动）。

## 文件清单（implementation/ 下与真实仓库同构镜像）

**本章脊柱（subtract-only 主角，四个文件）**

- `vllm/v1/sample/rejection_sampler.py` —— RejectionSampler（m5-m11）：v0.27
  组合持有普通 Sampler 采 bonus 位、双 Triton kernel、recovered 残差
  Gumbel-max、parse_output 还原变长、逐草稿位约束扩展。
- `vllm/v1/spec_decode/metadata.py` —— SpecDecodeMetadata 契约容器
  （m4 的数据面；**零删减逐字**，含 make_dummy）。
- `vllm/v1/spec_decode/ngram_proposer.py` —— NgramProposer 零模型 drafter
  （m12）：KMP/LPS 算法体逐字，@njit/@jit 装饰器保留（host numba 0.66 真跑）。
- `vllm/v1/spec_decode/llm_base_proposer.py` —— SpecDecodeBaseProposer
  契约骨架（m13）：EAGLE 主路径（首前向→k=0 空草稿/单步早退/自回归多步链式）
  + `_greedy_sample` 三支路 + 概率化 draft_probs 链。

**HOST SEAM 支撑面（消费面最小承载，逐字或注明退化）**

沿 ch30 已验证的采样栈镜像（同 pin v0.27.1、同书）+ 本章消费面回补：

- `vllm/v1/sample/sampler.py` —— 组合持有的普通 Sampler（bonus 位外采）。
  **回补 ch30 删掉的 spec 面**：`_combine_outputs_with_spec_tokens`
  （L358-L369 逐字——与 RejectionSampler 同名方法语义不同：这边拼整段
  spec、逐请求一行；那边造逐草稿位前缀行。读代码陷阱，测试对照双实现）、
  predict_bonus_token 合并段（L382-L388）、thinking holder 调用块
  （L404-L416，holder 恒 None 跳过）。
- `vllm/v1/sample/metadata.py` —— SamplingMetadata 冻结快照。**回补**
  `spec_token_ids`（L51-L52）与 `thinking_budget_state_holder`（L53-L55）
  两字段；ThinkingBudgetStateHolder import 不载（归 ch30，注解在
  `from __future__ import annotations` 下惰性）。
- `vllm/v1/sample/ops/bad_words.py` —— **回补** apply_bad_words_with_drafts
  （L39-L58 逐字）+ apply_bad_words（组合 Sampler 的 bonus 路径消费）。
- `vllm/v1/sample/logits_processor/builtin.py` —— **回补**
  MinTokens.add_request（L188-L195）/ update_state（L197-L224）/
  **apply_with_spec_decode（L235-L286 逐字，must_keep）** 与
  process_dict_updates（L289-L327）。MinP/LogitBias 的 update_state 沿
  ch30 镜像保持删除（本章无消费位）。
- `vllm/v1/sample/logits_processor/interface.py` —— **回补** BatchUpdate 族
  （L17-L57 逐字：MoveDirectionality/三类型别名/BatchUpdate）——MinTokens
  的 min_toks 填充走真实 update_state 面，测试不绕 API 自建内部状态。
- `vllm/config/__init__.py` —— HOST SEAM 字段面（ch30 版扩展）：
  ModelConfig/ParallelConfig/SpeculativeConfig/DraftModelConfig
  （get_hidden_size()/get_inputs_embeds_size()/hf_config 方法位）+
  SchedulerConfig.max_num_batched_tokens——ngram 与 drafter 契约骨架的
  构造消费面。
- 其余纯 seam（ch30 原样拷贝，未改动）：`vllm/{__init__,envs,logger,
  sampling_params,_custom_ops}.py`、`vllm/config/model.py`、
  `vllm/platforms/__init__.py`、`vllm/triton_utils/__init__.py`、
  `vllm/utils/{torch_utils,platform_utils,math_utils}.py`、
  `vllm/model_executor/layers/utils.py`、`vllm/v1/outputs.py`、
  `vllm/v1/attention/backends/flashinfer.py`、
  `vllm/v1/sample/ops/{topk_topp_sampler,topk_topp_triton,penalties,
  logprobs}.py`、`vllm/v1/sample/logits_processor/{__init__,state}.py`。

## 减法执行账（subtraction_plan.delete 六项逐一）

| delete | 内容 | 落点 |
|---|---|---|
| [0] synthetic 模式全链 | `unconditional_to_conditional_rates` import（L24）、`__init__` L77-L90（synthetic_conditional_rates/synthetic_mode）、forward 透传（L182-L183）、rejection_sample 形参（L409-L410）与 kernel 调用三/两参（L464-L466/L503-L505）、uniform 生成条件的 synthetic 句（L440-L445 只留 not all_greedy）、greedy kernel 的 uniform_probs_ptr/synthetic_conditional_rates_ptr 形参+SYNTHETIC_MODE 分支（L723-L724/L748-L754）、random kernel 的 synthetic_conditional_rates_ptr+SYNTHETIC_MODE（L786/L788/L812-L814） | rejection_sampler.py |
| [1] logprobs 装配 | `_get_logprobs_tensors` 整方法（L203-L250）、forward 尾部计算（L187-L196；L200 改传 None——**保留返回签名**）、parse_output 过滤分支+logprobs_tensors 形参（L277-L280）、import 的 LogprobsLists/LogprobsTensors（L16） | rejection_sampler.py（parse_output 返回二元组、第二元恒 None——真实调用面 gpu_model_runner.py:L3791 解包两值） |
| [2] thinking budget 支路 | apply_logits_processors 的 holder 调用块（L339-L345） | rejection_sampler.py |
| [3] llm_base_proposer 旁路 | mrope/xdrope（buffer 两分支 L175-L194、_get/_set_positions 分支、propose L629-L632、L854-L855）、多模态（L146-L150/L715-L722/L968-L978）、cudagraph padding（_determine_batch_execution_and_padding L1797-L1839 及调用位——精简版 input_batch_size==batch_size）、DFlash extra-slots（L106-L127 的槽位计算、L216-L257、set_inputs_first_pass else 分支 L861-L960、_init_parallel_drafting_params、prepare_next_token_ids_padded/prepare_inputs_padded）。**计划所指 tree attention 分支 v0.27.1 本文件已不存在**（该谱系位由 V2 树承担），已在文件头注明 | llm_base_proposer.py |
| [4] ngram numba 线程自适应与 JIT 预热 | __init__ L34-L53（阈值/cpu_count/线程数计算）与 L55-L62（JIT 预热调用）、batch_propose L97-L108/L122-L123（线程选择/恢复）、import os+get/set_num_threads | ngram_proposer.py——**@njit/@jit/prange 装饰器不在批准清单，原样保留**（host numba 真跑，与 ch30「只删批准项」同纪律） |
| [5] 非 精简范围声明 | gpu_model_runner.py / scheduler.py / core.py / structured_output/* 的 spec 段不做精简版 | 排批环/三组 index 摊平算术（_calc_spec_decode_metadata）/语法 rollback 以正文内嵌真源码解读；`metadata.py` 文件头有所有权环五锚的指路注记（propose_draft_token_ids/take_draft_token_ids/update_draft_token_ids/scheduled_spec_decode_tokens/num_spec_tokens_to_schedule） |

**llm_base_proposer 的第二类减法（引擎栈未内嵌，沿 v2 ch34「契约骨架」先例
+ delete[5] 对编排栈的处理，文件头有完整清单）**：注意力元数据构建
（build_per_group_and_layer_attn_metadata/_update_positions_dependent_metadata/
prepare_inputs 族）、EPLB、ROCm allowed_attn_types、模型加载与 embed/lm_head
权重共享（load_model/_maybe_share_*，正文内嵌解读——WC1 cost③ 显存账）、
cudagraph dispatcher（initialize_cudagraph_keys/dummy_run——PIECEWISE-only
契约正文内嵌 L411-L426）、_create_draft_vllm_config/_get_model。
保留 propose 主路径控制流：eagle3 combine → set_inputs_first_pass（左移
一格+末槽插 next_token）→ 首前向 → k=0 空草稿（动态 K）→ 单步/parallel
早退 → 自回归多步链式（上一步草稿 int() 喂回）→ stack [B,k]。

## HOST SEAM 偏差清单（writer/explainer 须知）

1. **config**：SpeculativeConfig/DraftModelConfig 是字段面载体（真实为
   ch03 域装配链数千行）；DraftModelConfig.get_hidden_size() 等以方法位
   直存标量。真实 SpeculativeConfig 校验链（drafter 谱系注册表）不在场。
2. **thinking_budget_state**：字段保留、import 不载（归 ch30）；本章
   RejectionSampler 侧 holder 块已按计划删，Sampler 镜像侧的 holder 块
   逐字保留但 holder 恒 None 整块跳过。
3. **parse_output 返回签名**：logprobs_tensors 形参已按计划删，但返回
   仍是 `(outputs, output_logprobs)` 二元组（第二元恒 None）——与真实
   调用面 gpu_model_runner.py:L3791 的解包一致，plan 明令「保留 forward
   的返回签名」的同族处理。
4. **flashinfer**：conftest 默认 VLLM_USE_FLASHINFER_SAMPLER=0（ch30 同款
   确定性 seam）；host 未装 flashinfer，bonus 位的 top_k/top_p 走
   forward_native 的 sort+Gumbel 路径。
5. **llm_base_proposer 可运行性**：构造期只消费 config 字段面与三块
   预分配 buffer（input_ids/positions/hidden_states）；`self.model` 由
   runner 装配后注入（真实装配在 gpu_model_runner L583-L658，正文内嵌），
   测试以固定映射替身模型驱动主路径（链式喂回可解析预测 f(x)=
   argmax(embed(x)@W)）——替身是测试协作者，精简版代码不含任何伪模型。

## 1:1 Source Map（精简版 ↔ vllm/...（v0.27.1 现核）↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码锚 | 改动 | 原因 |
|---|---|---|---|
| `RejectionSampler.__init__` | vllm/v1/sample/rejection_sampler.py:L61-L90 | 删 L77-L90 synthetic 构造 | delete[0]；spec_config=None（生产默认）不触发 |
| `RejectionSampler.forward` | 同上:L92-L201 | 删 L182-L183 透传、L187-L196 logprobs 计算（L200 改传 None） | delete[0]/[1]；bonus 外采/target fp32+clone/约束主线逐字 |
| `RejectionSampler.parse_output` | 同上:L252-L287 | 删 L277-L280 与 logprobs_tensors 形参 | delete[1]；valid_mask 过滤逐字 |
| `RejectionSampler.apply_logits_processors` | 同上:L289-L346 | 删 L339-L345 holder 块 | delete[2]；penalties repeat/bad_words_with_drafts/MinTokens.apply_with_spec_decode 逐字 |
| `RejectionSampler.apply_penalties` | 同上:L348-L374 | 无（逐字） | repeat_indices 展开到草稿位 |
| `RejectionSampler._combine_outputs_with_spec_tokens` | 同上:L376-L391 | 无（逐字） | 逐草稿位前缀行（与 Sampler 同名异义） |
| `rejection_sample` | 同上:L394-L507 | 删 synthetic 形参/条件/透传（见减法账 delete[0]） | 输出 buffer 预填 -1/all_greedy 早退/recovered 预采/双 kernel 同批共存逐字 |
| `apply_sampling_constraints` | 同上:L510-L565 | 无（逐字） | 温度/top-k/top-p 逐草稿位扩展（greedy 0→1 防除零） |
| `expand_batch_to_tokens`+`expand_kernel` | 同上:L568-L605/L848-L869 | 无（逐字） | [batch]→[num_tokens] kernel 化展开 |
| `generate_uniform_probs` | 同上:L608-L660 | 无（逐字） | float64 uniform（pytorch#16706）/n=0 跳过/seeded 覆写 |
| `sample_recovered_tokens`(_kernel) | 同上:L663-L710/L872-L953 | 无（逐字） | 每请求一份 q 的 v0.27 布局+免归一化 argmax+OOV clamp |
| `rejection_greedy_sample_kernel` | 同上:L713-L769 | 删 uniform_probs_ptr/synthetic 两形参+SYNTHETIC_MODE 分支 | delete[0]；标准 greedy 准则逐字 |
| `rejection_random_sample_kernel` | 同上:L772-L845 | 删 synthetic_conditional_rates_ptr/SYNTHETIC_MODE | delete[0]；接受判据 L829 逐字 |
| `SpecDecodeMetadata`(make_dummy) | vllm/v1/spec_decode/metadata.py:L9-L66 | 无（整文件逐字） | must_keep 核心数据结构 |
| `NgramProposer.__init__` | vllm/v1/spec_decode/ngram_proposer.py:L13-L62 | 删 L34-L53 线程计算+L55-L62 JIT 预热 | delete[3]；配置派生/buffer 预分配逐字 |
| `NgramProposer.batch_propose` | 同上:L64-L133 | 删 L97-L108/L122-L123 线程切换 | delete[3]；空列表卫兵/结果回读逐字 |
| `NgramProposer.propose`/`load_model` | 同上:L135-L170/L172-L174 | 无（逐字） | 有效性过滤（空 sampled/达 max_len 跳过） |
| `batch_propose_numba`/`_find_longest...` | 同上:L177-L203/L206-L293 | 无（逐字，含装饰器） | KMP/LPS 算法体——计划未批装饰器删除 |
| `SpecDecodeBaseProposer.__init__` | vllm/v1/spec_decode/llm_base_proposer.py:L69-L318 | 契约骨架（保留主路径字段派生+三 buffer；删两类未载栈，见减法账 delete[3] 与引擎栈清单） | drafter 契约面：k/method/hidden_size/采样开关/probabilistic 两态 |
| `SpecDecodeBaseProposer._greedy_sample` | 同上:L428-L438 | 无（逐字，三支路全保） | must_keep：local argmax 归约/异词表映射/主路径 |
| `_sample_from_logits`/`_sample_draft_tokens`/`take_last_draft_probs` | 同上:L440-L466/L468-L497/L499-L500 | 无（逐字） | 可选概率化 draft_probs（m13） |
| `SpecDecodeBaseProposer.propose` | 同上:L502-L767 | 契约骨架（控制流保留；删 attn 元数据/cudagraph padding/eplb/mm/rejected-mask，逐处 SUBTRACTED） | EAGLE 主路径：首前向→k=0/单步早退→链式→stack |
| `set_inputs_first_pass`（默认 EAGLE 分支） | 同上:L821-L860 | 删 xdrope 两行+extra-slots else 分支（L861-L960） | delete[2]/[3]；左移一格+末槽插 next_token 逐字 |
| `build_model_inputs_first_pass`/`model_returns_tuple` | 同上:L962-L991/L1007-L1015 | 删 mm 分支（前者） | 纯 input_ids 主路径逐字；mtp 按架构判逐字 |
| `compute_probs_and_sample_next_token` | 同上:L1848-L1886 | 无（逐字，含头部 NOTE） | 概率化 draft_probs（温度除→softmax→Exp(1) Gumbel-max） |

## 测试与运行

- host：`python -m pytest instances/vllm/artifacts-v3/ch34-spec-decode-implementation/tests -q`
  → **61 passed**（本机 `python`=Miniconda 3.11.11；torch 2.11.0+cu128、
  CUDA 可见（RTX PRO 6000 Blackwell）、triton 3.7.1、numba 0.66.0、
  flashinfer 未装——conftest 默认禁用支路）。三个 Triton kernel 与
  @njit(parallel=True) numba 批函数真跑；连跑 3 轮全绿无抖动。
- 结果台账：`tests/test-report.json`。
- 行为基准全部对真实源码现核：接受判据 `draft_prob > 0 and target_prob /
  draft_prob >= uniform_prob`（L829）、NO_DRAFT_PROBS 退化以 p_t(x) 接受
  （L816-L817）、padded draft(-1) 直接拒（L809-L811）、greedy 拒绝位写
  target argmax（one-hot 残差唯一 token）、全收补 bonus 于第 num_draft 位、
  ngram 残差=屏蔽 draft token 的 p_t（L913-L918）、免归一化 argmax 注释
  （L931-L932）、OOV mask −inf+clamp（L940-L952）、float64 uniform
  （L639-L648）、num_draft=0 请求 bonus 写第 0 位（kernel 循环 0 次+
  not rejected）、MinTokens.apply_with_spec_decode docstring 工例
  （num_draft=[2,3,1]→cumsum=[0,2,5,6]、n_mask=min(remaining, num_draft)）。
- 统计律测试：NO_DRAFT_PROBS 接受率≈p_t(x)（2 万样本）、
  min(1,p_t/p_d) 两端（全收/0.5）、**k=1 输出分布 TV<0.03 对拍 p_t
  （无损性——本章核心论断）**；确定性对拍：per-request generator 种子
  固定下 rejection_sample 输出与纯 torch 参考实现（min(1,p_t/p_d) 接受+
  残差 argmax 恢复）逐位一致。
