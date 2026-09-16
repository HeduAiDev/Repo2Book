# impl-notes — v3 ch29《Sampler 9 步管线》(Part VII)

标准代码章（非 primer）：对 vLLM **v0.27.1（6e448d0ea）** 的只做减法精简版——
同名、同结构、同控制流；只删 dossier `subtraction_plan.delete` 批准项，
`must_keep` 全保留；每 def/class 标 `# SOURCE: vllm/...:Lxxx`（行号对
v0.27.1 现核，v2 资产旧行号未沿用），删除处标 `# SUBTRACTED:`。
测试 host `python -m pytest` 全量可跑（82 passed + 1 容器专属 skip），GPU
容器（vllm/vllm-omni:latest，flashinfer 在场）83 passed 全绿。

## 文件清单（implementation/ 下与真实仓库同构镜像）

**本章脊柱（subtract-only 主角）**

- `vllm/v1/sample/sampler.py` —— Sampler 9 步编排（m1/m2/m3/m4/m15/m16）。
- `vllm/v1/sample/metadata.py` —— SamplingMetadata 冻结快照（m14 契约面）。
- `vllm/v1/sample/ops/topk_topp_sampler.py` —— TopKTopPSampler 构造期后端
  绑定（m11）+ 截断/掷骰分流（m9/m10/m12/m13）。
- `vllm/v1/sample/ops/topk_topp_triton.py` —— **整文件逐字保留**（must_keep
  原文；Qrita pivot 核，m12 只讲外部契约）——仅插入 4 处 `# SOURCE` 行标记。
- `vllm/v1/sample/ops/bad_words.py` —— step4 前缀匹配（m5）。
- `vllm/v1/sample/ops/penalties.py` —— step6 张量化入口 + −1 占位替换（m6）。
- `vllm/v1/sample/ops/logprobs.py` —— rank 不排序计数（m16）。
- `vllm/v1/sample/logits_processor/{__init__,interface,builtin,state}.py` ——
  argmax 不变性二分（m2）：两列分类容器、三件套构造器/声明/apply、
  build_logitsprocs（spec 互斥警告前指 ch32/33）+ 插件/FQCN 加载链。
- `vllm/model_executor/layers/utils.py` —— 惩罚真算式（m6：scatter_add_ 计数
  + OpenAI 两式 + repetition 自定义 op 调用）。
- `vllm/v1/outputs.py` —— LogprobsLists/LogprobsTensors/SamplerOutput 出件载体。

**HOST SEAM 支撑面（消费面最小承载，逐字或注明退化）**

- `vllm/platforms/__init__.py`（current_platform + DeviceCapability 逐字）
- `vllm/config/model.py`（LogprobsMode/PROCESSED_LOGPROBS_MODES 逐字）
- `vllm/config/__init__.py`（VllmConfig/SchedulerConfig 字段面——
  build_logitsprocs 与 MinP 构造器消费）
- `vllm/envs.py` / `vllm/logger.py` / `vllm/sampling_params.py`
- `vllm/utils/{torch_utils,platform_utils,math_utils}.py`
- `vllm/triton_utils/__init__.py`（HAS_TRITON 探测 + tl/triton 导出位）
- `vllm/_custom_ops.py`（apply_repetition_penalties 的 torch 参考算式）
- `vllm/v1/attention/backends/flashinfer.py`（FlashInferBackend.
  supports_compute_capability 逐字——构造期能力裁决的 import 面）

## 减法执行账（subtraction_plan.delete 六项逐一）

| delete | 内容 | 落点 |
|---|---|---|
| [0] logprob_token_ids 旁路 | gather_specific_token_logprobs 全方法 + forward 取值/消费段 | sampler.py；**保留** L113 `logprob_token_ids_tensors = None` 与 `if num_logprobs is None` 分支头（计划明令——默认路径靠它赋 None） |
| [1] thinking budget 整链 | 文件不创建 + sampler 调用块 L404-416 + metadata 字段/import | m7 登记轻讲，正文内嵌真源码（thinking_budget_state.py:L20-L81）不依赖精简版 |
| [2] spec decode 组合分支 | _combine_outputs_with_spec_tokens + predict_bonus_token 合并段 + metadata spec_token_ids + apply_bad_words_with_drafts + apply_with_spec_decode | sampler/bad_words/builtin/metadata |
| [3]（同 [0] 计划项内） | — | — |
| [4] 平台后端变体 | forward_cpu/_init_aiter_ops/forward_hip/aiter_sample/forward_xpu/compiled_random_sample/_skip_aiter_sampler_on_gfx1250/__init__ CPU·XPU·ROCm·兜底分支/apply_top_k_top_p CPU 分支/apply_top_k_only | topk_topp_sampler.py；**删后 self.forward 仅 is_cuda() 下绑定**（计划原文），forward_native/forward_cuda 两代表保留 |
| [5] 插件 API + [6] update_state 面 | cached_load/validate_params_fn/AdapterLogitsProcessor；BatchUpdateBuilder、三件套 update_state、add_request、process_dict_updates、interface 的 BatchUpdate 族类型与 update_state 抽象方法；__all__ 四名清理；各保留文件 import 连带修剪（计划明列） | logits_processor/* |

**注意**：`_to_tensor_scalar_tuple`（topk_topp_sampler.py:L515-L519）唯一
消费方 aiter_sample 已删，但它不在批准清单里——按「只删批准项」原样保留
（无行为影响，已注释说明）。`apply_top_k_top_p_pytorch` 内部的
`if allow_cpu_sync: return apply_top_k_only(...)` 死分支同理保留逐字
（allow_cpu_sync 恒 False 后不可达；apply_top_k_only 本体已按计划删）。

## HOST SEAM 偏差清单（writer/explainer 须知）

1. **platforms**：`is_cuda()` 用 `torch.cuda.is_available()` 等价探测（真实=
   加速器检测选类）；`get_device_capability` 用 torch 直询（真实走 NVML，
   cuda.py:L734-L742）；`simple_compile_backend="eager"`（真实 "inductor"，
   interface.py:L165——ch8 已立同款 seam，dynamo 仍 trace、数学不变）；
   `num_compute_units` CUDA 位逐字（cuda.py:L678-L679）+ CPU 位逐字
   （cpu.py:L423-424）。
2. **envs**：机制=字典+`__getattr__`（真实 L2059 起），但去掉 functools.cache
   （真实 L2092）——测试需要切换 `VLLM_USE_FLASHINFER_SAMPLER`，取值语义
   （默认 True / 显式 0-1）逐字一致。
3. **_custom_ops**：repetition 惩罚保留 torch 参考算式（_custom_ops.py:
   L309-L323 逐字），减去 torch.ops._C 编译核分派臂（L325-L333 及分派器
   CUDA 判断）——精简版无 vLLM 编译扩展；torch 算式即 CUDA 核的逐元素
   等价实现（正除负乘），数值不变。
4. **outputs**：只载 LogprobsLists/LogprobsTensors（filter/cat/empty_cpu
   三方法减去——D2H 装配面归 ch8）+ SamplerOutput。
5. **flashinfer backend**：只镜像类名 + supports_compute_capability
   （L461-L470 逐字，SM80-SM121 门槛）；flashinfer 本体仍由 flashinfer_sample
   内 `import flashinfer` 直连真实安装（容器内已验证可跑）。
6. **triton_utils**：HAS_TRITON=find_spec 双探测（importing.py:L13-L16）+
   未装时的最小占位（真实另有活跃驱动数校验 L18-L60）；未装 triton 时
   topk_topp_sampler 跳过 triton import——与真实回退语义一致。

## 非 CUDA 环境的绑定说明（重要）

减法后 `TopKTopPSampler.__init__` 只在 `is_cuda()` 下绑定 `self.forward`
（delete[4] 计划原文「删后 self.forward 仅 is_cuda() 下绑定」）。本类不定义
`forward` 方法（真实代码即如此——绑定的是实例属性），非 CUDA 环境下调用方
按构造期 native 绑定支路（L100-L102 的 else 位）显式取 `forward_native`；
测试的 `ensure_native_binding` 就是复现这一真实分支，不是发明。host 上
构造 Sampler 前若不设 `VLLM_USE_FLASHINFER_SAMPLER=0` 且 flashinfer 未装，
`forward_cuda` 支路会在掷骰时 `import flashinfer` 失败——这与真实 vLLM 的
行为一致（docstring：「Assumes flashinfer is installed, as guaranteed by
requirements/cuda.txt」）；conftest 默认设 0 走真实禁用支路（L45-L50）。

## 1:1 Source Map（精简版 ↔ vllm/...（v0.27.1 现核）↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码锚 | 改动 | 原因 |
|---|---|---|---|
| `sampler.Sampler.forward` | vllm/v1/sample/sampler.py:L72-L149 | 删 L111-112/L114-118/L133-136（logprob_token_ids 旁路三段），保 L113 与 `if num_logprobs is None` 头 | delete[0]；恒 None/空时旁路不执行；默认路径依赖保留的两行（计划明令） |
| `sampler.Sampler.apply_logits_processors` | 同上:L371-L417 | 删 L382-388（spec 合并）、L404-416（thinking budget） | delete[2]/[1]；非 spec 且未设 reasoning_config 时恒不触发 |
| `sampler.Sampler.sample` | 同上:L243-L302 | 无（逐字） | step7 全景 greedy 早退/温度/不变列/截断/torch.where 是本章核心 |
| `ops.topk_topp_sampler.TopKTopPSampler.__init__` | vllm/v1/sample/ops/topk_topp_sampler.py:L85-L129 | 删 L103-129 平台分支 | delete[4]；保留 CUDA+FlashInfer/native 两代表讲清绑定思想 |
| `ops.topk_topp_sampler.apply_top_k_top_p` | 同上:L349-L364 | 删 L355-358 CPU 分支 | delete[4]；批≥8 Triton / 小批 pytorch sort 两级分流保留 |
| `ops.topk_topp_sampler.apply_top_k_top_p_pytorch` | 同上:L367-L408 | 无（逐字，含 allow_cpu_sync 死分支） | m9 教学主实现（sort/阈值/cumsum/scatter） |
| `ops.topk_topp_sampler.random_sample` + `sample_with_exponential_noise` | 同上:L434-L472 | 无（逐字） | m10 Gumbel 掷骰（multinomial 同步之死的替代） |
| `ops.bad_words.apply_bad_words`/`_apply_bad_words_single_batch` | vllm/v1/sample/ops/bad_words.py:L9-L36 | 无（逐字） | step4 前缀匹配；入参契约=1-D 行视图（apply_bad_words 传 logits[i]） |
| `ops.penalties.apply_all_penalties`/`_convert_to_tensors` | vllm/v1/sample/ops/penalties.py:L10-L56 | 无（逐字） | m6 张量化+−1 占位替换；WC1 痛点③（约束源在 CPU） |
| `layers/utils.get_token_bin_counts_and_mask`/`apply_penalties` | vllm/model_executor/layers/utils.py:L34-L89 | 无（逐字；repetition 的 torch.ops._C 经 HOST SEAM _custom_ops 退化） | 惩罚真算式（OpenAI 两式+正除负乘） |
| `logits_processor.state.LogitsProcessors` | vllm/v1/sample/logits_processor/state.py:L148-L166 | 删同文件 BatchUpdateBuilder L18-145 | delete[6]；两列分类容器是 m2 结构基础、逐字保留 |
| `logits_processor.builtin.{MinP,MinTokens,LogitBias}*` | vllm/v1/sample/logits_processor/builtin.py:L23-L233 | 删三组 update_state/add_request/process_dict_updates/apply_with_spec_decode | delete[6]/[2]；构造器+_device_tensor+is_argmax_invariant+apply 全保留（计划禁删清单） |
| `logits_processor.__init__.build_logitsprocs` | vllm/v1/sample/logits_processor/__init__.py:L185-L218 | 无（逐字） | BUILTIN 三件套构造入口+spec 互斥警告（m17 前指 ch32/33） |
| `metadata.SamplingMetadata` | vllm/v1/sample/metadata.py:L14-L49 | 删 L51-55 两旁路字段+L11 import | delete[2]/[1]；19 字段−2=17 字段（v0.21→v0.27 快照字段零增长，WC4） |
| `ops.topk_topp_triton.apply_top_k_top_p_triton` | vllm/v1/sample/ops/topk_topp_triton.py:L856-L957 | 无（整文件逐字+4 行 SOURCE 标记） | must_keep「整文件保留」；m12 只讲外部契约 |
| `v1.outputs.{LogprobsTensors,SamplerOutput}` | vllm/v1/outputs.py:L28-L79/L212-L220 | 删 filter/cat/empty_cpu 与文件其余类 | 出件载体面；D2H 装配归 ch8 |

## 测试与运行

- host：`python -m pytest instances/vllm/artifacts-v3/ch29-sampler-pipeline/tests -q`
  → **82 passed, 1 skipped**（skip=flashinfer 容器专属，host 未装）。
- GPU 容器（flashinfer 在场）：
  `MSYS_NO_PATHCONV=1 VLLM_IMAGE=vllm/vllm-omni:latest bash scripts/vllm_docker.sh
  -m pytest /work/instances/vllm/artifacts-v3/ch29-sampler-pipeline/tests -q`
  → **83 passed**（真跑 flashinfer 三 API 分支 + forward_cuda 绑定正支路）。
- 结果台账：`tests/test-report.json`。
- 行为基准全部对真实源码现核：允许掩码极性（gpu_input_batch.py:L282-L283
  注释原话「if the corresponding token allowed, the value is False」）、
  frequency/presence 吃 output 计数而 repetition 另吃 prompt mask
  （layers/utils.py:L73-L88）、top-k 并列保留（严格 <）、top-p 边界
  （cumsum ≤ 1-p 反向 mask + 最末位恒保）、rank=#{x≥v} 含并列含自身、
  `sample_with_exponential_noise` 同 dtype 支路原地 `probs.div_(q)`
  （真实调用方每步喂新 softmax）。
