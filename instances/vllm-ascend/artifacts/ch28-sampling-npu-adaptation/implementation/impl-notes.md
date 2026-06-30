# ch28 实现说明 —— 只做减法的精简版（采样的 NPU 对位）

活动实例 `vllm-ascend`。host 无 NPU/CANN/vllm：真实 npu_stream_switch / wait_stream / Triton kernel /
AscendC 自定义算子不真跑；精简版只验**可读、可数值追踪的纯 Python 控制流**——
算法宝石 random_sample 的 Gumbel-max 与 torch.multinomial 同分布在 host CPU torch 即可经验验证。

本章对位三个真实源码文件：`vllm_ascend/sample/sampler.py`、`vllm_ascend/sample/penalties.py`、
`vllm_ascend/sample/rejection_sampler.py`（对照基座 `vllm/v1/sample/sampler.py` 与
`vllm/v1/sample/ops/topk_topp_sampler.py`、`vllm/v1/sample/rejection_sampler.py`）。

## 验收判据
把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本精简版（同名/同结构/同控制流，只删不增）。
- `python3 scripts/lint_fidelity.py instances/vllm-ascend/artifacts/ch28-sampling-npu-adaptation` 无 BLOCKING。
- `python3 -m pytest tests/` 17 passed。

## 核心立意（writer 须按此归属，勿写错）
- **Gumbel-max（probs.div_(q).argmax, q~Exp(1)）规避 multinomial 同步是 vLLM 上游既有手法，昇腾继承**，
  不是昇腾 NPU 对位创新（见 dossier.fidelity_alerts）。昇腾真正的 delta 只是把指数随机用
  `npu_stream_switch(global_stream())` 包成异步 + `wait_stream` 汇流两行（async-exponential 更进一步用
  Event 预算，本章按 subtraction_plan 删除，默认关）。
- **薄壳子类化**：三个采样器只覆写「碰 NPU 同步 / 能上 Triton」的少数热点，其余继承基类不动。
- **Triton 是加速、不是依赖**：每个热点 `if not HAS_TRITON: 走基类原版 / *_pytorch 回退`。

## 删除项（仅 subtraction_plan.delete 批准项）
1. `enable_reduce_sample` 的 TP all-gather 分布式分支（默认关、需多卡）：greedy_sample/forward_native/
   _apply_top_k_top_p 的 reduce 分支、rejection 模块级 greedy_sample、apply_sampling_constraints/
   rejection_sample 的 reduce 分支、`target_indices is not None` 整支、各 *_pytorch 的 enable_reduce_sampling 分支。
2. `_apply_top_k_top_p_ascendc` 整函数与 AscendC 派发（A2/A3 + CANN，host 不可跑）；保留 `_apply_top_k_top_p_pytorch` 唯一实现。
3. async-exponential 路径（do_async_exponential / set_q_event×2 / async_exponential_event / forward_native 的 enable_async_exponential 分支）。
4. block_verify / entropy_verify 高级接受模式（MagicMTP）：block_verify kernel/pytorch、blockwise 残差重采、posterior_* 阈值与分支。
5. rejection_sample 只保留一条 random 代表路径（单卡标准词表）。
6. 全部 `logger.*` 观测日志（保留 `if not HAS_TRITON:` 控制流，仅删日志行）。
随删除而失效的 import（get_tp_group / logger / get_ascend_config / AscendDeviceType / get_ascend_device_type /
block_verify kernel）一并标 `# SUBTRACTED:`。

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版 | 真实源码 `vllm_ascend/sample/...:Lxxx` | 改动 | 原因 |
|---|---|---|---|
| `sampler.py` `random_sample` | `sampler.py:L19-L42` | 逐字保留（含 npu_stream_switch/global_stream/wait_stream） | 全章算法核心，must_keep；NPU-only 调用由测试桩成 nullcontext/no-op |
| `sampler.py` `AscendSampler.apply_penalties` | `sampler.py:L46-L70` | 删 warning_once 日志行 | HAS_TRITON 优雅回退典范：不可用→`Sampler.apply_penalties` 基类原版 |
| `sampler.py` `AscendSampler.__init__` | `sampler.py:L72-L81` | 删 `async_exponential_event = torch.npu.Event()` + debug 日志 | async 路径默认关 |
| `sampler.py` `AscendSampler.{set_q_event,do_async_exponential}` | `sampler.py:L83-L102` | 整体删除 | async-exponential 预计算入口，默认关、需 NPU Stream/Event |
| `sampler.py` `AscendSampler.greedy_sample` | `sampler.py:L104-L124` | 删 enable_reduce_sample 分支，留 `argmax` | 单卡 argmax 与基类一致，薄壳实证 |
| `sampler.py` `AscendTopKTopPSampler.forward_native` | `sampler.py:L145-L190` | 删 reduce_sample 分支 + async_exponential 分支 + 日志 | 留 BATCH_INVARIANT 回退（与 ch26 同源）+ 默认 random_sample 主路 |
| `sampler.py` `_apply_top_k_top_p_pytorch` | `sampler.py:L193-L265` | 删 enable_reduce_sample 分支，留 else 单卡 sort/cumsum/masked_fill | host 可跑的 top-k/top-p 截断 |
| `sampler.py` `apply_top_k_top_p` | `sampler.py:L268-L302` | 删 `_apply_top_k_top_p_ascendc` 与按芯片派发，直接 `= _apply_top_k_top_p_pytorch` | AscendC 算子需 A2/A3+CANN，host 不可跑 |
| `penalties.py` `apply_all_penalties` / `_convert_to_tensors` | `penalties.py:L13-L45` | 无删减 | 薄壳「同接口换内核」实证：内核换 `apply_penalties_triton` |
| `rejection_sampler.py` `AscendRejectionSampler.apply_penalties` | `rejection_sampler.py:L44-L76` | 删 warning_once | 同样 HAS_TRITON 否则回退 `Sampler.apply_penalties`；repeat_indices 按 draft 展开 |
| `rejection_sampler.py` `AscendRejectionSampler.{prepare_sampling,__init__,forward}` | `rejection_sampler.py:L78-L198` | 删 debug 日志 | 薄壳：类体仅覆写这几处，forward 装配 bonus→constraints→rejection_sample |
| `rejection_sampler.py` `apply_sampling_constraints` | `rejection_sampler.py:L220-L286` | 删 reduce 分支 + 日志，留 `apply_top_k_top_p(logits,k,p)` | 温度/top-k/top-p 展开主路不变 |
| `rejection_sampler.py` `rejection_sample` | `rejection_sampler.py:L289-L724` | 删 block/entropy 开关与分支、reduce greedy、`target_indices` 整支、日志；留单卡 greedy + 一条 random 代表路径 | 每热点保留 `if HAS_TRITON: Triton kernel else: *_pytorch` 骨架 |
| `rejection_sampler.py` `expand_batch_to_tokens` / `expand_pytorch` | `rejection_sampler.py:L727-L767, L1092-L1136` | 无删减 | temperature/top-k/top-p 的 batch→token 展开；HAS_TRITON/pytorch 双实现 |
| `rejection_sampler.py` `sample_recovered_tokens` | `rejection_sampler.py:L770-L847` | 删 use_block_verify/target_indices/reduce 形参与 blockwise 分支 | 残差重采入口，留 `q.exponential_()` + Triton/pytorch 双实现 |
| `rejection_sampler.py` `rejection_greedy_sample_{spec_len_1_,}pytorch` | `rejection_sampler.py:L850-L916` | 无删减 | greedy 接受检验 host 回退，逐字保留 |
| `rejection_sampler.py` `rejection_random_sample_pytorch` | `rejection_sampler.py:L919-L1089` | 删 ENTROPY_VERIFY/reduce 分支与相关形参 | 留标准接受判据 `target/draft >= u`、被拒取 recovered、全接受补 bonus |
| `rejection_sampler.py` `sample_recovered_tokens_pytorch` | `rejection_sampler.py:L1139-L1260` | 删 reduce 分支与形参 | 留 normal mode 残差 `max(0,target-draft)/q` argmax |
| 模块级 `greedy_sample` / `rejection_random_sample_block_verify_pytorch` / `sample_recovered_tokens_blockwise_pytorch` | `rejection_sampler.py:L201-L217, L1263-L1518` | 整体删除 | reduce TP all-gather 与 block-verify 高级模式，默认关 |

## 注意
- random_sample 用 `probs.div_(q)` **原地**修改 probs；测试分布对照用单次大 batch 调用（B=40000）经验频率逼近 Categorical(p)。
- bonus_token_ids 真实为 int32（`SamplerOutput.sampled_token_ids`）——测试按 int32 构造与 output buffer 对齐。
- Triton kernel（`rejection_*_with_triton` / `*_kernel` / `apply_penalties_triton`）保留为 NPU-only 代表调用，
  与 `*_pytorch` 回退并列体现「加速非依赖」；host 由记录替身承接，只验 HAS_TRITON 分流，不真算。
