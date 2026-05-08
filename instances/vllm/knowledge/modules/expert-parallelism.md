# Expert Parallelism Knowledge — vLLM

Repo-specific facts about the EP / EPLB / FusedMoE surface in vLLM.
Source pin: `98661fe`. E-prefix IDs (E01..) — distinct from the K
(prefix-cache), P (preemption), T (tensor-parallelism) families to avoid
double-prefix collisions.

Roles: **I**=implementer, **T**=tester, **W**=writer, **R**=reviewer.

---

## E01: `_EP` group is created only for MoE models

- File: `vllm/distributed/parallel_state.py:L1670-L1696`
- Audience: I, R
- Fact: `initialize_model_parallel` checks `config.model_config is None or
  config.model_config.is_moe` BEFORE creating `_EP`. For dense models
  (Llama, Qwen3-base, Mistral) `_EP is None` and `get_ep_group()` raises
  an `AssertionError` with the message
  *"expert parallel group is not initialized. EP group is only created for
  MoE models with num_experts > 0."*
- Implication: any reimpl of EP must mirror this asymmetry vs `_TP`
  (which is always created). Asserting `_EP is not None` at FusedMoE
  init time is the correct pattern.

## E02: EPLB has its own process group `_EPLB`

- File: `vllm/distributed/parallel_state.py:L1700-L1719`
- Audience: I, R, W
- Fact: When `parallel_config.enable_eplb=True`, vLLM creates a SECOND
  process group with the SAME rank list as `_EP` but `group_name="eplb"`.
  Source comment: "to isolate EPLB communications from MoE forward pass
  collectives and prevent deadlocks."
- Implication: the chapter should call this out as a textbook example of
  W04 backpressure isolation. Sharing one group would let an in-flight
  EPLB rebalance broadcast block on the same NCCL stream as a forward
  dispatch all-to-all.

## E03: `expert_map[i] = -1` is the off-rank sentinel

- File: `vllm/model_executor/layers/fused_moe/layer.py:L117`
- Audience: I, T
- Fact: `determine_expert_map` returns a tensor of shape
  `(global_num_experts,)` initialized to `-1`. Owned global expert IDs
  are overwritten with their local indices. The `-1` is what the forward
  pass tests to skip tokens routed off-rank.
- Implication: `expert_map[topk_ids] != -1` is the standard mask. Tests
  must cover the `-1` branch (token routed entirely off-rank) AND the
  uneven case (`E % ep_size != 0`).

## E04: AgRsAll2AllManager is allgatherv + reduce_scatterv, NOT a true all-to-all

- File: `vllm/distributed/device_communicators/all2all.py:L40-L139`
- Audience: I, W
- Fact: The simplest of vLLM's 7 all-to-all backends does
  `dist_group.all_gatherv(...)` for dispatch and
  `dist_group.reduce_scatterv(...)` for combine. Same end state as a
  symmetric all-to-all, simpler dependency surface. The `_v` (variable
  size) suffix matters because per-rank token counts differ — fixed-size
  all_gather would not handle this.
- Implication: when explaining "all-to-all in vLLM", lead with this
  baseline; DeepEP HT/LL, Nixl, FlashInfer, Mori are all
  performance-optimised replacements with the same interface.

## E05: `FusedMoE` does NOT take a `routing_method`; it takes flag args

- File: `vllm/model_executor/layers/fused_moe/layer.py:L259-L286`
- Audience: I, W
- Fact: The `FusedMoE.__init__` signature has `use_grouped_topk: bool`,
  `num_expert_group: int | None`, `topk_group: int | None`,
  `custom_routing_function: Callable | None`, `scoring_func: str`,
  `e_score_correction_bias: torch.Tensor | None`. The actual
  `BaseRouter` instance is constructed internally via
  `create_fused_moe_router(...)` (router_factory.py).
- Implication: don't search source for a `routing_method=` kwarg — it
  doesn't exist. The router family is a flag-set product.

## E06: Mixtral and DeepSeek use different gate types

- Files: `vllm/model_executor/models/mixtral.py:L123`,
  `vllm/model_executor/models/deepseek_v2.py:L272`
- Audience: I, W, R
- Fact: Mixtral's gate is `ReplicatedLinear(hidden_size, num_experts,
  bias=False)` — a plain row-replicated linear. DeepSeek's gate is a
  custom `GateLinear` with optional `e_score_correction_bias` (the
  noaux_tc V3 path). The gate weight is REPLICATED across all EP ranks
  in both cases.
- Implication: the chapter side-by-side must show both gate types.
  Reader who only knows Mixtral will be confused by the DeepSeek
  `e_score_correction_bias` codepath.

## E07: Shared experts are constructed OUTSIDE FusedMoE on the model side

- File: `vllm/model_executor/models/deepseek_v2.py:L295-L317`
- Audience: I, W
- Fact: `shared_experts = DeepseekV2MLP(...)` is built BEFORE the
  `FusedMoE` and passed in via `shared_experts=...` constructor arg.
  Mixtral has no shared experts (`shared_experts=None`). The shared
  expert runs on every rank IN PARALLEL with the routed experts; it
  does NOT scale with `ep_size`.
- Implication: memory-savings claims for EP must caveat the shared
  expert. For DeepSeek-V2 `n_shared_experts=2`, that's two expert-MLPs
  worth of weights replicated on every rank.

## E08: `FusedMoEParallelConfig.make` collapses TP into EP

- File: `vllm/model_executor/layers/fused_moe/config.py:L1192-L1208`
- Audience: I, W
- Fact: When `use_ep=True`, `ep_size = tp_size; tp_size = 1` (the
  flatten_tp_size at L1077 is `dp × pcp × tp`, then re-assigned). EP is
  the COMPLEMENT axis of TP×DP×PCP, not a free hyperparameter. Operators
  set `tp_size`, `dp_size`, `pcp_size` and `enable_expert_parallel`;
  `ep_size` is determined.
- Implication: the chapter §5 mesh demo must show that `ep_size = world
  / (tp × pcp × dp)` is a derived quantity. Trying to set `ep_size`
  independently is a category error.

## E09: Round-robin placement is gated on backend support

- File: `vllm/model_executor/layers/fused_moe/layer.py:L160-L193`
- Audience: I, R
- Fact: `determine_expert_placement_strategy` falls back from
  `"round_robin"` to `"linear"` if (a) the model has only one expert
  group, (b) `num_redundant_experts > 0`, (c) `enable_eplb=True`, or
  (d) the all2all backend is not DeepEP-LL or NIXL.
  Logger warnings are emitted in each fallback.
- Implication: tests that exercise round-robin must use a config that
  satisfies all four conditions, otherwise the strategy silently
  reverts to linear.

## E10: Aux loss does NOT exist in vLLM source

- Files: across `vllm/model_executor/` (verified via grep)
- Audience: I, W, R
- Fact: vLLM is inference-only. `grep -r 'load_balance\|aux.*loss\|
  balance.*loss' vllm/` returns hits ONLY in EPLB-related paths. There
  is no Switch-Transformer-style auxiliary loss because there is no
  training loop.
- Implication: outline subsection 4 ("Expert Load Balancing Loss的梯度
  回传") MUST be reframed at chapter level. Sidebar the training-time
  technique, then pivot to runtime EPLB. Reviewer will check this
  reframe is documented in `impl-notes.md` §1.5.
- **TESTER REFINEMENT (2026-05-07)**: the trap-E "zero hits" claim has
  TWO false-positives outside `eplb/`: `phimoe.py:router_aux_loss_coef`
  (a *stored constant* used nowhere in inference) and
  `vision.py:get_load_balance_assignment` (image-tile placement, not
  expert routing). Trap E text should specify "*no aux-loss
  computation in MoE inference paths*", not "zero source matches".

## E11: Trap G non-commutation surfaces only with `renormalize=False`

- File: `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L94-L100`
- Audience: W, R, T
- Fact: With `renormalize=True` (Mixtral/Switch path), softmax-then-topk
  followed by renormalize is ALGEBRAICALLY EQUIVALENT to topk-first-then-
  softmax: `softmax(g_i)/(sum_top2 softmax(g_j)) == exp(g_i)/(exp(g_a)+exp(g_b))
  == softmax_over_2(top2_logits_i)`. The non-commutation Trap G refers to
  manifests only when `renormalize=False` — there the weights sum to <1
  (softmax tail), while topk-first-then-softmax always sums to 1.
- Implication: chapter §9.1's Trap G callout must specify the
  renormalize=False variant explicitly. Otherwise the math claim "do
  not commute" is technically false under the default Mixtral config.
  Tester pinned this with `test_trap_G_softmax_topk_does_not_commute_with_topk_softmax`.

## E12: Demo numeric pinning conventions for §3.2 alpha-beta

- Files: `instances/vllm/artifacts/09-expert-parallelism/implementation/all2all_baseline.py:alpha_beta_cost`
- Audience: T, W, R
- Fact: The α-β model in `alpha_beta_cost` uses formulas
  `T_AR = 2·(P-1)/P·(α + nbytes/β)` and `T_A2A = (P-1)/P·(α + nbytes/β)`
  → ratio is exactly 2.000 across all payload sizes BY CONSTRUCTION.
  The §3.2 demo's "ratio=2.000" column is a tautology of the model, not
  an empirical measurement.
- Implication: the writer must NOT present "ratio=2.000" as a discovered
  empirical result — frame it as "this is what the α-β model predicts
  for a ring topology; real all-to-all is imbalance-sensitive (Trap B)
  and can deviate". The honest demo caveat in impl-notes §3.6 already
  acknowledges this; the chapter §9.2 must echo it verbatim.

## E13: `ep_size==1` returns 3-tuple in source vs 2-tuple in our mirror

- Files: `vllm/model_executor/layers/fused_moe/layer.py:L107-L109` (source) and
  `instances/vllm/artifacts/09-expert-parallelism/implementation/expert_map.py:L49-L50` (impl)
- Audience: I, T, R
- Fact: vLLM source `determine_expert_map` at `ep_size==1` returns
  `(global_num_experts, None, None)` — a 3-tuple where the third item
  is the AITER-mode-only expert mask. Our pedagogical impl returns
  `(global_num_experts, None)` — 2-tuple, dropping the unused mask.
  This is documented in `impl-notes.md`; tests pin the 2-tuple shape.
- Implication: when readers diff our impl against source line-by-line
  in §9.3's 5-step rhythm, the writer must call out the dropped mask
  explicitly (under "Original adds X because..."). Otherwise the
  signature mismatch looks like a bug rather than a deliberate
  simplification for the AITER quant path that's out of scope.

## E11: Trap-G non-commutativity surfaces under renormalize=False, not renormalize=True

- File: `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L94-L100`
- Audience: T, W
- Fact: Under `renormalize=True`, the vLLM order
  ``softmax → topk → renorm`` is ALGEBRAICALLY EQUIVALENT to
  ``topk → softmax`` (both give `softmax(g_i)/Σ_topk softmax(g_j)`).
  The non-commutation surfaces only when `renormalize=False`, where
  vLLM returns softmax-tail mass (sum < 1) while topk-then-softmax
  always gives sum = 1.
- Implication: Trap-G evidence must use the `renormalize=False` arm
  (the actual Mixtral/DeepSeek default at certain layers) to expose
  the difference. Demo §3.1 already pins this:
  `renormalize=False → sum range [0.2730, 0.6171] mean 0.3899`.

## E12: ep_size==1 short-circuits expert_map to None — None-test required

- File: `vllm/model_executor/layers/fused_moe/layer.py:L107-L109`
- Audience: T
- Fact: `determine_expert_map(ep_size=1, ...)` returns
  `(global_num_experts, None)` — the second element is `None`, NOT
  an all-zeros tensor. The forward pass tests `if expert_map is None`
  and skips per-token branching.
- Implication: tests must cover BOTH the None branch (ep=1 fast path)
  AND the masked branch (ep>1) — they execute different code paths.
  `test_ep1_returns_E_and_None_map` and the corresponding integration
  pin `_run_local_experts` for both branches.

## E13: AgRs `combine` requires its caller to pre-sum partial outputs

- File: `vllm/distributed/device_communicators/all2all.py:L130-L135`
- Audience: T, I
- Fact: The pedagogical `AgRsAll2AllManager.combine` takes an
  *already-summed* full tensor and splits it back per-rank via
  `reduce_scatterv`. Real `reduce_scatter` would do the sum on-device,
  but in single-process simulation the harness sums first.
- Implication: round-trip tests (`dispatch → expert exec → combine`)
  must sum the per-rank contributions in test code BEFORE calling
  `combine`, otherwise the test doesn't model what real EP does.

## E14: §3.5 EPLB demo numerics depend on per-step seed=100+step pattern

- File: `implementation/demo.py:L290-L302`
- Audience: T, W
- Fact: The §3.5 EPLB demo uses `seed_for_step = 100 + step`, so
  step 0→seed=100, step 25→seed=125, step 50→seed=150, etc. The
  pinned per-rank loads (`[1292, 246, 257, 253]` at step 0;
  `[1295, 230, 261, 262]` at step 25) hold ONLY under this seed pattern.
- Implication: any test reproducing the §3.5 timeline must mirror the
  same `seed=100+step` indexing. `make_skewed_routing(seed=100)` for
  step 0, `seed=125` for step 25, `seed=150` for step 50 — this is the
  ground truth for the rebalance trajectory.

## E15: Trap-D evidence — _EP and _EPLB are different Python objects

- File: `implementation/ep_groups.py:L259-L267`
- Audience: T, R
- Fact: After `init_ep_group(... enable_eplb=True)`, `_EP` and `_EPLB`
  are SEPARATE `EPGroup` instances. `id(ep) != id(eplb)`. They share
  the same `rank_list` but have distinct `group_name` ("ep" vs "eplb").
- Implication: tests that verify Trap D must assert object-identity
  inequality (`ep is not eplb`), not just member-list equality. A
  shared rank list is necessary but NOT sufficient — the deadlock-
  prevention claim hinges on distinct heap objects (and in production,
  distinct NCCL communicators).

## E16: imbalance_ratio of all-zero load returns 0.0, not 1.0

- File: `implementation/eplb.py:imbalance_ratio` (L142-L152)
- Audience: T, W
- Fact: With all-zero load (`torch.zeros(...)`), the implementation does
  `mean = per_rank_load.mean().clamp_min(1.0)` so mean=1, max=0 → ratio=0.0.
  The numel==0 branch returns 1.0 sentinel, but the all-zeros branch returns
  0.0. The two edge cases produce DIFFERENT sentinels.
- Implication: tests must distinguish "no data ever" (numel=0 → 1.0) from
  "data is all zero" (zeros → 0.0). The chapter narrative should not say
  "imbalance is 1.0 when balanced" without the caveat that a fully-idle
  ring also reads 0.0 — the metric's interpretation depends on whether
  any token routed at all.

## E17: pytest collection requires conftest.py at tests/ AND testpaths=. in pytest.ini

- Files: `tests/conftest.py`, `tests/pytest.ini`
- Audience: T
- Fact: Mirroring Ch08's test scaffolding, Ch09 needs `conftest.py` to add
  the chapter root to sys.path (so `from implementation import ...` works)
  and `pytest.ini` with `testpaths = .` and `norecursedirs = _legacy
  __pycache__` to prevent the legacy directory from being collected.
  Without both, tests fail to import or collect _legacy artifacts.
- Implication: future chapters in the v6 standard MUST replicate this
  scaffolding. The implementer's smoke tests work via `sys.path.insert`
  in each test file, but a centralized conftest.py is cleaner for a
  large suite (203 tests in Ch09).

## E18: Writer narrative — render `\frac` 在 inline 不能塞复杂分式

- File: `narrative/chapter.md` 全文公式
- Audience: W
- Fact: `lint_formulas.py` 的 BLOCKING 规则之一是 `\frac` 出现在 inline `$...$` 中
  必须升级为 block `$$...$$`。Ch09 §9.1.2 的 trap-G 等价性证明、§9.5 的
  `mem_per_rank` 公式都需要 block-level 渲染。Inline 留给单符号或简单
  比例（如 `$1/\\sqrt{d_k}$`）。
- Implication: 以后写 MoE / EP / TP 系列章节涉及 sum 和 frac 时，默认走
  block；只有变量符号才用 inline。Trap-G 证明 `softmax(g_i)/Σ softmax(g_j)`
  必须 block。

## E19: trap recap §X 用 "claim → 错 → 为什么 → 源码证据" 模板更结实

- File: `narrative/chapter.md:§9.7`
- Audience: W, R
- Fact: Ch07 §7.6.4 / Ch08 §8.6.4 / Ch09 §9.7 都用同一个四件套结构：
  `**错**` `**为什么**` `**源码证据**` `**Demo 证据**`（或 `**测试**`）。
  reviewer 一眼能看到每条 trap 是否真的有源码 + 数字双重锚定。
- Implication: 后续 Ch11+ 的 trap recap 应该继承这个模板。每条 trap 没有
  源码证据 + demo/test 证据就是空 claim，会被 review 打回去。

## E20: §X 重构在章节开篇 hook 段落显式声明，不要藏在第 4 节

- File: `narrative/chapter.md:这章要讲什么？`
- Audience: W
- Fact: Ch07 "no radix tree" 在 hook 第 1 段说；Ch08 "no class TensorParallel"
  在 hook 第 1 段说；Ch09 hook 第 2 段就说"vLLM 没有 class ExpertParallel"
  并解释 outline §4 "梯度回传" 是 training 概念，要 reframe 成 EPLB。读者
  在前 3 段就知道 outline-vs-source 不一致是有意为之，不会以为 writer 读
  错了 outline。
- Implication: 任何 outline-vs-source mismatch 必须在 hook 显式标 reframe，
  不要藏到对应小节里"突然"切换主题。

## E21: 单进程 EP 模拟必须 verbatim 引用 honest-demo caveat

- File: `narrative/chapter.md:§9.6.2`
- Audience: W
- Fact: §3.2 的 α-β cost 模型 ratio=2.000 是模型恒等式（E12），不是测量
  结果。Ch09 §9.6.2 整段 honest caveat 直接 verbatim 复制 impl-notes §3.6 + tester E12 refinement。Reviewer 会校验
  这条 caveat 出现且用词跟 impl-notes 一致——少一句话就 REVISE。
- Implication: 任何"教学版 vs 生产版"差异必须把 impl-notes 的原文搬进
  narrative，而不是 paraphrase——paraphrase 容易丢精度。

## E22: Reviewer 验证 framing tip 时必须三处锚定（hook + 主体 + recap）

- File: `narrative/chapter.md` 全文，特别是 §9.7 traps recap
- Audience: R
- Fact: Ch09 review 中确认每条 framing tip 应用合格的判据是"三处锚定"：
  - **Tip 1（Trap-G renormalize=False qualifier）**: hook 的 take-aways 列表
    L57 + 主体 §9.1.2 推导 L143/L149 + Trap-G recap L1046 都必须显式提到
    "当且仅当 renormalize=False" 或代数等价句式。三处中缺任何一处 = REVISE。
  - **Tip 2（"MoE inference paths" 而不是 "zero matches"）**: §9.4 主体
    L598/L709 + Trap-E recap L1034/L1037 双锚，且必须 carve out E10 的两个
    false-positive（phimoe.py + vision.py）。
  - **Tip 4（α-β tautology framing）**: §9.6 主体 L924/L938 三处自我引用
    "模型恒等式" + Trap-B recap L1018/L1020 重复。
- Implication: 后续章节的 review 也要按"hook 提一次、主体推导一次、recap
  pin 一次"的三处锚定模式校验 framing tip。少一处都可能让读者半路忘记
  关键 caveat。Single-cycle APPROVED 的章节都满足这个模式（Ch07/Ch08/Ch09）。

## E23: Reviewer 校验 mapping rows 应数 `^|` 而不是信 writer 的求和

- File: `artifacts/{chapter}/narrative/chapter.md` 末尾的 source mapping table
- Audience: R
- Fact: Ch09 writer 自报 49 主 + 39 mini = 88 行 mapping。Reviewer 用
  `grep -c "^|" chapter.md` 数到 151 行。差异来源是 mini-mapping 内部的
  header/separator 行也算 `|` 起头 + writer 的"主 vs mini"分类没有把所有
  inline 表都算进去。**Floor 是 ≥10 行**——151 远超 ceiling 不必精算
  分类，只用 grep 总数确认 floor 即可。
- Implication: 不要花时间复算 writer 的"主 vs mini"分类——直接 `grep -c "^|"`
  对 floor 即可。如果 grep 总数 < 12（10 floor + 2 buffer for header lines），
  才需要细数。Ch07: 75; Ch08: 122; Ch09: 151——单调上升说明 v6 mapping
  cadence 在加深。

## E24: 非 blocking inline-density warning 可接受当且仅当符号都是单符号

- File: `scripts/lint_formulas.py` Too-Many-Inline-Formulas 检测
- Audience: R, W
- Fact: lint_formulas.py 把"一段里 ≥3 个 inline `$...$`"标为非 blocking
  warning。Ch09 命中 4 次（L608, L754, L773, L918），但每个 inline token 都是
  单符号（`$f_i$`, `$P_i$`, `$3Fh$`, `$\alpha$`, `$\beta$` 等），符合 wisdom
  writing.md 的 "single symbol allowed inline" 规则。
- Implication: reviewer 看到 inline-density warning 不要直接 REVISE，
  要打开对应行检查 inline 的内容是否都是单符号。如果有复杂表达式
  （sqrt, frac, sum 配合）就要 promote to block——但单符号密度高
  本身不是 blocking issue。Ch07/Ch08/Ch09 都触发过这种 warning，全部判过。

## E25: 三连"no class X"reframe 已固化为章节模式（archivist 跨 Ch07/Ch08/Ch09 观察）

- File: `trace/deliveries/2026-05-06_ch07-prefix-cache-ch07-v6-published.md`,
  `trace/deliveries/2026-05-06_ch08-tensor-parallelism-ch08-v6-published.md`,
  `trace/deliveries/2026-05-07_ch09-expert-parallelism-ch09-v6-published.md`
- Audience: A (archivist), L (team-lead), W (writer), R (reviewer)
- Fact: Ch07 "no radix tree" → Ch08 "no class TensorParallel" → Ch09
  "no class ExpertParallel/MoEParallel/TopKGate" 三连 reframe 全部用同样
  的三锚点结构：(1) 章节标题里直接 explicit "没有 class X"；(2) opener
  hook 段落用 grep 证据 + "(zero matches)"；(3) §X.2 body 列出 N 个真正
  实现 X 功能的协作文件。Ch10 multi-token-prediction 是第四个候选——
  `vllm/` 里没有 `class MultiTokenPrediction`、`class MTPHead`，DeepSeek
  MTP 类是 `DeepSeekMultiTokenPredictor`（DeepSeek-prefixed），加上 30+
  个 `*_mtp.py` per-model wrapper。
- Implication: archivist brief 写 Ch10 时直接默认应用三锚点 reframe
  template；reviewer 验证时数三锚点；writer 把模式列入 §10.2 mini-table
  的 K15 two-tier。**仍然不达 wisdom-promotion 门槛**（intra-instance
  N=3 / candidate N=4，不是跨 instance 的 2+），但 chapter-pattern 已
  reproducible enough that Ch10 brief 不需要重新 derive 这个模式。

## E26: §X.4-style training-vs-inference reframe 是 outline-as-topic-not-contract 的第三个轴（archivist）

- File: `trace/deliveries/2026-05-07_ch09-expert-parallelism-ch09-v6-published.md`
  §9.4 reframe；同时 Ch10 §10.3 "Training——多步 CE 损失的加权策略" 也
  是相同 pattern（已写入 brief）。
- Audience: A, L, W
- Fact: rule #6 ("outline 是 topic 不是 contract") 在 Ch07/Ch08 是
  "specific class doesn't exist" 轴，Ch09 §9.4 引入 NEW sub-pattern
  "training-time concept doesn't exist in inference codebase"。
  Ch10 §10.3 是第二个 instance ——multi-token-prediction 的训练 CE-loss
  在 vLLM source 里不存在（vLLM 是 inference engine）。Reframe 模板：
  (1) 一页 sidebar grounding 训练时怎么做的（Switch Transformer
  L_balance；multi-step CE with λ_k decay）；(2) pivot 到 vLLM 的
  inference response（EPLB / weight loading）；(3) 在 trap recap 里把
  "training mechanism is in vLLM" 列为错误。
- Implication: 若 outline 提及"loss"/"训练"/"梯度"/"backward"/"reward"/
  "RLHF"/"DPO"/"SFT" 等关键词，archivist 在 brief 里立刻 flag
  training-time 概念 OUT OF SCOPE，并提供 sidebar+pivot 模板。
  Ch11-Ch13 brief 时检查同样 pattern：DCP/PCP, KV-offload,
  prefix-cache-pooling 大概率没有 training reframe needed，但 Ch15+
  Llama 模型架构、Ch20 model_runner 应警惕。

## E27: framing tip three-anchor 验证模板可量化（archivist + reviewer 协作）

- File: `trace/deliveries/2026-05-07_ch09-expert-parallelism-ch09-v6-published.md`
  Gate 8 (5 tips × 3 anchors = 15 verifications)
- Audience: A, R, W
- Fact: Ch09 reviewer 在 Gate 8 把每个 framing tip 的三锚点（hook + body
  + recap）逐个列出 chapter line numbers，形成可数的 5×3=15 verification
  matrix。这把 tester 的"建议"变成了可机器验证的契约：writer 必须在
  hook 段落、对应主体段落、§X.7 trap recap 都至少出现一次。Ch08 用过
  类似但不显式列出 line numbers；Ch09 显式化后整个 verification 流程从
  "感觉它在那" → "L143 / L149 / L1046" 等具体行号。
- Implication: archivist 后续 brief 写 §6 candidate language traps 时，
  应按照 5×3=15 锚点结构提示 writer，并把 reviewer 的 anchor count
  作为 hard gate 8 的明确 metric（exceeds 15 = 3+ tips × 3 anchors all
  present）。Ch10 brief §6 已按此结构写。

## E28: knowledge file 增长曲线已超 compact() 修复优先级（archivist 跨章观察）

- File: `knowledge/modules/expert-parallelism.md` (24 facts),
  `tensor-parallelism.md` (19), `prefix-cache.md` (17);
  `scripts/learn.py:_parse_module_file` returns []
- Audience: A, L, framework
- Fact: 跨 4 章观察 compact() 都不能用：Ch07 prefix-cache 17 facts、
  Ch08 tensor-parallelism 19 facts、Ch09 expert-parallelism 24 facts、
  scheduler.md 12-after-2-compactions。每章都 manual workaround；
  从 Ch07 第一次 surface 到 Ch09 已经 4 chapters in succession。
  Ch10 multi-token-prediction.md 也将 ≥15 facts (M-prefix, 估计
  15-18 个)，会是第 5 个。
- Implication: archivist 在 Ch09 delivery 把 P2-2 升级为
  "must-fix-before-Ch12"。Ch10 brief 已显式提示 implementer
  "compact() is broken, manual workaround required if M-prefix exceeds
  15"。如果 Ch12 之前没修好，knowledge/modules/ 会变成不可维护的
  append-only 文件，违反 anti-bloat 规则（每个 module ≤15 facts）。
  Team-lead 应把 P2-2 排到 Ch10/Ch11 之间的间隙。
