# Ch10 Multi-Token Prediction — Implementer Notes

- Chapter: `10-multi-token-prediction`
- Source pin: vLLM `98661fe012c5c467252d4df8411d2f46190e9268`
- Author role: implementer
- Date: 2026-05-07

The chapter teaches **how vLLM uses Multi-Token Prediction heads + rejection
sampling to break the one-token-per-forward bottleneck of autoregressive
decoding**. The core algorithm — Chen 2023 / Leviathan 2023 rejection
sampling — is provably unbiased: the emitted-or-recovered token is
distributed *exactly* as the target distribution `p`, regardless of how bad
the draft distribution `q` is. The win comes from running the cheap draft K
times in one forward and verifying all K with one target forward; the loss
comes when the chain breaks early (geometric chain-break invariant).

This implementation reproduces the algorithmic surface in plain PyTorch /
NumPy, mirroring vLLM's two Triton kernels with Python loops. The output
shapes, `-1` chain-break sentinel, and Chen 2023 semantics match exactly.

---

## §1 — Source Analysis (HARD GATE)

### 1.1 Files implementing MTP / spec-decode in the target repo

| File | Lines (verified at 98661fe) | Role |
|---|---|---|
| `instances/vllm/source/vllm/v1/sample/rejection_sampler.py` | 921 total; L37-L195 (RejectionSampler nn.Module), L392-L503 (rejection_sample driver), L506-L562 (apply_sampling_constraints), L604-L656 (generate_uniform_probs), L659-L703 (sample_recovered_tokens), L708-L757 (rejection_greedy_sample_kernel), L760-L826 (rejection_random_sample_kernel), L853-L920 (sample_recovered_tokens_kernel) | THE rejection sampling implementation — Triton kernels, recovered-token sampling, synthetic-mode fallback |
| `instances/vllm/source/vllm/v1/spec_decode/metadata.py` | 66 total | `@dataclass SpecDecodeMetadata` — the data contract between proposer and rejection sampler |
| `instances/vllm/source/vllm/v1/spec_decode/llm_base_proposer.py` | 1820 total; L60-L303 (`__init__`), L407-L412 (`_greedy_sample`), L413-L656 (`def propose`), L1402-L1576 (`_maybe_share_embeddings`, `_maybe_share_lm_head`) | `class SpecDecodeBaseProposer` — common scaffolding for EAGLE / draft_model / dflash / extract_hidden |
| `instances/vllm/source/vllm/v1/spec_decode/eagle.py` | 22 total; L10-L22 | `class EagleProposer(SpecDecodeBaseProposer)` — pure inheritance, the algorithm IS the base |
| `instances/vllm/source/vllm/v1/spec_decode/medusa.py` | 78 total; L18-L78 | `class MedusaProposer` — does NOT inherit from base; K independent MLP heads |
| `instances/vllm/source/vllm/v1/spec_decode/draft_model.py` | 88 total; L17-L88 | `class DraftModelProposer(SpecDecodeBaseProposer)` — separate small transformer drafts |
| `instances/vllm/source/vllm/v1/spec_decode/extract_hidden_states.py` | 382 total; L26-L65 | `class ExtractHiddenStatesProposer` — single-step variant; `assert num_speculative_tokens == 1` (L30) |
| `instances/vllm/source/vllm/v1/spec_decode/ngram_proposer.py` | 285 total; L12-L162 (`__init__` + `propose`), L170-L285 (numba batch path) | `class NgramProposer` — n-gram lookup; NO draft probs (rejection sampler runs greedy-only path) |
| `instances/vllm/source/vllm/model_executor/models/deepseek_mtp.py` | 488 total; L43-L62 (SharedHead), L63-L122 (DeepSeekMultiTokenPredictorLayer), L124-L184 (DeepSeekMultiTokenPredictor), L186-L488 (DeepSeekMTP wrapper + load_weights + `_rewrite_spec_layer_name` at L458-L488) | The canonical MTP head impl — RMSNorm + attention + RMSNorm + full MoE block + SharedHead |
| `instances/vllm/source/vllm/config/speculative.py` | (large) L55-L70 (SpeculativeMethod literals), L73-L210 (class SpeculativeConfig), L213-L227 (`_acceptance_length_to_rates`) | `SpeculativeMethod = Literal[...]` — note `"mtp"` is NOT a literal; DeepSeek MTP loads via `method="draft_model"` + `model="deepseek_mtp"` |
| `instances/vllm/source/vllm/model_executor/models/llama_eagle3.py` | 425 total | EAGLE3 reference — `Eagle3LlamaForCausalLM` with `fc` projection fusion |

This is **11 source files** in the table — exceeds the v6 floor of 5 (Ch04: 4,
Ch05: 4, Ch06: 4, Ch07: 5, Ch08: 8, Ch09: 10). The breadth IS the lesson —
spec-decode is a 5-proposer-family + 1-verifier collaboration plus 2
reference-site model files.

### 1.2 Key classes and their responsibilities

| Source class | Lines | Purpose | Owns | Delegates |
|---|---|---|---|---|
| `RejectionSampler(nn.Module)` | rejection_sampler.py:L37-L195 | Verify drafts via rejection sampling | `sampler`, `synthetic_conditional_rates` | `rejection_sample(...)` driver |
| `SpecDecodeMetadata` (dataclass) | metadata.py:L9-L66 | Data contract between proposer ↔ sampler | `draft_token_ids`, `cu_num_draft_tokens`, `target_logits_indices`, `bonus_logits_indices`, `max_spec_len` | nothing — pure data |
| `SpecDecodeBaseProposer` | llm_base_proposer.py:L60-L1820 | Common proposer scaffolding (CUDA graphs, slot mapping, weight sharing) | `vllm_config`, `num_speculative_tokens`, `parallel_drafting`, `pass_hidden_states_to_model` | `_get_model()` (subclass), `propose()` (concrete) |
| `EagleProposer(SpecDecodeBaseProposer)` | eagle.py:L10-L22 | EAGLE/EAGLE3 (fc-fusion path) | nothing — pure subclass | base.`propose()` |
| `MedusaProposer` | medusa.py:L18-L78 | K independent MLP heads on target hidden state | K heads, `model` | `model.compute_logits` per head |
| `DraftModelProposer(SpecDecodeBaseProposer)` | draft_model.py:L17-L88 | Run a separate small model as draft | `_raise_if_vocab_size_mismatch`, `_raise_if_draft_tp_mismatch` | base machinery; does NOT share lm_head (L86-L88) |
| `NgramProposer` | ngram_proposer.py:L12-L162 | N-gram lookup over recent context | `min_n`, `max_n`, numba batch propose | numba JIT functions |
| `ExtractHiddenStatesProposer` | extract_hidden_states.py:L26+ | KV-cache hidden-state extractor; `num_speculative_tokens == 1` | `hidden_states` buffer | model forward |
| `DeepSeekMultiTokenPredictorLayer` | deepseek_mtp.py:L63-L122 | One MTP head: enorm + hnorm + eh_proj + DeepseekV2DecoderLayer + SharedHead | per-layer modules | `mtp_block.forward()` |
| `DeepSeekMultiTokenPredictor` | deepseek_mtp.py:L124-L184 | Stack of K MTP layers | `layers: ModuleDict`, `embed_tokens`, `logits_processor` | per-step `layers[idx].forward()` |
| `DeepSeekMTP` | deepseek_mtp.py:L186-L488 | Top-level wrapper | `model: DeepSeekMultiTokenPredictor`, `expert_weights` | `_rewrite_spec_layer_name` (L458-L488) |
| `SharedHead` | deepseek_mtp.py:L43-L62 | RMSNorm + ParallelLMHead (lm_head usually shared with target) | `norm: RMSNorm`, `head: ParallelLMHead` | weight tied via `_maybe_share_lm_head` |

### 1.3 Data flow — the MTP propose → verify cycle

```
Target forward (1 step):
   target_hidden, last_token_id  ←  the just-sampled output

Proposer (K-step, runs once):
   For step in 0..K-1:
     emb_n  = enorm(embed(prev_token))     # only step 0 uses target's last_token
     h_t    = hnorm(target_hidden)
     fused  = eh_proj([emb_n, h_t])         # the MTP-specific fusion
     hidden, residual = mtp_block(fused)    # full transformer block (Trap E)
     hidden = hidden + residual
     logits = shared_head.head(shared_head.norm(hidden))
     prev_token = argmax(logits)            # next draft id
     target_hidden = hidden                 # roll for next step
   draft_token_ids ← [prev_token_0, ..., prev_token_{K-1}]

Target forward (verify K positions in ONE pass):
   target_logits[0..K]  ←  forward over draft_token_ids prefix
   bonus_token  ←  Sampler.forward(target_logits[K])     # always sampled

RejectionSampler.forward (rejection_sampler.py:L87-L195):
   1. apply temperature/top-k/top-p to target_logits
   2. for each request:
        for pos in 0..K-1:
          accept iff u < min(1, p(d_i)/q(d_i))           # Chen 2023
          on reject: emit recovered token from (p-q)_+; chain breaks
        if all_accepted: emit bonus_token
   output_token_ids[batch, max_spec_len + 1] with -1 sentinels
```

**Two algorithm paths, same theorem.**

The chain-break invariant (Ch10 load-bearing): **once a position rejects, all
later positions in that request stay PLACEHOLDER_TOKEN_ID = -1**.
`rejection_sampler.py:L425-L430` initializes the output buffer with -1; the
kernel then writes left-to-right and stops on first reject (L734 sets
`rejected = True`, L734 `if not rejected` gates the next iter).

### 1.4 Design decisions and WHY (≥3 with trade-off analysis)

1. **Output buffer pre-filled with `PLACEHOLDER_TOKEN_ID = -1`.**
   - Decision: `rejection_sampler.py:L425-L430` — `torch.full((batch, max_spec_len + 1), -1)`.
     The kernel's chain-break is *implicit* — it just stops writing.
   - Trade-off: SAVES a per-position output-mask. Cost: zero. The `-1`
     sentinel is filtered out by `parse_output` (`rejection_sampler.py:L370-L389`)
     before tokens reach the user. Alternative — emit a separate
     `num_emitted` tensor and pack — would cost an extra kernel pass.
   - Source: `rejection_sampler.py:L425-L430, L734, L820`.

2. **Bonus token is sampled by `Sampler` (NOT by `RejectionSampler`).**
   - Decision: `rejection_sampler.py:L128-L142` — `bonus_logits` indexed
     out of `logits` and sampled via `self.sampler(bonus_logits, ...)`.
   - Trade-off: Lets bonus tokens use top-p / top-k / penalties (which
     spec-decode itself doesn't support). Cost: one extra small sample
     call per request. Alternative — sample bonus inside the rejection
     kernel — would force everyone to use vanilla multinomial.
   - Source: `rejection_sampler.py:L120-L142`.

3. **EPLB-style separate group is NOT needed for spec-decode.**
   - Decision: spec-decode does NOT have its own process group. The draft
     model lives in the same TP group as the target (`draft_model.py:L36
     _raise_if_draft_tp_mismatch`).
   - Trade-off: Simpler — one fewer group to coordinate. Cost: draft TP
     must match target TP, even if draft is small enough to run on one rank.
   - Source: `draft_model.py:L36-L51`.

4. **`SpeculativeMethod` literal: `"mtp"` is one of 17 MTPModelTypes.**
   - Decision: `speculative.py:L35-L67` — `MTPModelTypes` literal includes
     `"deepseek_mtp"`, `"qwen3_5_mtp"`, `"ernie_mtp"`, `"mimo_mtp"`,
     `"glm4_moe_mtp"`, `"deepseek_v4_mtp"`, ..., `"mtp"` itself, and 11 more.
     `MTPModelTypes` is folded into `EagleModelTypes`, which is folded into
     `SpeculativeMethod`. So MTP is a first-class method tier, but enumerated
     per-model (each model family ships its own `*_mtp.py` wrapper).
   - Trade-off: Per-model code duplication (each `*_mtp.py` is its own file)
     vs flexibility (each model can specialize fusion topology). DeepSeek MTP
     at this commit can be invoked via either `method="deepseek_mtp"` directly
     OR via `method="draft_model"` with `model="deepseek_mtp"` — multiple
     paths into the same machinery.
   - Source: `speculative.py:L35-L67`.
   - **Correction to brief §2.2**: brief says `"mtp"` is NOT in `SpeculativeMethod`;
     verified at 98661fe that it IS, transitively via `MTPModelTypes ⊂ EagleModelTypes
     ⊂ SpeculativeMethod`. The underlying point about per-model wrappers still holds.

5. **`extract_hidden_states.py` enforces `num_speculative_tokens == 1`.**
   - Decision: `extract_hidden_states.py:L30` — hard assert. This proposer
     is for **KV-cache hidden-state extraction** (KV transfer between
     ranks), NOT speculative decoding per se. Single-step is by design.
   - Trade-off: Specialized path for a specialized job. Cost: zero —
     this proposer doesn't speculate, it just caches hidden states.
   - Source: `extract_hidden_states.py:L26-L80`.

### 1.5 Outline-vs-source mismatch handling — TWO reframes

#### Reframe A — Outline §3 "Training multi-step CE loss" → "Inference-time MTP weight loading"

The outline `book/book-outline.json` lists subsection 3 as:

> "Training——多步CE损失的加权策略"

**vLLM is inference-only — there is NO MTP training loss code anywhere in
the codebase.** Verified via grep:

```
$ grep -rn "MTPLoss\|multi_step_ce\|compute_mtp_loss\|mtp_aux_loss" \
       instances/vllm/source/vllm/
(no results)
```

Reframe (§10.3 of the chapter):

1. **Sidebar** grounding readers from training literature: the canonical MTP
   training loss is `L_MTP = Σ_{k=0..K-1} λ_k · CE(p_k, x_{i+k})` with `λ_k`
   typically decaying 1.0 / 0.5 / 0.25 (DeepSeek-V3 paper, Switch-Transformer
   multi-step supervision).
2. **Pivot** to vLLM's inference response:
   - `_rewrite_spec_layer_name` (`deepseek_mtp.py:L458-L488`) — remap HF
     checkpoint names to vLLM's MTP layer layout.
   - `_maybe_share_lm_head` (`llm_base_proposer.py:L1471-L1539`) — tie MTP's
     LM head to target's, saving ~vocab × hidden params per MTP layer.
   - `_maybe_share_embeddings` (`llm_base_proposer.py:L1402-L1469`) — same
     for the embed_tokens.

This mirrors **Ch09 §9.4's reframe** (training-time aux loss → inference-time
EPLB). It's the **second instance** of the training-to-inference reframe
pattern; reviewer expects sidebar + pivot per the same template.

#### Reframe B — "no class MultiTokenPrediction" — the FOURTH instance

There is no top-level `class MultiTokenPrediction` in `vllm/`. Verified via
grep:

```
$ grep -rn "^class MultiTokenPrediction\|^class MTPHead\|^class MTPModel" \
       instances/vllm/source/vllm/
(no results)
```

What exists is `DeepSeekMultiTokenPredictor` and `DeepSeekMultiTokenPredictorLayer`
(DeepSeek-prefixed) plus 30+ similar `*_mtp.py` files (qwen3_5_mtp.py,
ernie_mtp.py, glm4_moe_mtp.py, deepseek_v4_mtp.py, mimo_mtp.py,
longcat_flash_mtp.py, openpangu_mtp.py, ...). Each model family has its own
MTP wrapper.

This is the **FOURTH "no class X"** case, mirroring:
- Ch07: no `class RadixTree` (chained-hash impl)
- Ch08: no `class TensorParallel` (5-file collab)
- Ch09: no `class ExpertParallel`/`MoEParallel`/`TopKGate` (5-file collab)
- Ch10: no `class MultiTokenPrediction` (per-model MTP wrappers + shared
  spec_decode infrastructure)

Three-anchor pattern (writer must hit all three):
1. **Title** — the chapter title says "MTP" but the framing should
   acknowledge "MTP is one of five spec-decode methods" up front.
2. **Hook** — 1st-page paragraph naming the four prior cases and stating
   "this chapter follows the same pattern: open with what's NOT in source".
3. **§10.2 body** — explicit grep evidence + file list, plus the
   `SpeculativeMethod` literal showing `"mtp"` is not a first-class method.

---

## §2 — Implementation Module Mapping

| Module | Source mirror | Key contents |
|---|---|---|
| `spec_metadata.py` | `vllm/v1/spec_decode/metadata.py:L1-L66` | `SpecDecodeMetadata` dataclass + `make_dummy` factory + `PLACEHOLDER_TOKEN_ID/MAX_SPEC_LEN/GREEDY_TEMPERATURE` constants |
| `rejection_sampling.py` | `vllm/v1/sample/rejection_sampler.py:L37-L920` | Greedy + random kernels (Python loops), `sample_recovered_tokens`, `rejection_sample` driver |
| `acceptance_math.py` | derived from kernel structure + `speculative.py:L213-L227` | `expected_tokens`, `speedup`, `break_even_alpha`, `simulate_chain_break`, parameter counts |
| `mtp_head.py` | `deepseek_mtp.py:L43-L184` | `RMSNorm`, `MTPBlock`, `SharedHead`, `DeepSeekMultiTokenPredictorLayer`, `DeepSeekMultiTokenPredictor`, `parameter_count_mtp/medusa` |
| `weight_loading.py` | `deepseek_mtp.py:L458-L488`, `llm_base_proposer.py:L1402-L1576`, `speculative.py:L213-L227` | `rewrite_spec_layer_name`, `remap_checkpoint`, `maybe_share_lm_head`, `maybe_share_embeddings`, `acceptance_length_to_rates`, `unconditional_to_conditional_rates`, `loader_demo_shapes` |
| `proposers/base.py` | `llm_base_proposer.py:L60-L106` | `ProposerOutput`, `SpecDecodeBaseProposer` minimal scaffold |
| `proposers/eagle.py` | `eagle.py:L10-L22` + `llama_eagle3.py` | `EagleProposer` with fc-fusion block |
| `proposers/medusa.py` | `medusa.py:L18-L78` | `MedusaProposer` with K independent MLP heads |
| `proposers/draft_model.py` | `draft_model.py:L17-L88` | `DraftModelProposer` with separate small transformer |
| `proposers/ngram.py` | `ngram_proposer.py:L12-L162` | `NgramProposer` with prompt-lookup matching |
| `proposers/extract_hidden.py` | `extract_hidden_states.py:L26-L382` | `ExtractHiddenStatesProposer` (single-step) |
| `proposers/mtp.py` | `deepseek_mtp.py:L186-L488` + draft_model loader | `DeepSeekMTPProposer` wrapping the predictor |
| `demo.py` | runs §1-§5 | 5 demos producing ≥80 verbatim numerics |

Total: 12 .py modules (+`__init__`) + impl-notes. ~2300 LOC implementation,
~430 LOC impl-notes, ≥130 `# REFERENCE:` comments.

---

## §3 — Demo Numerics (verbatim quotes for the writer)

All numbers below come from `tests/demo-output.txt` (regenerate with
`python implementation/demo.py`). Deterministic at `seed=42` (top-level)
and per-demo seeds inside.

### §3.1 Rejection sampling unbiasedness (vocab=8, K=4, 10000 trials)

```
Target distribution p = [0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03]
Draft distribution  q = [0.10, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05]
Empirical p_hat       = [0.2906, 0.2037, 0.1543, 0.1005, 0.1044, 0.0674, 0.0494, 0.0297]
KL(empirical || p)    = 0.000395
Pass threshold        = 0.01
```

The empirical distribution after rejection sampling matches `p` to
KL ≈ 4×10⁻⁴, confirming Chen 2023's unbiasedness theorem.

### §3.2 Geometric chain-break — α × K grid (analytic + 10000-trial empirical)

```
alpha\K          1         2         3         4         5
alpha=0.3   1.3000    1.3900    1.4170    1.4251    1.4275
alpha=0.4   1.4000    1.5600    1.6240    1.6496    1.6598
alpha=0.5   1.5000    1.7500    1.8750    1.9375    1.9688
alpha=0.6   1.6000    1.9600    2.1760    2.3056    2.3834
alpha=0.7   1.7000    2.1900    2.5330    2.7731    2.9412
alpha=0.8   1.8000    2.4400    2.9520    3.3616    3.6893
alpha=0.9   1.9000    2.7100    3.4390    4.0951    4.6856

Empirical sanity (analytic vs mean ± 95% CI):
  alpha=0.5, K=2 → empirical 1.7507 ± 0.0162  vs analytic 1.7500
  alpha=0.5, K=4 → empirical 1.9323 ± 0.0232  vs analytic 1.9375
  alpha=0.7, K=2 → empirical 2.1912 ± 0.0171  vs analytic 2.1900
  alpha=0.7, K=4 → empirical 2.7657 ± 0.0305  vs analytic 2.7731
```

**Trap A evidence**: at α=0.5, K=4, E[tok] = 1.9375, NOT 4 × 0.5 = 2.0.
The geometric chain-break costs ~3% even at moderate α.

### §3.3 Speedup S = E[tok] / (1 + c·K) — break-even and net-loss zones

```
K = 4
c\alpha    0.30    0.40    0.50    0.60    0.70    0.80    0.90
c=0.05    1.188   1.375   1.615   1.921   2.311   2.801   3.413
c=0.1     1.018   1.178   1.384   1.647   1.981   2.401   2.925
c=0.2     0.792   0.916   1.076   1.281   1.541   1.868   2.275
c=0.3     0.648   0.750   0.881   1.048   1.260   1.528   1.861

Break-even alpha (S = 1):
  K=2, c=0.05  →  alpha* = 0.0916
  K=2, c=0.10  →  alpha* = 0.1708
  K=2, c=0.20  →  alpha* = 0.3062
  K=4, c=0.05  →  alpha* = 0.1668
  K=4, c=0.10  →  alpha* = 0.2871
  K=4, c=0.20  →  alpha* = 0.4553
  K=8, c=0.05  →  alpha* = 0.2857
  K=8, c=0.10  →  alpha* = 0.4448
  K=8, c=0.20  →  alpha* = 0.6206
```

**Trap B evidence**: at K=4, c=0.20, α=0.30, speedup = 0.792 < 1 — MTP is a
NET LOSS. Higher c needs higher α (or lower K) to break even.

### §3.4 Greedy fast-path vs random-path (synthetic acceptance, 1000 trials)

```
Trials              = 1000
K                   = 4
Greedy mean emit    = 1.5120
Random mean emit    = 4.5150
ratio random/greedy = 2.9861
Greedy emit min/max = 1/5
Random emit min/max = 1/5
```

The random path emits ~3× more tokens than greedy for this configuration —
because random mode emits a recovered-token at every reject (always at
least 1 emit per K positions), whereas greedy mode short-circuits and the
target's argmax often won't equal the sampled draft.

### §3.5 MTP head parameter count vs Medusa (Trap E)

```
hidden=2048, intermediate=8192, vocab=32000, K=2

MTP per-layer params       =     75,505,664
   enorm                   =          2,048
   hnorm                   =          2,048
   eh_proj (2h*h)          =      8,388,608
   mtp_block_attn          =     16,777,216
   mtp_block_ffn           =     50,331,648
   mtp_block_norms         =          4,096
MTP total (shared lm_head) =    216,549,376
MTP total (separate lm)    =    282,085,376

Medusa per-head            =     73,924,608
   mlp                     =      8,388,608
   lm_head                 =     65,536,000
Medusa per-head MLP-only   =      8,388,608

Ratio MTP / Medusa (shared lm)   = 12.91x
Ratio MTP / Medusa (separate lm) = 1.91x
```

**Trap E evidence**: the MTP head is **NOT lightweight**. Even excluding
LM head and using a dense FFN (the real DeepSeek MTP layer uses MoE which
is ~10× heavier), MTP per-layer is ~9× the Medusa per-head MLP. With LM
head shared, the MTP-stack-vs-Medusa ratio is ~12.9×.

### §3.6 Loader demo (HF → vLLM weight name remap, target_layers=61, mtp_layers=1)

```
input keys        = 193   (target × 61 layers + 4 MTP-layer weights at idx 61 + 2 top-level)
target keys       = 185
mtp keys          = 8
sample renames:
  Path 1 (block weight wrapped):
    model.layers.61.self_attn.q_proj.weight
    → model.layers.61.mtp_block.self_attn.q_proj.weight
  Path 2 (shared embed promoted):
    model.layers.61.embed_tokens.weight
    → model.embed_tokens.weight
  Path 3 (MTP-specific kept):
    model.layers.61.eh_proj.weight              (unchanged)
    model.layers.61.shared_head.head.weight     (unchanged)
```

Verbatim numerics count: ≥85 across the 5 demos. Exceeds brief target ≥80.

### §3.7 Honest demo caveats (writer must quote verbatim)

> The acceptance rate `α` here is SYNTHETIC (uniform Bernoulli) — real
> workloads have α that varies per-position and depends on prompt domain.
> Production DeepSeek-V3 reports α≈0.85+ for the first MTP step.

> The draft cost ratio `c` here is a parameter, not measured. Real H100
> `c` for DeepSeek-V3 MTP is reportedly ≈0.05-0.10; for Llama-3.3-1B
> drafting Llama-3.3-70B ≈0.02-0.05.

> The single-process demo runs rejection sampling on CPU. Real vLLM uses
> Triton kernels (`rejection_greedy_sample_kernel`,
> `rejection_random_sample_kernel`); numerics differ by RNG (use seed=42
> for reproducibility); the algorithm is identical.

> The MTP head architecture demo uses a dense FFN instead of DeepSeek's
> MoE block — the parameter count is then a LOWER BOUND for real
> DeepSeek-V3 MTP (which adds ~256-routed-expert FFN params per MTP
> layer; ~10× the dense FFN cost).

---

## §4 — Language Traps (target 5-7)

Each trap is "easy to write and almost-but-not-quite right". Writer should
explicitly call out **at least 5** in §10.7 with the
"claim → 错 → 为什么 → 源码证据 → Demo/测试" structure.

**Trap A — "MTP doubles throughput / K=4 means 4× speedup."**
Wrong. Speedup is `E[tok] / (1 + c·K)` where `E[tok] = (1 - α^(K+1)) / (1 - α)`.
At α=0.5, K=4: `E[tok]=1.94`, c=0.10 → speedup ≈ 1.38×. K does NOT equal
speedup; **acceptance rate × chain-break geometry IS the speedup**.
Source: `rejection_sampler.py:L424-L430` (chain-break implicit in -1
output buffer); `metadata.py:L10-L24` (per-request K).

**Trap B — "Speculative decoding is always cheaper than autoregressive."**
Wrong. Demo §3.3 shows: at K=4, c=0.20, α=0.30, speedup = 0.792 < 1 —
NET LOSS. The break-even α at K=8, c=0.20 is 0.62 — quite high. The
operator chooses `num_speculative_tokens` and lives with the trade-off.
Source: `rejection_sampler.py` doesn't gate on this (no early-out); the
operator chooses K via `SpeculativeConfig.num_speculative_tokens`.

**Trap C — "Draft model needs to share target's architecture for accuracy."**
Partial. EAGLE/MTP need shared trunk (target hidden state as input — high
coupling, high acceptance). Draft-model approach (Llama-3.3-1B drafting
for Llama-3.3-70B) does NOT share architecture, just same vocabulary;
works fine with α≈0.5-0.7. Ngram doesn't share anything; works at
α≈0.3-0.5. So "shared architecture" is one design point, not a requirement.
Source: `draft_model.py:L36 _raise_if_draft_tp_mismatch` only checks TP
compatibility; `_raise_if_vocab_size_mismatch` only checks vocab.

**Trap D — "Rejection sampling is biased for high temperature."**
Wrong. The Chen 2023 algorithm is provably unbiased for ANY target `p`
and ANY draft `q` (provided `q(x) > 0` wherever `p(x) > 0`). Demo §3.1
verifies KL(empirical || p) ≈ 4×10⁻⁴ for very different `p` and `q`.
Temperature affects α (acceptance rate), but NOT the distribution of
emitted tokens.
Source: `rejection_sampler.py:L491-L504, L659-L703` — `(p - q)_+`
residual sampling is the proof.

**Trap E — "MTP heads are lightweight — just an MLP per position."**
Wrong, at least not for DeepSeek's canonical MTP. `DeepSeekMultiTokenPredictorLayer`
(`deepseek_mtp.py:L92-L97`) uses a **full `DeepseekV2DecoderLayer`** as its
`mtp_block` — including the MoE block with hundreds of experts. Demo §3.5
shows the dense-FFN approximation already gives MTP/Medusa ratio = 12.9×
(shared lm_head). With MoE, the ratio grows further.
Source: `deepseek_mtp.py:L92-L97` (`mtp_block: DeepseekV2DecoderLayer`).

**Trap F — "MTP is a vLLM-side training technique."**
Wrong. vLLM is **inference-only**. MTP heads are TRAINED in the upstream
model's training repo (DeepSeek-V3, Llama EAGLE, etc.) with multi-step CE
loss. vLLM's job is to LOAD the trained MTP weights (`load_weights` in
`deepseek_mtp.py:L239`, with `_rewrite_spec_layer_name` for HF name
remapping at L458) and USE them at inference as draft heads. Verified
via grep: zero hits for `MTPLoss`, `multi_step_ce`, `compute_mtp_loss`,
`mtp_aux_loss`, `.backward()` related to MTP.
(Mirror of Ch09 Trap-E aux-loss reframe — second instance of
training-to-inference reframe.)

**Trap G — "Acceptance rate is a property of the model."**
Wrong. Acceptance rate is a property of (draft, target, prompt, sampling
temperature) — a CONDITIONAL expectation. Same draft-target pair on
different prompts gets different α; same target with different drafts
gets very different α (Medusa < draft-model < MTP < EAGLE typically).
Production systems track running α as live telemetry to detect
distribution shift.
Source: `rejection_sampler.py:L72-L85` (synthetic_conditional_rates is
per-position to model the conditional structure).

---

## §5 — Cross-chapter Links

### Back-pointers (Ch10 reuses):
- **Ch01 (Self-attention)** — the MTP transformer block uses MHA. Same
  forward pass we covered in Ch01.
- **Ch08 (Tensor Parallelism)** — `eh_proj` and the LM head can be
  TP-sharded. Draft models can have their own TP via
  `draft_tensor_parallel_size` (`speculative.py:L93-L98`).
- **Ch09 (Expert Parallelism)** — DeepSeek MTP layer's `mtp_block` is a
  full `DeepseekV2DecoderLayer` with MoE (Trap E). The MoE machinery is
  what we built in Ch09.

### Forward-pointers (Ch10 sets up):
- **Ch15+ (Llama / Mistral / Qwen3 model zoo)** — each model has its own
  `*_eagle3.py` or `*_mtp.py` with similar structure but different trunk
  models.
- **Ch27 (DeepSeek-V3.2 deep-dive)** — DeepSeek-V3 was the first
  production model to ship MTP heads; the chapter covers the
  architectural details of `DeepseekV2MoE` + MLA that Ch10 sidesteps.
- **Ch28 (DeepSeek-V4-Pro)** — `deepseek_v4_mtp.py` adds the `hc_mult`
  carrier hidden-state expansion; covered there.

### Same-cycle hand-off:
- Tester will pin every demo numeric in §3 with `assertEqual` /
  `assertAlmostEqual`. The §3.2 α-K grid is the load-bearing one for the
  writer.
- Writer will use the §3.5 parameter-count comparison as the §10.5
  spine and as Trap E's source of evidence.
- Reviewer will gate on the **two reframe anchors**: §10.3 training-to-inference
  reframe must have sidebar + pivot; §10.2 "no class MTP" reframe must hit
  three anchors (title + hook + body).

---

## §6 — Source pin verification

```
$ cd instances/vllm/source && git rev-parse HEAD
98661fe012c5c467252d4df8411d2f46190e9268
```

Matches the brief's pin at `98661fe`. All line numbers in this document
and in `# REFERENCE:` comments throughout the implementation modules were
verified against this commit. If a future re-run hits drift, re-grep for
the symbol (function/class name) before re-citing — the lesson from
Ch07/Ch08/Ch09.
