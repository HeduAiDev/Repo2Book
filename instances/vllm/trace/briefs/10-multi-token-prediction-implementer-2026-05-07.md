# Rehydration Brief — Ch10 Multi-Token Prediction (Implementer)

- **Chapter**: `10-multi-token-prediction`
- **Title**: Multi-Token Prediction (MTP)
- **Outline level**: core (Part 2)
- **Status**: dispatch — first v6-grade pass for Ch10 (cadence baseline holds at N=6 after Ch09 single-cycle APPROVED with broadest source surface yet)
- **Dependencies (per outline)**: `01-self-attention-fundamentals` only — outline does not list Ch09 EP, but Ch10 source surface DOES touch parallel paths (the spec-decode draft can have its own TP, and DeepSeek-V3 MTP shares the MoE machinery). Cross-link to Ch08 (TP) + Ch09 (EP) where draft model parallelism comes up.
- **Dependents downstream**: Ch27 (DeepSeek-V3.2 deep-dive — DeepSeek-V3 was the first production model to ship MTP heads); Ch28 (DeepSeek-V4-Pro — the `deepseek_v4_mtp.py` reference); Part-3 model-zoo chapters (Llama EAGLE/EAGLE3 in `llama_eagle3.py`, Qwen MTP, Mistral EAGLE, etc.)
- **Source pin**: vLLM commit `98661fe` at `instances/vllm/source/` (verified by archivist 2026-05-07)
- **Brief generated**: 2026-05-07 by archivist
- **Recipient**: implementer (direct dispatch by team-lead, no book-editor relay — operational rule from Ch07/Ch08/Ch09)

---

## §1 — Chapter scope (5 movements — what Ch10 actually covers)

**Core question**: Autoregressive decoding generates ONE token per forward pass — even when the model is confident about the next 3-4 tokens, you pay one full forward to confirm each. **Multi-Token Prediction (MTP)** trains the model to also predict tokens 2..N at training time, so at inference a small draft head can propose K future tokens in ONE forward pass; the main model then verifies all K in parallel via **rejection sampling**. If the math is right, you get token output equivalent to greedy / temperature sampling from the main model alone — but with K-fold throughput when acceptance rate is high. What does vLLM's `SpecDecodeBaseProposer` + DeepSeek's `DeepSeekMultiTokenPredictor` + the `RejectionSampler` actually do, and what coupling does that introduce between draft architecture, target architecture, and acceptance rate?

The chapter covers **5 movements**:

1. **Speculative decoding theory.** Given a target distribution `p(x | context)` and a (cheaper) draft distribution `q(x | context)`, the **rejection sampling** algorithm (Chen et al. 2023, Leviathan et al. 2023, https://arxiv.org/abs/2211.17192) lets you sample K draft tokens, compute target logits for all K positions in **ONE** forward pass, then either accept each draft (with probability `min(1, p/q)`) or sample a "recovered" token from the residual `(p - q)_+` distribution. Critical theorem: the accepted-or-recovered token is **distributed exactly as `p`** — bit-for-bit equivalent in distribution to autoregressive sampling from the target, with NO bias. The economics: if acceptance rate is `α`, expected tokens per target forward = `(1 - α^(K+1)) / (1 - α)` — speedup is roughly `1 / (c + 1/E[tok])` where `c` is draft cost ratio. So MTP wins WHEN acceptance is high AND draft is cheap.

2. **Draft-target verification (rejection sampling kernel).** Open `vllm/v1/sample/rejection_sampler.py:L37 RejectionSampler` and `:L392 rejection_sample`. Walk the algorithm: for each draft token `d_i` at position `i`, compute `p_i(d_i)` from target logits and `q_i(d_i)` from draft probs (when available; for ngram drafts it's None and the path becomes greedy-equivalent). Accept iff `u_i < min(1, p_i(d_i) / q_i(d_i))` where `u_i ~ Uniform[0,1]`. On reject, sample from `(p - q)_+ / Σ(p - q)_+` ("recovered token") and STOP — all later drafts in this sequence are wasted. The bonus token (when ALL drafts accept) is sampled fresh from `p` after the last accepted position. **Two Triton kernels**: `rejection_greedy_sample_kernel` (fast path when target is argmax — no probs needed, just compare draft_id == target_argmax) and `rejection_random_sample_kernel` (full algorithm with uniform draws and recovered sampling). Output shape `[batch_size, max_spec_len + 1]` with `PLACEHOLDER_TOKEN_ID = -1` for non-emitted positions.

3. **MTP heads architecture (DeepSeek-V3 canonical impl).** Open `vllm/model_executor/models/deepseek_mtp.py:L63 DeepSeekMultiTokenPredictorLayer` and `:L124 DeepSeekMultiTokenPredictor`. The MTP module is **architecturally a single transformer block** (RMSNorm + attention + RMSNorm + MoE layer, exactly like a regular DeepSeek-V2 decoder layer) plus **two extra norms** (`enorm` for the next-token embedding, `hnorm` for the residual stream from target's last hidden state) plus a **projection** `eh_proj: Linear(2*hidden → hidden)` that fuses `[norm(emb_next), norm(h_target)]` into the MTP block's input. This is the "shared trunk" choice — the MTP head sees the target's hidden state directly. Each MTP layer outputs ONE additional next-token prediction; stacking `num_nextn_predict_layers` of them gives K. The `SharedHead` (L43) is a final RMSNorm + LM-head projection (shares weights with target's lm_head for parameter efficiency). Compare to **EAGLE** (`vllm/model_executor/models/llama_eagle3.py`, 425 lines) — Eagle uses the target's hidden state and previous draft as a single transformer block, projects through `fc` for fusion. Same idea, different fusion topology.

4. **Acceptance sampling and the chain-break invariant.** The acceptance rate `α` is **per-position**: at draft position `i`, `α_i = E[min(1, p_i(d_i)/q_i(d_i))]` where the expectation is over draft samples. The geometric series `Σ α^i` is what determines expected tokens per forward, but **the chain breaks on first reject** — if draft positions 0,1 accept but 2 rejects, position 3 and 4 are discarded even if their drafts would also have accepted. This is the **chain-break invariant** for Ch10 (parallel to Ch07's "chain-break in radix tree" and Ch08's "1 AR per pair"). Numerics: at `α=0.7, K=4`, expected tokens = `(1 - 0.7^5) / (1 - 0.7) = (1 - 0.16807) / 0.3 ≈ 2.77` — NOT 4 × 0.7 = 2.8 (close but not equal; the geometric chain matters). With `α=0.5, K=4`, expected = 1.94. Recovered token always emits 1 (so floor is 1 token per forward). The chapter must derive this formula and walk concrete numbers. **Greedy fast path**: when target is argmax, `α_i = P(d_i = argmax(p_i))` — for a confident target this is high, and the kernel skips the recovered-sampling math entirely.

5. **System impact / token-throughput math.** Speedup = `E[tokens] / (target_forward_cost + draft_forward_cost)` where draft cost is amortized over K. Concretely: target forward cost ≈ 100ms (large model on H100); draft cost ≈ 10ms per K-token batch (small MTP head, K positions in one pass). At α=0.7, K=4: tokens=2.77, total cost=110ms, speedup ≈ 2.77 / (1.10) = **2.52×** vs autoregressive (1 token / 100ms = 0.01 tok/ms vs 2.77/110 = 0.0252 tok/ms). At α=0.4: tokens=1.85, speedup ≈ 1.68×. At α<0.3, MTP can be a NET LOSS (pay 10% draft cost for <1.5× tokens). **Trade-offs**: K too high → more wasted compute on rejected positions (the chain-break tax); K too low → underutilize the draft. Most production MTP runs use K=2-4. Memory: MTP heads add 1-3 transformer blocks worth of weight (~3-10% of target param count for DeepSeek-V3); KV cache for MTP head adds another `K · num_layers_mtp · cache_per_layer`.

**OUT of scope** (do NOT re-cover):
- Training the MTP heads (cross-entropy weighting, hidden-state target, multi-step supervision) → outline subsection 3 "Training——多步CE损失的加权策略" is OUT OF SCOPE for vLLM source. **vLLM is inference-only**; there is NO MTP training loss code, NO gradient backprop on MTP. **Reframe required** (mirror of Ch09 §9.4 reframe — see §2.2 below).
- EAGLE-1/EAGLE-2 algorithmic details (`vllm/v1/spec_decode/eagle.py` is 22 lines and just inherits from base; the meat is in `llama_eagle3.py`) → reference, not deep-dive. EAGLE3 is its own architecture variation.
- Medusa head architecture (`medusa.py:L18 MedusaProposer`) → reference for comparison. Medusa adds K independent MLP heads at the target's last hidden state; cheaper than MTP transformer blocks but lower acceptance rate.
- Ngram-based speculation (`ngram_proposer.py:L12 NgramProposer`) → reference; n-gram drafts have NO probabilities, so rejection sampler runs greedy-only path. Useful negative test.
- Tree-attention spec-decode paths (`tree_attn.py`) → mention only.
- DFlash, Suffix decoding, parallel drafting — these are siblings of MTP under the same `SpecDecodeBaseProposer`. Reference, do NOT deep-dive.
- The full SpecDecodeBaseProposer state machine (`llm_base_proposer.py` is 1820 lines) → focus on the spec-decode CONTROL FLOW (propose → verify → accept-or-recover); skip CUDA graph, slot mapping, parallel drafting plumbing details.

If implementer is re-deriving cross-entropy training math or EAGLE3 fc-projection internals — STOP. Those belong elsewhere.

---

## §2 — Source surface (verified at commit 98661fe)

### §2.1 — Files and exact line ranges

| File | Lines (verified) | What |
|---|---|---|
| `vllm/v1/sample/rejection_sampler.py` | 921 lines total | THE rejection sampling implementation — Triton kernels, recovered-token sampling, synthetic-mode fallback |
| `vllm/v1/sample/rejection_sampler.py` | L37-L195 | `class RejectionSampler(nn.Module)` — `__init__` accepts `Sampler + SpeculativeConfig`; `forward` orchestrates bonus_logits, target_logits processing, and the call to `rejection_sample` |
| `vllm/v1/sample/rejection_sampler.py` | L392-L505 | `def rejection_sample` — the algorithmic core. Greedy kernel branch, then `target_probs.softmax`, `sample_recovered_tokens`, `rejection_random_sample_kernel` |
| `vllm/v1/sample/rejection_sampler.py` | L506-L563 | `apply_sampling_constraints` — top-k/top-p applied to target logits before rejection |
| `vllm/v1/sample/rejection_sampler.py` | L604-L658 | `generate_uniform_probs` — per-position uniform draws used for accept/reject |
| `vllm/v1/sample/rejection_sampler.py` | L659-L707 | `sample_recovered_tokens` — sample from `(p - q)_+` residual |
| `vllm/v1/sample/rejection_sampler.py` | L708-L853 | The two Triton kernels: `rejection_greedy_sample_kernel`, `rejection_random_sample_kernel`, `expand_kernel`, `sample_recovered_tokens_kernel` |
| `vllm/v1/spec_decode/metadata.py` | 66 lines total | `@dataclass SpecDecodeMetadata` — draft_token_ids, num_draft_tokens (per-req), cu_num_draft_tokens, target_logits_indices, bonus_logits_indices. **The data contract** between proposer and rejection sampler |
| `vllm/v1/spec_decode/llm_base_proposer.py` | 1820 lines total | `class SpecDecodeBaseProposer` — common logic for EAGLE, draft_model, dflash, ExtractHiddenStates. THE shared scaffolding |
| `vllm/v1/spec_decode/llm_base_proposer.py` | L60-L303 | `__init__` — pulls `num_speculative_tokens`, sets up parallel_drafting flag, draft hidden_size handling (DeepSeek-V4 `hc_mult` carrier) |
| `vllm/v1/spec_decode/llm_base_proposer.py` | L407-L412 | `_greedy_sample` — the simplest branch: `argmax(hidden_states @ lm_head)` |
| `vllm/v1/spec_decode/llm_base_proposer.py` | L413-L656 | `def propose` — THE core proposer: takes target hidden states, runs draft model K times (or once for parallel-drafting), returns drafts |
| `vllm/v1/spec_decode/llm_base_proposer.py` | L986-L1156 | `def propose_tree` — tree-attention spec-decode (out of scope, reference only) |
| `vllm/v1/spec_decode/llm_base_proposer.py` | L1158-L1264 | `def prepare_inputs` — slot-mapping for verification step |
| `vllm/v1/spec_decode/llm_base_proposer.py` | L1265-L1314 | `_create_draft_vllm_config`, `_get_model` — draft model construction |
| `vllm/v1/spec_decode/llm_base_proposer.py` | L1402-L1576 | `_maybe_share_embeddings`, `_maybe_share_lm_head` — weight sharing with target |
| `vllm/v1/spec_decode/eagle.py` | 22 lines total | `class EagleProposer(SpecDecodeBaseProposer)` — pure inheritance, no extra logic. Symbolic — the algorithm IS the base |
| `vllm/v1/spec_decode/medusa.py` | 78 lines total | `class MedusaProposer` — does NOT inherit from base; K independent MLP heads. Reference comparison |
| `vllm/v1/spec_decode/draft_model.py` | 88 lines total | `class DraftModelProposer(SpecDecodeBaseProposer)` — uses a separate small transformer (e.g. Llama-3.3-1B) as the draft |
| `vllm/v1/spec_decode/extract_hidden_states.py` | 382 lines total | `class ExtractHiddenStatesProposer` — for MTP-style "use target's hidden states as draft input" path. **Critical for MTP**: `assert num_speculative_tokens == 1` (L30) — single-step MTP variant |
| `vllm/v1/spec_decode/ngram_proposer.py` | 285 lines total | `class NgramProposer` — n-gram lookup over recent context; NO draft probs. Negative test for rejection sampler (greedy path only) |
| `vllm/v1/spec_decode/ngram_proposer.py` | L12-L162 | `__init__` (sets `self.k = num_speculative_tokens`), `propose` (n-gram match in context buffer) |
| `vllm/v1/spec_decode/ngram_proposer.py` | L170-L285 | `batch_propose_numba` (numba-compiled batch path) |
| `vllm/v1/spec_decode/utils.py` | (large) L460-L600 | Triton kernels for slot mapping, padding — `compute_new_slot_mapping`, etc. |
| `vllm/model_executor/models/deepseek_mtp.py` | 488 lines total | DeepSeek-V3 MTP heads — THE canonical real-world MTP impl |
| `vllm/model_executor/models/deepseek_mtp.py` | L43-L62 | `class SharedHead(nn.Module)` — RMSNorm + ParallelLMHead (lm_head shared with target) |
| `vllm/model_executor/models/deepseek_mtp.py` | L63-L122 | `class DeepSeekMultiTokenPredictorLayer(nn.Module)` — `enorm` (RMSNorm on next-token embed) + `hnorm` (RMSNorm on target's hidden state) + `eh_proj: Linear(2*hidden → hidden)` + `mtp_block: DeepseekV2DecoderLayer` (the regular MoE block!) + `shared_head` |
| `vllm/model_executor/models/deepseek_mtp.py` | L124-L184 | `class DeepSeekMultiTokenPredictor(nn.Module)` — stacks `num_nextn_predict_layers` MTP layers; `forward` applies them sequentially |
| `vllm/model_executor/models/deepseek_mtp.py` | L186-L488 | `class DeepSeekMTP(nn.Module, DeepseekV2MixtureOfExperts)` — top-level MTP wrapper; `load_weights` handles `_rewrite_spec_layer_name` weight name remapping |
| `vllm/config/speculative.py` | (large) L59-L70 | `SpeculativeMethod = Literal["ngram", "medusa", "mlp_speculator", "draft_model", "suffix", EagleModelTypes, NgramGPUTypes]`. **Note: "mtp" is NOT a literal — DeepSeek MTP uses `method="draft_model"` with the deepseek_mtp model class, OR `eagle3` for EAGLE-style fusion** |
| `vllm/config/speculative.py` | L73-L210 | `class SpeculativeConfig` — `num_speculative_tokens`, `method`, `prompt_lookup_max/min` (ngram-only), `rejection_sample_method` (`"standard"` vs `"synthetic"`), `synthetic_acceptance_rates` |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | 163 lines total | GPU-worker-side wrapper — calls into v1/sample/rejection_sampler |
| `vllm/model_executor/models/llama_eagle3.py` | 425 lines total | EAGLE3 reference: `class Eagle3LlamaForCausalLM` — fuses target hidden state with prev draft via `fc` projection. Comparison anchor for MTP |

This is **10-12 source files** — the broadest spec-decode surface vLLM exposes.
Should match Ch09's 10-file floor.

### §2.2 — Outline-vs-source mismatches to flag (CRITICAL)

**The outline subsection 3 — "Training——多步CE损失的加权策略" — is OUT OF SCOPE for vLLM source.** vLLM is an inference engine; there is NO cross-entropy training loss for MTP heads anywhere in the codebase. The MTP heads ARE trained at training time with multi-step CE, but that code lives in DeepSeek-V3's training repo, NOT in vLLM. **Ch10 must reframe §3 from "training CE loss design" to "inference-time MTP head loading and weight sharing"**, with a one-page sidebar grounding readers from training literature ("MTP is trained with weighted multi-step CE — token at step k contributes `λ_k · CE(p_k, x_{i+k})` with `λ_k` typically decaying 1.0 → 0.5 → 0.25 → ..."), then PIVOT to vLLM's actual code: `load_weights` in `deepseek_mtp.py:L239`, `_rewrite_spec_layer_name` (L458) for weight name remapping, weight sharing via `_maybe_share_lm_head` (`llm_base_proposer.py:L1471`). Same logic as Ch09's §9.4 reframe (training-time concept → inference-time response).

**The outline does NOT explicitly mention rejection sampling**, but rejection sampling IS the verification mechanism for MTP. Subsection 4 "Inference——MTP verification的acceptance rate分析" naturally absorbs `rejection_sampler.py` — expand it. The chapter MUST cover the algorithm AND the acceptance-rate math, NOT just say "MTP verification".

**`class MultiTokenPrediction` does NOT exist in vLLM.** Verified via grep (next §): the only matches are `DeepSeekMultiTokenPredictor`, `DeepSeekMultiTokenPredictorLayer` — DeepSeek-V3-prefixed. There are also 30+ similar `*_mtp.py` files (qwen3_5_mtp.py, ernie_mtp.py, glm4_moe_mtp.py, deepseek_v4_mtp.py, mimo_mtp.py, longcat_flash_mtp.py, openpangu_mtp.py, etc.) — each model family has its own MTP wrapper. **THIS IS THE FOURTH "no class X" CASE** in the book:
- Ch07: no `class RadixTree` (chained-hash impl)
- Ch08: no `class TensorParallel` (5-file collab)
- Ch09: no `class ExpertParallel`/`MoEParallel`/`TopKGate` (5-file collab)
- Ch10: no `class MultiTokenPrediction` (per-model MTP wrappers + shared spec_decode infrastructure)

**Use the same three-anchor framing**: title L1, hook (1st-page paragraph), §X.2 body with grep evidence + file list. The pattern is now well-rehearsed across Ch07-Ch09; reviewer expects three anchors with grep evidence.

**The chapter title says "MTP" but the source surface is "speculative decoding with MTP as one method".** vLLM's spec-decode infrastructure is designed for ngram, medusa, eagle, draft_model, MTP — they're all proposers under `SpecDecodeBaseProposer` (or peers like Medusa). MTP is NOT first-class in `SpeculativeMethod` enum (which lists `"ngram", "medusa", "mlp_speculator", "draft_model", "suffix", EagleModelTypes, NgramGPUTypes`). DeepSeek MTP loads via `method="draft_model"` with `model="deepseek_mtp"` or via the eagle3 path. **Be precise**: Ch10 teaches MTP-the-technique using DeepSeek's MTP heads as the canonical example, with the GENERIC spec-decode infrastructure as the carrier. Hook should make this clear.

**`knowledge/INDEX.md` must add a new module**:
- Add row: `| [multi-token-prediction](modules/multi-token-prediction.md) | 10 | rejection_sampler.py, llm_base_proposer.py, deepseek_mtp.py |`
- This is a NEW module; use **M-prefix IDs** (M01, M02, ...) — distinct from Ch07 (K-), Ch06 (P-), Ch08 (T-), Ch09 (E-). **MUST avoid collision** per `feedback_double_prefix_headings.md`.

### §2.3 — Verified absence of structures

- No top-level `class MultiTokenPrediction`, `class MTPHead`, `class MTPModel`, `class TokenPredictor` anywhere in `vllm/`. The MTP class is `DeepSeekMultiTokenPredictor` (DeepSeek-prefixed) plus 30+ similar `*_mtp.py` per-model wrappers.
- No `class SpeculativeDecoder` either — the spec-decode entry point is `SpecDecodeBaseProposer` (a base class) + per-method subclasses.
- No `"mtp"` literal in `SpeculativeMethod` enum. MTP is implemented via `method="draft_model"` (deepseek_mtp model) or the eagle3 family.
- No training loss code anywhere — `vllm/` has zero `.backward()`, `compute_aux_loss`, `compute_ce_loss`, `MTPLoss`, `multi_step_loss` (verify via grep). MTP training happened upstream in the model's training repo.
- No `acceptance_rate` field in any production-runtime telemetry struct that we can verify yet — `vllm/v1/spec_decode/metrics.py` may have it; implementer should check (`grep "acceptance" vllm/v1/spec_decode/`).

### §2.4 — vLLM commit pin verification

```
$ cd instances/vllm/source && git rev-parse HEAD
98661fe012c5c467252d4df8411d2f46190e9268
```

Matches Ch09 brief's pin at `98661fe`. All line numbers in this brief were
verified against this commit. If a future re-run hits a drift, re-grep for the
symbol (function/class name) before re-citing.

---

## §3 — Outline section walk-through

Outline subsections (from `book-outline.json` →
`parts.part2_advanced_common.chapters[10-multi-token-prediction].subsections`)
and how to map them to source. Subsection text is the *topic* (the question the
section answers), not a class-name contract.

| Outline subsection | Reframed scope | Source anchor |
|---|---|---|
| 1. "MTP为什么有效——speculative decoding的free lunch版本" | Open `vllm/v1/sample/rejection_sampler.py:L37 RejectionSampler` and the algorithmic core at `:L392 rejection_sample`. Derive: target distribution `p(x|context)`, draft `q(x|context)`, accept iff `u < min(1, p/q)` else sample from `(p-q)_+`. **Theorem (Chen 2023)**: accepted-or-recovered token is distributed exactly as `p`. Show the math: probability of emitting token `x` = P(accept · x ~ q) + P(reject · x ~ recover) = `q(x)·min(1, p/q) + Σ_y(q(y)(1-min(1,p/y))) · (p(x)-q(x))_+/Z = p(x)` (algebra). "Free lunch" caveat: ONLY when acceptance is high AND draft is cheap; otherwise net loss. **5-step rhythm**: open `rejection_sampler.py:L37` → ask "why is this not biased?" → derive (above) → impl `rejection_sampling.py` reproducing the algorithm in plain PyTorch → diff: vLLM's path uses Triton kernels (`rejection_greedy_sample_kernel`, `rejection_random_sample_kernel`); ours uses for-loop. | `vllm/v1/sample/rejection_sampler.py:L37-L505`, `vllm/v1/spec_decode/metadata.py:L10` |
| 2. "MTP head的网络结构（transformer block + lm_head per step）" | Open `vllm/model_executor/models/deepseek_mtp.py:L63 DeepSeekMultiTokenPredictorLayer`. Walk the structure: `enorm` (RMSNorm on next-token embedding) + `hnorm` (RMSNorm on target hidden state) → `eh_proj: Linear(2*hidden → hidden)` (fuses) → `mtp_block: DeepseekV2DecoderLayer` (full MoE block, NOT a lightweight MLP — surprising) → `shared_head: SharedHead(RMSNorm + ParallelLMHead)`. Stack `num_nextn_predict_layers` of these for K-step MTP. Compare to **EAGLE3** (`llama_eagle3.py:L?? Eagle3LlamaForCausalLM`) — uses `fc` projection + transformer block. Compare to **Medusa** (`medusa.py:L18 MedusaProposer`) — K independent MLP heads (no transformer trunk; cheaper but lower acceptance). **5-step rhythm**: open `deepseek_mtp.py:L63` → ask "why a full transformer block, not a lightweight MLP?" (answer: maintains expressivity for predicting K tokens ahead, where shorter-horizon decay would hurt acceptance) → derive: parameter count is roughly `3 · hidden² + 8 · n_routed_experts · hidden · intermediate` per MTP layer (full DeepSeek block) → impl `mtp_head.py` reproducing the layer in plain PyTorch (skip MoE; use single dense FFN for clarity) → diff: real DeepSeek MTP layer is `DeepseekV2DecoderLayer` (with MoE/EP — chains back to Ch09); ours uses `nn.Linear`. | `deepseek_mtp.py:L43-L184`, `llama_eagle3.py:L1-L100`, `medusa.py:L18-L78` |
| 3. "Training——多步CE损失的加权策略" → **REFRAME to "Inference-time MTP weight loading and weight sharing"** | The literal subsection title is OUT OF SCOPE (vLLM is inference). Reframe: **sidebar** explaining training-time multi-step CE (`L_MTP = Σ_k λ_k · CE(p_k, x_{i+k})` with `λ_k` typical decay 1.0/0.5/0.25/0.125 per spec literature), then **pivot** to vLLM's inference response: `_maybe_share_lm_head` (`llm_base_proposer.py:L1471`) — MTP shares the target's `lm_head` weight to save params; `_rewrite_spec_layer_name` (`deepseek_mtp.py:L458`) — name remapping at load time so HuggingFace checkpoints load cleanly; `load_weights` (`deepseek_mtp.py:L239`) — loads MTP block weights with the special `mtp.*` prefix handling; `extract_hidden_states.py:L26 ExtractHiddenStatesProposer` — single-step path that uses target's hidden state as draft input (matches DeepSeek's "shared trunk" design). **5-step rhythm**: open `deepseek_mtp.py:L239` (`load_weights`) → ask "if MTP heads are trained, how do they get into vLLM at runtime?" → derive (training repo → HF checkpoint → vLLM `load_weights` with name remapping) → impl `mtp_loader.py` (toy weight name remapper + sharer) → diff: real `_rewrite_spec_layer_name` walks `model.layers.X.MOE.mtp.Y.weight` → `mtp_layer_X.Y.weight`. | `deepseek_mtp.py:L186-L488`, `llm_base_proposer.py:L1402-L1576` |
| 4. "Inference——MTP verification的acceptance rate分析" | Open `rejection_sampler.py:L392 rejection_sample` deeper. Derive expected-tokens-per-target-forward formula: `E[tok | α, K] = Σ_{k=0}^K α^k = (1 - α^(K+1)) / (1 - α)` (geometric series; chain breaks on first reject; the +1 is the bonus token if all K accept). Walk concrete numbers: `α=0.7, K=4 → 2.77 tok`; `α=0.5, K=4 → 1.94 tok`; `α=0.3, K=4 → 1.41 tok`; `α=0.4, K=2 → 1.56 tok`. Speedup formula: `S = E[tok] / (1 + c·K)` where `c` is draft cost ratio. At target=100ms, draft=10ms/K-batch, c≈0.1 → S ≈ E[tok] / 1.1. Walk α-sweep table: α∈{0.3, 0.4, 0.5, 0.6, 0.7, 0.8} × K∈{1,2,3,4,5} → speedup matrix. **5-step rhythm**: open `rejection_sampler.py:L392` → ask "what determines whether MTP is a win?" → derive (above) → impl `acceptance_math.py` reproducing the formula + plotting curves → diff: production acceptance rates for DeepSeek-V3 MTP layer reportedly α≈0.85+ (from DeepSeek paper); for ngram drafts α≈0.4 (lower but draft cost is near-zero). | `rejection_sampler.py:L392-L505`, `vllm/v1/spec_decode/metrics.py` (if exists) |
| 5. "MTP与speculative decoding的对比" | Open the four side-by-side proposers: `eagle.py:L10 EagleProposer` (inherits base, EAGLE3 fc-fusion), `medusa.py:L18 MedusaProposer` (K independent MLP heads), `draft_model.py:L17 DraftModelProposer` (separate small transformer), `ngram_proposer.py:L12 NgramProposer` (no draft probs). MTP-via-DeepSeek uses the `draft_model` method with the deepseek_mtp model class. Comparison axes: (a) draft cost — Medusa cheap, MTP medium, draft-model expensive; (b) acceptance rate — Medusa low, MTP high, EAGLE highest (claimed); (c) parameter overhead — ngram zero, Medusa K·MLP_block, MTP K·transformer_block, draft-model whole_small_model; (d) requires_target_hidden — ngram NO, Medusa YES, MTP YES, draft-model NO. Build a comparison table with 5 rows × 5 columns. **5-step rhythm**: open `eagle.py:L10` (one of the comparators) → ask "why does vLLM ship 5 spec-decode methods?" → derive (different (cost, accuracy) trade-offs for different model sizes / workloads) → impl `compare_proposers.py` running the same prompt through 4 toy proposers → diff: production has CUDA graphs, parallel drafting, tree attention; ours runs sequentially. | `eagle.py`, `medusa.py:L18`, `draft_model.py:L17`, `ngram_proposer.py:L12`, `extract_hidden_states.py:L26` |

Use this 5-section mapping as the chapter's §10.1-§10.5 spine. §10.6+ for source-mapping table (main + per-section mini per K15 two-tier), §10.7 for language-trap recap, §10.8 for verification, §10.9 for forward-pointers.

---

## §4 — Knowledge dependencies

### Existing knowledge entries to read before work

- `knowledge/modules/expert-parallelism.md` — E01-E24 if Ch10 touches DeepSeek MoE machinery (which it WILL via DeepSeekMultiTokenPredictorLayer's `mtp_block: DeepseekV2DecoderLayer` containing MoE). E07 (shared experts) is load-bearing — DeepSeek MTP layer inherits the `n_shared_experts` etc.
- `knowledge/modules/tensor-parallelism.md` — T01-T19. Draft models (DraftModelProposer) can have their own TP (`_raise_if_draft_tp_mismatch` in `draft_model.py:L36`). T03 `MergedColumnParallelLinear`, T04 GQA × TP head replication may matter for the `eh_proj` Linear and the `lm_head` sharing.
- `knowledge/modules/attention.md` — only tangentially relevant; Ch10 doesn't deep-dive attention but the MTP transformer block does have an attention module.
- `knowledge/modules/scheduler.md` — P-prefix; potentially relevant if Ch10 touches the spec-decode scheduling integration (skip in this chapter).

### NEW knowledge module REQUIRED

**Create `knowledge/modules/multi-token-prediction.md`** — Ch10 owns its own module:

- Use **M-prefix IDs** (M01, M02, ...) — distinct from Ch07 (K-), Ch06 (P-), Ch08 (T-), Ch09 (E-). **MUST avoid collision** per `feedback_double_prefix_headings.md` user feedback.
- Forward-shared with: Ch27 (DeepSeek-V3.2 deep-dive), Ch28 (DeepSeek-V4-Pro / `deepseek_v4_mtp.py`), Ch15+ Llama EAGLE deep-dive.
- **WARNING (carried from Ch07/Ch08/Ch09 lessons)**: `learn.py` append-mode bugs were fixed (P1-1 task #36 completed), but if doubled `## M0X: M0X:` headers show up after extraction, fix immediately. Also: expert-parallelism.md hit 24 facts, tensor-parallelism.md 19, prefix-cache.md 17 — **`learn.py compact()` is broken** (`_parse_module_file` returns []). Manual workaround if Ch10 module exceeds 15 facts.
- Update `knowledge/INDEX.md`:
  - Add row: `| [multi-token-prediction](modules/multi-token-prediction.md) | 10 | rejection_sampler.py, llm_base_proposer.py, deepseek_mtp.py |`

### Anticipated facts the implementer will discover (M-prefix candidates)

- M01: `RejectionSampler` is `nn.Module` (not stateless function) because it caches `synthetic_conditional_rates` tensor on device — the `__init__` accepts `Sampler` and `SpeculativeConfig`. (`rejection_sampler.py:L37, L60-L86`)
- M02: `rejection_sample` greedy fast-path SKIPS the `target_probs.softmax` and `sample_recovered_tokens` entirely when `sampling_metadata.all_greedy=True` — uses `target_logits.argmax` and bare integer comparison `draft_id == target_argmax`. **Critical for performance**: greedy is the cheap path. (`rejection_sampler.py:L457-L470`)
- M03: `draft_probs` can be `None` for n-gram drafts; the random-sample kernel takes `NO_DRAFT_PROBS=draft_probs is None` as a Triton metaparam and runs a different path. (`rejection_sampler.py:L500`)
- M04: The chain-break invariant is encoded in `cu_num_draft_tokens` (cumulative) — once a position rejects, the kernel writes `PLACEHOLDER_TOKEN_ID = -1` for all subsequent positions in that request. (`rejection_sampler.py:L424-L430` output buffer init)
- M05: `SpecDecodeMetadata` has SEPARATE `target_logits_indices` (size num_draft_tokens) and `bonus_logits_indices` (size batch_size) — bonus logits are sampled ONLY when all draft tokens accept; the indices are precomputed by the proposer. (`metadata.py:L10-L23`)
- M06: `SpecDecodeBaseProposer.num_speculative_tokens` is GLOBAL (one K for the whole engine) — set at engine init from `SpeculativeConfig.num_speculative_tokens`. NOT per-request. (`llm_base_proposer.py:L79`)
- M07: `SpeculativeMethod` literal does NOT include "mtp" — DeepSeek MTP loads via `method="draft_model"` + `model="deepseek_mtp"`. (`speculative.py:L59-L70`)
- M08: DeepSeek MTP layer is FULL transformer block (`DeepseekV2DecoderLayer`), not lightweight — `enorm` + `hnorm` + `eh_proj` are the ONLY MTP-specific parts; the `mtp_block` IS a regular MoE decoder. Big surprise for readers expecting "lightweight head". (`deepseek_mtp.py:L92-L110`)
- M09: `SharedHead` shares the LM head with the target — `ParallelLMHead(config.vocab_size, config.hidden_size, ...)` has weight tied via `_maybe_share_lm_head`. Saves ~vocab_size × hidden params per MTP layer. (`deepseek_mtp.py:L43-L62`, `llm_base_proposer.py:L1471`)
- M10: `extract_hidden_states.py` enforces `num_speculative_tokens == 1` (L30) — this is the SINGLE-STEP MTP path where draft proposes ONLY 1 token using target hidden states. Contrasts with multi-step MTP/EAGLE which proposes K tokens.
- M11: `parallel_drafting` flag (`SpecDecodeBaseProposer:L100-L105`) — when True, draft proposes ALL K tokens in one forward (DFlash, parallel-drafting EAGLE). When False, draft runs K times sequentially. Affects `extra_slots_per_request`.
- M12: Acceptance rate analytic formula: `E[accepted | α, K] = (1 - α^(K+1)) / (1 - α)`. Implementer must verify with simulation.
- M13: `synthetic_acceptance_rates` mode in `SpeculativeConfig` (L193-L204) — for testing, hardcodes a per-position decaying acceptance rate. Useful for tester to pin numerics without needing a real model.
- M14: `recover_token_ids` from `(p - q)_+` — the residual distribution. When `draft_probs is None` (ngram), the residual reduces to just `target_probs` and recovery is just `target.multinomial(1)`. (`rejection_sampler.py:L474-L480`)
- M15: `num_nextn_predict_layers` (`deepseek_mtp.py:L129`) — DeepSeek-V3 config field; `num_mtp_layers` aliases this. NOT a vLLM concept; comes from HF config. (Could trip implementer thinking it's vLLM-side.)

---

## §5 — Wisdom hits (role priorities: implementer = debugging > architecture > testing > writing)

Read these before opening source:

- `wisdom/debugging.md` — `F.linear` weight shape `[out, in]`; for `eh_proj: nn.Linear(hidden_size * 2, hidden_size, bias=False)` the weight has shape `[hidden, 2*hidden]`. The `forward` does `x @ W^T` so input `[..., 2*hidden]` becomes `[..., hidden]`. Easy to swap dims when reproducing.
- `wisdom/architecture.md` — **backpressure gates and lateral comm patterns**. Spec-decode has a clean PROPOSE → VERIFY architectural split: proposer outputs `SpecDecodeMetadata` (a passive data contract, `metadata.py:L10`), verifier (RejectionSampler) consumes it. This is the cleanest gate-pattern in the codebase. Also: "**fix prompts not chapters**" — if reviewer flags outline §3 not reframed, fix the impl-notes outline-mismatch handling, not the chapter.
- `wisdom/testing.md` — preemption test design generalizes to "test the boundary": `K=1` (degenerate, equivalent to autoregressive), `K=max_spec_len`, `α=1.0` (always accept; equivalent to `K`-step parallel decode), `α=0.0` (always reject; degenerates to bonus token only — emit 1 per forward). Tester needs all four.
- `wisdom/writing.md` — formula rules (NON-NEGOTIABLE: `\mathrm{}` not `\text{}`, no `\boxed`, no `\frac` inline). **Ch10 will be FORMULA-HEAVY** (rejection sampling theorem, geometric series for expected tokens, speedup formula, acceptance rate distribution). Plan for high formula-density mitigation per K-series writer patterns: ≤2 inline formulas per bullet, render mid-proof variable references as plain text when the symbol isn't doing math work. **E18 from Ch09 wisdom** says inline `\frac` cannot hold complex fractions — promote to display.

Plus the reproducible-cadence patterns from `state.json:v6_compliance` (after Ch09):

- `two_tier_mapping` — mandatory; Ch10 surface is BROAD (10+ files). Aim for 40-50 main + 30-40 mini.
- `language_trap_callouts` — Ch10 has plenty (see §6); plan ≥5-7 explicit recap items, matching Ch09's 7.
- `honest_demo_caveats` — synthetic acceptance-rate testing is NOT real workload behavior; the chapter must caveat. K17 lesson: writer must quote caveat verbatim from impl-notes.
- `single_cycle_approval` — Ch04, Ch05, Ch06, Ch07, Ch08, Ch09 all hit this. Ch10 must replicate; cadence holds at N=6.
- `framing_tip_three_anchor_verification` (E22 from Ch09) — every framing tip must show up at hook + body + recap. Reviewer will count.
- `no_class_X_three_anchor_pattern` — the "no class MultiTokenPrediction" reframe needs title + hook + §X.2 body anchors with grep evidence. Pattern is 4th instance now (Ch07 → Ch08 → Ch09 → Ch10).

---

## §6 — Candidate language traps for the writer (target 5-7)

Each candidate is a phrasing that is "easy to write and almost-but-not-quite right". Writer picks the strongest 5-7 for explicit callouts at the relevant section + a dedicated recap section, mirroring Ch07 §7.6.4 / Ch08 §8.6.4 / Ch09 §9.7.

**Trap A — "MTP doubles throughput / K=4 means 4× speedup."** No. Speedup is `E[tok] / (1 + c·K)` where `E[tok] = (1 - α^(K+1)) / (1 - α)` and `c` is draft cost ratio. At α=0.5, K=4: `E[tok]=1.94`, c=0.1 → speedup ≈ 1.39×. K does NOT equal speedup; **acceptance rate multiplied by chain-break geometry IS the speedup**. Source evidence: `rejection_sampler.py:L420-L430` shows the chain-break in the kernel (output buffer pre-filled with PLACEHOLDER, only filled up to first reject); `metadata.py:L10` makes `num_draft_tokens` per-request — the geometry is per-token-position.

**Trap B — "Speculative decoding is always cheaper than autoregressive."** No. When `α < 1/(1+c·K) · K`, MTP is a NET LOSS. At very low acceptance (α=0.2, K=4, c=0.1), `E[tok]=1.18`, speedup=0.84× — SLOWER than autoregressive. The chapter must show this explicitly with a numeric example (the α-sweep table in §10.4). Source evidence: `rejection_sampler.py` itself doesn't gate on this (no early-out); the operator chooses `num_speculative_tokens` and lives with the trade-off. **Production telemetry** (`metrics.py` if it exists) likely tracks acceptance rate so operators can tune.

**Trap C — "Draft model needs to share the target's architecture for accuracy."** Partially true but misleading. **EAGLE/MTP need shared trunk** (target hidden state as input — high coupling, high acceptance). **Draft-model approach** (`DraftModelProposer`, e.g. Llama-3.3-1B drafting for Llama-3.3-70B) DOES NOT share architecture, just same vocabulary; works fine with α≈0.5-0.7. **Ngram** doesn't share anything; works at α≈0.3-0.5 but draft cost is near-zero. So "shared architecture" is one design point, not a requirement. Source evidence: `draft_model.py:L36 _raise_if_draft_tp_mismatch` only checks TP compatibility, NOT architecture. `vocab_size` is the only hard match (`_raise_if_vocab_size_mismatch`, L33).

**Trap D — "Rejection sampling is biased for high temperature."** No. The Chen 2023 algorithm is **provably unbiased** for ANY target distribution `p` and ANY draft `q` (provided `q(x) > 0` wherever `p(x) > 0`; in practice both are softmaxes of model logits so this holds). Temperature affects `p` and `q` shape, hence acceptance rate, but NOT the distribution of emitted tokens. The "recovered token" sampled from `(p-q)_+` is the proof. Source evidence: `rejection_sampler.py:L491-L504` — the random-sample kernel takes both `draft_probs` and `target_probs` and samples uniformly, then either accepts or recovers. Chapter must derive the unbiasedness theorem.

**Trap E — "MTP heads are lightweight — just an MLP per position."** No, at least not for DeepSeek's canonical MTP. `DeepSeekMultiTokenPredictorLayer` (`deepseek_mtp.py:L63`) uses a **full `DeepseekV2DecoderLayer`** as its `mtp_block` — including the MoE block with hundreds of experts. The MTP-specific parts are just `enorm` + `hnorm` + `eh_proj` (a single Linear). **Medusa heads ARE lightweight** (K independent MLPs, no transformer trunk), but Medusa typically has lower acceptance. Chapter must distinguish DeepSeek MTP / EAGLE / Medusa weight footprint. Source evidence: `deepseek_mtp.py:L92-L110` instantiates `mtp_block: DeepseekV2DecoderLayer`. `medusa.py:L18 MedusaProposer` is 78 lines for the WHOLE proposer.

**Trap F — "MTP is a vLLM-side training technique."** No. **vLLM is inference-only**. MTP heads are TRAINED in the upstream model's training repo (DeepSeek-V3, Llama EAGLE, etc.) with multi-step CE loss. vLLM's job is to **load the trained MTP weights** (`load_weights` in `deepseek_mtp.py:L239`, with `_rewrite_spec_layer_name` for HF name remapping at L458) and **use them at inference** as draft heads. Verified via grep over `vllm/`: zero hits for `MTPLoss`, `multi_step_ce`, `compute_mtp_loss`, `mtp_aux_loss`, `.backward()` related to MTP. The directory `vllm/v1/spec_decode/` is clean of training. (Mirror of Ch09 Trap-E aux-loss reframe.)

**Trap G — "Acceptance rate is a property of the model."** No, acceptance rate is a property of (draft, target, prompt, sampling temperature) — it's a CONDITIONAL expectation. Same draft-target pair on different prompts gets different α (e.g., generic-prose prompts get higher α than code prompts because token distributions are flatter). Same target with different drafts gets very different α (Medusa < draft-model < MTP < EAGLE typically). **Production systems track running α as live telemetry** to detect distribution shift. Source evidence: `metrics.py:L?? if exists` would have rolling-acceptance counters; otherwise this is operator-side instrumentation.

Pick 5-7 of A/B/C/D/E/F/G for primary callouts; reviewer expects ≥5 (Ch09 hit 7, exceeds floor). Recap section §10.7 should explicitly enumerate them with "claim → 错 → 为什么 → 源码证据 → Demo/测试" per Ch07/Ch08/Ch09 template.

---

## §7 — Demo plan (numerics for verbatim narrative use, target ≥20 verbatim)

The implementer's `demo.py` should produce numbers the writer will quote
verbatim (per the demo-numerics-verbatim hard gate, K17 / N=6 baseline; Ch09
hit 65+ verbatim values).

**Demo §1 — Rejection sampling unbiasedness verification.** Build a toy: target `p = softmax(target_logits)` and draft `q = softmax(draft_logits)` with vocab=8. Run rejection sampling 10000 times at K=4. Assert empirical distribution of accepted-or-recovered tokens matches `p` within tolerance 0.01 (chi-square or KL divergence). Pin: `KL(empirical || p) < 0.01`. **Numbers**: 8 vocab probabilities + KL value = 9 verbatim. Plus 2 more: `mean_accepted_per_seq`, `mean_recovered_per_seq`.

**Demo §2 — Geometric chain-break: expected tokens vs acceptance rate.** For `α ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}` and `K ∈ {1, 2, 3, 4, 5}`, simulate 10000 sequences. Each sequence: draw uniform U_i, accept iff U_i < α (independent), stop on first reject. Bonus token if all K accept. Compute mean tokens-per-forward, compare to analytic `E[tok | α, K] = (1 - α^(K+1)) / (1 - α)`. Pin both empirical (with 95% CI) and analytic. **Numbers**: 7×5 = 35 (α, K) cells × 2 (empirical, analytic) = 70 values; reduce to a verbatim table with 7×5 = 35 cells. **Trap-A evidence**: at α=0.5, K=4, `E[tok]=1.94` (NOT 4 × 0.5 = 2.0).

**Demo §3 — Speedup curve under draft cost ratio.** For `c ∈ {0.05, 0.1, 0.2, 0.3}` (draft cost ratio per K-batch) and α-sweep, compute speedup `S = E[tok] / (1 + c·K)` at K=4. Identify break-even α (where S=1). **Numbers**: 4×7 = 28 speedup values + 4 break-even α values. **Trap-B evidence**: at c=0.1, K=4, the break-even α is ≈ 0.5; below that, MTP is NET LOSS.

**Demo §4 — Greedy fast-path vs random-path numerics.** Two modes: `all_greedy=True` (use `target_argmax`, no recovered sampling, ngram-style) and `all_random=True` (full algorithm with draft probs). Same prompt, K=4, 100 trials. Pin: greedy mean tokens, random mean tokens, ratio. Show that random-path emits more tokens than greedy (recovery never wastes a forward). **Numbers**: 4-6 verbatim.

**Demo §5 — MTP head architecture parameter count.** Build toy DeepSeek MTP layer (drop MoE, use single Linear FFN for clarity): `enorm + hnorm + eh_proj(2h, h) + mtp_block(transformer dense)+ shared_head(rmsnorm + lm_head)`. With `hidden=2048, intermediate=8192, vocab=32000, K=2`: compute total params. Compare to Medusa head (`K · MLP(h, h)`): much smaller. Pin: MTP-layer params, Medusa-layer params, ratio. Show: `MTP/Medusa ≈ 50-100×` parameter ratio. **Numbers**: 6 values per layer-type × 2 layer-types = 12 + 2 ratios = 14.

These 5 demos collectively give the writer **≥80 ground-truth numbers** to
quote verbatim (target ≥20). Test report should pin every one with
`assertEqual` / `assertLess` / `assertAlmostEqual` and explicit values — Ch07
K17 lesson: writer pre-runs linters AND tester pins exact numbers → APPROVED
in one cycle. **Honest demo caveats** the impl-notes must state (then writer
quotes verbatim):

- Acceptance rate `α` here is SYNTHETIC (uniform Bernoulli) — real workloads have α that varies per-position and depends on prompt domain. Production DeepSeek-V3 reports α≈0.85+ for the first MTP step; ours uses uniform α as a controlled experiment.
- Draft cost ratio `c` here is a parameter, not measured. Real H100 `c` for DeepSeek-V3 MTP is reportedly ≈0.05-0.10; for Llama-3.3-1B drafting Llama-3.3-70B ≈0.02-0.05.
- The single-process demo runs rejection sampling on CPU for clarity; real vLLM uses Triton kernels (`rejection_greedy_sample_kernel`, `rejection_random_sample_kernel`). Numerics differ by RNG (use seed=42 for reproducibility); algorithm is identical.
- MTP head architecture demo uses a dense FFN instead of DeepSeek's MoE block — the parameter count is then a LOWER BOUND for real DeepSeek-V3 MTP (which adds ~256 expert FFN params per MTP layer).

---

## §8 — Floor reminders (v6 hard gates, confirmed at N=6 after Ch09)

Implementer commit must satisfy:

- **≥5 source files in impl-notes "Source Analysis" section.** Ch10 natural surface is **10-12 files**: `rejection_sampler.py`, `metadata.py`, `llm_base_proposer.py`, `eagle.py`, `medusa.py`, `draft_model.py`, `extract_hidden_states.py`, `ngram_proposer.py`, `deepseek_mtp.py`, `speculative.py`, `llama_eagle3.py`. **Aim for 10-12**; the breadth IS the lesson (per Ch07/Ch08/Ch09 cadence; Ch09 hit 10).
- **≥60 `# REFERENCE: <path>:Lxxx` comments across impl modules.** Match the floor. Ch04: 65, Ch05: 61, Ch06: 60, Ch07: ~60, Ch08: 64, Ch09: 66. Target ≥65 for Ch10.
- **≥10 mapping rows; aim for 40 main + 30 mini per K15 two-tier.** Ch07: 27+45 = 72. Ch08: 122. Ch09: 49 main + 39 mini + helper = 151 (`^|` count). Ch10 should adopt two-tier from §10.1 through §10.5: main mapping at §10.6 + mini-tables in §10.1 (rejection-sampler kernels — greedy vs random), §10.2 (MTP head architectures — DeepSeek vs EAGLE vs Medusa), §10.3 (load-weight name remapping examples), §10.4 (α-K sweep grid), §10.5 (proposer comparison table).
- **Demo numerics verbatim** in tests/test-report.md, then narrative quotes them character-for-character.
- **Both linters PASS at the BLOCKING bar** before handoff (writer + reviewer re-run; mismatches trigger preemptive REVISE per K17). Non-blocking inline-density warnings acceptable IFF every inline token is single symbol (E24 from Ch09). The bar is "0 blocking AND every inline token single-symbol", NOT "0/0".
- **5-step rhythm in every major section §10.1-§10.5.**
- **5-7 language traps with explicit "claim → 错 → 为什么 → 源码证据 → Demo/测试" per §6 above**, plus dedicated recap §10.7. Ch09 hit 7 traps exceeding floor of 5.
- **Forward/back-pointers wired**:
  - Back to Ch01 (self-attention — MTP transformer block uses attention), Ch08 (TP — draft_model can have its own TP), Ch09 (EP — DeepSeek MTP layer uses MoE).
  - Forward to Ch15+ (model zoo; Llama EAGLE, Mistral EAGLE), Ch27 (DeepSeek-V3.2 deep-dive — MTP first-class), Ch28 (DeepSeek-V4-Pro / `deepseek_v4_mtp.py`).
- **Source pin verification**: implementer's first command is `cd instances/vllm/source && git rev-parse HEAD` — must equal `98661fe`. Any line numbers that drift between brief and source → re-grep before citing.
- **Outline §3 reframe documented in impl-notes**: explicit "outline-vs-source mismatch handling" subsection that names the issue (training CE loss is out of scope) and the resolution (sidebar + pivot to inference-time weight loading). Reviewer will check this — Ch07/Ch08/Ch09 lesson (each had its own reframe doc).
- **§10.2 "no class MultiTokenPrediction" reframe**: title + hook + §10.2 body anchors with grep evidence and 4-anchor file list. Fourth instance after Ch07/Ch08/Ch09 — pattern is now well-rehearsed, reviewer will count anchors.

### What APPROVED at cycle 1 looks like (K17 / N=6 baseline)

Writer's handoff message must contain BOTH linter outputs verbatim:

```
$ python3 scripts/lint_formulas.py instances/vllm/artifacts/10-multi-token-prediction/narrative/chapter.md
[expected: 🟢 No blocking issues]
[acceptable: ≤4 non-blocking inline-density warnings IFF every inline token is single symbol per E24]

$ python3 scripts/lint_source_grounding.py instances/vllm/artifacts/10-multi-token-prediction/
[expected: ✓ All grounding checks passed!]
```

Plus the hard gates: ≥10 mapping rows, all source files in impl-notes
referenced in narrative, 5-step rhythm in every §10.X, demo numerics verbatim,
≥5 trap callouts (target 7), forward-pointers wired, no `class
MultiTokenPrediction` reframe applied at three anchors (title + hook + §10.2),
§10.3 training-CE → inference-loading reframe documented.

### Cadence projection from Ch04-Ch09

| Metric | Ch04 | Ch05 | Ch06 | Ch07 | Ch08 | Ch09 | Ch10 (target) |
|---|---|---|---|---|---|---|---|
| Lines | 712 | 757 | 655 | 859 | 1051 | 1204 | ≥1100 |
| Words | 3064 | 3849 | 3351 | 4440 | 6058 | 7792 | ≥6500 |
| Mapping rows (`^\|` count) | 13 | 21 | 40 | 72 | 122 | 151 | ≥130 |
| Tests | 48 | 74 | 97 | 83 | 144 | 204 | ≥150 |
| Source files | 5 | 7 | 6 | 5 | 8 | 10 | ≥10 |
| Cycles to APPROVED | 1 | 1 | 1 | 1 | 1 | 1 | 1 (target) |
| Lang trap callouts | 0 | 0 | 4 | 4 | 5 | 7 | ≥5 (target 6-7) |

Ch10 should be in the Ch08-Ch09 ballpark on lines/words (broad surface), with
mapping density continuing to scale with breadth. Test count should match Ch09
(parametrising α-K sweep is the lever, just like Ch08 tp_size sweep and Ch09
ep_size sweep).

---

## §9 — Cadence carry-forward from Ch09

**Ch09 hit single-cycle APPROVED with broadest source surface yet (10 vLLM
modules) and quality bar holding (lints clean, mapping density scales, traps
exceed floor). Ch10 must replicate.** Specific carry-forwards:

1. **Three "no class X" reframes graduate to a recognized pattern.** Ch07 (radix tree) → Ch08 (TensorParallel) → Ch09 (ExpertParallel). Ch10 is the FOURTH instance (no `class MultiTokenPrediction`). Pattern is now well-rehearsed; reviewer expects three anchors with grep evidence + opener "this chapter, like Ch07/Ch08/Ch09, opens with what's NOT in source".

2. **Outline-vs-source training-vs-inference reframe (Ch09 §9.4 NEW pattern).** Ch09 introduced "training-time concept doesn't exist in inference codebase" reframe (aux-loss → EPLB). Ch10 §10.3 has the IDENTICAL pattern: training CE loss → inference weight loading. Apply same template: sidebar grounding + pivot to vLLM's actual code.

3. **Three-anchor framing-tip verification (Ch09 E22 reviewer wisdom).** Every framing tip from tester must appear at hook + body + recap. Reviewer counts. Ch09 had 5 tips × 3 anchors = 15 verifications. Ch10 should target the same.

4. **Tester framing-guidance loop (Ch06-Ch09, N=4 in a row).** Tester is expected to produce 5+ surgical narrative-shaping tips from test code. Implementer's brief should HINT at what tester will discover — the demo plan in §7 above is structured to surface those tips:
   - Demo §2 chain-break (geometric series) → Tip "K does not equal speedup; chain-break geometry matters"
   - Demo §3 break-even α → Tip "MTP is a NET LOSS below break-even"
   - Demo §1 unbiasedness → Tip "rejection sampling is provably unbiased; recovered-token math is the proof"
   - Demo §5 parameter-count comparison → Tip "MTP head ≠ lightweight; MoE block makes it ~50-100× Medusa"

5. **Honest-demo caveat OR-skip discipline (Ch09 K17/E11 OR-skip strict).** §7 lists 4 caveats; impl-notes must state them, writer quotes verbatim, reviewer cross-checks. K17 lesson holds.

6. **Knowledge module M-prefix discipline.** Avoid double-prefix headings (`## M0X: M0X:`). The `learn.py compact()` is broken — manual workaround in use across Ch07-Ch09. **P2-2 (system-improvements) flagged "must-fix-before-Ch12"** in Ch09 delivery; if M-prefix module exceeds 15 facts, manual compact required.

---

## §10 — Direct-dispatch operational notes

**Per `feedback_direct_dispatch.md` rule**: book-editor's idle-summary handoffs
were unreliable in Ch07; team-lead direct-SendMessages each agent. Ch09
followed this; Ch10 will too.

**Handoff sequence for Ch10**:

1. **Team-lead → implementer**: SendMessage with this brief's path
   (`/home/zjq/Repo2Book/instances/vllm/trace/briefs/10-multi-token-prediction-implementer-2026-05-07.md`)
   plus the §1 chapter scope summary as inline context.
2. **Implementer → tester**: SendMessage when implementation + impl-notes
   complete; include linter passes in handoff message.
3. **Tester → writer**: SendMessage with framing tips and demo verbatim
   numerics; tester is expected to produce 5+ tips per Ch06-Ch09 cadence.
4. **Writer → reviewer**: SendMessage with both linters' outputs verbatim
   (K17 protocol).
5. **Reviewer → archivist**: SendMessage on APPROVED with verdict report path.
6. **Archivist → team-lead**: SendMessage with delivery summary + state.json
   diff.

**Brief-on-approval discipline (`feedback_brief_on_approval.md`)**: when Ch10
is APPROVED, archivist immediately writes the Ch11 (DCP/PCP) brief without
waiting for explicit user prompt.

**Source-grounding-verify-before-dispatch (`feedback_outline_topic_not_contract.md`,
rule #6)**: this brief was written with archivist running source-verification
queries first (vllm/v1/spec_decode/ enumeration, deepseek_mtp.py class walk,
rejection_sampler.py inspection, SpeculativeMethod enum check). All file:line
refs in §2.1 verified at commit 98661fe. If implementer hits drift, re-grep
the symbol.

**Loop-escalation rule**: if reviewer/writer ping-pong > 3 cycles, escalate to
team-lead per `repo2book.json:pipeline.topology.decision_protocol.escalate`.
Ch04-Ch09 each completed in 1 cycle; Ch10 should too.

---

**END OF BRIEF**

Brief author: archivist (2026-05-07).
Source pin: 98661fe012c5c467252d4df8411d2f46190e9268.
Outline source: `instances/vllm/book/book-outline.json` →
`parts.part2_advanced_common.chapters[10-multi-token-prediction]`.
Reframes flagged: §10.2 "no class MultiTokenPrediction" (4th instance);
§10.3 "training CE loss → inference weight loading" (2nd instance after Ch09 §9.4).
