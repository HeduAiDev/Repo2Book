# ch35 impl-notes — quantization math reference implementation (primer)

`kind: primer` — this is **not** a subtract-only companion. Per the
implementer contract's primer branch, `implementation/*.py` is a small,
paper-faithful reference port (NumPy, no torch, host-runnable) of the
quantization mathematics from three papers (GPTQ, AWQ, SmoothQuant), not a
line-for-line subset of `vllm/model_executor/layers/quantization/`. Gate is
`lint_paper_grounding.py --expect-primer`, not `lint_fidelity.py`.

## What it is

Four modules, one per mechanism cluster in the dossier:

| Module | Papers/sections covered | Landing-code cross-reference (vllm) |
|---|---|---|
| `uniform_quant.py` | SmoothQuant (arXiv:2211.10438) §2 Eq.1, Fig.3, §3; AWQ (arXiv:2306.00978) §3.2 Eq.1 | `vllm/model_executor/layers/quantization/utils/quant_utils.py:L586-L621` (`quantize_weights` — symmetric branch: `w_s=max(|max|/qmax,|min|/qmin)`, no zero-point; asymmetric branch: `w_s=(max-min)/qmax` + zero-point) |
| `smoothquant.py` | SmoothQuant §4 Eq.3 (migration), Eq.4 (difficulty factor s) | `vllm/model_executor/layers/quantization/utils/quant_utils.py:L322-L338` (`scaled_quantize`'s symmetric amax-scale path consumes activations whose quantization difficulty has already been migrated offline) |
| `awq.py` | AWQ §3.2 Eq.1 (quantize-dequantize), Eq.2-3 (scaled error ratio), Eq.4-5 (scale search over alpha) | `vllm/model_executor/layers/quantization/awq.py:L262-L286` (`AWQLinearMethod.apply` — `ops.awq_dequantize`/`ops.awq_gemm` consume the already-scaled packed qweight/scales/qzeros) |
| `gptq.py` | GPTQ (arXiv:2210.17323) §3 Eq.1-3 (OBQ background), §4 Eq.4-5 + Algorithm 1, §5 Setup (per-row asymmetric min-max grid) | `vllm/model_executor/layers/quantization/gptq.py:L376-L399` (`GPTQLinearMethod.apply` → `ops.gptq_gemm`); `vllm/model_executor/layers/quantization/utils/quant_utils.py:L655-L686` (`gptq_quantize_weights` reuses the same uniform grid + act_order permutation) |

All three papers' algorithms (GPTQ's Hessian-based reconstruction, AWQ's
scale search, SmoothQuant's migration) are **offline calibration**
procedures — none of them run inside vllm itself. vllm only consumes their
*output*: quantized weight tensors (packed INT4/INT8, grouped, with
per-group/per-channel scales) or migrated/rescaled activation-weight pairs
consumed by the shared amax-based quantizer in `quant_utils.py`. This is why
every "code" `embed_excerpt` in the dossier is a *consumer* of one of these
papers' outputs, not an implementation of the paper's algorithm — and why
this reference implementation exists: to give the reader something they can
actually step through and hand-verify the papers' equations against, since
the real calibration code isn't in this repository.

Two mechanisms in the dossier deliberately have **no** reference-implementation
counterpart:
- `m7-quant-method-calling-surface` (`QuantizeMethodBase.apply` /
  `QuantizationConfig.get_quant_method`) is a pure vllm framework protocol,
  not paper math — illustrated directly via its real vllm excerpt in the
  narrative, no `# PAPER:` anchor is possible for it.
- `m8-fp8-e8m0-scale` (block-quantization scale stored as `float8_e8m0fnu`,
  rounded to the nearest power of two) is an OCP Microscaling(MX) hardware
  format convention, not one of this chapter's three papers' equations
  (dossier's own `paper_origin_note` says so explicitly) — also illustrated
  only via its real vllm excerpt, not reimplemented here.

## PAPER anchors (def-level; see also inline `# PAPER:` comments per equation)

`uniform_quant.py`:
- `smoothquant_scale` / `quantize_smoothquant` / `dequantize` — SmoothQuant §2 Eq.1
- `awq_scale` / `quantize_awq` — AWQ §3.2 Eq.1
- `per_tensor_scale` / `per_token_scale` / `per_channel_scale` — SmoothQuant §2 Fig.3
- `effective_quant_levels` — SmoothQuant §3 (outlier collapse argument)

`smoothquant.py`:
- `difficulty_factor` — SmoothQuant §4 Eq.4
- `migrate` — SmoothQuant §4 Eq.3
- `smoothquant_pipeline_error` — cross-validation helper (not a paper equation), ties `migrate`+`quantize_smoothquant` together to reproduce the "does migration reduce quantization error" claim numerically

`awq.py`:
- `round_err` — AWQ §3.2 ("RoundErr(.) ~ 0.25")
- `quantize_dequantize` — AWQ §3.2 Eq.1
- `scaled_error_ratio` — AWQ §3.2 Eq.2-3
- `search_alpha` — AWQ §3.2 Eq.4-5

`gptq.py`:
- `reconstruction_error` — GPTQ §3 Eq.1
- `hessian_from_activations` — GPTQ §3 ("H_F = 2 X_F X_F^T")
- `dampen` — GPTQ §4 "Step 3: Cholesky Reformulation" (1% average-diagonal dampening)
- `make_asymmetric_per_row_quantizer` — GPTQ §5 "Setup" (per-row asymmetric min-max grid)
- `remove_hessian_row_col` — GPTQ §3 Eq.3 (OBQ)
- `obq_pick_and_compensate` — GPTQ §3 Eq.2 (OBQ)
- `obq_quantize_row` — full greedy-order OBQ pass (the §4 Step 1 baseline GPTQ compares its "arbitrary/same order" insight against)
- `gptq_lazy_batch_compensate` — GPTQ §4 Eq.4-5 (lazy batch-update)
- `gptq_quantize` — GPTQ §4 Algorithm 1 (full pseudocode: dampen + Cholesky-reformulated Hinv + block-processed columns)

## Two quantization-scale conventions (deliberately kept separate)

SmoothQuant Eq.1 uses `Delta = max(|X|) / (2^(N-1) - 1)` (reserves one code
point); AWQ Eq.1 uses `Delta = max(|w|) / 2^(N-1)` (full symmetric range).
`uniform_quant.py` keeps `smoothquant_scale`/`quantize_smoothquant` and
`awq_scale`/`quantize_awq` as distinct functions rather than unifying them
under one "symmetric quantize" helper — collapsing them would either
silently pick one paper's convention for the other, or require a parameter
that doesn't correspond to anything in either paper. `test_uniform_quant.py`
pins down the exact ratio between the two (`128/127` for N=8). vllm's own
`quantize_weights` (`quant_utils.py:L595-L608`) uses yet a third convention
for the symmetric case (`max(|max|/qmax, |min|/qmin)`, no `-1` on the
denominator) — the narrative's landing section should point this out rather
than let a reader assume all three are the same formula.

## Worked-example numbers the chapter narrative can cite directly

- **SmoothQuant outlier collapse** (`effective_quant_levels`): a channel
  ~100x smaller than an outlier channel collapses from 256 (8-bit) levels to
  under 5 — reproduces the paper's own "2-3 effective levels" claim (§3).
- **AWQ scaling** (`scaled_error_ratio`): scaling a salient weight by s=2
  (aggregated over many trials, since RoundErr is an *expected*-value claim)
  tracks the naive `1/s` prediction to within simulation noise; s=8 on a
  group where scaling shifts the absmax demonstrates the "too-large-s hurts
  non-salient channels" effect from §3.2 (`Delta' > Delta`).
- **SmoothQuant migration** (`smoothquant_pipeline_error`): on a toy layer
  with one ~80x outlier input channel, `alpha=0.5` per-tensor quantization
  error after migration is strictly lower than quantizing the raw
  (unmigrated) activations/weights.
- **GPTQ** (`gptq_quantize` vs `quant_fn(W)` alone): on a small (3x6, 3x8)
  toy `(W, X)` problem, GPTQ's Eq.1 reconstruction error is never worse than
  plain RTN (`quant_fn` applied directly, no compensation), and the result
  is (near-)invariant to the lazy-batch `blocksize` — confirming Eq.4-5 is
  an efficiency reformulation of the same per-column process, not a
  different algorithm.

## Tests

`tests/test_uniform_quant.py`, `tests/test_smoothquant.py`,
`tests/test_awq.py`, `tests/test_gptq.py` — 27 tests total, all pure NumPy,
host-runnable:

```
cd instances/vllm/artifacts/ch35-primer-quantization
PYTHONPATH=implementation python3 -m pytest tests -q
```

27 passed (uniform_quant 9 + smoothquant 4 + awq 5 + gptq 9), no xfail/skip,
deterministic (all RNG seeded).

## Provenance

This chapter's paper pack (`book/papers/ch35-primer-quantization/`) is the
same three papers (GPTQ arXiv:2210.17323, AWQ arXiv:2306.00978, SmoothQuant
arXiv:2211.10438) used by the `vllm-ascend` instance's analogous primer
chapter (`ch31-primer-quantization`, née `ch35`). The reference
implementation here is adapted from that chapter's — same paper math, same
tests — with the "landing anchor" doc-comments and this notes file rewritten
to point at **vllm**'s own quantization call sites (`quant_utils.py`,
`gptq.py`, `awq.py`, `base_config.py`) instead of `vllm_ascend`'s, per this
chapter's dossier `code_spine`/`embed_excerpts`. No paper-math function was
changed; only the "where does this land in the real repo" commentary differs.
