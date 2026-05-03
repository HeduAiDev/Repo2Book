# vLLM Book Factory — Multi-Agent Writing System

## ⛔ HARD RULE: Narrative Guardian

**You (the main orchestrator) CANNOT directly write or edit any
`artifacts/*/narrative/chapter.md` file.** These files are OWNED by the Writer agent.

Every narrative edit MUST go through the Writer subagent or the Chapter Pipeline.
If you find yourself wanting to `Edit` or `Write` a narrative file — STOP.
Run `python3 scripts/guard_narrative.py check <chapter_id>` first.
If locked, invoke Writer instead.

**Only modify:** prompts/, agents/, pipelines/, schemas/, scripts/, CLAUDE.md, book-outline.json.
**FORBIDDEN to modify directly:** artifacts/*/narrative/chapter.md.

If the chapter quality is wrong, fix the PROMPTS, not the chapter content.
The prompts produce the content. Fix the cause, not the symptom.

Architecture: MetaGPT-style Role agents + Ralph Backpressure gates.
本书是一套**多 Agent 协作写作系统**产生的技术书籍。每章由 4 个 Agent 串行完成。

---

## Project Map

```
vllm-from-scratch/
├── CLAUDE.md                   ← YOU ARE HERE
├── orchestrator.py             # CLI entry: outline/write/fix/ask/status
│
├── agents/                     # MetaGPT Role definitions
│   ├── base.py                 # BaseAgent + GateResult + AgentResult
│   ├── implementer.py          # Agent-1: 基于vLLM源码的手撕实现
│   ├── tester.py               # Agent-2: Backpressure gate (pytest)
│   ├── writer.py               # Agent-3: 认知顺序叙事
│   ├── reviewer.py             # Agent-4: 0基础读者审查 (FINAL GATE)
│   └── book_editor.py          # 总编: 意图解析+调度+状态追踪
│
├── prompts/                    # Agent system prompts (THE SOURCE OF TRUTH)
│   ├── implementer.md          # MUST read vLLM source before coding
│   ├── writer.md               # MUST anchor every section to vLLM file:line
│   ├── reviewer.md             # MUST check source_grounding + formulas
│   ├── tester.md               # MUST generate test-report.json
│   └── book_editor.md          # Intent parsing + dispatch logic
│
├── pipelines/
│   └── chapter_pipeline.py     # 4-agent pipeline with Ralph gates
│
├── schemas/
│   ├── chapter_context.json    # Per-chapter state + gate tracking
│   └── book_outline.json       # 28-chapter, 5-part outline
│
├── scripts/
│   ├── lint_formulas.py        # Auto-detect LaTeX rendering issues
│   └── lint_source_grounding.py # Auto-detect vLLM source ref gaps
│
├── artifacts/                  # ALL chapter content lives here
│   ├── 01-self-attention-fundamentals/
│   │   ├── implementation/     # .py files with # REFERENCE: vllm/xxx.py:L123
│   │   ├── tests/              # pytest tests (run in vLLM Docker container)
│   │   ├── narrative/          # chapter.md (the actual book chapter)
│   │   ├── reviews/            # review-report.json
│   │   └── context.json        # Gate state + changelog
│   ├── 02-kv-cache/
│   ├── ... (chapters 01-13 complete)
│   └── 14-triton-primer/       ← NEXT CHAPTER
│
├── book/
│   └── book-outline.json       # Master outline
│
└── vllm/                       # Cloned vLLM source (reference only)
```

---

## The V3 Writing Standard (CRITICAL — read before writing ANY chapter)

经过三次迭代（v1: theory-only → v2: source-only → v3: BALANCED），当前写作标准：

### Every chapter section must have BOTH:
1. **Source Trail:** Open a specific vLLM file, show the code, explain where it lives
2. **Theory Deep Dive:** Derive the math, prove the correctness, explain the WHY

### The "5-step rhythm" for each major section:
```
1. 打开 vllm/xxx.py:L123 → ClassName.method()     ← Source Trail (入口)
2. 这个方法做了什么？为什么？                       ← Bridge (连接)
3. 让我们从零推导它背后的原理...                     ← Theory Deep Dive (深入)
4. 我们的简化实现: [code]                            ← Implementation (实践)
5. 原版比我们多了 X 和 Y，因为...                    ← Source Diff (差异)
```

### Anti-patterns (both extremes are WRONG):
- ❌ ALL source references, zero theory → "source dump" — reader learns nothing
- ❌ ALL theory, zero source refs → "textbook copy" — not about vLLM
- ✅ Every section: source entry point + theory derivation

---

## Agent Pipeline (Gates)

```
Implementer ──[implementation_exists]──→
  Tester ──[tests_pass BACKPRESSURE]──→
    Writer ──[narrative_complete]──→
      Reviewer ──[review_approved FINAL]──→ Published
```

### Agent 1 — Implementer (prompts/implementer.md)
- HARD GATE: Must complete 5-item Source Analysis before writing any code
- Every function MUST have `# REFERENCE: vllm/path/to/file.py:L123` comment
- Implementation must mirror vLLM's actual classes (same names, same API shape)
- Write to `artifacts/{chapter_id}/implementation/`
- Output: `{module}.py` + `impl-notes.md` (with Source Mapping Table, 5+ rows)

### Agent 2 — Tester (prompts/tester.md)
- THE BACKPRESSURE GATE: if tests fail, Writer NEVER starts
- Tests run in vLLM Docker container: `docker run --rm --entrypoint bash -v $(pwd):/workspace --network host vllm/vllm-openai:latest`
- Test levels: unit (core logic) + integration (cross-chapter compatibility) + teaching examples
- Output: `test_{module}.py` + `test-report.json`

### Agent 3 — Writer (prompts/writer.md)
- CRITICAL: Balance source grounding AND theoretical depth — NOT one or the other
- **Diagrams: use `svg-diagram` skill.** Invoke with Skill tool: `Skill(skill="svg-diagram", args="...")`.
  The skill encapsulates Python→SVG→xmllint→PNG workflow. NEVER use Excalidraw or Mermaid for
  diagrams with >3 connected elements. Markdown tables for numerical data.
- Every Cell (2-7) must have at least one `vllm/xxx.py:L123` reference
- Chapter structure: Hook (Cell 2) → Problem Demo (3) → Theory (4) → Walkthrough (5) → Implementation (6) → Demo/Viz (7) → Source Mapping (9) → Verification (10) → Summary (11)
- Chinese: 大白话, no 书面语. Formality spectrum per cell (see style-guide in prompts)
- Output: `chapter.md` + `outline.md`

### Agent 4 — Reviewer (prompts/reviewer.md)
- 7 review dimensions (in order): source_grounding, formula_renderability, coherence, readability, engagement, cross_chapter_consistency, concept_precision
- source_grounding: auto-REJECT if any Cell 2-7 has 0 vLLM references for 3+ paragraphs
- formula_renderability: auto-REJECT on `\text{}`, `\boxed{}`, `\tag*{}`, inline `\frac`, single-line `$$`
- Output: `review-report.json`

---

## Formula Rules (NON-NEGOTIABLE)

Run `python3 scripts/lint_formulas.py artifacts/{chapter_id}/narrative/chapter.md` before marking narrative complete.

### Blocking (auto-REJECT):
- `\text{}` → use `\mathrm{}`
- `\boxed{}` → use markdown bold header above formula
- `\tag*{}` → put annotation outside `$$` block
- `\frac` inside inline `$...$` → promote to `$$...$$` block
- `$$` on same line as formula content → separate lines

### Non-blocking (warnings):
- 3+ inline formulas in one paragraph
- Inline formula > 30 characters

### Allowed inline ($...$):
- Single symbols: `$x$`, `$\alpha$`
- Simple expressions: `$d_k$`, `$\sqrt{2}$`, `$1/\sqrt{d_k}$`

---

## Source Grounding Rules (NON-NEGOTIABLE)

Run `python3 scripts/lint_source_grounding.py artifacts/{chapter_id}/` before marking narrative complete.

### Requirements:
- Every Cell (2-7) must have at least 1 vLLM file:line reference
- Every implementation function must have `# REFERENCE: vllm/...` comment
- Source Mapping Table must have 5+ rows
- impl-notes.md must list 3+ vLLM files

### Reference format:
- `vllm/v1/core/kv_cache_manager.py:L225` — full path + line
- `kv_cache_manager.py:L225` — short path (after full path established)
- `flash_attn.py → FlashAttentionImpl.forward()` — class.method

---

## Test Execution

All tests MUST pass in the vLLM Docker container:

```bash
docker run --rm --entrypoint bash \
  -v /mnt/e/Laboratory/vllm-from-scratch:/workspace \
  --network host \
  vllm/vllm-openai:latest \
  -c "pip3 install pytest -q 2>&1 | tail -1 && \
      cd /workspace/artifacts/{chapter_id} && \
      python3 -m pytest tests/ -q 2>&1"
```

The vLLM container has: Python 3.12, PyTorch, CUDA, pytest pre-installed.

---

## Chapter Skeleton Creation

For new chapters, create the skeleton FIRST:

```bash
mkdir -p artifacts/{chapter_id}/{implementation,tests,narrative,reviews}
touch artifacts/{chapter_id}/implementation/__init__.py
```

Then update `context.json` with gates and chapter metadata.

---

## Artifact Structure per Chapter

```
artifacts/{chapter_id}/
├── implementation/
│   ├── __init__.py
│   ├── {module}.py           # Code with # REFERENCE: vllm/xxx.py:L123 on every function
│   └── impl-notes.md         # Source analysis + design decisions + Source Mapping Table
├── tests/
│   ├── test_{module}.py      # pytest tests
│   └── test-report.json      # Generated after test execution
├── narrative/
│   ├── chapter.md            # THE ACTUAL BOOK CHAPTER
│   └── outline.md            # Section-level outline
├── reviews/
│   └── review-report.json    # 7-dimension review
└── context.json              # Gate state, version, changelog, summary
```

---

## Current Progress (2026-05-03)

### Part 1: 公共基础能力 — 11/11 chapters COMPLETE
```
01 ✅ self-attention-fundamentals    | Self-Attention 算子深度解析
02 ✅ kv-cache                       | KV Cache 内存模型与实现
03 ✅ flashattention-pagedattention  | FlashAttention & PagedAttention
04 ✅ continuous-batching            | Continuous Batching 动态调度
 - ✅ chunked-prefill                | Chunked Prefill (inserted after Ch4)
05 ✅ memory-management              | GPU 显存管理系统
06 ✅ scheduling                     | 请求调度系统
07 ✅ prefix-cache                   | Prefix Cache
08 ✅ tensor-parallelism             | Tensor Parallelism
09 ✅ expert-parallelism             | Expert Parallelism
10 ✅ multi-token-prediction         | Multi-Token Prediction
```
Total Part 1: 119 tests

### Part 2: 公共能力进阶 — 3/3 chapters COMPLETE
```
11 ✅ dcp-pcp              | DCP/PCP 上下文并行
12 ✅ kv-offload           | KV Cache Offload 层级存储
13 ✅ prefix-cache-pooling | Prefix Cache Pooling 全局共享池
```
Total Part 2: 31 tests

### Part 3-5: REMAINING (15 chapters)
```
Part 3: 从 Triton 算子构建 Llama-3.2-1B
  14 ⬜ triton-primer
  15 ⬜ llama-model-architecture
  16 ⬜ triton-rmsnorm
  17 ⬜ triton-rope
  18 ⬜ triton-attention
  19 ⬜ triton-mlp
  20 ⬜ model-runner
  21 ⬜ end-to-end-llama

Part 4: Prefill-Decode 分离
  22 ⬜ pd-architecture
  23 ⬜ pd-prefix-cache
  24 ⬜ layerwise-connectors
  25 ⬜ pd-ratio

Part 5: 模型定制优化
  26 ⬜ qwen-3.5-397b
  27 ⬜ deepseek-v3.2
  28 ⬜ deepseek-v4-pro
```

---

## Common Pitfalls (Lessons Learned)

1. **Don't write generic LLM textbook chapters.** Every concept must trace to a vLLM file:line.
   The acid test: "After reading this section, can the reader open the vLLM repo and find the code?"

2. **Don't over-correct to source-only.** Theory (derivations, proofs, numerical examples) and source
   references are NOT mutually exclusive. Both are mandatory.

3. **Run formula linter before claiming complete.** `\text{}` in formulas is the #1 issue.

4. **Run source grounding linter before claiming complete.** Sections without vLLM refs → auto-REJECT.

5. **Tests MUST pass in Docker.** Local test runs are not sufficient — the vLLM container has
   specific PyTorch/CUDA versions.

6. **F.linear weight shape is [out, in].** `nn.Linear(in, out)` stores weight as `[out, in]` and
   `F.linear(x, weight)` does `x @ weight^T`. Getting this wrong causes silent shape bugs.

7. **Chapter IDs must be unique across old and new outlines.** Old chapter artifacts must be
   cleaned up when the outline changes.

8. **`ref_cnt = -1` means "not ready"** in offload/BLockPool contexts. `get()` checks `is_ready`
   (ref_cnt >= 0). Use `_get_any()` for internal operations that need to find non-ready blocks.

---

## Quick Reference: Writing a New Chapter

```bash
# 1. Create skeleton
mkdir -p artifacts/{chapter_id}/{implementation,tests,narrative,reviews}
touch artifacts/{chapter_id}/implementation/__init__.py

# 2. Explore vLLM source (use Agent tool with subagent_type=Explore)
#    → Understand the real implementation before writing

# 3. Write implementation (Implementer)
#    → Every function: # REFERENCE: vllm/xxx.py:L123
#    → Write impl-notes.md with Source Mapping Table

# 4. Write tests (Tester)
#    → Unit + integration + teaching example tests

# 5. Run tests
docker run --rm --entrypoint bash \
  -v /mnt/e/Laboratory/vllm-from-scratch:/workspace \
  --network host vllm/vllm-openai:latest \
  -c "pip3 install pytest -q 2>&1 | tail -1 && \
      cd /workspace/artifacts/{chapter_id} && python3 -m pytest tests/ -q"

# 6. Write narrative (Writer)
#    → Every section: Source Trail + Theory Deep Dive
#    → 5-step rhythm per major section

# 7. Lint
python3 scripts/lint_formulas.py artifacts/{chapter_id}/narrative/chapter.md
python3 scripts/lint_source_grounding.py artifacts/{chapter_id}/

# 8. Fix all BLOCKING issues

# 9. Update context.json → status: published, gates all True
```
