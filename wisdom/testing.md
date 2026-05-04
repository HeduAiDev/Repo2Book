# Wisdom: Testing Patterns

Universal testing patterns. These prevent the #1 pipeline killer: tests that pass for the wrong reasons.

---

## W02: Preemption tests — both requests must fit initially

**Discovered by**: tester
**Confirmed in**: vllm (04-continuous-batching)
**Severity**: BLOCKER — test trivially passes without testing preemption

### Symptom
- A preemption test passes but never actually triggers preemption
- Test sets up a scenario where requests can't all fit, but the first request already
  consumes all resources, so preemption is never attempted

### Root Cause
For preemption to trigger: (1) multiple requests must ALL be running first, (2) THEN a
resource shortage occurs, (3) THEN the scheduler chooses a victim. If step 1 fails (not
enough resources for all), no preemption happens.

### Correct Pattern
```python
# WRONG: r1 takes all blocks, r2 never gets in → preemption never triggers
r1 = Request(prompt=64 tokens, blocks=4)  # All blocks
r2 = Request(prompt=32 tokens, blocks=2)  # No room

# RIGHT: both fit initially, THEN r1 needs more → preempt r2
r1 = Request(prompt=32 tokens, blocks=2)  # Half blocks
r2 = Request(prompt=32 tokens, blocks=2)  # Half blocks
# Step 2: r1 needs decode block → preempt r2
```

### Affected Roles
- **tester**: Verify preemption ACTUALLY triggered (assert preempted_req_ids)
- **implementer**: Design resource limits so tests can trigger preemption
- **reviewer**: Flag preemption tests that don't assert preemption occurred

---

## W03: Numerical trace — 2+ iterations, ALL intermediates

**Discovered by**: writer, reviewer
**Confirmed in**: vllm (03-flashattention, 04-continuous-batching)
**Severity**: BLOCKING in review dimension `algorithm_comprehension`

### Pattern
Every non-trivial algorithm in a chapter MUST include a numerical trace with:
1. Concrete small numbers (traceable mentally: L=12, BLOCK=4, not L=4096)
2. At least 2 complete iterations (first iteration = exploration, second = pattern recognition)
3. ALL intermediate variable values at each step (m, l, P, correction, O_acc for softmax)
4. Values that match the chapter's formulas exactly

### Example (bad)
"At step 2, the correction factor updates the accumulation."
(No values, no formula, no connection to proof)

### Example (good)
"At step 2 (j=1): m_old=2.0, m_new=3.5, correction=exp(2.0-3.5)=0.2231.
 O_acc = 0.2231 × old_O + exp(S₁ - 3.5) × V₁ / l_new"

### Affected Roles
- **writer**: Include numerical trace BEFORE writing the proof
- **reviewer**: AUTO-REJECT if numerical trace missing or incomplete
- **tester**: Test the exact numbers from the trace

---

## W10: ref_cnt = -1 means "not ready"

**Discovered by**: implementer, tester
**Confirmed in**: vllm (12-kv-offload)
**Severity**: HIGH

### Pattern
In resource pool/cache systems, a sentinel value (usually -1) means "not yet initialized"
or "not ready for use." The `get()` method checks `is_ready` (which tests `ref_cnt >= 0`).
Internal operations use `_get_any()` to bypass this check.

### Affected Roles
- **implementer**: Document sentinel values in impl-notes.md
- **tester**: Test both ready and not-ready states
- **writer**: Explain the sentinel pattern in the narrative

---

## Other Testing Patterns

### Always test OOM paths
Memory exhaustion is the #1 source of real-world bugs. Every resource allocator
should have a test where resources run out.

### Docker command in test-report.json
Always record the exact Docker command and image tag used. This makes test
reproducibility trivial — anyone can re-run with the recorded command.
