# Wisdom: Architecture Patterns

Universal architectural patterns for multi-agent book production pipelines.
These patterns apply to ANY repo2book instance.

---

## W04: Backpressure gates prevent cascading errors

**Discovered by**: book-editor, all
**Confirmed in**: vllm (all 13 published chapters)
**Severity**: BLOCKING — without gates, broken code reaches the reader

### Pattern
Each pipeline stage has a binary gate. If the gate fails, the pipeline STOPS.
The downstream agent NEVER sees broken work. The upstream agent MUST fix.

```
Implementer ──[implementation_exists]──→
  Tester ──[tests_pass BACKPRESSURE]──→
    Writer ──[narrative_complete]──→
      Reviewer ──[review_approved]──→ Published
```

### Why Backpressure (not fire-and-forget)
Without backpressure, the Writer writes about broken code, the Reviewer reviews
meaningless text, and the reader gets garbage. With backpressure, the pipeline
stops at the FIRST failure, saving all downstream work.

### Affected Roles
- **book-editor**: Enforce gates, never skip a stage
- **implementer**: Fix before tester re-runs
- **tester**: Binary judgment: PASS or REJECT (no "mostly passes")
- **writer**: Never start until tests_pass gate is true
- **reviewer**: Never start until narrative_complete gate is true

---

## W08: Lateral communication — Reviewer↔Writer bypasses Lead

**Discovered by**: reviewer, writer, book-editor
**Confirmed in**: vllm (04-continuous-batching rewrite)
**Severity**: HIGH — reduces iteration latency from O(Lead) to O(1)

### Pattern
The Reviewer can SendMessage directly to the Writer with REVISE instructions.
The Writer can SendMessage directly to the Implementer with change requests.
The Lead is only involved for loop detection (>3 rounds on same issue).

### Why Direct Communication
Routing through the Lead adds latency and context loss. The Reviewer has the full
context of why something needs fixing. Passing it through the Lead loses details.

### Loop Detection (WHEN to escalate to Lead)
If Reviewer↔Writer exceeds 3 rounds on the same issue:
1. The issue is likely a fundamental design problem, not a fixable bug
2. The Lead must decide: rewrite the chapter, update the prompts, or accept the trade-off
3. The Lead has authority the agents don't — they can change the rules

### Affected Roles
- **reviewer**: SendMessage to writer for REVISE, to Lead for loop escalation
- **writer**: SendMessage to implementer for code changes
- **book-editor**: Monitor loop count, intervene at round 4
- **implementer/tester**: Less lateral communication needed (their gates are objective)

---

## W12: Chapter pipeline — skeleton + 4 gates → publish

**Discovered by**: book-editor
**Confirmed in**: vllm (pipeline design iteration v1→v3)
**Severity**: MEDIUM

### Pattern
Every chapter follows the same lifecycle:
1. Create skeleton: `mkdir -p artifacts/{id}/{implementation,tests,narrative,reviews}`
2. Implementer writes code → Gate: implementation_exists
3. Tester writes and runs tests → Gate: tests_pass (BACKPRESSURE)
4. Writer writes narrative → Gate: narrative_complete
5. Reviewer reviews → Gate: review_approved (FINAL)
6. Published

### State Tracking
Each chapter has `context.json` with gate states. This is the single source of truth
for chapter progress. Never trust a verbal claim — check context.json.

### Affected Roles
- **book-editor**: Create skeleton, track state, enforce gate order
- **all agents**: Update context.json when your stage completes

---

## Other Architecture Patterns

### Fix prompts, not chapters
When the same quality issue appears across multiple chapters, the PROMPT is wrong.
Fixing individual chapters treats the symptom. Fixing the prompt cures the cause.

### Unlockable narrative protection
Use `guard_narrative.py` to prevent direct narrative edits by the orchestrator.
Only the Writer agent can modify narrative files. This prevents "just this one fix"
from destroying the Writer-Reader contract.

### Per-role tools, not universal tools
Each agent gets exactly the tools it needs. Implementer gets Agent (for sub-exploration).
Writer gets Skill (for svg-diagram). Reviewer gets SendMessage (for lateral REVISE).
Tester only needs Bash + Read. Principle of least privilege.
