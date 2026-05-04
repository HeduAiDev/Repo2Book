# repo2book Architecture — Complete System Design

- **Type**: design_change
- **Chapter**: N/A (framework-wide)
- **Date**: 2026-05-04
- **Agents involved**: team-lead (orchestrator), user
- **User present**: true
- **Tags**: [architecture, agent-teams, knowledge-system, topology, archivist]

## What happened

Designed and implemented the complete repo2book framework — a general-purpose system that turns any code repository into a source-grounded technical book via multi-agent pipeline. This is a fundamental restructuring from the previous vLLM-specific single-project system.

### Systems built

1. **Two-tier memory (knowledge + wisdom)**: Agents self-improve through experience. Knowledge = repo-specific facts (TTL'd, per-instance). Wisdom = universal patterns (gated, cross-instance).

2. **Organizational topology**: Pipeline is NOT fixed linear. Five modes (linear, pair, panel, writer_editor, swarm) with per-chapter topology override. Voting/discussion/escalation protocols for organizational decisions.

3. **Trace + Archivist**: Long-term project memory that survives Claude Code context compression. Dedicated archivist agent per instance records every event, generates rehydration briefs, and answers queries about past decisions.

4. **Project-scoped configuration**: `.claude/settings.local.json` + `.claude/agents/*.md` + `.claude/skills/` — all project-level, not global. Framework uses `repo2book.json` as master config.

5. **By-repo instance structure**: `instances/{repo}/` with own knowledge, trace, team, and archivist. Framework wisdom shared across all instances.

## Why it matters

- **Context loss prevention**: Without the archivist+ trace, a project spanning months loses every lesson from early chapters. The archivist ensures continuity.
- **Multi-repo scalability**: The framework can now run multiple book projects simultaneously, each with its own team and memory.
- **Quality improvement over time**: Knowledge and wisdom accumulate. Agents get smarter as the project progresses. Bugs are prevented, not re-discovered.
- **Organizational flexibility**: Complex chapters can allocate more personnel. The topology adapts to chapter difficulty.

## What to remember

The archivist is the single most important system component for long-running projects. Without it, everything degrades. With it, the project's institutional memory is preserved.

When working on Chapter 14+, always query the archivist first: `python3 scripts/archivist.py brief --chapter 14 --role implementer`
