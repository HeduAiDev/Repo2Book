# Cross-Chapter Decisions

Framework-wide decisions that apply to ALL chapters.

## 2026-05-04 — repo2book Architecture

- **Decision**: Adopt repo2book as the general framework (not vLLM-specific)
- **Rationale**: Enable multi-repo book production, reuse patterns across projects
- **Impact**: All agents, all chapters
- **See**: `sessions/2026-05-04.md` for full session summary

### Key architectural decisions:
1. Two-tier memory: knowledge (repo-specific, TTL'd) + wisdom (universal, gated)
2. Organizational topology: 5 modes with per-chapter override capability
3. Trace system: chapter-first organization with raw session backups
4. Archivist agent: per-instance, maintains long-term memory
5. Project-scoped configuration: `.claude/settings.local.json` + agents + skills
6. Lateral communication: Reviewer→Writer, Writer→Implementer (no Lead routing)
7. Hook-driven pipeline handoff: TaskCompleted → inbox message → next agent
