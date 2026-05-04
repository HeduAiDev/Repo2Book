# User Preferences

Accumulated across all sessions. Updated when user provides explicit feedback.

## Style Preferences
- Language: zh-CN (Simplified Chinese)
- Tone: 大白话 (colloquial, not formal/书面语)
- Formality spectrum: Cell 2 casual → Cell 4 rigorous → Cell 11 casual return
- Diagrams: svg-diagram skill (Python→SVG→xmllint→PNG). NEVER Excalidraw.

## Technical Preferences
- Source grounding: EVERY section must have source file:line reference
- Theory depth: Mathematical proof with intuition scaffolding (numerical trace first, then symbols)
- Code walkthrough: Mandatory, with verified line numbers from actual implementation
- Implementation language: Match source (CUDA→Triton, Python→Python)

## Process Preferences
- Agent teams: Use persistent teammates with lateral communication
- Pipeline: Backpressure gates (never skip a stage)
- Quality: Formula lint + source grounding lint must pass before publish
- Fix prompts, not chapters: When quality issue is widespread, fix the SOURCE (prompts)

## Architectural Preferences
- By-repo instances: Each repo gets its own team, knowledge, trace
- Project-scoped config: Everything project-level lives in `.claude/`
- Archivist per instance: Dedicated long-term memory agent
- Topology flexibility: Simple chapters = linear, complex = pair/panel

## Known Dislikes
- Manual narrative editing by orchestrator (→ created guard_narrative.py)
- Mermaid for dense diagrams (→ created svg-diagram skill)
- Generic textbook chapters without source grounding (→ mandatory REFERENCE comments)
- Subagent-based teams (→ migrated to true Agent Teams with tmux)
