"""
Writer Agent — 按认知顺序将技术内容写成让零基础读者能看懂、看进去、看完的故事。

Heavily draws from reimpl-tutorial:
- style-guide.md (narrator voice, formality spectrum, humor guidelines)
- walkthrough-prompt.md (code walkthrough structure, three layers pattern)
- feature-template.md (notebook cell structure)
- derivation-prompt.md (theory section requirements)
"""

from pathlib import Path
from typing import Any, Dict

from .base import AgentResult, BaseAgent, GateResult


class WriterAgent(BaseAgent):
    """Agent-3: Writes the educational narrative following cognitive order."""

    def __init__(self, chapter_id: str, read_only: bool = False):
        super().__init__("writer", chapter_id)
        self.read_only = read_only
        self.narrative_dir = None

    def _pre_gate(self) -> GateResult:
        """Tests must pass before writing (unless read_only mode)."""
        if self.read_only:
            return GateResult(passed=True, reason="Read-only mode — no gate required")

        ctx = self.load_context()
        if not ctx.get("gates", {}).get("tests_pass"):
            return GateResult(
                passed=False,
                reason="Tests gate not passed — Tester must approve implementation first",
            )
        return GateResult(passed=True, reason="Ready to write")

    def _act(self, context: Dict[str, Any]) -> AgentResult:
        """Write the chapter narrative."""
        self.narrative_dir = self.ensure_dir("narrative")

        chapter_ctx = self.load_context()
        prev_ctxs = self.load_previous_contexts()
        user_question = context.get("user_question", "")
        rewrite_target = context.get("rewrite_target", "")

        if self.read_only and user_question:
            # Answer a user question based on existing artifacts
            prompt = self._build_qa_prompt(chapter_ctx, prev_ctxs, user_question)
            prompt_path = self.narrative_dir / "ANSWER.md"
            prompt_path.write_text(prompt, encoding="utf-8")
            return AgentResult(
                agent_name=self.name,
                status="success",
                output_paths=[str(prompt_path)],
                summary=f"Answer prompt written to {prompt_path}",
            )

        if rewrite_target:
            # Incremental rewrite of a specific section
            prompt = self._build_rewrite_prompt(
                chapter_ctx, prev_ctxs, rewrite_target, context
            )
        else:
            # Full chapter writing
            prompt = self._build_write_prompt(chapter_ctx, prev_ctxs)

        prompt_path = self.narrative_dir / "WRITE.md"
        prompt_path.write_text(prompt, encoding="utf-8")

        if not self.read_only:
            self.update_gate("narrative_complete", True)
            self.add_changelog("narrative written")

        return AgentResult(
            agent_name=self.name,
            status="success",
            output_paths=[str(prompt_path)],
            summary=f"Writing prompt written to {prompt_path}",
        )

    def _build_write_prompt(
        self, chapter_ctx: Dict, prev_ctxs: Dict[str, Dict]
    ) -> str:
        """Build the full chapter writing prompt."""
        prompt = self.prompt
        prompt += f"\n\n---\n\n## Chapter Spec\n\n"
        prompt += f"**Chapter ID:** {chapter_ctx.get('chapter_id')}\n"
        prompt += f"**Chapter Number:** {chapter_ctx.get('chapter_number')}\n"
        prompt += f"**Title:** {chapter_ctx.get('title')}\n"
        prompt += f"**Level:** {chapter_ctx.get('level', 'core')}\n"
        prompt += f"**Has theory:** {chapter_ctx.get('has_theory', False)}\n"
        prompt += f"**Difficulty:** {chapter_ctx.get('estimated_difficulty', 'intermediate')}\n"

        # Running example stage
        prompt += f"\n## Running Example\n"
        prompt += f"**This chapter's stage:** {chapter_ctx.get('summary', {}).get('running_example_stage', 'N/A')}\n"

        # What previous chapters taught
        if prev_ctxs:
            prompt += "\n## Previous Chapter Context (for continuity)\n"
            for cid, pctx in prev_ctxs.items():
                concepts = pctx.get("summary", {}).get("key_concepts", [])
                tone = pctx.get("summary", {}).get("narrative_tone", "")
                prompt += f"- **{cid}**: concepts={concepts}, tone='{tone}'\n"

        # Implementation and test artifacts to reference
        prompt += "\n## Artifacts to Work From\n"
        prompt += f"- Implementation: `artifacts/{chapter_ctx.get('chapter_id')}/implementation/`\n"
        prompt += f"- Tests: `artifacts/{chapter_ctx.get('chapter_id')}/tests/`\n"
        prompt += f"- Impl notes: `artifacts/{chapter_ctx.get('chapter_id')}/implementation/impl-notes.md`\n"

        prompt += "\n## Instructions\n"
        prompt += (
            "Write the complete chapter following the Chapter Structure in your "
            "system prompt. Every section is mandatory unless marked OPTIONAL.\n\n"
            "1. Start by reading the implementation code and impl-notes.md\n"
            "2. Read the tests to understand what behaviors are verified\n"
            "3. Read context.json files from ALL previous chapters for continuity\n"
            "4. Write the chapter to: `artifacts/{chapter_id}/narrative/chapter.md`\n"
            "5. Write the section outline to: `artifacts/{chapter_id}/narrative/outline.md`\n"
            "6. Follow the formality spectrum EXACTLY for each section\n"
            "7. Every formula MUST have BOTH a numerical example AND a life analogy\n"
            "8. The Source Mapping Table is MANDATORY\n"
        )

        return prompt

    def _build_rewrite_prompt(
        self,
        chapter_ctx: Dict,
        prev_ctxs: Dict[str, Dict],
        target: str,
        context: Dict[str, Any],
    ) -> str:
        """Build a targeted rewrite prompt for a specific section."""
        prompt = self.prompt
        prompt += f"\n\n---\n\n## REWRITE MODE — Only modify: {target}\n\n"
        prompt += f"**Existing chapter:** `artifacts/{chapter_ctx.get('chapter_id')}/narrative/chapter.md`\n"
        prompt += f"**Section to rewrite:** {target}\n"
        prompt += f"**Reason:** {context.get('revision_reason', 'Not specified')}\n"
        prompt += f"**Specific instructions:** {context.get('revision_instructions', 'Not specified')}\n"
        prompt += "\n## Instructions\n"
        prompt += (
            "1. Read the EXISTING chapter narrative\n"
            "2. ONLY modify the specified section — preserve everything else\n"
            "3. Maintain the same narrator voice and formality register\n"
            "4. Ensure the modified section still flows naturally into the next section\n"
            "5. If code examples are changed, verify they still run\n"
        )
        return prompt

    def _build_qa_prompt(
        self, chapter_ctx: Dict, prev_ctxs: Dict[str, Dict], question: str
    ) -> str:
        """Build a Q&A prompt — answer a user question based on chapter artifacts."""
        prompt = self.prompt
        prompt += f"\n\n---\n\n## Q&A MODE (Read-Only) — Answer a reader's question\n\n"
        prompt += f"**Question:** {question}\n"
        prompt += f"**Context chapter:** {chapter_ctx.get('chapter_id')} — {chapter_ctx.get('title')}\n"
        prompt += f"**Available artifacts:** `artifacts/{chapter_ctx.get('chapter_id')}/`\n"
        prompt += (
            "\n## Instructions\n"
            "1. Read the chapter narrative, implementation, and impl-notes\n"
            "2. Answer the question in the voice of the 'knowledgeable friend' narrator\n"
            "3. Use plain language — if the answer involves math, include a numerical example\n"
            "4. Cite the exact location in the chapter where the reader can learn more\n"
            "5. Do NOT modify any files — this is read-only\n"
        )
        return prompt

    def _post_gate(self, result: AgentResult) -> GateResult:
        """Verify chapter.md exists and has required sections."""
        if self.read_only:
            return GateResult(passed=True, reason="Read-only mode")

        chapter_path = self.narrative_dir / "chapter.md" if self.narrative_dir else None
        outline_path = self.narrative_dir / "outline.md" if self.narrative_dir else None

        checks = {
            "chapter_md_exists": chapter_path and chapter_path.exists(),
            "outline_md_exists": outline_path and outline_path.exists(),
        }
        failed = [k for k, v in checks.items() if not v]
        if failed:
            return GateResult(passed=False, reason=f"Missing files: {failed}", details=checks)
        return GateResult(passed=True, reason="Narrative complete", details=checks)
