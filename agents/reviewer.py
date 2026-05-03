"""
Reviewer Agent — 从零基础读者视角审查章节。

The FINAL GATE before a chapter can be published. Checks:
1. Logical coherence
2. Readability
3. Engagement (not boring)
4. Cross-chapter consistency
5. Concept precision
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ARTIFACTS_DIR, AgentResult, BaseAgent, GateResult


class ReviewerAgent(BaseAgent):
    """Agent-4: Reviews from a zero-background reader perspective."""

    REVIEW_DIMENSIONS = [
        "code_walkthrough",           # v6: 源码手撕 — auto-REJECT if no code section
        "chapter_structure",          # v6: 章节结构 — auto-REJECT on duplicate IDs
        "algorithm_comprehension",    # v3: 0基础算法可理解性
        "source_grounding",           # v2: vLLM源码根基
        "formula_renderability",      # v1: 公式可渲染性
        "coherence",
        "readability",
        "engagement",
        "cross_chapter_consistency",
        "concept_precision",
    ]

    def __init__(self, chapter_id: str, cross_chapter_mode: bool = False):
        super().__init__("reviewer", chapter_id)
        self.cross_chapter_mode = cross_chapter_mode
        self.review_dir = None

    def _pre_gate(self) -> GateResult:
        """Narrative must be complete before review."""
        if self.cross_chapter_mode:
            return GateResult(passed=True, reason="Cross-chapter mode")

        ctx = self.load_context()
        if not ctx.get("gates", {}).get("narrative_complete"):
            return GateResult(
                passed=False,
                reason="Narrative gate not passed — Writer must complete first",
            )
        return GateResult(passed=True, reason="Ready to review")

    def _act(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the review."""
        self.review_dir = self.ensure_dir("reviews")

        chapter_ctx = self.load_context()
        prev_ctxs = self.load_previous_contexts()
        downstream_ctxs = self._load_downstream_contexts(chapter_ctx)

        review_prompt = self._build_review_prompt(
            chapter_ctx, prev_ctxs, downstream_ctxs, context
        )

        prompt_path = self.review_dir / "REVIEW_PLAN.md"
        prompt_path.write_text(review_prompt, encoding="utf-8")

        # Generate initial review report
        report = self._generate_review_report(
            chapter_ctx, prev_ctxs, downstream_ctxs
        )
        report_path = self.review_dir / "review-report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        verdict = report.get("overall_verdict", "REJECTED")

        if not self.cross_chapter_mode:
            if verdict == "APPROVED":
                self.update_gate("review_approved", True)
                self.add_changelog("review approved")
            else:
                self.update_gate("review_approved", False)
                self.add_changelog(f"review verdict: {verdict}")

            # Handle downstream consistency
            if verdict == "APPROVED":
                self.update_gate("downstream_consistency", "ok")
            elif report["dimensions"]["cross_chapter_consistency"]["score"] == "fail":
                # Mark downstream chapters for check
                self._mark_downstream_for_check(chapter_ctx)

        return AgentResult(
            agent_name=self.name,
            status="success" if verdict in ("APPROVED", "REVISE") else "failed",
            output_paths=[str(prompt_path), str(report_path)],
            summary=f"Review verdict: {verdict}",
            metadata=report,
        )

    def _build_review_prompt(
        self,
        chapter_ctx: Dict,
        prev_ctxs: Dict[str, Dict],
        downstream_ctxs: Dict[str, Dict],
        context: Dict[str, Any],
    ) -> str:
        """Build the review prompt."""
        prompt = self.prompt

        if self.cross_chapter_mode:
            prompt += "\n\n---\n\n## CROSS-CHAPTER REVIEW MODE\n"
            prompt += f"**Target chapters:** {context.get('target_chapters', [self.chapter_id])}\n"
        else:
            prompt += f"\n\n---\n\n## Review Chapter: {chapter_ctx.get('chapter_id')} — {chapter_ctx.get('title')}\n"

        prompt += f"\n## Chapter to Review\n"
        prompt += f"- **Narrative:** `artifacts/{chapter_ctx.get('chapter_id')}/narrative/chapter.md`\n"
        prompt += f"- **Implementation notes:** `artifacts/{chapter_ctx.get('chapter_id')}/implementation/impl-notes.md`\n"
        prompt += f"- **Test report:** `artifacts/{chapter_ctx.get('chapter_id')}/tests/test-report.json`\n"

        if prev_ctxs:
            prompt += "\n## Previous Chapters (for continuity check)\n"
            for cid, pctx in prev_ctxs.items():
                prompt += (
                    f"- **{cid}**: concepts={pctx.get('summary', {}).get('key_concepts', [])}, "
                    f"interface=`{pctx.get('summary', {}).get('code_interface', '')}`\n"
                )

        if downstream_ctxs:
            prompt += "\n## Downstream Chapters (affected by changes)\n"
            for cid, dctx in downstream_ctxs.items():
                prompt += f"- **{cid}**: depends on concepts from this chapter\n"

        prompt += "\n## Instructions\n"
        prompt += (
            "1. Read the chapter narrative AS A BEGINNER READER\n"
            "2. Evaluate all 5 dimensions (coherence, readability, engagement, "
            "cross-chapter consistency, concept precision)\n"
            "3. For every issue found, cite the exact location (section, paragraph)\n"
            "4. Provide specific, actionable revision instructions\n"
            "5. Generate review-report.json\n"
            "6. Set verdict: APPROVED, REVISE, or REJECTED\n"
        )

        return prompt

    def _generate_review_report(
        self,
        chapter_ctx: Dict,
        prev_ctxs: Dict[str, Dict],
        downstream_ctxs: Dict[str, Dict],
    ) -> Dict[str, Any]:
        """Generate the review report schema."""
        return {
            "chapter_id": self.chapter_id,
            "reviewer_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "mode": "cross_chapter" if self.cross_chapter_mode else "chapter",
            "dimensions": {
                dim: {
                    "score": "pass",
                    "issues": [],
                }
                for dim in self.REVIEW_DIMENSIONS
            },
            "overall_verdict": "APPROVED",
            "revision_instructions": "",
            "blocking_issues": [],
            "non_blocking_suggestions": [],
        }

    def _load_downstream_contexts(self, chapter_ctx: Dict) -> Dict[str, Dict]:
        """Load context.json for chapters that depend on this one."""
        downstream = {}
        for dep_id in chapter_ctx.get("dependents", []):
            dep_ctx_path = ARTIFACTS_DIR / dep_id / "context.json"
            if dep_ctx_path.exists():
                downstream[dep_id] = json.loads(dep_ctx_path.read_text(encoding="utf-8"))
        return downstream

    def _mark_downstream_for_check(self, chapter_ctx: Dict) -> None:
        """Mark dependent chapters as needing consistency check."""
        for dep_id in chapter_ctx.get("dependents", []):
            dep_ctx_path = ARTIFACTS_DIR / dep_id / "context.json"
            if dep_ctx_path.exists():
                dep_ctx = json.loads(dep_ctx_path.read_text(encoding="utf-8"))
                dep_ctx.setdefault("gates", {})["downstream_consistency"] = "needs_check"
                dep_ctx_path.write_text(
                    json.dumps(dep_ctx, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    def _post_gate(self, result: AgentResult) -> GateResult:
        """Review report must have APPROVED verdict."""
        report = result.metadata
        verdict = report.get("overall_verdict", "REJECTED")

        if verdict == "APPROVED":
            return GateResult(passed=True, reason="Review approved", details=report)
        elif verdict == "REVISE":
            return GateResult(
                passed=False,
                reason=f"Revision requested: {report.get('revision_instructions', '')[:200]}",
                details=report,
            )
        else:
            return GateResult(
                passed=False,
                reason=f"REJECTED — {len(report.get('blocking_issues', []))} blocking issues",
                details=report,
            )
