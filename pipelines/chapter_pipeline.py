"""
Single-chapter pipeline: Implementer → Tester → Writer → Reviewer.

Orchestrates the 4-agent pipeline with Ralph Backpressure gates between stages.
Each stage can only start if the previous stage's gate is passed.
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import ARTIFACTS_DIR, AgentResult
from agents.implementer import ImplementerAgent
from agents.tester import TesterAgent
from agents.writer import WriterAgent
from agents.reviewer import ReviewerAgent


class ChapterPipeline:
    """
    Orchestrates the 4-agent pipeline for a single chapter.

    Pipeline stages with Ralph Backpressure:
      Implementer ──[implementation_exists]──→ Tester ──[tests_pass]──→
        Writer ──[narrative_complete]──→ Reviewer ──[review_approved]──→ Published
    """

    PIPELINE_GATES = [
        ("implementation_exists", "Implementer"),
        ("tests_pass", "Tester"),
        ("narrative_complete", "Writer"),
        ("review_approved", "Reviewer"),
    ]

    def __init__(self, chapter_id: str, vllm_source_dir: str = None):
        self.chapter_id = chapter_id
        self.vllm_source_dir = vllm_source_dir
        self.results: List[AgentResult] = []

    def run(
        self,
        stages: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the pipeline.

        Args:
            stages: Which stages to run. Default: all 4.
            context: Additional context for the agents.
            progress_callback: Called with (stage_name, status) for progress tracking.

        Returns:
            Dict with overall status, per-stage results, and gate states.
        """
        stages = stages or ["implementer", "tester", "writer", "reviewer"]
        context = context or {}

        def progress(stage: str, status: str):
            if progress_callback:
                progress_callback(stage, status)

        pipeline_result = {
            "chapter_id": self.chapter_id,
            "stages_run": [],
            "stages_skipped": [],
            "gate_results": {},
            "final_status": "unknown",
        }

        # Stage 1: Implementer
        if "implementer" in stages:
            progress("implementer", "running")
            impl = ImplementerAgent(self.chapter_id, self.vllm_source_dir)
            result = impl.run(context)
            self.results.append(result)
            pipeline_result["stages_run"].append("implementer")

            if result.status == "failed":
                progress("implementer", "failed")
                pipeline_result["final_status"] = "failed_at_implementer"
                pipeline_result["gate_results"]["implementation_exists"] = False
                return pipeline_result

            progress("implementer", "completed")
            pipeline_result["gate_results"]["implementation_exists"] = True
        else:
            pipeline_result["stages_skipped"].append("implementer")

        # Gate check before Tester
        ctx = self._load_context()
        if not ctx.get("gates", {}).get("implementation_exists") and "tester" in stages:
            pipeline_result["final_status"] = "gate_blocked: implementation_exists"
            return pipeline_result

        # Stage 2: Tester (THE BACKPRESSURE GATE)
        if "tester" in stages:
            progress("tester", "running")
            tester = TesterAgent(self.chapter_id)
            result = tester.run(context)
            self.results.append(result)
            pipeline_result["stages_run"].append("tester")

            if result.status == "failed":
                progress("tester", "failed")
                pipeline_result["final_status"] = "failed_at_tester"
                pipeline_result["gate_results"]["tests_pass"] = False
                pipeline_result["test_backpressure"] = {
                    "message": "⚠️ BACKPRESSURE: Tests failed. Writer will NOT be started.",
                    "action_required": "Fix implementation and re-run Tester",
                }
                return pipeline_result

            progress("tester", "completed")
            pipeline_result["gate_results"]["tests_pass"] = True
        else:
            pipeline_result["stages_skipped"].append("tester")

        # Gate check before Writer
        ctx = self._load_context()
        if not ctx.get("gates", {}).get("tests_pass") and "writer" in stages:
            pipeline_result["final_status"] = "gate_blocked: tests_pass"
            return pipeline_result

        # Stage 3: Writer
        if "writer" in stages:
            progress("writer", "running")
            read_only = context.get("read_only", False)
            writer = WriterAgent(self.chapter_id, read_only=read_only)
            result = writer.run(context)
            self.results.append(result)
            pipeline_result["stages_run"].append("writer")

            if result.status == "failed":
                progress("writer", "failed")
                pipeline_result["final_status"] = "failed_at_writer"
                pipeline_result["gate_results"]["narrative_complete"] = False
                return pipeline_result

            progress("writer", "completed")
            pipeline_result["gate_results"]["narrative_complete"] = True
        else:
            pipeline_result["stages_skipped"].append("writer")

        # Gate check before Reviewer
        ctx = self._load_context()
        if not ctx.get("gates", {}).get("narrative_complete") and "reviewer" in stages:
            pipeline_result["final_status"] = "gate_blocked: narrative_complete"
            return pipeline_result

        # Stage 4: Reviewer (THE FINAL GATE)
        if "reviewer" in stages:
            progress("reviewer", "running")
            cross_chapter = context.get("cross_chapter_mode", False)
            reviewer = ReviewerAgent(
                self.chapter_id, cross_chapter_mode=cross_chapter
            )
            result = reviewer.run(context)
            self.results.append(result)
            pipeline_result["stages_run"].append("reviewer")

            if result.status == "failed":
                verdict = result.metadata.get("overall_verdict", "REJECTED")
                progress("reviewer", f"failed ({verdict})")
                pipeline_result["final_status"] = f"review_{verdict.lower()}"
                pipeline_result["gate_results"]["review_approved"] = False

                if verdict == "REVISE":
                    pipeline_result["revision_instructions"] = result.metadata.get(
                        "revision_instructions", ""
                    )
                return pipeline_result

            progress("reviewer", "completed")
            pipeline_result["gate_results"]["review_approved"] = True

        pipeline_result["final_status"] = "published"
        pipeline_result["gate_results"]["downstream_consistency"] = "ok"

        # Update context status
        ctx = self._load_context()
        ctx["status"] = "published"
        self._save_context(ctx)

        return pipeline_result

    def run_incremental(
        self,
        trigger: str,
        rewrite_target: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run only the stages needed for an incremental change.

        This is called by the Book Editor when a user requests a modification
        to an existing chapter.
        """
        context = context or {}

        if trigger == "code_bug":
            # Code changed → re-test → re-write affected narrative → re-review
            return self.run(
                stages=["implementer", "tester", "writer", "reviewer"],
                context=context,
            )
        elif trigger == "narrative_only":
            # Only narrative changed → re-review
            return self.run(
                stages=["writer", "reviewer"],
                context={**context, "rewrite_target": rewrite_target},
            )
        elif trigger == "review_only":
            return self.run(stages=["reviewer"], context=context)
        elif trigger == "qa":
            return self.run(
                stages=["writer"],
                context={**context, "read_only": True},
            )
        else:
            raise ValueError(f"Unknown trigger: {trigger}")

    def _load_context(self) -> Dict[str, Any]:
        ctx_path = ARTIFACTS_DIR / self.chapter_id / "context.json"
        if ctx_path.exists():
            return json.loads(ctx_path.read_text(encoding="utf-8"))
        return {}

    def _save_context(self, ctx: Dict[str, Any]) -> None:
        ctx_path = ARTIFACTS_DIR / self.chapter_id / "context.json"
        ctx_path.parent.mkdir(parents=True, exist_ok=True)
        ctx_path.write_text(
            json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8"
        )
