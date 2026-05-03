"""
Implementer Agent — 从零手撕实现本章 vLLM 功能模块。

Follows the reimpl-tutorial patterns:
- Deep source analysis (analysis-deep.md Step 1-5)
- Algorithm derivation (derivation-prompt.md)
- Source mapping table
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .base import ROOT, AgentResult, BaseAgent, GateResult


class ImplementerAgent(BaseAgent):
    """Agent-1: Reimplements a vLLM feature from scratch."""

    def __init__(self, chapter_id: str, vllm_source_dir: str = None):
        super().__init__("implementer", chapter_id)
        self.vllm_source_dir = Path(vllm_source_dir or ROOT / "vllm")
        self.impl_dir = None

    def _pre_gate(self) -> GateResult:
        """Ensure chapter spec and source code exist."""
        ctx = self.load_context()
        if not ctx:
            return GateResult(
                passed=False,
                reason="No context.json found — run Book Editor first to define chapter spec",
            )
        if not self.vllm_source_dir.exists():
            return GateResult(
                passed=False,
                reason=f"vLLM source directory not found: {self.vllm_source_dir}",
            )
        return GateResult(passed=True, reason="Ready to implement")

    def _act(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the implementation using Claude Code."""
        self.impl_dir = self.ensure_dir("implementation")

        # Build the implementation prompt
        chapter_ctx = self.load_context()
        prev_ctxs = self.load_previous_contexts()

        impl_prompt = self._build_impl_prompt(chapter_ctx, prev_ctxs, context)

        # Write prompt to file for Claude Code to consume
        prompt_path = self.impl_dir / "IMPLEMENT.md"
        prompt_path.write_text(impl_prompt, encoding="utf-8")

        # Execute Claude Code (or use the direct approach via Claude's API)
        # For now, we output the structured prompt to be consumed
        result = self._execute_implementation(impl_prompt)

        if result["success"]:
            self.update_gate("implementation_exists", True)
            self.add_changelog("implementation complete")

        return AgentResult(
            agent_name=self.name,
            status="success" if result["success"] else "failed",
            output_paths=result.get("files", []),
            summary=result.get("summary", ""),
            errors=result.get("errors", []),
            metadata={"impl_dir": str(self.impl_dir)},
        )

    def _build_impl_prompt(
        self,
        chapter_ctx: Dict,
        prev_ctxs: Dict[str, Dict],
        context: Dict[str, Any],
    ) -> str:
        """Build the complete implementation prompt using reimpl-tutorial patterns."""
        prompt = self.prompt  # The implementer.md system prompt
        prompt += f"\n\n---\n\n## Chapter Spec\n\n"
        prompt += f"**Chapter ID:** {chapter_ctx.get('chapter_id')}\n"
        prompt += f"**Chapter Number:** {chapter_ctx.get('chapter_number')}\n"
        prompt += f"**Title:** {chapter_ctx.get('title')}\n"
        prompt += f"**What to implement:** {chapter_ctx.get('summary', {}).get('code_interface', 'See chapter spec')}\n"
        prompt += f"**Key concepts to cover:** {chapter_ctx.get('summary', {}).get('key_concepts', [])}\n"

        # Previous chapter interfaces (for compatibility)
        if prev_ctxs:
            prompt += "\n## Previous Chapter Interfaces (MUST be compatible)\n\n"
            for cid, pctx in prev_ctxs.items():
                iface = pctx.get("summary", {}).get("code_interface", "")
                if iface:
                    prompt += f"- **{cid}**: `{iface}`\n"

        # Source file hints from chapter spec
        source_hints = context.get("source_files", [])
        if source_hints:
            prompt += f"\n## Relevant vLLM Source Files\n\n"
            for f in source_hints:
                full_path = self.vllm_source_dir / f
                if full_path.exists():
                    prompt += f"- `{f}` ({full_path})\n"
                else:
                    prompt += f"- `{f}` (NOT FOUND — search in vllm/ directory)\n"

        # Running example context
        prompt += f"\n## Running Example\n"
        prompt += f"**Stage:** {chapter_ctx.get('summary', {}).get('running_example_stage', 'N/A')}\n"
        prompt += f"**Previous stage:** Will be read from dependencies\n"

        prompt += "\n## Instructions\n"
        prompt += (
            "1. Read the vLLM source files listed above\n"
            "2. Understand the core algorithm and design decisions\n"
            "3. Reimplement from scratch — simpler than original but NOT wrong\n"
            "4. Write code to: `artifacts/{chapter_id}/implementation/{module}.py`\n"
            "5. Write design notes to: `artifacts/{chapter_id}/implementation/impl-notes.md`\n"
            "6. Include a SOURCE MAPPING TABLE in impl-notes.md\n"
            "7. Verify your code runs with sample inputs before marking done\n"
        )

        return prompt

    def _execute_implementation(self, prompt: str) -> Dict[str, Any]:
        """
        Execute the implementation.

        In this architecture, the agent writes a structured implementation prompt
        to a file, and Claude Code (or another AI backend) executes it.

        The prompt file serves as both the instruction and the audit trail.
        """
        # Write the implementation code
        # This is where Claude Code would be invoked programmatically
        # For now, we structure the output and let the pipeline runner handle execution
        return {
            "success": True,
            "files": [
                str(self.impl_dir / "IMPLEMENT.md"),
            ],
            "summary": f"Implementation prompt written to {self.impl_dir / 'IMPLEMENT.md'}",
            "errors": [],
        }

    def _post_gate(self, result: AgentResult) -> GateResult:
        """Verify implementation files exist."""
        impl_files = list(self.impl_dir.glob("*.py")) if self.impl_dir else []
        notes_file = self.impl_dir / "impl-notes.md" if self.impl_dir else None

        checks = {
            "implementation_code_exists": len(impl_files) > 0,
            "impl_notes_exist": notes_file and notes_file.exists(),
        }

        failed = [k for k, v in checks.items() if not v]
        if failed:
            return GateResult(
                passed=False,
                reason=f"Missing: {failed}",
                details=checks,
            )
        return GateResult(passed=True, reason="All implementation files present", details=checks)
