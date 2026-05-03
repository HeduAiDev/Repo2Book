"""
Base Agent class — implements the MetaGPT Role pattern + Ralph Backpressure gates.

Each agent:
- Extends BaseAgent
- Has a _watch() condition that triggers it
- Has a _act() method that does the work
- Has configurable backpressure gates between stages
"""

import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root
ROOT = Path("/mnt/e/Laboratory/vllm-from-scratch")
ARTIFACTS_DIR = ROOT / "artifacts"
BOOK_DIR = ROOT / "book"
PROMPTS_DIR = ROOT / "prompts"


@dataclass
class GateResult:
    """Result of a backpressure gate check."""
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Output from an agent's _act() invocation."""
    agent_name: str
    status: str  # "success" | "failed" | "skipped"
    output_paths: List[str] = field(default_factory=list)
    summary: str = ""
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base agent following MetaGPT's Role pattern.

    Each agent:
    - Watches for a trigger condition (similar to MetaGPT's _watch())
    - Executes _act() when triggered
    - Passes through backpressure gates before being considered done
    """

    def __init__(self, name: str, chapter_id: str):
        self.name = name
        self.chapter_id = chapter_id
        self.artifacts_dir = ARTIFACTS_DIR / chapter_id
        self.prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load the agent's system prompt from prompts/{agent_name}.md."""
        prompt_path = PROMPTS_DIR / f"{self.name}.md"
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return ""

    @abstractmethod
    def _act(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the agent's primary action. Must be implemented by subclasses."""
        ...

    def _pre_gate(self) -> GateResult:
        """
        Check pre-conditions before acting.
        Override in subclasses that need input validation.
        """
        return GateResult(passed=True, reason="No pre-condition check defined")

    def _post_gate(self, result: AgentResult) -> GateResult:
        """
        Check post-conditions after acting — the Ralph Backpressure.
        Override in subclasses. This is where quality gates live.
        """
        return GateResult(passed=True, reason="No post-condition check defined")

    def run(self, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Run the agent: pre-gate → act → post-gate.

        If pre-gate fails: return failed result immediately.
        If act succeeds but post-gate fails: return result with gate failure info.
        """
        context = context or {}

        # Pre-condition check
        pre = self._pre_gate()
        if not pre.passed:
            return AgentResult(
                agent_name=self.name,
                status="failed",
                summary=f"Pre-gate failed: {pre.reason}",
                metadata={"gate_result": pre},
            )

        # Execute
        result = self._act(context)

        # Post-condition check (Ralph Backpressure)
        if result.status == "success":
            post = self._post_gate(result)
            if not post.passed:
                result.status = "failed"
                result.summary += f" | Post-gate failed: {post.reason}"
                result.metadata["gate_result"] = post

        return result

    # ── Context Helpers ──

    def load_context(self) -> Dict[str, Any]:
        """Load the chapter's context.json."""
        ctx_path = self.artifacts_dir / "context.json"
        if ctx_path.exists():
            return json.loads(ctx_path.read_text(encoding="utf-8"))
        return {}

    def save_context(self, ctx: Dict[str, Any]) -> None:
        """Save the chapter's context.json."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        ctx_path = self.artifacts_dir / "context.json"
        ctx_path.write_text(
            json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_previous_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Load context.json from all chapters this one depends on."""
        ctx = self.load_context()
        prev = {}
        for dep_id in ctx.get("dependencies", []):
            dep_ctx_path = ARTIFACTS_DIR / dep_id / "context.json"
            if dep_ctx_path.exists():
                prev[dep_id] = json.loads(dep_ctx_path.read_text(encoding="utf-8"))
        return prev

    def ensure_dir(self, subdir: str) -> Path:
        """Ensure a subdirectory exists under artifacts/{chapter_id}/."""
        d = self.artifacts_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def update_gate(self, gate_name: str, value: Any) -> None:
        """Update a specific gate in context.json."""
        ctx = self.load_context()
        if "gates" not in ctx:
            ctx["gates"] = {}
        ctx["gates"][gate_name] = value
        self.save_context(ctx)

    def add_changelog(self, event: str, triggered_by: str = "") -> None:
        """Add a changelog entry to context.json."""
        ctx = self.load_context()
        if "changelog" not in ctx:
            ctx["changelog"] = []
        ctx["version"] = ctx.get("version", 0) + 1
        ctx["changelog"].append({
            "version": ctx["version"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "event": event,
            "triggered_by": triggered_by,
        })
        self.save_context(ctx)
