"""
Book Editor Agent — 常驻入口 + 总调度。

The human reader's single point of contact. Handles:
- Interactive outline design
- User feedback / question parsing
- Agent team dispatch
- State tracking
- Proactive downstream management
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .base import ROOT, ARTIFACTS_DIR, BOOK_DIR, AgentResult, GateResult


class IntentType(Enum):
    CODE_BUG = "code_bug"
    READABILITY_ISSUE = "readability_issue"
    CONSISTENCY_ISSUE = "consistency_issue"
    QUESTION = "question"
    ADD_SECTION = "add_section"
    RESTRUCTURE = "restructure"
    STATUS_CHECK = "status_check"
    NEW_CHAPTER = "new_chapter"


class ScopeType(Enum):
    IMPLEMENTATION = "implementation"
    NARRATIVE = "narrative"
    CROSS_CHAPTER = "cross_chapter"
    READ_ONLY = "read_only"
    FULL_CHAPTER = "full_chapter"
    MULTI_CHAPTER = "multi_chapter"


@dataclass
class ParsedIntent:
    """Structured user intent."""
    type: IntentType
    scope: ScopeType
    chapter_id: str
    description: str
    agents_to_dispatch: List[str]
    metadata: Dict[str, Any]


class BookEditorAgent:
    """
    The Book Editor — always-on entry point for the book system.

    Unlike other agents, this one doesn't extend BaseAgent because it's
    not part of a single-chapter pipeline. It orchestrates across all chapters.
    """

    # Dispatch matrix: (IntentType, ScopeType) → [agent_names]
    DISPATCH_MATRIX = {
        (IntentType.CODE_BUG, ScopeType.IMPLEMENTATION): ["implementer", "tester", "writer"],
        (IntentType.CODE_BUG, ScopeType.NARRATIVE): ["writer", "reviewer"],
        (IntentType.READABILITY_ISSUE, ScopeType.NARRATIVE): ["writer", "reviewer"],
        (IntentType.CONSISTENCY_ISSUE, ScopeType.CROSS_CHAPTER): ["reviewer"],
        (IntentType.QUESTION, ScopeType.READ_ONLY): ["writer"],
        (IntentType.ADD_SECTION, ScopeType.FULL_CHAPTER): [
            "implementer", "tester", "writer", "reviewer"
        ],
        (IntentType.NEW_CHAPTER, ScopeType.FULL_CHAPTER): [
            "implementer", "tester", "writer", "reviewer"
        ],
        (IntentType.RESTRUCTURE, ScopeType.MULTI_CHAPTER): ["reviewer"],
        (IntentType.STATUS_CHECK, ScopeType.READ_ONLY): [],
    }

    def __init__(self):
        self.outline_path = BOOK_DIR / "book-outline.json"

    # ── Outline Management ──

    def load_outline(self) -> Dict[str, Any]:
        """Load the book outline."""
        if self.outline_path.exists():
            return json.loads(self.outline_path.read_text(encoding="utf-8"))
        return {"title": "", "chapters": [], "running_example": {}}

    def save_outline(self, outline: Dict[str, Any]) -> None:
        """Save the book outline."""
        BOOK_DIR.mkdir(parents=True, exist_ok=True)
        self.outline_path.write_text(
            json.dumps(outline, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def create_chapter_skeleton(self, chapter_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create a context.json skeleton for a new chapter."""
        chapter_id = chapter_spec["id"]
        ctx_path = ARTIFACTS_DIR / chapter_id / "context.json"
        ctx_path.parent.mkdir(parents=True, exist_ok=True)

        ctx = {
            "chapter_id": chapter_id,
            "chapter_number": chapter_spec.get("number", 0),
            "title": chapter_spec.get("title", ""),
            "version": 0,
            "status": "draft",
            "dependencies": chapter_spec.get("dependencies", []),
            "dependents": [],
            "summary": {
                "key_concepts": [],
                "code_interface": "",
                "prerequisites_assumed": [],
                "narrative_tone": "",
                "running_example_stage": "",
            },
            "gates": {
                "implementation_exists": False,
                "tests_pass": False,
                "narrative_complete": False,
                "review_approved": False,
                "downstream_consistency": "needs_check",
            },
            "artifacts": {
                "implementation_dir": f"artifacts/{chapter_id}/implementation",
                "test_dir": f"artifacts/{chapter_id}/tests",
                "narrative_path": f"artifacts/{chapter_id}/narrative/chapter.md",
                "review_path": f"artifacts/{chapter_id}/reviews/review-report.json",
            },
            "changelog": [],
        }
        ctx_path.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")

        # Update dependents on dependencies
        for dep_id in ctx["dependencies"]:
            dep_ctx_path = ARTIFACTS_DIR / dep_id / "context.json"
            if dep_ctx_path.exists():
                dep_ctx = json.loads(dep_ctx_path.read_text(encoding="utf-8"))
                if chapter_id not in dep_ctx.get("dependents", []):
                    dep_ctx.setdefault("dependents", []).append(chapter_id)
                    dep_ctx_path.write_text(
                        json.dumps(dep_ctx, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

        return ctx

    # ── Intent Parsing ──

    def parse_intent(self, user_input: str, chapter_id: str = "") -> ParsedIntent:
        """
        Parse user feedback into a structured intent.

        This is where the Book Editor's intelligence lives — understanding
        what the user actually wants and determining the right scope/agents.
        """
        inp = user_input.lower()

        # Keywords → IntentType mapping
        code_keywords = ["跑不通", "报错", "bug", "error", "代码", "实现", "wrong"]
        readability_keywords = ["太难", "看不懂", "复杂", "confusing", "不清楚",
                                "不通顺", "长", "啰嗦", "hard", "difficult"]
        consistency_keywords = ["矛盾", "不一致", "对不上", "contradict", "inconsistent",
                               "前面说", "后面说", "冲突"]
        question_keywords = ["解释", "是什么", "为什么", "怎么理解", "意思是",
                            "explain", "what is", "why", "how does", "?"]
        add_keywords = ["加一节", "加一章", "补充", "增加", "add section", "add chapter"]
        restructure_keywords = ["重新组织", "调整顺序", "合并", "拆分", "reorganize",
                               "restructure", "merge", "split"]

        # Detect intent type
        if any(kw in inp for kw in consistency_keywords):
            intent_type = IntentType.CONSISTENCY_ISSUE
            scope = ScopeType.CROSS_CHAPTER
        elif any(kw in inp for kw in code_keywords):
            intent_type = IntentType.CODE_BUG
            scope = ScopeType.IMPLEMENTATION
        elif any(kw in inp for kw in readability_keywords):
            intent_type = IntentType.READABILITY_ISSUE
            scope = ScopeType.NARRATIVE
        elif any(kw in inp for kw in question_keywords):
            intent_type = IntentType.QUESTION
            scope = ScopeType.READ_ONLY
        elif any(kw in inp for kw in add_keywords):
            intent_type = IntentType.ADD_SECTION
            scope = ScopeType.FULL_CHAPTER
        elif any(kw in inp for kw in restructure_keywords):
            intent_type = IntentType.RESTRUCTURE
            scope = ScopeType.MULTI_CHAPTER
        else:
            intent_type = IntentType.STATUS_CHECK
            scope = ScopeType.READ_ONLY

        agents = self.DISPATCH_MATRIX.get((intent_type, scope), [])

        return ParsedIntent(
            type=intent_type,
            scope=scope,
            chapter_id=chapter_id,
            description=user_input,
            agents_to_dispatch=agents,
            metadata={"raw_input": user_input},
        )

    # ── Status Reporting ──

    def get_chapter_status(self, chapter_id: str) -> Dict[str, Any]:
        """Get the current status of a chapter."""
        ctx_path = ARTIFACTS_DIR / chapter_id / "context.json"
        if not ctx_path.exists():
            return {"exists": False, "chapter_id": chapter_id}

        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        return {
            "exists": True,
            "chapter_id": ctx["chapter_id"],
            "title": ctx.get("title", ""),
            "version": ctx.get("version", 0),
            "status": ctx.get("status", "unknown"),
            "gates": ctx.get("gates", {}),
            "last_modified": ctx.get("changelog", [{}])[-1].get("date", "N/A")
            if ctx.get("changelog")
            else "N/A",
        }

    def get_book_status(self) -> List[Dict[str, Any]]:
        """Get the status of all chapters in the outline."""
        outline = self.load_outline()
        chapters = outline.get("chapters", [])
        return [self.get_chapter_status(ch["id"]) for ch in chapters]

    def get_chapters_needing_check(self) -> List[str]:
        """Find chapters with downstream_consistency: needs_check."""
        needing = []
        for ctx_dir in ARTIFACTS_DIR.iterdir():
            if ctx_dir.is_dir():
                ctx_path = ctx_dir / "context.json"
                if ctx_path.exists():
                    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                    if ctx.get("gates", {}).get("downstream_consistency") == "needs_check":
                        needing.append(ctx["chapter_id"])
        return needing

    # ── Pipeline Dispatch ──

    def dispatch(self, intent: ParsedIntent) -> Dict[str, Any]:
        """
        Dispatch the appropriate agent team based on the parsed intent.
        Returns a dispatch plan that the pipeline runner executes.
        """
        plan = {
            "intent": {
                "type": intent.type.value,
                "scope": intent.scope.value,
                "chapter_id": intent.chapter_id,
                "description": intent.description,
            },
            "agents": intent.agents_to_dispatch,
            "mode": "incremental" if intent.type != IntentType.NEW_CHAPTER else "full",
            "gates": [],
            "warnings": [],
        }

        # Build the gate chain
        if "implementer" in intent.agents_to_dispatch:
            plan["gates"].append("implementation_exists")
        if "tester" in intent.agents_to_dispatch:
            plan["gates"].append("tests_pass")
        if "writer" in intent.agents_to_dispatch:
            plan["gates"].append("narrative_complete")
        if "reviewer" in intent.agents_to_dispatch:
            plan["gates"].append("review_approved")

        # Check for downstream impacts
        if intent.scope in (ScopeType.IMPLEMENTATION, ScopeType.FULL_CHAPTER):
            ctx_path = ARTIFACTS_DIR / intent.chapter_id / "context.json"
            if ctx_path.exists():
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                dependents = ctx.get("dependents", [])
                if dependents:
                    plan["warnings"].append({
                        "type": "downstream_impact",
                        "message": f"Changes may affect: {dependents}",
                        "affected_chapters": dependents,
                    })

        return plan
