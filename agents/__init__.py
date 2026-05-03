"""
Multi-Agent Book Factory — MetaGPT + Ralph Backpressure architecture.

Agent Roles:
  BookEditor  — 常驻入口 + 总调度
  Implementer — 从零手撕实现 (reimpl-tutorial analysis-deep patterns)
  Tester      — 验证正确性 (Ralph Backpressure gate)
  Writer      — 认知顺序写作 (reimpl-tutorial style-guide + walkthrough patterns)
  Reviewer    — 读者视角审查 (0-basis reader perspective)
"""

from .base import BaseAgent, AgentResult, GateResult
from .book_editor import BookEditorAgent, IntentType, ScopeType, ParsedIntent
from .implementer import ImplementerAgent
from .tester import TesterAgent
from .writer import WriterAgent
from .reviewer import ReviewerAgent

__all__ = [
    "BaseAgent",
    "AgentResult",
    "GateResult",
    "BookEditorAgent",
    "IntentType",
    "ScopeType",
    "ParsedIntent",
    "ImplementerAgent",
    "TesterAgent",
    "WriterAgent",
    "ReviewerAgent",
]
