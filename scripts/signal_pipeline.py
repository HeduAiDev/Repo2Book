#!/usr/bin/env python3
"""
Signal-based pipeline: Writer ↔ Reviewer via file signals.
No inbox polling. No deadlocks.

CRITICAL: Writer signals "done" ONLY after passing all self-checks.
          NO premature handoff. Ralph Backpressure: writer gate MUST pass.

Files in /tmp/book-factory-{chapter}/:
  writer-done       Writer: all checks passed, ready for review
  reviewer-feedback Reviewer: REVISE with blocking issues (JSON)
  reviewer-approved Reviewer: APPROVED (empty file = signal)
"""

import json, os, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TEAM = "book-factory"

WRITER_PROMPT = """Writer 自检闸门——全部通过才能发"done"信号。

信号目录: /tmp/book-factory-{chapter}/
mkdir -p /tmp/book-factory-{chapter}/

=== 写作 ===
1. 读取 .claude/agents/writer.md
2. 读取 researcher 简报(如果有): instances/vllm/trace/chapters/{chapter}/research-brief.md
3. 读取实现代码(确认行号): instances/vllm/artifacts/{chapter}/implementation/
4. 写叙事到 {narrative}
5. 生成所有需要的图表——svg-diagram skill，先图后文，图在文中引用

=== 自检闸门 === 全部通过才能发信号
6. 每个图表引用都对应实际文件: ls instances/vllm/artifacts/{chapter}/diagrams/
7. lint_formulas.py {narrative} → 0 blocking（不是"没有 blocking"，是运行后确认 0 blocking）
8. lint_source_grounding.py instances/vllm/artifacts/{chapter}/ → all pass
9. 以上全部通过 → touch /tmp/book-factory-{chapter}/writer-done

=== 等待审查 ===
10. 等 reviewer 信号（最多等 1 小时）:
    while [ ! -f /tmp/book-factory-{chapter}/reviewer-feedback ] && [ ! -f /tmp/book-factory-{chapter}/reviewer-approved ]; do sleep 30; done
11. reviewer-approved → exit（章节完成）
12. reviewer-feedback → 读文件，按 blocking 列表逐条修复 → rm reviewer-feedback → rm writer-done → 回到步骤3"""

REVIEWER_PROMPT = """Reviewer 审查——0-basis readability 标准。

信号目录: /tmp/book-factory-{chapter}/

1. 读取 .claude/agents/reviewer.md
2. 等待 writer 的 done 信号（最多 1 小时）:
   while [ ! -f /tmp/book-factory-{chapter}/writer-done ]; do sleep 30; done
3. 读取 {narrative}
4. lint_formulas + lint_source_grounding（独立验证，不信任 writer 的自检）
5. 9 维审查，核心标准：零基础读者能看懂吗？
6. 判定:
   APPROVED → touch /tmp/book-factory-{chapter}/reviewer-approved
            → 写 {reviews}/review-report.json (verdict: APPROVED)
            → exit
   REVISE   → 写 /tmp/book-factory-{chapter}/reviewer-feedback
            → 格式: {{"blocking":["行X: 具体问题"],"suggestions":["建议"]}}
            → 回到步骤2（等 writer 修复后重新 writer-done）"""


def clean(chapter: str):
    """Remove signal files and old agents from config."""
    import shutil
    sig_dir = Path(f"/tmp/book-factory-{chapter}")
    if sig_dir.exists():
        shutil.rmtree(sig_dir)

    cf = Path.home() / ".claude/teams/book-factory/config.json"
    if cf.exists():
        with open(cf) as f:
            c = json.load(f)
        c['members'] = [m for m in c['members'] if 'agentId' not in m]
        with open(cf, 'w') as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
        print(f"Config: {len(c['members'])} members")


def generate(chapter: str):
    narrative = f"instances/vllm/artifacts/{chapter}/narrative/chapter.md"
    reviews = f"instances/vllm/artifacts/{chapter}/reviews"

    print(f"""
# ============================================================
# Chapter: {chapter}
# Signal dir: /tmp/book-factory-{chapter}/
# ============================================================

# Step 1 — Clean
python3 scripts/signal_pipeline.py --clean {chapter}

# Step 2 — Spawn WRITER
Agent(
  subagent_type="general-purpose",
  team_name="{TEAM}",
  name="writer",
  run_in_background=True,
  prompt={json.dumps(WRITER_PROMPT.format(chapter=chapter, narrative=narrative))}
)

# Step 3 — Spawn REVIEWER
Agent(
  subagent_type="general-purpose",
  team_name="{TEAM}",
  name="reviewer",
  run_in_background=True,
  prompt={json.dumps(REVIEWER_PROMPT.format(chapter=chapter, narrative=narrative, reviews=reviews))}
)

# Step 4 — Spawn RESEARCHER (optional, can run in parallel)
Agent(
  subagent_type="general-purpose",
  team_name="{TEAM}",
  name="researcher",
  run_in_background=True,
  prompt="读取 .claude/agents/researcher.md。调研 {chapter}。输出到 instances/vllm/trace/chapters/{chapter}/research-brief.md。完成后通知 writer（touch /tmp/book-factory-{chapter}/research-done）。"
)
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: signal_pipeline.py <chapter_id>")
        print("       signal_pipeline.py --clean <chapter_id>")
        sys.exit(1)

    if sys.argv[1] == "--clean":
        clean(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        generate(sys.argv[1])
