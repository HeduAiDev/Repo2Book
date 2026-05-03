#!/usr/bin/env python3
"""
vLLM Book Factory — MetaGPT + Ralph Backpressure Orchestrator.

Main entry point for the multi-agent book-writing system.

Usage:
  python orchestrator.py outline           # Interactive outline design
  python orchestrator.py write <chapter>   # Write a full chapter
  python orchestrator.py fix <chapter>     # Fix a chapter based on feedback
  python orchestrator.py ask <chapter>     # Ask a question about a chapter
  python orchestrator.py status            # Show book status
  python orchestrator.py status <chapter>  # Show chapter status
  python orchestrator.py check-downstream  # Find chapters needing consistency check

Architecture:
  BookEditor (常驻入口)
    ├── Interactive outline design
    ├── Intent parsing + team dispatch
    └── State tracking + downstream management

  Chapter Pipeline (per-chapter):
    Implementer ──[gate]──→ Tester ──[gate]──→ Writer ──[gate]──→ Reviewer
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agents.book_editor import BookEditorAgent, IntentType, ScopeType, ParsedIntent
from pipelines.chapter_pipeline import ChapterPipeline


def cmd_outline(args):
    """Interactive outline design."""
    editor = BookEditorAgent()

    print("=" * 60)
    print("  vLLM Book Factory — 交互式目录设计")
    print("=" * 60)
    print()

    # Ask the key questions
    print("在设计目录之前，我想了解几个问题：\n")

    background = input("1. 目标读者有什么技术背景？（如：有Python和PyTorch基础）\n  > ")
    language = input("\n2. 偏好语言风格？（默认：中文为主+英文术语）\n  > ") or "zh-CN + en"
    mode = input("\n3. 教学模式？\n   [1] 手撕实现（从零重写vLLM核心功能,推荐）\n   [2] 源码导读（分析vLLM源码结构）\n  > ") or "1"

    print(f"\n好的。现在让我们逐级设计章节...\n")

    # Level 0: Foundation
    print("─" * 40)
    print("Level 0 — Foundation (基础数据结构和概念)")
    print("─" * 40)
    print("建议的章节：")
    print("  00-welcome: Why vLLM — 为什么需要专门的LLM推理引擎")
    print("  01-basics: LLM推理基础 — Transformer回顾, KV Cache直观理解")
    print()

    chapters = [
        {"id": "00-welcome", "number": 0, "title": "为什么需要 vLLM", "level": "foundation",
         "covers": "LLM推理的挑战，vLLM的设计哲学", "dependencies": [],
         "has_theory": False, "estimated_difficulty": "beginner"},
        {"id": "01-basics", "number": 1, "title": "LLM推理基础", "level": "foundation",
         "covers": "Transformer回顾, KV Cache直观理解, 推理内存瓶颈分析",
         "dependencies": ["00-welcome"], "has_theory": True, "estimated_difficulty": "beginner"},
    ]

    # Level 1: Core
    print("─" * 40)
    print("Level 1 — Core Algorithm (核心算法)")
    print("─" * 40)
    core_chapters_input = input(
        "输入核心章节的标题，用逗号分隔\n"
        "（建议: Self-Attention从零实现, KV Cache, PagedAttention, Continuous Batching）\n"
        "  > "
    )
    core_titles = [t.strip() for t in core_chapters_input.split(",")] if core_chapters_input.strip() else [
        "Self-Attention从零实现", "KV Cache原理与实现", "PagedAttention核心机制",
        "Continuous Batching连续批处理"
    ]
    for i, title in enumerate(core_titles):
        ch_num = len(chapters)
        ch_id = f"{ch_num:02d}-{title.lower().replace(' ', '-')[:30]}"
        dependencies = [chapters[-1]["id"]] if chapters else []
        chapters.append({
            "id": ch_id, "number": ch_num, "title": title, "level": "core",
            "covers": title, "dependencies": dependencies,
            "has_theory": True, "estimated_difficulty": "intermediate",
        })

    # Level 2: Enhancements
    print("\n" + "─" * 40)
    print("Level 2 — Enhancements (性能优化和工程增强)")
    print("─" * 40)
    enhance_input = input(
        "输入增强章节的标题，用逗号分隔\n"
        "（建议: 内存管理, 调度策略, 量化技术, Prefix Caching）\n"
        "  > "
    )
    enhance_titles = [t.strip() for t in enhance_input.split(",")] if enhance_input.strip() else [
        "内存管理与显存优化", "调度策略与请求排队", "量化技术入门"
    ]
    for i, title in enumerate(enhance_titles):
        ch_num = len(chapters)
        ch_id = f"{ch_num:02d}-{title.lower().replace(' ', '-')[:30]}"
        chapters.append({
            "id": ch_id, "number": ch_num, "title": title, "level": "enhancement",
            "covers": title,
            "dependencies": [chapters[-1]["id"]],
            "has_theory": True, "estimated_difficulty": "intermediate",
        })

    # Level 3: Advanced
    print("\n" + "─" * 40)
    print("Level 3 — Advanced (高级特性)")
    print("─" * 40)
    adv_input = input(
        "输入高级章节的标题，用逗号分隔\n"
        "（建议: 张量并行, 流水线并行, 投机解码, 端到端集成）\n"
        "  > "
    )
    adv_titles = [t.strip() for t in adv_input.split(",")] if adv_input.strip() else [
        "张量并行与分布式推理", "投机解码加速推理", "端到端集成与性能分析"
    ]
    for i, title in enumerate(adv_titles):
        ch_num = len(chapters)
        ch_id = f"{ch_num:02d}-{title.lower().replace(' ', '-')[:30]}"
        chapters.append({
            "id": ch_id, "number": ch_num, "title": title, "level": "advanced",
            "covers": title,
            "dependencies": [chapters[-1]["id"]],
            "has_theory": True, "estimated_difficulty": "advanced",
        })

    # Build outline
    outline = {
        "title": "vLLM 从零到专家",
        "reader_profile": {
            "assumed_knowledge": [background],
            "target_level": "intermediate",
            "language": language,
        },
        "running_example": {
            "name": "Llama-3.2-1B 推理全流程",
            "description": "用最小的 Llama 模型演示从 tokenize 到 output 的完整推理过程",
            "evolution": [{"chapter_id": ch["id"], "role_in_chapter": ch["covers"]} for ch in chapters],
        },
        "chapters": chapters,
    }

    editor.save_outline(outline)
    print(f"\n✅ 目录已保存到 {editor.outline_path}")
    print(f"共 {len(chapters)} 章")
    print("\n章节列表：")
    for ch in chapters:
        print(f"  {ch['id']}: {ch['title']} [{ch['level']}]")


def cmd_write(args):
    """Write a chapter."""
    chapter_id = args.chapter
    editor = BookEditorAgent()

    # Ensure chapter skeleton exists
    outline = editor.load_outline()
    chapter_spec = None
    for ch in outline.get("chapters", []):
        if ch["id"] == chapter_id:
            chapter_spec = ch
            break

    if not chapter_spec:
        print(f"❌ 章节 '{chapter_id}' 不在目录中。先运行 `python orchestrator.py outline`")
        return

    # Create skeleton
    editor.create_chapter_skeleton(chapter_spec)

    # Run pipeline
    print(f"\n{'='*60}")
    print(f"  开始写 {chapter_id}: {chapter_spec['title']}")
    print(f"{'='*60}\n")

    pipeline = ChapterPipeline(chapter_id, vllm_source_dir=str(ROOT / "vllm"))
    result = pipeline.run(
        progress_callback=lambda stage, status: print(f"  [{stage}] {status}")
    )

    print(f"\n{'='*60}")
    print(f"  Pipeline result: {result['final_status']}")
    if result.get("test_backpressure"):
        print(f"\n  {result['test_backpressure']['message']}")
    print(f"{'='*60}")


def cmd_fix(args):
    """Fix a chapter based on user feedback."""
    chapter_id = args.chapter
    feedback = args.feedback
    editor = BookEditorAgent()

    intent = editor.parse_intent(feedback, chapter_id)
    dispatch = editor.dispatch(intent)

    print(f"\n意图分析：")
    print(f"  类型: {intent.type.value}")
    print(f"  范围: {intent.scope.value}")
    print(f"  调度: {dispatch['agents']}")

    if dispatch.get("warnings"):
        for w in dispatch["warnings"]:
            print(f"  ⚠️  {w['message']}")

    # Run incremental pipeline
    pipeline = ChapterPipeline(chapter_id, vllm_source_dir=str(ROOT / "vllm"))
    result = pipeline.run_incremental(
        trigger=intent.type.value,
        context={
            "user_feedback": feedback,
            "revision_reason": feedback,
        },
    )
    print(f"\n结果: {result['final_status']}")


def cmd_ask(args):
    """Ask a question about a chapter."""
    chapter_id = args.chapter
    question = args.question

    pipeline = ChapterPipeline(chapter_id, vllm_source_dir=str(ROOT / "vllm"))
    result = pipeline.run_incremental(
        trigger="qa",
        context={"user_question": question},
    )
    print(f"\nQ&A prompt written. Result: {result['final_status']}")


def cmd_status(args):
    """Show book/chapter status."""
    editor = BookEditorAgent()

    if args.chapter:
        status = editor.get_chapter_status(args.chapter)
        if not status.get("exists"):
            print(f"❌ 章节 '{args.chapter}' 不存在")
        else:
            print(f"\n  {status['chapter_id']}: {status.get('title', 'N/A')}")
            print(f"  Status: {status['status']}")
            print(f"  Version: {status['version']}")
            print(f"  Gates:")
            for g, v in status.get("gates", {}).items():
                icon = "✅" if v == True else ("❌" if v == False else "⚠️")
                print(f"    {icon} {g}: {v}")
            print(f"  Last modified: {status['last_modified']}")
    else:
        book_status = editor.get_book_status()
        needing_check = editor.get_chapters_needing_check()

        print(f"\n{'='*60}")
        print(f"  vLLM Book Factory — Status")
        print(f"{'='*60}\n")

        for s in book_status:
            if s.get("exists"):
                gates = s.get("gates", {})
                all_pass = all(
                    v == True for v in gates.values()
                    if isinstance(v, bool)
                )
                icon = "✅" if all_pass else "🔄"
                print(f"  {icon} {s['chapter_id']}: {s.get('title', 'N/A')} (v{s['version']})")
            else:
                print(f"  ⬜ {s['chapter_id']}: [not started]")

        if needing_check:
            print(f"\n  ⚠️  Chapters needing consistency check: {needing_check}")


def main():
    parser = argparse.ArgumentParser(description="vLLM Book Factory Orchestrator")
    subparsers = parser.add_subparsers(dest="command")

    # outline
    subparsers.add_parser("outline", help="Interactive outline design")

    # write
    p_write = subparsers.add_parser("write", help="Write a chapter")
    p_write.add_argument("chapter", help="Chapter ID")

    # fix
    p_fix = subparsers.add_parser("fix", help="Fix a chapter")
    p_fix.add_argument("chapter", help="Chapter ID")
    p_fix.add_argument("feedback", help="What needs fixing")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask about a chapter")
    p_ask.add_argument("chapter", help="Chapter ID")
    p_ask.add_argument("question", help="Your question")

    # status
    p_status = subparsers.add_parser("status", help="Show book status")
    p_status.add_argument("chapter", nargs="?", help="Chapter ID (optional)")

    # check-downstream
    subparsers.add_parser("check-downstream", help="Find chapters needing consistency check")

    args = parser.parse_args()

    if args.command == "outline":
        cmd_outline(args)
    elif args.command == "write":
        cmd_write(args)
    elif args.command == "fix":
        cmd_fix(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "check-downstream":
        editor = BookEditorAgent()
        needing = editor.get_chapters_needing_check()
        if needing:
            print(f"Chapters needing consistency check: {needing}")
        else:
            print("All chapters consistent.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
