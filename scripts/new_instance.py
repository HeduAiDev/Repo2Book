#!/usr/bin/env python3
"""repo2book —— 新建一本书（实例）。把任意代码仓库 scaffold 成一个待写的 book 实例。

用法:
  python3 scripts/new_instance.py <name> --repo <git-url> \
      [--title "书名"] [--repo-name vLLM] [--prefix vllm] \
      [--lang Python] [--reader advanced] [--lang-code zh-CN] \
      [--clone | --no-clone] [--activate]

做的事:
  1. 建 instances/<name>/{book/{cartography,bible,assets},knowledge/modules,trace/{decisions,deliveries},artifacts}
  2. 写 instances/<name>/repo2book.json（实例配置）+ INSTANCE.md（实例状态/规则）
  3. 种空的 Book Bible（glossary/interfaces/arc-map/voice-guide）+ trace/state.json + knowledge/INDEX.md
  4. （默认）把目标仓 blobless clone 进 instances/<name>/source（--no-clone 跳过）
  5. （--activate）把顶层 repo2book.json 的 active_instance 设为 <name>

之后：按 docs/superpowers/ARCHITECT-RUNBOOK.md §0 出架构地图 + 大纲，再逐章发车。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def w(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  + {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="实例名（短 slug，如 redis / sqlite / react）")
    ap.add_argument("--repo", required=True, help="目标仓库 git URL")
    ap.add_argument("--title", default="")
    ap.add_argument("--repo-name", default="")
    ap.add_argument("--prefix", default="", help="正文规范源码路径前缀（默认取仓库名）")
    ap.add_argument("--lang", default="")
    ap.add_argument("--reader", default="advanced")
    ap.add_argument("--lang-code", default="zh-CN")
    ap.add_argument("--clone", dest="clone", action="store_true", default=True)
    ap.add_argument("--no-clone", dest="clone", action="store_false")
    ap.add_argument("--activate", action="store_true", help="顺手把它设为 active_instance")
    a = ap.parse_args()

    name = a.name
    inst = ROOT / "instances" / name
    if inst.exists():
        sys.exit(f"实例已存在: {inst}")
    repo_name = a.repo_name or a.repo.rstrip("/").split("/")[-1].replace(".git", "")
    prefix = a.prefix or repo_name.lower()
    title = a.title or f"{repo_name} 源码解读"

    print(f"scaffold instances/{name}/ ...")
    for d in ["book/cartography", "book/bible", "book/assets/roadmap",
              "knowledge/modules", "trace/decisions", "trace/deliveries", "artifacts"]:
        (inst / d).mkdir(parents=True, exist_ok=True)
        (inst / d / ".gitkeep").touch()

    cfg = {
        "instance": name,
        "source": {
            "repo_url": a.repo, "repo_name": repo_name, "source_dir": f"instances/{name}/source",
            "canonical_prefix": prefix, "language": a.lang, "clone": "blobless",
            "tracked_version": "", "tracked_commit": "", "baseline_ancestor": "",
        },
        "book": {
            "title": title, "kind": "source-reading",
            "reader_profile": {"assumed_knowledge": [], "target_level": a.reader,
                               "language": a.lang_code, "style": "大白话 + 技术深度"},
            "outline_file": f"instances/{name}/book/cartography/outline-final.json",
            "artifacts_dir": f"instances/{name}/artifacts",
            "cartography_dir": f"instances/{name}/book/cartography",
            "bible_dir": f"instances/{name}/book/bible",
            "knowledge_dir": f"instances/{name}/knowledge",
            "trace_dir": f"instances/{name}/trace",
            "instance_doc": f"instances/{name}/INSTANCE.md",
        },
        "status": "新建实例 — 待出架构地图 + 大纲",
    }
    w(inst / "repo2book.json", json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")

    w(inst / "book/bible/glossary.json", "{}\n")
    w(inst / "book/bible/interfaces.json", "{}\n")
    w(inst / "book/bible/arc-map.json", "[]\n")
    # voice-guide 承袭既往实例打磨出的通用约定（templates/instance/voice-guide.md），
    # 只替换本仓特定占位（仓名/读者/语言/规范路径前缀）——绝不退回白板，否则丢掉跨书经验。
    vg_tpl = ROOT / "templates" / "instance" / "voice-guide.md"
    if vg_tpl.exists():
        vg = (vg_tpl.read_text(encoding="utf-8")
              .replace("{{REPO}}", repo_name).replace("{{PREFIX}}", prefix)
              .replace("{{READER}}", a.reader).replace("{{LANG_CODE}}", a.lang_code))
    else:
        vg = (f"# 叙述声线与风格指南（{repo_name} Book Bible）\n\n"
              "> 模板缺失，回退最小骨架——请从 instances/vllm/book/bible/voice-guide.md 补全通用约定。\n")
    w(inst / "book/bible/voice-guide.md", vg)
    w(inst / "knowledge/INDEX.md", f"# {repo_name} knowledge（仓库特定事实，带 TTL）\n\n（空）\n")
    w(inst / "trace/state.json", json.dumps({
        "project": f"{name}-source-reading-book", "outline": cfg["book"]["outline_file"],
        "status": "instance bootstrap", "chapters": {}, "updated_by": "new_instance",
    }, ensure_ascii=False, indent=2) + "\n")

    w(inst / "INSTANCE.md",
      f"# 实例：{repo_name}（{title}）\n\n"
      "> 本文件 = 当前实例的「当前状态 + 实例专属规则」。通用方法论见仓库根 `CLAUDE.md`。\n\n"
      f"- **源码**：`{a.repo}` → `instances/{name}/source/`（blobless clone）。规范路径前缀 `{prefix}/…`。\n"
      "- **跟踪版本**：（待填：tag/commit）。\n"
      f"- **读者**：{a.reader}（{a.lang_code}）。\n\n"
      "## 实例专属硬规则\n（如：某语言/某栈的运行环境约束、调试进容器等——按需补。）\n\n"
      "## 当前状态\n新建实例。下一步：出架构地图 + 大纲（RUNBOOK §0）。\n")

    if a.clone:
        src = inst / "source"
        print(f"blobless clone {a.repo} -> {src.relative_to(ROOT)} ...")
        rc = subprocess.run(["git", "clone", "--filter=blob:none", a.repo, str(src)]).returncode
        print("  clone OK" if rc == 0 else f"  clone 失败 (rc={rc})，可稍后手动 clone")
    else:
        print("  (--no-clone) 跳过克隆；记得手动把源码放到 source/")

    if a.activate:
        reg_p = ROOT / "repo2book.json"
        reg = json.loads(reg_p.read_text(encoding="utf-8"))
        reg["active_instance"] = name
        reg.setdefault("instances", {})[name] = {"config": f"instances/{name}/repo2book.json", "title": title}
        reg_p.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  active_instance -> {name}")

    print(f"\n完成。下一步：\n"
          f"  1. 在 source/ pin 一个 commit；填 instances/{name}/repo2book.json 的 tracked_commit + INSTANCE.md。\n"
          f"  2. {'已设为 active' if a.activate else f'把顶层 repo2book.json 的 active_instance 改成 ' + name}。\n"
          f"  3. 按 docs/superpowers/ARCHITECT-RUNBOOK.md：出架构地图 cartography/ + 大纲，再逐章发车。")


if __name__ == "__main__":
    main()
