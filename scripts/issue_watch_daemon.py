#!/usr/bin/env python3
"""Repo2Book GitHub issue 系统级值守守护（2026-09-29 重构：无弹窗，事件驱动的无头处理）。

由 Windows 计划任务 Repo2BookIssueWatch 每 5 分钟拉起（pythonw 无窗）。
行为：检测新事件（独立状态文件）——
  - 无事件 / CHECK-FAILED：静默退出（零成本零打扰）。
  - 有事件（新 issue/重开/新评论）：无头拉起 `claude -p`（bypassPermissions，
    prompt 限定为 issue 处理工作流），由它跑 scripts/watch_issues.py（主状态）
    对账并处理——若交互会话已处理过，主状态 QUIET、无头进程即退出，
    天然防两会话重复处理。
凭据同源（git credential manager）。日志：scripts/.issue-watch-os.log。
无任何 UI（不弹窗，用户 2026-09-29 指定）。
"""
import json
import os
import subprocess
import sys
import datetime

# pythonw 下 sys.stdout/stderr 为 None——被导入模块（watch_issues 的失败分支
# 会 print）一旦 print 就 AttributeError 崩溃，pythonw 崩溃可能弹系统错误框
# （2026-09-29 弹窗风暴教训）。开局一律重定向到 devnull，物理杜绝任何输出路径。
sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
sys.stderr = sys.stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import watch_issues as wi

STATE = os.path.join(HERE, ".issue-watch-os-state.json")
LOG = os.path.join(HERE, ".issue-watch-os.log")

HEADLESS_PROMPT = (
    "你是 Repo2Book 的 issue 值守员（计划任务无头拉起）。仓库根 E:\\Laboratory\\Repo2Book。"
    "第一步：cd /e/Laboratory/Repo2Book && python scripts/watch_issues.py（本机用 python，python3 是坏桩）。"
    "若 QUIET/BASELINE：说明交互会话已处理或无事件，直接结束，不写任何文件不提交。"
    "若 CHECK-FAILED：直接结束。"
    "若 ALERT：逐事件执行——python scripts/watch_issues.py --show N 读全文+评论；"
    "对照仓库代码/文档深入核实（有歧义先用 --comment N 发澄清问题、保持 open）；"
    "确认成立则修复（脚本/文档/契约/linter 类直接修并跑相关 linter 自证；"
    "涉及章节正文 instances/*/artifacts*/narrative/ 的派 writer 角色子代理修，"
    "修完跑 lint_chapter_structure+lint_formulas+lint_trace_consistency）；"
    "已修复的 git add 相关文件→commit（信息末行 Co-Authored-By: Claude Opus 5 (1M context)"
    " <noreply@anthropic.com>）→git push https://github.com/HeduAiDev/Repo2Book.git"
    " vllm-book-v3-rewrite→python scripts/watch_issues.py --close N \"处理结论+commit 短号\"。"
    "铁律：只动 E:\\Laboratory\\Repo2Book 目录树内文件；python 改文件 newline='\\n' 保 LF；"
    "数字不许编；单轮最多 5 个事件。返回 ≤10 行摘要。"
)


def log(msg: str) -> None:
    line = f"{datetime.datetime.now():%F %H:%M:%S} {msg}\n"
    try:
        with open(LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
    except OSError:
        pass


def main() -> None:
    try:
        issues = wi.api(f"/repos/{wi.REPO}/issues?state=open&sort=updated&direction=desc&per_page=50")
        issues = [i for i in issues if "pull_request" not in i]
    except SystemExit:
        return  # 网络抖动静默重试
    prev = {}
    if os.path.exists(STATE):
        try:
            prev = json.load(open(STATE, encoding="utf-8"))
        except ValueError:
            pass
    baseline = "_init" not in prev
    has_events, cur = False, {}
    for i in issues:
        n = str(i["number"])
        cur[n] = {"title": i["title"], "comments": i["comments"], "state": "open"}
        p = prev.get(n)
        if p is None:
            if not baseline:
                has_events = True
        elif p.get("state") == "closed" or i["comments"] > p.get("comments", 0):
            has_events = True
    cur["_init"] = True
    try:
        json.dump(cur, open(STATE, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    except OSError as e:
        log(f"state-write-failed: {e}")
        return
    if baseline:
        log(f"baseline: {len(issues)} open")
        return
    if not has_events:
        return
    log("event: detected (log-only mode — 2026-09-29 用户reopen实锤无头修复质量不足，处理收回主会话；恢复自主处理需 Lead 改回)")
    return
    # 无头拉起：--permission-mode bypassPermissions（用户部署的值守机器人，
    # prompt 已限定工作流与目录树）；输出追加日志供事后审计。
    try:
        r = subprocess.run(
            ["claude", "-p", HEADLESS_PROMPT, "--permission-mode", "bypassPermissions"],
            cwd="E:\\Laboratory\\Repo2Book",
            capture_output=True, text=True, timeout=1800,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log(f"headless rc={r.returncode}: {(r.stdout or '').strip()[:400]}")
    except Exception as e:
        log(f"headless-failed: {e}")


if __name__ == "__main__":
    main()
