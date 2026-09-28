#!/usr/bin/env python3
"""Repo2Book GitHub issue 系统级值守通知器（2026-09-28 用户批准部署）。

与 scripts/watch_issues.py 的分工：本脚本由 Windows 计划任务每 5 分钟拉起，
**只检测+桌面弹窗通知用户**，不处理 issue；用独立状态文件（不与 Claude
会话内的值守 cron 抢事件——两侧各自独立发现同一个新 issue 是预期行为，
会话侧负责处理）。凭据同源（git credential manager，token 不落盘）。

弹窗走 PowerShell ToastNotificationManager（Win10/11 原生，无需第三方模块）。
pythonw 运行（计划任务无窗）。事件与错误追加 log（scripts/.issue-watch-os.log）。
"""
import json
import os
import subprocess
import sys
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import watch_issues as wi  # 复用凭据/API/数据结构

STATE = os.path.join(HERE, ".issue-watch-os-state.json")
LOG = os.path.join(HERE, ".issue-watch-os.log")


def log(msg: str) -> None:
    line = f"{datetime.datetime.now():%F %H:%M:%S} {msg}\n"
    try:
        with open(LOG, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
    except OSError:
        pass


def toast(title: str, body: str) -> None:
    ps = (
        "$x=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        "$t=$x.GetXml();$x.SelectSingleNode('//text[1]').AppendChild($x.CreateTextNode('REPO2BOOK'))|Out-Null;"
        "$x.SelectSingleNode('//text[2]').AppendChild($x.CreateTextNode('BODY'))|Out-Null;"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Repo2Book IssueWatch')"
        ".Show([Windows.UI.Notifications.ToastNotification]::new($x))"
    )
    ps = ps.replace("REPO2BOOK", title.replace("'", "''")).replace("BODY", body.replace("'", "''"))
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            timeout=30, capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as e:  # 弹窗失败不误检测
        log(f"toast-failed: {e}")


def main() -> None:
    try:
        issues = wi.api(f"/repos/{wi.REPO}/issues?state=open&sort=updated&direction=desc&per_page=50")
        issues = [i for i in issues if "pull_request" not in i]
    except SystemExit:
        log("check-failed（API/网络，详见 watch_issues 输出口径）")
        return  # 静默重试，不弹窗打扰
    prev = {}
    if os.path.exists(STATE):
        try:
            prev = json.load(open(STATE, encoding="utf-8"))
        except ValueError:
            pass
    baseline = "_init" not in prev
    alerts = []
    cur = {}
    for i in issues:
        n = str(i["number"])
        cur[n] = {"title": i["title"], "comments": i["comments"], "state": "open"}
        p = prev.get(n)
        if p is None:
            if not baseline:
                alerts.append(("新 issue", i))
        elif p.get("state") == "closed":
            alerts.append(("重开", i))
        elif i["comments"] > p.get("comments", 0):
            alerts.append(("新评论", i))
    cur["_init"] = True
    try:
        json.dump(cur, open(STATE, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    except OSError as e:
        log(f"state-write-failed: {e}")
        return
    if baseline:
        log(f"baseline: {len(issues)} open")
        return
    if not alerts:
        return
    log("alert: " + "; ".join(f"#{i['number']}({k})" for k, i in alerts))
    body = " ｜ ".join(f"#{i['number']} {i['title'][:40]}" for _, i in alerts[:3])
    if len(alerts) > 3:
        body += f" …等 {len(alerts)} 项"
    toast("GitHub issue 值守", body)


if __name__ == "__main__":
    if "--test-toast" in sys.argv:
        toast("GitHub issue 值守", "弹窗链路自检：看到我即正常")
    else:
        main()
