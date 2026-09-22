#!/usr/bin/env python3
"""Repo2Book GitHub issue 监听器（维护模式，2026-09-23 用户部署）。

职责：轮询 HeduAiDev/Repo2Book 的 open issues，与本地状态文件对账，
报告三类事件：NEW（新 issue）/ REOPENED（关闭后重开）/ COMMENT（开放
issue 有新评论——深入讨论回环需要）。凭据复用 git credential manager
（与 push 同源，token 不落盘不打印）。

用法：
  python scripts/watch_issues.py            # 常规轮询：输出 BASELINE/QUIET/ALERT 块
  python scripts/watch_issues.py --show 12  # 看某 issue 全文 + 全部评论（处理时用）

退出码：0=正常（无论有无事件）；1=CHECK-FAILED（网络/凭据/限流，下轮重试）。
首次运行为 BASELINE：只记录当前 open issues、不告警。
"""
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印符号免疫

REPO = "HeduAiDev/Repo2Book"
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".issue-watch-state.json")
UA = "repo2book-issue-watch"


def gh_token() -> str | None:
    out = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    return None


def api(path: str):
    tok = gh_token()
    if not tok:
        print("CHECK-FAILED: git credential fill 未取到 GitHub token")
        sys.exit(1)
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "User-Agent": UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        print(f"CHECK-FAILED: GitHub API {e.code} {path}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"CHECK-FAILED: 网络 {e.reason}")
        sys.exit(1)


def brief(i: dict) -> str:
    body = (i.get("body") or "").strip().replace("\r", "")
    excerpt = body[:800] + ("…" if len(body) > 800 else "")
    who = (i.get("user") or {}).get("login", "?")
    labels = ",".join(l["name"] for l in i.get("labels", [])) or "-"
    return (
        f"#{i['number']} [{labels}] {i['title']}\n"
        f"  by {who} · created {i['created_at']} · {i['comments']} comments\n"
        f"  {i['html_url']}\n"
        f"  body: {excerpt or '(空)'}"
    )


def load_state() -> dict:
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE, encoding="utf-8"))
        except ValueError:
            pass
    return {}


def save_state(st: dict) -> None:
    json.dump(st, open(STATE, "w", encoding="utf-8", newline="\n"),
              ensure_ascii=False, indent=1)


def poll() -> None:
    issues = api(f"/repos/{REPO}/issues?state=open&sort=updated&direction=desc&per_page=50")
    issues = [i for i in issues if "pull_request" not in i]  # issues API 混入 PR，剔除
    prev = load_state()
    baseline = "_init" not in prev
    alerts, cur = [], {}
    for i in issues:
        n = str(i["number"])
        cur[n] = {"title": i["title"], "comments": i["comments"],
                  "updated": i["updated_at"], "state": "open"}
        p = prev.get(n)
        if p is None:
            if not baseline:
                alerts.append(("NEW", i))
        elif p.get("state") == "closed":
            alerts.append(("REOPENED", i))
        elif i["comments"] > p.get("comments", 0):
            alerts.append((f"COMMENT +{i['comments'] - p.get('comments', 0)}", i))
    cur["_init"] = True
    save_state(cur)

    if baseline:
        print(f"BASELINE: 记录当前 {len(issues)} 个 open issues（不告警）")
        for i in issues:
            print(f"  #{i['number']} {i['title']}")
        return
    if not alerts:
        print(f"QUIET: 无新事件（{len(issues)} open）")
        return
    print(f"ALERT: {len(alerts)} 个事件")
    for kind, i in alerts:
        print(f"\n== [{kind}] ==")
        print(brief(i))


def show(n: str) -> None:
    i = api(f"/repos/{REPO}/issues/{n}")
    print(f"#{i['number']} {i['title']}  ({i['state']}, by {(i.get('user') or {}).get('login', '?')})")
    print(i.get("html_url", ""))
    print(f"\n--- body ---\n{(i.get('body') or '').strip()}")
    if i["comments"]:
        cs = api(f"/repos/{REPO}/issues/{n}/comments?per_page=100")
        print(f"\n--- comments ({len(cs)}) ---")
        for c in cs:
            print(f"\n[{(c.get('user') or {}).get('login', '?')} @ {c['created_at']}]")
            print(c.get("body", "").strip())


def rest(path: str, method: str, payload: dict | None = None):
    tok = gh_token()
    if not tok:
        print("CHECK-FAILED: git credential fill 未取到 GitHub token")
        sys.exit(1)
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=data, method=method,
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        print(f"CHECK-FAILED: GitHub API {e.code} {method} {path} {e.read()[:200]!r}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"CHECK-FAILED: 网络 {e.reason}")
        sys.exit(1)


def comment(n: str, text: str) -> None:
    rest(f"/repos/{REPO}/issues/{n}/comments", "POST", {"body": text})
    print(f"OK: 已评论 #{n}")


def close(n: str, text: str | None) -> None:
    if text:
        comment(n, text)
    rest(f"/repos/{REPO}/issues/{n}", "PATCH", {"state": "closed"})
    print(f"OK: 已关闭 #{n}")


def reopen(n: str) -> None:
    rest(f"/repos/{REPO}/issues/{n}", "PATCH", {"state": "open"})
    print(f"OK: 已重开 #{n}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--show":
        show(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--comment":
        comment(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--close":
        close(sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else None)
    elif len(sys.argv) >= 3 and sys.argv[1] == "--reopen":
        reopen(sys.argv[2])
    else:
        poll()
