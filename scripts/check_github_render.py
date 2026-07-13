#!/usr/bin/env python3
"""真值口径:把章节送进 GitHub 自家的 markdown API,看有没有「本该渲染却吐成字面文本」的标记。

为什么需要它:lint_formulas.py 是**离线正则近似**,而 GitHub(cmark-gfm + math 扩展)的真实行为
只有 GitHub 自己知道。2026-07-13 正是靠这个 oracle 才发现全书 20 章里 93 处行内数学 + 16 处
粗体在 GitHub 上根本没渲染——而当时的 linter 全部放行(退出码 0)。

做法(基于结果,不基于规则):渲染 → 把 GitHub **确实**渲染成数学/代码的节点剔掉 → 正文里若还
站着 `$` 或 `**`,就是没渲染成功的标记。它不关心「为什么」失败,因此不会像正则那样漏掉未知病种。

用法:
    python3 scripts/check_github_render.py <chapter.md> [...]
    python3 scripts/check_github_render.py --all          # 活动实例全书
依赖:网络 + `gh` 已鉴权。退出码 1 = 有未渲染标记。
"""
import glob
import html as htmllib
import re
import subprocess
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])


def render(md_path: str) -> str:
    text = open(md_path, encoding="utf-8").read()
    p = subprocess.run(
        ["gh", "api", "-X", "POST", "/markdown", "-f", "mode=gfm", "-f", "text=" + text],
        capture_output=True, text=True, timeout=180,
    )
    if p.returncode != 0:
        raise RuntimeError("gh api 失败(网络/鉴权?): " + p.stderr[:200])
    return p.stdout


def _residue(h: str) -> str:
    """剔掉 GitHub 确实渲染成功的数学/代码节点,只留正文。"""
    h = re.sub(r"<math-renderer\b.*?</math-renderer>", " ", h, flags=re.S)
    h = re.sub(r"<pre\b.*?</pre>", " ", h, flags=re.S)
    h = re.sub(r"<code\b.*?</code>", " ", h, flags=re.S)
    h = re.sub(r"<div[^>]*data-math-style.*?</div>", " ", h, flags=re.S)
    h = re.sub(r"<[^>]+>", " ", h)
    return htmllib.unescape(h)


def check(md_path: str) -> list:
    hits = []
    for line in _residue(render(md_path)).split("\n"):
        for m in re.finditer(r".{0,40}\$.{0,40}", line):
            hits.append("未渲染数学: " + m.group(0).strip())
        for m in re.finditer(r".{0,40}\*\*.{0,40}", line):
            hits.append("未渲染粗体: " + m.group(0).strip())
    return hits


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--all" in sys.argv:
        from instance import active_instance_dir  # noqa: F401
        args = sorted(glob.glob(str(active_instance_dir()) + "/artifacts/*/narrative/*.md"))
    if not args:
        print(__doc__)
        sys.exit(2)

    total = 0
    for p in args:
        try:
            hits = check(p)
        except Exception as e:  # noqa: BLE001
            print("ERR  " + p + "  " + str(e)[:120])
            continue
        if hits:
            total += len(hits)
            print("### " + p + "  (" + str(len(hits)) + ")")
            for h in hits[:10]:
                print("    " + h)
    if total:
        print("\n🔴 " + str(total) + " 处标记在 GitHub 上没渲染 —— auto-REJECT")
    else:
        print("🟢 GitHub 真值:全部数学/粗体均正常渲染")
    sys.exit(1 if total else 0)
