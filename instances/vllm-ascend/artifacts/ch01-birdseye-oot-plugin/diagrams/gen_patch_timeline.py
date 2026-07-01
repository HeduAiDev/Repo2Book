#!/usr/bin/env python3
"""两段式 patch 的生命周期时间轴：platform 段（worker 启动前）vs worker 段（worker 内）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="40" font-family="sans-serif" font-size="24" font-weight="bold" fill="#0f172a" text-anchor="middle">两段式 monkey-patch 的时机轴：platform 段在 worker 启动前，worker 段在 worker 内</text>')

# 主时间轴
axis_y = 300
L.append(f'<line x1="60" y1="{axis_y}" x2="{W-60}" y2="{axis_y}" stroke="#334155" stroke-width="2.4" marker-end="url(#a)"/>')
L.append(f'<text x="{W-60}" y="{axis_y+26}" font-family="sans-serif" font-size="12.5" fill="#64748b" text-anchor="end">时间 →</text>')

# platform 段 band（上方）
L.append(f'<rect x="60" y="86" width="800" height="150" rx="12" fill="#faf5ff" stroke="#e9d5ff" stroke-width="1.6"/>')
L.append(f'<text x="80" y="112" font-family="sans-serif" font-size="15" font-weight="bold" fill="#7e22ce">platform 段　is_global_patch=True　作用域：engine-core / scheduler</text>')

# worker 段 band（下方）
L.append(f'<rect x="900" y="404" width="360" height="150" rx="12" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.6"/>')
L.append(f'<text x="920" y="430" font-family="sans-serif" font-size="15" font-weight="bold" fill="#1d4ed8">worker 段　is_global_patch=False</text>')
L.append(f'<text x="920" y="452" font-family="sans-serif" font-size="13" fill="#2563eb">作用域：worker 进程内 · 模型 / 算子</text>')


def stage(x, y, w, h, title, sub, col, bg):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{bg}" stroke="{col}" stroke-width="2"/>')
    L.append(f'<text x="{x+w/2}" y="{y+26}" font-family="sans-serif" font-size="14" font-weight="bold" fill="{col}" text-anchor="middle">{esc(title)}</text>')
    for i, s in enumerate(sub):
        L.append(f'<text x="{x+w/2}" y="{y+48+i*20}" font-family="{"monospace" if s[1] else "sans-serif"}" font-size="12" fill="#475569" text-anchor="middle">{esc(s[0])}</text>')
    return (x, y, w, h)


def tick(x, t):
    L.append(f'<circle cx="{x}" cy="{axis_y}" r="6" fill="#334155"/>')
    L.append(f'<text x="{x}" y="{axis_y-14}" font-family="sans-serif" font-size="12" fill="#334155" text-anchor="middle">{esc(t)}</text>')


# 时刻点
tick(180, "T1 平台选定")
tick(430, "T2 pre_register_and_update")
tick(700, "T3 engine-core 子进程")
tick(1030, "T4 worker.__init__")

# platform 段的两个触发点（band 内，连到 axis）
s1 = stage(110, 132, 280, 88, "pre_register_and_update()",
           [("adapt_patch(is_global_patch=True)", True), ("→ import patch.platform", True)], "#7e22ce", "#f3e8ff")
L.append(f'<line x1="{s1[0]+s1[2]/2}" y1="{s1[1]+s1[3]}" x2="430" y2="{axis_y-6}" stroke="#7e22ce" stroke-width="1.8" stroke-dasharray="4 4"/>')

s2 = stage(540, 132, 300, 88, "general plugins（每个子进程）",
           [("register_* 先 _ensure_global_patch()", False), ("进程级幂等锁：同进程只打一次", False)], "#7e22ce", "#f3e8ff")
L.append(f'<line x1="{s2[0]+s2[2]/2}" y1="{s2[1]+s2[3]}" x2="700" y2="{axis_y-6}" stroke="#7e22ce" stroke-width="1.8" stroke-dasharray="4 4"/>')

# worker 段触发点
s3 = stage(920, 456, 320, 82, "每个 worker.__init__",
           [("adapt_patch(is_global_patch=False)", True), ("→ import patch.worker", True)], "#1d4ed8", "#dbeafe")
L.append(f'<line x1="{s3[0]+s3[2]/2}" y1="{s3[1]}" x2="1030" y2="{axis_y+6}" stroke="#1d4ed8" stroke-width="1.8" stroke-dasharray="4 4"/>')

# 底注
L.append(f'<text x="60" y="600" font-family="sans-serif" font-size="13" fill="#64748b">判据：这个 vLLM 符号在哪一层、何时被用到——engine/scheduler 全局对象走 platform 段；worker 进程内的模型/算子走 worker 段。</text>')

L.append('</svg>')
open("patch_timeline.svg", "w").write("\n".join(L))
print("wrote patch_timeline.svg")
