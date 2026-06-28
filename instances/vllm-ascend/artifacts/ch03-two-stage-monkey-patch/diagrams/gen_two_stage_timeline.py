#!/usr/bin/env python3
"""两段式 patch 时序/分层图：platform 段（构图前·进程级）vs worker 段（worker 构造时）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#6d28d9"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, lines, rx=10, fs=15, tw="normal", tcol="#1e293b"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * (fs + 4) / 2 + fs / 2 - 2
    for i, (t, w2, c2) in enumerate(lines):
        L.append(
            f'<text x="{x + w/2}" y="{cy + i*(fs+4)}" font-family="sans-serif" font-size="{fs}" '
            f'font-weight="{w2}" fill="{c2}" text-anchor="middle">{esc(t)}</text>'
        )


def arrow(x1, y1, x2, y2, col="#475569", marker="a", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2"{d} marker-end="url(#{marker})"/>')


# Title
L.append(f'<text x="{W/2}" y="38" font-family="sans-serif" font-size="24" font-weight="bold" fill="#0f172a" text-anchor="middle">两段式 monkey-patch：单一入口 adapt_patch · 二分两段时机</text>')

# Divider line between two stages
divx = W / 2
L.append(f'<line x1="{divx}" y1="70" x2="{divx}" y2="{H-30}" stroke="#cbd5e1" stroke-width="2" stroke-dasharray="6 6"/>')

# Stage headers
L.append(f'<text x="295" y="92" font-family="sans-serif" font-size="18" font-weight="bold" fill="#1d4ed8" text-anchor="middle">platform 段 · 进程早期、构图前</text>')
L.append(f'<text x="885" y="92" font-family="sans-serif" font-size="18" font-weight="bold" fill="#b45309" text-anchor="middle">worker 段 · 每个 worker 进程构造时</text>')

# ---- LEFT (platform) triggers ----
box(40, 120, 230, 78, "#eff6ff", "#3b82f6",
    [("触发点①", "bold", "#1d4ed8"), ("NPUPlatform.", "normal", "#334155"),
     ("pre_register_and_update()", "normal", "#334155")], fs=13)
box(40, 210, 230, 78, "#eff6ff", "#3b82f6",
    [("触发点②", "bold", "#1d4ed8"), ("_ensure_global_patch()", "normal", "#334155"),
     ("（守卫 _GLOBAL_PATCH_APPLIED）", "normal", "#64748b")], fs=13)
box(40, 314, 230, 78, "#f5f3ff", "#8b5cf6",
    [("被三个 general_plugins 入口调：", "bold", "#6d28d9"), ("register_connector /", "normal", "#475569"),
     ("model_loader / service_profiling", "normal", "#475569")], fs=12.5)

# adapt_patch(True)
box(330, 200, 215, 70, "#dbeafe", "#2563eb",
    [("adapt_patch(", "normal", "#1e293b"), ("is_global_patch=True)", "bold", "#1d4ed8")], fs=14)
# import platform pkg
box(330, 320, 215, 72, "#1e293b", "#0f172a",
    [("from vllm_ascend.patch", "normal", "#e2e8f0"), ("import platform", "bold", "#93c5fd")], fs=13)

arrow(270, 159, 328, 218)
arrow(270, 249, 328, 235)
arrow(270, 353, 328, 252, col="#6d28d9", marker="ag")
arrow(437, 270, 437, 318)

# ---- RIGHT (worker) triggers ----
box(630, 165, 230, 70, "#fff7ed", "#f59e0b",
    [("触发点", "bold", "#b45309"), ("NPUWorker.__init__", "normal", "#334155")], fs=14)
box(920, 200, 215, 70, "#fef3c7", "#d97706",
    [("adapt_patch()", "bold", "#b45309"), ("（默认 is_global_patch=False）", "normal", "#475569")], fs=13)
box(920, 320, 215, 72, "#1e293b", "#0f172a",
    [("from vllm_ascend.patch", "normal", "#e2e8f0"), ("import worker", "bold", "#fcd34d")], fs=13)
arrow(860, 200, 918, 225, col="#b45309")
arrow(1027, 270, 1027, 318)

# bottom note band
by = 440
L.append(f'<rect x="40" y="{by}" width="{W-80}" height="120" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.4"/>')
L.append(f'<text x="62" y="{by+30}" font-family="sans-serif" font-size="15" font-weight="bold" fill="#0f172a">import 副作用 = 执行 patch</text>')
notes = [
    "patch 没有任何显式 apply()：import 这两个包时，包 __init__ 里一串「裸 import」级联触发各 patch_*.py 的模块级重绑代码。",
    "platform 段必须早于「构建 Scheduler / MultiprocExecutor / KV-cache 工厂表」之前——它替换的正是构图阶段就被引用的类。",
    "worker 子进程（spawn）是全新解释器，父进程的 patch 不继承，故必须在每个 worker __init__ 里重新触发 worker 段。",
]
for i, t in enumerate(notes):
    L.append(f'<text x="62" y="{by+58+i*22}" font-family="sans-serif" font-size="13" fill="#334155">• {esc(t)}</text>')

L.append('</svg>')
open("two_stage_timeline.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote two_stage_timeline.svg")
