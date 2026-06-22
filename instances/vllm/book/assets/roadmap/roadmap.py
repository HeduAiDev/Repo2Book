#!/usr/bin/env python3
"""Book roadmap master — renders the vLLM v1 request-lifecycle spine as an SVG,
highlighting the current chapter's stage. Reused as each chapter's Roadmap ("你在这里").

Usage: python3 roadmap.py --highlight async-engine --out roadmap.svg
Coordinates are computed (svg-diagram skill convention); text is escaped.
"""
import argparse
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# (highlight-key, 标题, 副标题) — 请求生命周期主线
# ⚠️ 键名即语义：每个键对应它所高亮的那个框。调度器(ch13/14)并入「EngineCore 循环」
#    框，故用 'engine-core'；IPC 章(ch07)用 'ipc'。（历史上 engine-core/scheduler 两键
#    与标签错位、坑过 ch12，已理顺：键名与框语义一一对应。）
STAGES = [
    ("entrypoints", "入口", "LLM.generate / OpenAI server"),
    ("input-processor", "Stage 1 输入", "tokenize → EngineCoreRequest"),
    ("async-engine", "AsyncLLM 解耦", "三段式 / output_handler"),
    ("ipc", "IPC 边界", "ZMQ + msgpack 跨进程"),
    ("engine-core", "EngineCore 循环", "schedule → execute → sample"),
    ("output-processor", "Stage 3 输出", "detokenize → RequestOutput"),
    ("stream", "流式返回", "SSE / generate() 产出"),
]


def build(highlight: str) -> str:
    keys = [k for k, _, _ in STAGES]
    if highlight and highlight not in keys:
        raise SystemExit(
            f"未知 --highlight {highlight!r}。可用键: {', '.join(keys)}。"
            "（调度器/EngineCore 循环类章节用 'engine-core'；IPC 章用 'ipc'。"
            "off-spine 子系统暂用 '' 不高亮，待分层 roadmap 落地。）"
        )
    bw, bh, gap, x0, y0 = 168, 72, 38, 30, 92
    w = x0 * 2 + len(STAGES) * bw + (len(STAGES) - 1) * gap
    h = 200
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
             'markerWidth="7" markerHeight="5" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="{w // 2}" y="38" text-anchor="middle" font-size="20" '
             f'font-weight="bold" fill="#0f172a">vLLM v1 请求生命周期 · 全书地图</text>')
    L.append(f'<text x="{w // 2}" y="62" text-anchor="middle" font-size="13" '
             f'fill="#64748b">你在这里（高亮处为本章所在阶段）</text>')
    for i, (key, label, sub) in enumerate(STAGES):
        x = x0 + i * (bw + gap)
        on = (key == highlight)
        fill = "#2563eb" if on else "#f1f5f9"
        stroke = "#1d4ed8" if on else "#cbd5e1"
        tcol = "white" if on else "#0f172a"
        scol = "#dbeafe" if on else "#64748b"
        sw = 3 if on else 2
        L.append(f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="10" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{x + bw // 2}" y="{y0 + 31}" text-anchor="middle" '
                 f'font-size="15" font-weight="bold" fill="{tcol}">{esc(label)}</text>')
        L.append(f'<text x="{x + bw // 2}" y="{y0 + 53}" text-anchor="middle" '
                 f'font-size="10.5" fill="{scol}">{esc(sub)}</text>')
        if i < len(STAGES) - 1:
            ax = x + bw
            ax2 = x + bw + gap
            L.append(f'<line x1="{ax}" y1="{y0 + bh // 2}" x2="{ax2 - 3}" '
                     f'y2="{y0 + bh // 2}" stroke="#64748b" stroke-width="2" '
                     f'marker-end="url(#a)"/>')
    L.append('</svg>')
    return '\n'.join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlight", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build(a.highlight))
    print("wrote", a.out)
