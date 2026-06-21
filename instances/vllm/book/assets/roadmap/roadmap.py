#!/usr/bin/env python3
"""Book roadmap master — renders the vLLM v1 request-lifecycle spine as an SVG,
highlighting the current chapter's stage. Reused as each chapter's Roadmap ("你在这里").

Usage: python3 roadmap.py --highlight async-engine --out roadmap.svg
Coordinates are computed (svg-diagram skill convention); text is escaped.
"""
import argparse
import re
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


_CJK = re.compile(r'[㐀-䶿一-鿿　-〿＀-￯]')


_CJK_FONT = "Droid Sans Fallback"  # host 上唯一带 CJK 字形的字体


def cjk_font(svg: str) -> str:
    """PNG 渲染器（ImageMagick）对 sans-serif/monospace 的中文不做逐字回退 → 中文丢字形。
    凡 <text> 内容含中文的，把其 font-family 强制为单独的 "Droid Sans Fallback"
    （绝不与 latin 族逗号并列，否则又会丢字），让中文在 PNG 里正确渲染。
    latin/数字文本（不含 CJK）保持原 font-family 不变。惠及全书所有章节的 Roadmap。"""
    def repl(m):
        attrs, content = m.group(1), m.group(2)
        if not _CJK.search(content):
            return m.group(0)
        new_attrs, n = re.subn(r'font-family="[^"]*"', f'font-family="{_CJK_FONT}"', attrs)
        if n == 0:
            new_attrs = attrs + f' font-family="{_CJK_FONT}"'
        return f'<text{new_attrs}>{content}</text>'
    return re.sub(r'<text\b([^>]*)>(.*?)</text>', repl, svg, flags=re.S)


# (highlight-key, 标题, 副标题) — 请求生命周期主线
STAGES = [
    ("entrypoints", "入口", "LLM.generate / OpenAI server"),
    ("input-processor", "Stage 1 输入", "tokenize → EngineCoreRequest"),
    ("async-engine", "AsyncLLM 解耦", "三段式 / output_handler"),
    ("engine-core", "IPC 边界", "ZMQ + msgpack 跨进程"),
    ("scheduler", "EngineCore 循环", "schedule → execute → sample"),
    ("output-processor", "Stage 3 输出", "detokenize → RequestOutput"),
    ("stream", "流式返回", "SSE / generate() 产出"),
]


def build(highlight: str) -> str:
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
        f.write(cjk_font(build(a.highlight)))
    print("wrote", a.out)
