#!/usr/bin/env python3
"""Book roadmap master — renders the vLLM v1 request-lifecycle spine as an SVG,
highlighting the current chapter's stage. Reused as each chapter's Roadmap ("你在这里").

Usage:
  python3 roadmap.py --highlight async-engine --out roadmap.svg     # 主线章: 高亮一个生命周期框
  python3 roadmap.py --highlight kv-cache     --out roadmap.svg     # off-spine 章: 高亮所属主线框 + "深入:<子系统>"标注

主线键见 STAGES；off-spine 子系统键见 SUBSYS（映射到它所属的主线阶段 + 中文深入标签）。
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

# off-spine 子系统 → (所属主线阶段键, 中文深入标签)
# off-spine 章(ch15+)的 highlight 用这里的键：高亮其所属主线框 + 在下方画一个
# 「深入：<标签>」标注框，让"你在这里"既保留全局定位、又点明本章钻入的子系统。
SUBSYS = {
    "config-and-wiring": ("entrypoints", "配置与装配"),
    "kv-cache": ("engine-core", "分页 KV 缓存"),
    "worker-and-executor": ("engine-core", "Worker 与执行器"),
    "model-runner": ("engine-core", "ModelRunner 执行"),
    "distributed-parallelism": ("engine-core", "分布式并行 TP/PP/EP"),
    "model-definitions": ("engine-core", "模型定义层"),
    "custom-ops-and-compilation": ("engine-core", "自定义算子与编译"),
    "attention": ("engine-core", "注意力后端"),
    "model-architecture": ("engine-core", "模型架构"),
    "sampling": ("engine-core", "采样"),
    "spec-decode": ("engine-core", "投机解码"),
    "pd-disaggregation": ("ipc", "P/D 分离"),
}


def build(highlight: str) -> str:
    spine_keys = [k for k, _, _ in STAGES]
    sub_label = None
    hl_key = highlight
    if highlight in SUBSYS:
        hl_key, sub_label = SUBSYS[highlight]
    elif highlight and highlight not in spine_keys:
        raise SystemExit(
            f"未知 --highlight {highlight!r}。\n"
            f"  主线键: {', '.join(spine_keys)}\n"
            f"  子系统键: {', '.join(SUBSYS)}\n"
            "（主线章用主线键；off-spine 章用子系统键；ch01/03 等无定位的用 ''。）"
        )

    bw, bh, gap, x0, y0 = 168, 72, 38, 30, 92
    w = x0 * 2 + len(STAGES) * bw + (len(STAGES) - 1) * gap
    h = 270 if sub_label else 200
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
             'markerWidth="7" markerHeight="5" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="{w // 2}" y="38" text-anchor="middle" font-size="20" '
             f'font-weight="bold" fill="#0f172a">vLLM v1 请求生命周期 · 全书地图</text>')
    # 无 highlight（ch01/03 等 meta 导读章）：图是全局鸟瞰，不承诺"你在这里"高亮，
    # 否则副标题承诺高亮、视觉却无高亮，自相矛盾。有 highlight 才用定位副标题。
    subtitle = ("你在这里（高亮处为本章所在阶段）" if hl_key
                else "全程总览（后续各章逐站放大）")
    L.append(f'<text x="{w // 2}" y="62" text-anchor="middle" font-size="13" '
             f'fill="#64748b">{subtitle}</text>')
    hl_x = None
    for i, (key, label, sub) in enumerate(STAGES):
        x = x0 + i * (bw + gap)
        on = (key == hl_key)
        if on:
            hl_x = x
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

    # off-spine: 在高亮主线框下方画「深入：<子系统>」标注框 + 连线
    if sub_label and hl_x is not None:
        cy = y0 + bh + 36          # callout 顶
        ch = 50
        cx = hl_x - 6              # 略宽于主线框
        cw = bw + 12
        L.append(f'<line x1="{hl_x + bw // 2}" y1="{y0 + bh}" '
                 f'x2="{hl_x + bw // 2}" y2="{cy}" stroke="#7c3aed" '
                 f'stroke-width="2" stroke-dasharray="4 3" marker-end="url(#a)"/>')
        L.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="9" '
                 f'fill="#f5f3ff" stroke="#7c3aed" stroke-width="2.5"/>')
        L.append(f'<text x="{cx + cw // 2}" y="{cy + 21}" text-anchor="middle" '
                 f'font-size="11" fill="#7c3aed">本章深入</text>')
        L.append(f'<text x="{cx + cw // 2}" y="{cy + 40}" text-anchor="middle" '
                 f'font-size="14.5" font-weight="bold" fill="#6d28d9">{esc(sub_label)}</text>')

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
