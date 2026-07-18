#!/usr/bin/env python3
"""Book map master — renders the Triton-Ascend 书脊（7 Part 结构化下降链）as a
narrow horizontal strip (breadcrumb), highlighting the current chapter's Part.
Reused as each chapter's 开篇「你在这里」横幅。

本书主线命题：同一份 Triton 前端（@triton.jit 里的一行 Python），昇腾后端**不走
GPU 的 SIMT/PTX 路**，而是抛弃 tensor-of-pointers 指针模型，一路
追踪 → ttadapter(Triton-MLIR→Linalg 结构化) → HFusion 融合 → HIVM 达芬奇硬件 IR
→ AscendC 库调用，落到达芬奇 cube/vector 双核。7 个 Part 就是这条结构化下降链的
7 级台阶——排成**单行面包屑**，从左到右（I→VII），中间用 → 箭头连成一条链。当前章
所在 Part 高亮（填色加粗），其余淡置；高亮 chip 下方挂一个细小的「本章深入」小标。
版式为**窄长条**（宽 ≈1400、高 ≈240，宽高比 ≈6:1），只占很少版面。

Usage:
  python3 roadmap.py --highlight watershed-linalg --out roadmap.svg  # Part 键
  python3 roadmap.py --highlight P3               --out roadmap.svg  # Part 别名 P1..P7
  python3 roadmap.py --highlight ch10             --out roadmap.svg  # 章号键 + "本章深入" 小标
  python3 roadmap.py                              --out roadmap.svg  # 空: 全书总览（meta 章）

Part 键见 STAGES；章号键见 ALIASES。错键报错并列出全部可用键。
Coordinates are computed (svg-diagram skill convention); text is escaped.
"""
import argparse
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# (Part 键, 标题, 副标题) — Triton-Ascend 书脊 7 Part 作结构化下降链的 7 级台阶
STAGES = [
    ("orient-hw",         "I 鸟瞰·达芬奇",   "fork 非插件 · cube/vector 双核 + 显式搬运"),
    ("language-cann",     "II 语言层 CANN",  "双 builder · UB/GM 显式搬运 · scope/同步"),
    ("watershed-linalg",  "III 分水岭",      "指针张量 → 结构化 Linalg（triton-shared）"),
    ("dual-core-opt",     "IV 异构双核",     "核亲和定点 · 跨核同步 · UB 多缓冲软流水"),
    ("hivm-hfusion",      "V 硬件 IR HIVM",  "HFusion 融合 → HIVM → AscendC 库调用"),
    ("backend-runtime",   "VI 后端运行时",   "挂载 · bishengir 边界 · 驱动/发射器"),
    ("metrics-practice",  "VII 度量·实战",   "flash-attention CV 融合 · 能力边界"),
]

# 别名 → (Part 键, 中文"本章深入"标签)。P1..P7 简记 + 每个 chapter_id（ch01..ch33）。
# meta 鸟瞰章（ch01）标签留 None，只做纯 Part 高亮。
ALIASES = {
    "P1": ("orient-hw", None), "P2": ("language-cann", None),
    "P3": ("watershed-linalg", None), "P4": ("dual-core-opt", None),
    "P5": ("hivm-hfusion", None), "P6": ("backend-runtime", None),
    "P7": ("metrics-practice", None),
    # ── P1 鸟瞰与达芬奇硬件模型 ──
    "ch01": ("orient-hw", None),                                       # 全书心智模型：纯 Part I 高亮
    "ch02": ("orient-hw", "原理篇：达芬奇 NPU cube/vector 双核 + UB/GM"),
    "ch03": ("orient-hw", "上手第一课：vector-add 的 GPU→NPU 改写"),
    # ── P2 语言层 CANN 扩展 ──
    "ch04": ("language-cann", "双 builder 与 Ascend 内建分发"),
    "ch05": ("language-cann", "显式内存层级：UB/GM/L1/L0C · copy/fixpipe"),
    "ch06": ("language-cann", "昇腾内建算子：索引搬运/向量/cast"),
    "ch07": ("language-cann", "自定义算子框架与 Ascend libdevice"),
    "ch08": ("language-cann", "scope/核间同步/流水线提示"),
    # ── P3 分水岭 Triton→Linalg ──
    "ch09": ("watershed-linalg", "原理篇：MLIR 与 Linalg 结构化 codegen"),
    "ch10": ("watershed-linalg", "分水岭：指针张量 → 结构化张量"),
    "ch11": ("watershed-linalg", "PtrAnalysis：addptr 链→stride/offset"),
    "ch12": ("watershed-linalg", "BlockPtr→memref · load/store→linalg"),
    "ch13": ("watershed-linalg", "MaskAnalysis：mask→extract_slice"),
    "ch14": ("watershed-linalg", "Unstructure 兜底与标量化"),
    # ── P4 昇腾优化 pass：异构双核 ──
    "ch15": ("dual-core-opt", "AutoBlockify：网格实例合并"),
    "ch16": ("dual-core-opt", "Cube 还是 Vector：核亲和定点传播"),
    "ch17": ("dual-core-opt", "Scope 切分与 cube↔vector 同步搬运"),
    "ch18": ("dual-core-opt", "DAGSSBuffer：UB 多缓冲软流水"),
    "ch19": ("dual-core-opt", "离散掩码拆分与交错访存"),
    # ── P5 HFusion/HIVM 硬件 IR 与下降 ──
    "ch20": ("hivm-hfusion", "TritonAscend 方言与三逃生舱"),
    "ch21": ("hivm-hfusion", "HFusion 方言：Linalg 之上的融合 IR"),
    "ch22": ("hivm-hfusion", "FusionKind 与 Cube/Vector 自动调度"),
    "ch23": ("hivm-hfusion", "HIVM 方言：达芬奇硬件 IR"),
    "ch24": ("hivm-hfusion", "HIVM 显式同步：set_flag/wait_flag"),
    "ch25": ("hivm-hfusion", "下降收官：HFusion→HIVM→AscendC 库调用"),
    # ── P6 后端与运行时 ──
    "ch26": ("backend-runtime", "AscendBackend 契约与 hacc.target 注入"),
    "ch27": ("backend-runtime", "三段下降链：add_stages 编排"),
    "ch28": ("backend-runtime", "闭源边界 bishengir-compile"),
    "ch29": ("backend-runtime", "NPU 驱动与二进制装载"),
    "ch30": ("backend-runtime", "动态生成的发射器 · rtKernelLaunch"),
    "ch31": ("backend-runtime", "一套后端两个框架：torch_npu/mindspore"),
    # ── P7 度量与实战 ──
    "ch32": ("metrics-practice", "收官实战：flash-attention CV 融合"),
    "ch33": ("metrics-practice", "能力边界：测试套件揭示的支持谱系"),
}

# ── 横条（面包屑）版式常量 ────────────────────────────────────────────────
MARGIN_X = 34
CHIP_TOP = 96
CHIP_H = 46
CHIP_PAD = 20
CHIP_MIN_W = 84
GAP = 40
LABEL_FS = 15
CALLOUT_GAP = 20
CALLOUT_H = 34
CALLOUT_FS = 12.5
CANVAS_H = 240


def text_width(s, fs):
    """估算文本像素宽（CJK≈fs*0.98，拉丁字母数字≈fs*0.56，其余≈fs*0.4）。"""
    return fs * sum(
        (0.98 if ('⺀' <= c <= '鿿' or '＀' <= c <= '￯') else
         0.56 if (c.isascii() and c.isalnum()) else 0.4)
        for c in s
    )


def build(highlight: str) -> str:
    spine_keys = [k for k, _, _ in STAGES]
    sub_label = None
    hl_key = highlight
    if highlight in ALIASES:
        hl_key, sub_label = ALIASES[highlight]
    elif highlight and highlight not in spine_keys:
        chs = [k for k in ALIASES if k.startswith("ch")]
        ps = [k for k in ALIASES if k.startswith("P") and len(k) == 2]
        raise SystemExit(
            f"未知 --highlight {highlight!r}。\n"
            f"  Part 键: {', '.join(spine_keys)}\n"
            f"  Part 别名: {', '.join(ps)}\n"
            f"  章号键: {', '.join(chs)}\n"
            "（Part 章用 Part 键/别名；按章发车用章号键 chNN；meta 总览用 ''。）"
        )

    widths = [max(CHIP_MIN_W, int(text_width(label, LABEL_FS)) + 2 * CHIP_PAD)
              for _, label, _ in STAGES]
    xpos, x = [], MARGIN_X
    for wi in widths:
        xpos.append(x)
        x += wi + GAP
    w = x - GAP + MARGIN_X
    h = CANVAS_H
    cy_mid = CHIP_TOP + CHIP_H // 2

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
              'markerWidth="7" markerHeight="5" orient="auto">'
              '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="{w // 2}" y="38" text-anchor="middle" font-size="20" '
              f'font-weight="bold" fill="#0f172a">Triton-Ascend 结构化下降链 · 全书地图（7 Part 书脊）</text>')
    subtitle = ("你在这里：高亮处为本章所在 Part（← 同一 Triton 前端一路降级成 AscendC →）" if hl_key
                else "全书总览：同一 Triton 前端如何走结构化下降链落到达芬奇 cube/vector（后续各章逐 Part 放大）")
    L.append(f'<text x="{w // 2}" y="64" text-anchor="middle" font-size="13" '
              f'fill="#64748b">{esc(subtitle)}</text>')

    for i in range(len(STAGES) - 1):
        x1 = xpos[i] + widths[i]
        x2 = xpos[i + 1]
        L.append(f'<line x1="{x1 + 6}" y1="{cy_mid}" x2="{x2 - 4}" y2="{cy_mid}" '
                  f'stroke="#94a3b8" stroke-width="2" marker-end="url(#a)"/>')

    for i, (key, label, _sub) in enumerate(STAGES):
        x0, wi = xpos[i], widths[i]
        on = (key == hl_key)
        fill = "#2563eb" if on else "#f1f5f9"
        stroke = "#1d4ed8" if on else "#cbd5e1"
        tcol = "white" if on else "#475569"
        sw = 2.5 if on else 1.5
        fw = "bold" if on else "normal"
        L.append(f'<rect x="{x0}" y="{CHIP_TOP}" width="{wi}" height="{CHIP_H}" '
                  f'rx="13" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{x0 + wi // 2}" y="{cy_mid + 5}" text-anchor="middle" '
                  f'font-size="{LABEL_FS}" font-weight="{fw}" fill="{tcol}">{esc(label)}</text>')

    if sub_label and hl_key in spine_keys:
        hi = spine_keys.index(hl_key)
        hx, hw = xpos[hi], widths[hi]
        anchor = hx + hw // 2
        text = "本章深入 · " + sub_label
        cw = max(hw, int(text_width(text, CALLOUT_FS)) + 30)
        cx = hx + hw // 2 - cw // 2
        cx = max(8, min(cx, w - cw - 8))
        cy = CHIP_TOP + CHIP_H + CALLOUT_GAP
        lx = max(cx + 16, min(anchor, cx + cw - 16))
        L.append(f'<line x1="{anchor}" y1="{CHIP_TOP + CHIP_H}" x2="{lx}" y2="{cy}" '
                  f'stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="4 3"/>')
        L.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{CALLOUT_H}" '
                  f'rx="{CALLOUT_H // 2}" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.8"/>')
        L.append(f'<text x="{cx + cw // 2}" y="{cy + CALLOUT_H // 2 + 4}" '
                  f'text-anchor="middle" font-size="{CALLOUT_FS}" font-weight="bold" '
                  f'fill="#6d28d9">{esc(text)}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlight", default="")
    ap.add_argument("--out", default="roadmap.svg")
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build(a.highlight))
    print("wrote", a.out)
