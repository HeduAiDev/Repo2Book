#!/usr/bin/env python3
"""ch09 章内图：层级三路分派 / holdback 时间线 / prefix-read 双窗口。
坐标全部由 Python 计算；text 全部 esc()；中文用普通 sans-serif，rsvg 自动回退。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


HEAD = ('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
        'markerWidth="7" markerHeight="5" orient="auto">'
        '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')


def box(x, y, w, h, fill, stroke, rx=10, sw=2):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=13, anchor="middle", fill="#0f172a", weight="normal",
         family="sans-serif", mono=False):
    fam = "monospace" if mono else family
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" font-family="{fam}" fill="{fill}">{esc(s)}</text>')


# ───────────────────────── 图1：三路分派 + 层级 ─────────────────────────
def diagram_hierarchy():
    w, h = 880, 600
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', HEAD,
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w // 2, 34, "from_new_request 工厂：三路分派", 19, weight="bold"))
    L.append(text(w // 2, 56, "按 tokenizer 是否存在、tokenizers 版本、是否 fast 三条件分派",
                  12, fill="#64748b"))

    # 决策节点
    cx = w // 2
    L.append(box(cx - 130, 78, 260, 46, "#fef9c3", "#ca8a04"))
    L.append(text(cx, 100, "from_new_request(tokenizer, request)", 12.5, mono=True,
                  fill="#713f12", weight="bold"))
    L.append(text(cx, 117, "vllm/v1/engine/detokenizer.py:L48", 10, fill="#a16207"))

    # 三分支
    bx = [70, 330, 590]
    bw = 220
    by = 200
    branches = [
        ("tokenizer is None", "空壳 IncrementalDetokenizer",
         "只累积 token_ids；update→None", "#f1f5f9", "#94a3b8", "#334155"),
        ("USE_FAST_DETOKENIZER\nand PreTrainedTokenizerFast", "FastIncrementalDetokenizer",
         "tokenizers 库 DecodeStream", "#dbeafe", "#2563eb", "#1e3a8a"),
        ("else（慢路径回退）", "SlowIncrementalDetokenizer",
         "Python detokenize_incrementally", "#dcfce7", "#16a34a", "#14532d"),
    ]
    for i, (cond, name, backend, fill, stroke, tc) in enumerate(branches):
        x = bx[i]
        # 条件连线
        sx, sy = cx, 124
        tx, ty = x + bw // 2, by
        L.append(f'<line x1="{sx}" y1="{sy}" x2="{tx}" y2="{ty - 2}" '
                 f'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
        # 条件标签
        for j, line in enumerate(cond.split("\n")):
            L.append(text(tx, 150 + j * 14, line, 10.5, fill="#475569", mono=True))
        # 分支盒
        L.append(box(x, by, bw, 70, fill, stroke, sw=2.5))
        L.append(text(x + bw // 2, by + 28, name, 12.5, mono=True, weight="bold", fill=tc))
        L.append(text(x + bw // 2, by + 50, backend, 10.5, fill="#475569"))

    # 层级（继承）
    hy = 360
    L.append(text(w // 2, hy - 6, "继承层级：公共逻辑只写一次", 14, weight="bold"))
    # IncrementalDetokenizer (root)
    rootx = cx - 150
    L.append(box(rootx, hy + 10, 300, 44, "#f1f5f9", "#94a3b8"))
    L.append(text(cx, hy + 30, "IncrementalDetokenizer", 12.5, mono=True, weight="bold"))
    L.append(text(cx, hy + 47, "层级根 + 空壳实现 + 工厂", 10.5, fill="#475569"))
    # Base
    basey = hy + 90
    L.append(box(rootx, basey, 300, 50, "#fef9c3", "#ca8a04"))
    L.append(text(cx, basey + 20, "BaseIncrementalDetokenizer (ABC)", 12.5, mono=True,
                  weight="bold", fill="#713f12"))
    L.append(text(cx, basey + 38, "update / holdback / min_tokens / check_stop_strings",
                  10, fill="#a16207"))
    L.append(f'<line x1="{cx}" y1="{hy + 54}" x2="{cx}" y2="{basey - 2}" '
             f'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
    # 两子类
    suby = basey + 86
    for x, name, be, fill, stroke, tc in [
        (rootx, "FastIncrementalDetokenizer", "decode_next → DecodeStream.step",
         "#dbeafe", "#2563eb", "#1e3a8a"),
        (rootx + 160, "SlowIncrementalDetokenizer", "decode_next → detokenize_incrementally",
         "#dcfce7", "#16a34a", "#14532d"),
    ]:
        L.append(box(x, suby, 140, 56, fill, stroke, sw=2))
        # 类名分两行
        L.append(text(x + 70, suby + 20, name, 9.6, mono=True, weight="bold", fill=tc))
        L.append(text(x + 70, suby + 38, "↑ 各自实现", 9.6, fill="#475569"))
        L.append(text(x + 70, suby + 52, "decode_next", 9.2, mono=True, fill=tc))
        L.append(f'<line x1="{cx}" y1="{basey + 50}" x2="{x + 70}" y2="{suby - 2}" '
                 f'stroke="#64748b" stroke-width="1.4" marker-end="url(#a)"/>')
    L.append('</svg>')
    return '\n'.join(L)


# ───────────────────────── 图2：holdback 时间线 ─────────────────────────
def diagram_holdback():
    w, h = 900, 520
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', HEAD,
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w // 2, 32, "stop-string holdback 时间线", 19, weight="bold"))
    L.append(text(w // 2, 54, 'stop="END"（len=3 → 扣留 max(len)-1 = 2 个尾字符），流式不吐扣留区',
                  12, fill="#64748b"))

    # 逐步：output_text 累积，尾部 2 字符为扣留区
    steps = [
        ("step k", "好的E", "好", "的E"),
        ("step k+1", "好的EN", "的", "EN"),
        ("step k+2", "好的END", "", "(命中→截断)"),
    ]
    cell = 30
    x0 = 120
    y = 110
    rowh = 92
    for i, (label, full, emitted, held) in enumerate(steps):
        yy = y + i * rowh
        L.append(text(70, yy + 28, label, 12, anchor="start", weight="bold", fill="#334155"))
        chars = list(full)
        # 命中行特殊处理
        hit = (i == 2)
        for j, ch in enumerate(chars):
            cx = x0 + j * cell
            n = len(chars)
            in_held = (not hit) and (j >= n - 2)
            if hit:
                fill, stroke, tc = ("#fecaca", "#dc2626", "#7f1d1d")
            elif in_held:
                fill, stroke, tc = ("#fed7aa", "#ea580c", "#7c2d12")
            else:
                fill, stroke, tc = ("#dcfce7", "#16a34a", "#14532d")
            L.append(box(cx, yy, cell, 38, fill, stroke, rx=5, sw=1.6))
            L.append(text(cx + cell // 2, yy + 25, ch, 15, weight="bold", fill=tc, mono=True))
        # 右侧说明
        ex = x0 + len(chars) * cell + 24
        if hit:
            L.append(text(ex, yy + 16, "check_stop_strings 命中 END", 11.5, anchor="start",
                          fill="#b91c1c", weight="bold"))
            L.append(text(ex, yy + 33, "→ output_text 截到 END 之前（“好的”）", 11,
                          anchor="start", fill="#7f1d1d"))
        else:
            L.append(text(ex, yy + 16, f"本步 delta 吐出：“{emitted}”", 11.5, anchor="start",
                          fill="#166534", weight="bold"))
            L.append(text(ex, yy + 33, f"尾部扣留：“{held}”（末 2 字符，可能是 stop 前缀）", 11,
                          anchor="start", fill="#9a3412"))

    # 图例
    ly = y + 3 * rowh + 18
    leg = [("#dcfce7", "#16a34a", "累计安全区（已可吐）"),
           ("#fed7aa", "#ea580c", "扣留区（末 max(len)-1 字符）"),
           ("#fecaca", "#dc2626", "命中 stop → 截断")]
    lx = x0
    for fill, stroke, lab in leg:
        L.append(box(lx, ly, 22, 18, fill, stroke, rx=4, sw=1.5))
        L.append(text(lx + 30, ly + 14, lab, 11.5, anchor="start", fill="#334155"))
        lx += 230

    # finished 对照
    fy = ly + 50
    L.append(box(x0, fy, 660, 56, "#eef2ff", "#6366f1", sw=2))
    L.append(text(x0 + 16, fy + 23, "finished=True 时：buffer_length=0，扣留区一次性全吐",
                  12.5, anchor="start", weight="bold", fill="#3730a3"))
    L.append(text(x0 + 16, fy + 44,
                  "min_tokens 期间：stop_check_offset 持续推到文末 → 这段时间的 stop 一律不算",
                  11.5, anchor="start", fill="#4338ca"))
    L.append('</svg>')
    return '\n'.join(L)


# ───────────────────── 图3：prefix/read 双窗口 + UTF-8 边界 ─────────────────────
def diagram_window():
    w, h = 940, 560
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', HEAD,
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w // 2, 32, "增量去 token 化的 prefix / read 双窗口", 19, weight="bold"))
    L.append(text(w // 2, 54,
                  "new_text = decode(tokens[prefix:])，prefix_text = decode(tokens[prefix:read])，相减得增量",
                  11.5, fill="#64748b"))

    # token 横条
    toks = ["…", "wor", "ld", "!", " 你", "<半>", "<半>"]
    cell = 92
    x0 = 70
    y = 110
    prefix_idx = 1   # prefix_offset 落在 index 1
    read_idx = 4     # read_offset 落在 index 4（即已读到这里）
    for j, t in enumerate(toks):
        cx = x0 + j * cell
        # 着色：[prefix:read) 上下文窗 = 浅黄；[read:] 待读 = 浅蓝
        if j < prefix_idx:
            fill, stroke = "#f1f5f9", "#cbd5e1"
        elif j < read_idx:
            fill, stroke = "#fef9c3", "#ca8a04"
        else:
            fill, stroke = "#dbeafe", "#2563eb"
        L.append(box(cx, y, cell - 8, 44, fill, stroke, rx=6, sw=1.8))
        L.append(text(cx + (cell - 8) // 2, y + 28, t, 13, mono=True, weight="bold"))
        L.append(text(cx + (cell - 8) // 2, y + 60, f"idx {j}", 10, fill="#94a3b8"))

    # prefix / read 竖线
    px = x0 + prefix_idx * cell - 4
    rx = x0 + read_idx * cell - 4
    for xx, lab, col in [(px, "prefix_offset", "#ca8a04"), (rx, "read_offset", "#2563eb")]:
        L.append(f'<line x1="{xx}" y1="{y - 14}" x2="{xx}" y2="{y + 70}" '
                 f'stroke="{col}" stroke-width="2.4" stroke-dasharray="5 3"/>')
        L.append(text(xx, y - 20, lab, 11.5, mono=True, weight="bold", fill=col))

    # 两次 decode 说明
    dy = y + 110
    L.append(box(x0, dy, 400, 70, "#fef9c3", "#ca8a04", sw=2))
    L.append(text(x0 + 16, dy + 24, "prefix_text = decode(tokens[prefix:read])", 11.5,
                  anchor="start", mono=True, weight="bold", fill="#713f12"))
    L.append(text(x0 + 16, dy + 46, "= “world” —— 仅作上下文锚，", 11.5, anchor="start", fill="#a16207"))
    L.append(text(x0 + 16, dy + 62, "  让空格 cleanup 有相邻 token 参考", 11, anchor="start", fill="#a16207"))

    L.append(box(x0 + 430, dy, 420, 70, "#dbeafe", "#2563eb", sw=2))
    L.append(text(x0 + 446, dy + 24, "new_text = decode(tokens[prefix:])", 11.5,
                  anchor="start", mono=True, weight="bold", fill="#1e3a8a"))
    L.append(text(x0 + 446, dy + 46, "= “world! 你” —— 增量 = new_text[len(prefix_text):]", 11.5,
                  anchor="start", fill="#1d4ed8"))
    L.append(text(x0 + 446, dy + 62, "= “! 你”（截掉锚段后的真正新文字）", 11, anchor="start", fill="#1d4ed8"))

    # UTF-8 边界
    uy = dy + 110
    L.append(text(w // 2, uy, "UTF-8 多字节边界：半截字节先扣住，等下一 token 补全", 14, weight="bold"))
    rows = [
        ("收到 <半>（一个多字节字符的前半段）", 'new_text 末尾 = "\\ufffd"（U+FFFD 替换字符）',
         "→ 吐空串、offset 不动，等下一 token", "#fed7aa", "#ea580c", "#7c2d12"),
        ("收到第二个 <半>（补全字节序列）", 'new_text = "你"（完整字符，不再以 \\ufffd 结尾）',
         "→ 吐出 “你”，read_offset 前移", "#dcfce7", "#16a34a", "#14532d"),
    ]
    for i, (a, b, c, fill, stroke, tc) in enumerate(rows):
        yy = uy + 18 + i * 64
        L.append(box(x0, yy, 780, 54, fill, stroke, sw=1.8))
        L.append(text(x0 + 16, yy + 21, a, 12, anchor="start", weight="bold", fill=tc))
        L.append(text(x0 + 16, yy + 41, b + "    " + c, 11, anchor="start", fill=tc, mono=False))
    L.append('</svg>')
    return '\n'.join(L)


for name, fn in [("01-detok-hierarchy", diagram_hierarchy),
                 ("02-stop-holdback", diagram_holdback),
                 ("03-offset-window", diagram_window)]:
    svg = fn()
    with open(f"/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch09-detokenization/diagrams/{name}.svg", "w") as f:
        f.write(svg)
    print("wrote", name)
