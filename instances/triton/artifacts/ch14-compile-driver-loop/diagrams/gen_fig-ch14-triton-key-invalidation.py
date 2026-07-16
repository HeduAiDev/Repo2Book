#!/usr/bin/env python3
"""before-after 模板:改编译器 1 个文件 1 行 -> triton_key 指纹翻转 -> 全部 kernel 磁盘缓存失效。
双面板,仅"改动"步骤高亮。改造点:PANELS(标题,步骤,高亮下标)。
步骤项可以是字符串(单行)或字符串列表(同一个框内多行,居中对齐)——用于放不下单行的长句,
避免把一句话拆成两个独立的框(读者会误以为文字被截断)。框宽/框高按 cjk_text_width() 派生,
零硬编码魔数。两栏步骤数可以不同(基线枚举合并成 1 个多行框后行数天然少于改动栏),
各自独立按自身行数向下排列,不强行对齐行号。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):全角字符按 1.0x size,半角按 0.58x size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

def as_lines(step):
    return step if isinstance(step, (list, tuple)) else [step]

PANELS = [
    ("基线:未改动", [
        ["枚举 19 项(frontend 1+compiler 5", "+backends 3+language 9+so 1)"],
        "逐项 sha256,'-' 拼接",
        "得基线指纹 fingerprint_before",
    ], None),
    ("改动后:1 个文件改 1 行", [
        "同样枚举 19 项",
        "其中 1 项(code_generator.py)hash 变",
        "拼接串随之整串变",
        "fingerprint_after != fingerprint_before",
    ], 1),
]
STEP_FONT = 12.5
BOX_PAD_X = 48  # 框内左右留白合计
# BOX_W 由全部步骤文字里最宽的一行派生(cjk_text_width),不硬编码——保证每行都在框内留白之内
max_line_w = max(cjk_text_width(line, STEP_FONT)
                 for _, steps, _ in PANELS for step in steps for line in as_lines(step))
BOX_W = max_line_w + BOX_PAD_X
BOX_H, VGAP, PAD, TOP = 50, 20, 40, 118
PANEL_W = BOX_W + 40
LH = 18  # 多行框行距

def box_height(step):
    n = len(as_lines(step))
    return BOX_H if n <= 1 else BOX_H + LH * (n - 1)

# 校验:每个框内每行文字宽度不超过框宽留白
MAX_LINE_W = BOX_W - BOX_PAD_X
for _, steps, _ in PANELS:
    for step in steps:
        for line in as_lines(step):
            lw = cjk_text_width(line, STEP_FONT)
            assert lw <= MAX_LINE_W, f"line too wide for box: {line!r} ({lw:.0f} > {MAX_LINE_W})"

panel_heights = [sum(box_height(s) for s in steps) + VGAP * (len(steps) - 1) for _, steps, _ in PANELS]
n_rows_max = max(len(steps) for _, steps, _ in PANELS)
w = PAD * 2 + PANEL_W * 2 + 100
h = TOP + max(panel_heights) + PAD + 90

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="36" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("triton_key 内容寻址:改编译器源码任一行,指纹翻转,全部 kernel 磁盘缓存失效")}</text>']

hot_row_center = None  # 高亮框(改动步骤)垂直中心,用于定位两栏之间的横向指示箭头
panel_end_y = []  # 每栏最后一个框的下边缘,用于定位脚注
for p, (title, steps, hot) in enumerate(PANELS):
    px = PAD + p * (PANEL_W + 100)
    cx = px + PANEL_W / 2
    is_hot_panel = hot is not None
    title_fill = "#b45309" if is_hot_panel else "#1e40af"
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="{title_fill}">{esc(title)}</text>')
    y = TOP
    for i, step in enumerate(steps):
        lines = as_lines(step)
        bh = box_height(step)
        hl = (i == hot)
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{bh}" rx="8" '
                  f'fill="{"#fef3c7" if hl else "#e2e8f0"}" '
                  f'stroke="{"#d97706" if hl else "#64748b"}" stroke-width="{2 if hl else 1}"/>')
        n = len(lines)
        text_fill = "#92400e" if hl else "#0f172a"
        top_line_y = y + bh / 2 - (n - 1) * LH / 2 + 5
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx}" y="{top_line_y + k * LH}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="{STEP_FONT}" '
                      f'fill="{text_fill}">{esc(line)}</text>')
        if hl:
            hot_row_center = y + bh / 2
        if i < len(steps) - 1:
            L.append(f'<line x1="{cx}" y1="{y+bh}" x2="{cx}" y2="{y+bh+VGAP-4}" '
                      'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
        y += bh + VGAP
    panel_end_y.append(y - VGAP)

midy = hot_row_center if hot_row_center is not None else TOP + max(panel_heights) / 2
L.append(f'<line x1="{PAD+PANEL_W+10}" y1="{midy}" x2="{PAD+PANEL_W+90}" y2="{midy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+50}" y="{midy-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#d97706">{esc("编译器源码变 1 行")}</text>')

foot_y = max(panel_end_y) + 46
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#1e293b" font-weight="bold">'
          f'{esc("结论:triton_key 按编译器身份内容寻址,不看版本号——改 1/19 项即触发全部 kernel 磁盘缓存集体 miss、必重编。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">'
          f'{esc("与按实参特化的内存 launch 缓存键(另一章)正交——那把键管的是同一 kernel 不同实参,这把键管的是编译器本身变没变。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch14-triton-key-invalidation.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
