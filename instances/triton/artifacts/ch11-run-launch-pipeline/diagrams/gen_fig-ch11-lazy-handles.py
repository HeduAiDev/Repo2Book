#!/usr/bin/env python3
"""before-after 模板:CompiledKernel 惰性设备句柄。左=编译后(本书 headless 实测,
module/function=None);中间触发器 __getattribute__ 拦截 name=='run';
右=首次发射后(需真设备,module/function 被填实)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

LEFT_TITLE = "编译后（本书 headless 实测）"
LEFT_STEPS = [
    ("CompiledKernel", "asm 5 段 + metadata", False),
    ("module = None", "", True),
    ("function = None", "", True),
    ("run 未初始化", "无需 GPU 也能走到这里", False),
]
RIGHT_TITLE = "首次发射后（需真设备）"
RIGHT_STEPS = [
    ("driver.active.launcher_cls", "造 C++ launcher 挂到 self.run", False),
    ("load_binary(cubin)", "装进 GPU，查 max_shared_mem", False),
    ("module = <已装载>", "", True),
    ("function = <已装载>", "", True),
]

BOX_W, BOX_H, VGAP, PANEL_W, PAD, TOP = 300, 50, 20, 360, 44, 110
n = len(LEFT_STEPS)
w = PAD * 2 + PANEL_W * 2 + 160
h = TOP + n * (BOX_H + VGAP) + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
          '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
          'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{PAD}" y="38" font-family="sans-serif" font-size="17" font-weight="bold" '
          f'fill="#0f172a">{esc("惰性设备句柄：首次读 .run 才把 cubin 真正装上 GPU")}</text>')


def panel(px, title, steps, base_fill, base_stroke, hot_fill, hot_stroke):
    out = []
    cx = px + PANEL_W / 2
    out.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    for i, (label, detail, hot) in enumerate(steps):
        y = TOP + i * (BOX_H + VGAP)
        fill, stroke = (hot_fill, hot_stroke) if hot else (base_fill, base_stroke)
        out.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
                    f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hot else 1.4}"/>')
        ty = y + BOX_H/2 - (7 if detail else 0) + 4
        out.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="sans-serif" '
                    f'font-size="12.5" font-weight="bold" fill="{"#7f1d1d" if hot else "#0f172a"}">'
                    f'{esc(label)}</text>')
        if detail:
            out.append(f'<text x="{cx}" y="{ty+17}" text-anchor="middle" font-family="sans-serif" '
                        f'font-size="10.5" fill="#334155">{esc(detail)}</text>')
        if i < len(steps) - 1:
            out.append(f'<line x1="{cx}" y1="{y+BOX_H}" x2="{cx}" y2="{y+BOX_H+VGAP-4}" '
                        'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    return out


px_left = PAD
px_right = PAD + PANEL_W + 160
L += panel(px_left, LEFT_TITLE, LEFT_STEPS, "#e2e8f0", "#64748b", "#fee2e2", "#b91c1c")
L += panel(px_right, RIGHT_TITLE, RIGHT_STEPS, "#dbeafe", "#1d4ed8", "#dcfce7", "#15803d")

# 中间触发器
mid_y = TOP + (n * (BOX_H + VGAP) - VGAP) / 2
tx0 = px_left + PANEL_W
tx1 = px_right
L.append(f'<line x1="{tx0+8}" y1="{mid_y}" x2="{tx1-8}" y2="{mid_y}" '
          'stroke="#7c3aed" stroke-width="2.4" marker-end="url(#a)"/>')
L.append(f'<rect x="{(tx0+tx1)/2-70}" y="{mid_y-46}" width="140" height="34" rx="6" '
          'fill="#ede9fe" stroke="#7c3aed"/>')
L.append(f'<text x="{(tx0+tx1)/2}" y="{mid_y-24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#5b21b6">{esc("读 kernel.run")}</text>')
L.append(f'<text x="{(tx0+tx1)/2}" y="{mid_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#5b21b6">{esc("__getattribute__")}</text>')
L.append(f'<text x="{(tx0+tx1)/2}" y="{mid_y+36}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#5b21b6">{esc("拦截 -> _init_handles()")}</text>')

foot_y = h - 66
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("数字核对：module_after_compile=None，function_after_compile=None（本书 headless 实测）；")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="12" fill="#334155">'
          f'{esc("metadata.shared=0 字节——编译期即定，与是否装载设备无关。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+42}" font-family="sans-serif" font-size="11" fill="#64748b">'
          f'{esc("本书在 host 上只能停在左态；右态需真 GPU，如实标注未执行。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch11-lazy-handles.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
