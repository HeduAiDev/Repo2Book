#!/usr/bin/env python3
"""before-after 模板：outline-scope pass 把 scope.scope 从函数体内提成独立 func.func + call。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PAD, TOP = 40, 118
PANEL_W = 470
GAP = 150
w = PAD * 2 + PANEL_W * 2 + GAP
h = 560

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("outline-scope pass：scope.scope 只是临时容器")}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12.5" fill="#64748b">'
     f'{esc("核类型最终落在函数属性上，不是语句属性——原地换成一次 call")}</text>']

# ---- Left panel: BEFORE (ttir) ----
lx = PAD
L.append(f'<text x="{lx+PANEL_W/2}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">'
          f'{esc("Before —— ttir 阶段")}</text>')
outer_h = 360
L.append(f'<rect x="{lx}" y="{TOP}" width="{PANEL_W}" height="{outer_h}" rx="10" '
          'fill="#f8fafc" stroke="#334155" stroke-width="1.6" stroke-dasharray="6,4"/>')
L.append(f'<text x="{lx+18}" y="{TOP+26}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">{esc("func.func @test_scope")}</text>')

inner_boxes = [
    ("scope.scope { tcore_type<CUBE> }", "#dbeafe", "#1d4ed8"),
    ("scope.scope { tcore_type<VECTOR> }", "#ede9fe", "#6d28d9"),
]
ib_w, ib_h, ib_gap = PANEL_W - 60, 70, 30
iy = TOP + 46
for text, fill, stroke in inner_boxes:
    L.append(f'<rect x="{lx+30}" y="{iy}" width="{ib_w}" height="{ib_h}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{lx+30+ib_w/2}" y="{iy+ib_h/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="{stroke}" '
              f'font-weight="bold">{esc(text)}</text>')
    iy += ib_h + ib_gap

L.append(f'<text x="{lx+30}" y="{TOP+outer_h-50}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">{esc("SizedRegion<1>、NoRegionArguments，")}</text>')
L.append(f'<text x="{lx+30}" y="{TOP+outer_h-36}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">{esc("SingleBlockImplicitTerminator<scope::ReturnOp>")}</text>')
L.append(f'<text x="{lx+30}" y="{TOP+outer_h-18}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">'
          f'{esc("ScopeOp 声明属性仅 1 个（UnitAttr no_inline）；")}</text>')
L.append(f'<text x="{lx+30}" y="{TOP+outer_h-4}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">'
          f'{esc("tcore_type 等以 discardable attr 挂上")}</text>')

# ---- Right panel: AFTER (post outline-scope) ----
rx0 = PAD + PANEL_W + GAP
L.append(f'<text x="{rx0+PANEL_W/2}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#0f172a">'
          f'{esc("After —— ttadapter 阶段（outline-scope pass 之后）")}</text>')

caller_h = 150
L.append(f'<rect x="{rx0}" y="{TOP}" width="{PANEL_W}" height="{caller_h}" rx="10" '
          'fill="#f8fafc" stroke="#334155" stroke-width="1.6" stroke-dasharray="6,4"/>')
L.append(f'<text x="{rx0+18}" y="{TOP+26}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">{esc("func.func @test_scope")}</text>')
call_texts = ["call @test_scope_0", "call @test_scope_1"]
cy = TOP + 44
for t in call_texts:
    L.append(f'<rect x="{rx0+30}" y="{cy}" width="{PANEL_W-60}" height="40" rx="7" '
              'fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>')
    L.append(f'<text x="{rx0+PANEL_W/2}" y="{cy+25}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="#78350f" '
              f'font-weight="bold">{esc(t)}</text>')
    cy += 40 + 16

outlined_y = TOP + caller_h + 34
outlined = [
    ("func.func @test_scope_0", "attributes {tcore_type<CUBE>}", "#dbeafe", "#1d4ed8"),
    ("func.func @test_scope_1", "attributes {tcore_type<VECTOR>}", "#ede9fe", "#6d28d9"),
]
oy = outlined_y
for name, attr, fill, stroke in outlined:
    L.append(f'<rect x="{rx0}" y="{oy}" width="{PANEL_W}" height="66" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="{rx0+PANEL_W/2}" y="{oy+27}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="{stroke}" '
              f'font-weight="bold">{esc(name)}</text>')
    L.append(f'<text x="{rx0+PANEL_W/2}" y="{oy+47}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="{stroke}">'
              f'{esc(attr)}</text>')
    oy += 66 + 16

# central transform arrow (kept clear below the call-box dashed border)
mid_y = TOP + caller_h + 42
ax1 = lx + PANEL_W + 8
ax2 = rx0 - 8
L.append(f'<line x1="{ax1}" y1="{mid_y}" x2="{ax2}" y2="{mid_y}" '
          'stroke="#d97706" stroke-width="2.6" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{mid_y-14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#b45309">'
          f'{esc("outline-scope pass")}</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{mid_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#b45309">'
          f'{esc("（反向：InlineScope）")}</text>')

foot_y = h - 26
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">'
          f'{esc("外提出 2 个函数（@test_scope_0/@test_scope_1）：tcore_type 从 op 属性搬到 func.func 的 attributes，原地留下 call。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m5-outline-scope.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
