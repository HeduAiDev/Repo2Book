#!/usr/bin/env python3
"""fig-m5-transitive-reduction: 已插同步集是依赖 DAG 的传递归约。左=无条件例
(load→vadd→store,load→store 依赖被传递覆盖,虚线,不插直连同步);右=条件例
(vadd 落入 scf.if,覆盖路径可能断,补 1 对 MTE2→MTE3 直连 set_flag/wait_flag)。
取自 inject-sync.mlir @test_mem_injcet_sync_basic / @test_injcet_sync_if。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "已插同步集是依赖 DAG 的传递归约:可达即冗余,断路即补插"
SUBTITLE = "对照 inject-sync.mlir @test_mem_injcet_sync_basic(无条件)vs @test_injcet_sync_if(vadd 落入 scf.if)"

NODE_W, NODE_H, PAD, TOP = 190, 46, 180, 140
PANEL_W = 340
PANEL_GAP = 130
left_cx = PAD + PANEL_W / 2
right_x0 = PAD + PANEL_W + PANEL_GAP
right_cx = right_x0 + PANEL_W / 2

CHAIN_Y = [TOP, TOP + 150, TOP + 300]
NODE_LABELS = ["load(MTE2)\n写 %0", "vadd(V)\n读/写 %0", "store(MTE3)\n读 %0"]

h = TOP + 300 + NODE_H + 130
w = right_x0 + PANEL_W + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

def draw_chain(cx, panel_title, cond_note):
    out = []
    out.append(f'<text x="{cx}" y="{TOP-40}" text-anchor="middle" font-family="sans-serif" '
                f'font-size="14" font-weight="bold" fill="#0f172a">{esc(panel_title)}</text>')
    if cond_note:
        out.append(f'<text x="{cx}" y="{TOP-20}" text-anchor="middle" font-family="sans-serif" '
                    f'font-size="11" fill="#b45309">{esc(cond_note)}</text>')
    for i, (y, label) in enumerate(zip(CHAIN_Y, NODE_LABELS)):
        lines = label.split("\n")
        out.append(f'<rect x="{cx-NODE_W/2}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="8" '
                    'fill="#e2e8f0" stroke="#334155" stroke-width="1.5"/>')
        y0 = y + NODE_H/2 - (len(lines)-1)*8 + 4
        for k, ln in enumerate(lines):
            out.append(f'<text x="{cx}" y="{y0+k*16}" text-anchor="middle" font-family="sans-serif" '
                        f'font-size="12" fill="#0f172a">{esc(ln)}</text>')
    return out

L += draw_chain(left_cx, "无条件例:vadd 恒执行", None)
L += draw_chain(right_cx, "条件例:vadd 落入 scf.if", "分支不取时 vadd 可能不执行")

def edge_arrow(cx, y0, y1, color, mkid, dash=False):
    da = ' stroke-dasharray="6,4"' if dash else ''
    return (f'<line x1="{cx}" y1="{y0}" x2="{cx}" y2="{y1}" stroke="{color}" '
            f'stroke-width="1.8" marker-end="url(#{mkid})"{da}/>')

# 左面板:相邻边实同步(flag),跨节点虚线依赖(被覆盖,不插)
L.append(edge_arrow(left_cx, CHAIN_Y[0]+NODE_H, CHAIN_Y[1], "#334155", "a"))
L.append(f'<text x="{left_cx+14}" y="{(CHAIN_Y[0]+NODE_H+CHAIN_Y[1])/2}" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">set/wait[MTE2,V]</text>')
L.append(edge_arrow(left_cx, CHAIN_Y[1]+NODE_H, CHAIN_Y[2], "#334155", "a"))
L.append(f'<text x="{left_cx+14}" y="{(CHAIN_Y[1]+NODE_H+CHAIN_Y[2])/2}" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">set/wait[V,MTE3]</text>')
curve_x = left_cx - NODE_W/2 - 70
L.append(f'<path d="M {left_cx-NODE_W/2} {CHAIN_Y[0]+NODE_H/2} '
          f'C {curve_x} {CHAIN_Y[0]+NODE_H/2}, {curve_x} {CHAIN_Y[2]+NODE_H/2}, '
          f'{left_cx-NODE_W/2} {CHAIN_Y[2]+NODE_H/2}" fill="none" stroke="#94a3b8" '
          f'stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#g)"/>')
L.append(f'<text x="{curve_x-8}" y="{(CHAIN_Y[0]+CHAIN_Y[2])/2+NODE_H/2}" text-anchor="end" '
          f'font-family="sans-serif" font-size="10.5" fill="#64748b">load→store 依赖</text>')
L.append(f'<text x="{curve_x-8}" y="{(CHAIN_Y[0]+CHAIN_Y[2])/2+NODE_H/2+16}" text-anchor="end" '
          f'font-family="sans-serif" font-size="10.5" fill="#64748b">被传递覆盖,0 对直连 flag</text>')

# 右面板:相邻边同样实同步,额外补一条实线跨节点直连(红色)
L.append(edge_arrow(right_cx, CHAIN_Y[0]+NODE_H, CHAIN_Y[1], "#334155", "a"))
L.append(f'<text x="{right_cx+14}" y="{(CHAIN_Y[0]+NODE_H+CHAIN_Y[1])/2}" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">set/wait[MTE2,V]</text>')
L.append(edge_arrow(right_cx, CHAIN_Y[1]+NODE_H, CHAIN_Y[2], "#334155", "a"))
L.append(f'<text x="{right_cx+14}" y="{(CHAIN_Y[1]+NODE_H+CHAIN_Y[2])/2}" font-family="sans-serif" '
          f'font-size="10.5" fill="#334155">set/wait[V,MTE3]</text>')
curve_x2 = right_cx - NODE_W/2 - 70
L.append(f'<path d="M {right_cx-NODE_W/2} {CHAIN_Y[0]+NODE_H/2} '
          f'C {curve_x2} {CHAIN_Y[0]+NODE_H/2}, {curve_x2} {CHAIN_Y[2]+NODE_H/2}, '
          f'{right_cx-NODE_W/2} {CHAIN_Y[2]+NODE_H/2}" fill="none" stroke="#b91c1c" '
          f'stroke-width="2.2" marker-end="url(#r)"/>')
L.append(f'<text x="{curve_x2-8}" y="{(CHAIN_Y[0]+CHAIN_Y[2])/2+NODE_H/2}" text-anchor="end" '
          f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
          f'fill="#b91c1c">set/wait[MTE2,MTE3]</text>')
L.append(f'<text x="{curve_x2-8}" y="{(CHAIN_Y[0]+CHAIN_Y[2])/2+NODE_H/2+16}" text-anchor="end" '
          f'font-family="sans-serif" font-size="10.5" fill="#b91c1c">覆盖路径可能断,补 1 对直连</text>')

foot_y = h - 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">左:load→store 直连 flag 对数 = 0'
          f'(inject-sync.mlir:L4-L29 无 MTE2→MTE3 项)</text>')
L.append(f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">右:补插的 MTE2→MTE3 直连 flag 对数 = 1'
          f'(inject-sync.mlir:L151 set_flag + L160 wait_flag);'
          f'每对 flag 占 1 个 event id,故此处省下的 event id 消耗 = 1</text>')
L.append('</svg>')

out = Path(__file__).with_name('fig-m5-transitive-reduction.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
