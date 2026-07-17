#!/usr/bin/env python3
"""before-after 模板:同一逻辑列(idx=3)随行号被 phase 换相后落到不同物理列。
数据全部取自 explainer/traces/swizzle_phase.out(vec=8,perPhase=2,maxPhase=4)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

IDX = 3
ROWS = [0, 2, 4, 6]
PHASE = {0: 0, 2: 1, 4: 2, 6: 3}           # (r//perPhase)%maxPhase
LDMX_COL = {r: IDX ^ PHASE[r] for r in ROWS}     # 3,2,1,0
LL_OFFSET = {r: 8 * PHASE[r] for r in ROWS}      # vec*phase = 0,8,16,24
NUM_COLS = 8
ROW_COLORS = {0: "#3b82f6", 2: "#16a34a", 4: "#eab308", 6: "#ef4444"}

CELL, GAP = 40, 4
LABEL_W = 56
PAD, TOP = 44, 132
grid_w = LABEL_W + NUM_COLS * (CELL + GAP) - GAP
row_h = CELL + 10

panel_gap = 90
NOTE_W = 190
w = PAD * 2 + grid_w * 2 + panel_gap + NOTE_W
h = TOP + len(ROWS) * row_h + 40 + 120

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
     f'fill="#0f172a">{esc("同一逻辑列 idx=3,四行被 phase 换相后摊到不同物理列")}</text>',
     f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("参数 (vec,perPhase,maxPhase)=(8,2,4);phase(r)=⌊r/perPhase⌋ mod maxPhase;物理列 = idx XOR phase(r)")}</text>']

panel_x = [PAD, PAD + grid_w + panel_gap]
titles = ["无 swizzle:四行都请求物理列 3", "换相后:物理列 3→2→1→0 摊开"]
subs = ["4 行同一 bank 列,访存排队 4 拍", "phase 不同 → 物理列不同,4 路并发无冲突"]

for pi in range(2):
    px = panel_x[pi]
    L.append(f'<text x="{px}" y="{TOP-46}" font-family="sans-serif" font-size="13" '
              f'font-weight="bold" fill="#0f172a">{esc(titles[pi])}</text>')
    L.append(f'<text x="{px}" y="{TOP-28}" font-family="sans-serif" font-size="11" '
              f'fill="#64748b">{esc(subs[pi])}</text>')
    # 列号表头
    for c in range(NUM_COLS):
        cx = px + LABEL_W + c * (CELL + GAP)
        L.append(f'<text x="{cx+CELL/2}" y="{TOP-6}" text-anchor="middle" '
                  f'font-family="monospace" font-size="11" fill="#94a3b8">{c}</text>')
    for ri, r in enumerate(ROWS):
        ry = TOP + ri * row_h
        L.append(f'<text x="{px+LABEL_W-10}" y="{ry+CELL/2+4}" text-anchor="end" '
                  f'font-family="monospace" font-size="12" fill="#334155">'
                  f'{esc(f"行 {r}")}</text>')
        highlight_col = IDX if pi == 0 else LDMX_COL[r]
        for c in range(NUM_COLS):
            cx = px + LABEL_W + c * (CELL + GAP)
            is_hl = (c == highlight_col)
            fill = ROW_COLORS[r] if is_hl else "#f1f5f9"
            stroke = "#334155" if is_hl else "#e2e8f0"
            L.append(f'<rect x="{cx}" y="{ry}" width="{CELL}" height="{CELL}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="{1.5 if is_hl else 1}"/>')
            if is_hl:
                L.append(f'<text x="{cx+CELL/2}" y="{ry+CELL/2+4}" text-anchor="middle" '
                          f'font-family="monospace" font-size="12" font-weight="bold" '
                          f'fill="white">{c}</text>')
        # 右侧注记:phase / LL 偏移(仅右面板)
        if pi == 1:
            note = f"phase={PHASE[r]}  vec·phase={LL_OFFSET[r]}"
            nx = px + LABEL_W + NUM_COLS * (CELL + GAP) + 8
            L.append(f'<text x="{nx}" y="{ry+CELL/2+4}" font-family="monospace" '
                      f'font-size="11" fill="#475569">{esc(note)}</text>')

# 中间箭头
ax1 = panel_x[0] + grid_w + 12
ax2 = panel_x[1] - 12
amidy = TOP + (len(ROWS) * row_h) / 2 - row_h / 2
L.append(f'<line x1="{ax1}" y1="{amidy}" x2="{ax2}" y2="{amidy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{amidy-10}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#92400e">'
          f'{esc("列 XOR phase(r)")}</text>')

# 底部说明条
foot_y = TOP + len(ROWS) * row_h + 34
foot_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{foot_y}" width="{foot_w}" height="70" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y+26}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("ldmatrix 加载器:物理列 = xor_(idx, phase) —— 右侧格子内数字(SharedToDotOperandMMAv2.cpp:L181)")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+46}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("LinearLayout 列偏移 = vec·phase(r):0,8,16,24(行 0/2/4/6,右侧注记;LinearLayoutConversions.cpp:L369)")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+64}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">'
          f'{esc("相位周期 = perPhase·maxPhase = 8 行")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch34-m2-swizzle-remap.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
