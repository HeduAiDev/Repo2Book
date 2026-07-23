#!/usr/bin/env python3
"""fig-m6-eventid-lifetime: event id 分配 = 按 (srcPipe,dstPipe) 分池 + 生命周期
不冲突复用。上半:两条并行 load->vadd 链在 (MTE2,V) 池的生命周期区间图(横轴=
指令序),重叠则分不同 event id。下半:表格核对 (MTE2,V) 与 (V,MTE3) 两个独立
池、以及 widen 例把单池顶到 8 个 event id 的上限。取自 inject-sync.mlir
@test_injcet_sync_two_event_id / @test_widen_sync。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "event id 分配:同池(srcPipe,dstPipe)内生命周期重叠才分不同 id,不重叠可复用"
SUBTITLE = "对照 inject-sync.mlir @test_injcet_sync_two_event_id(两条并行链)+ @test_widen_sync(8 路占满)"

PAD, TOP = 46, 130

# ---- 上半:生命周期区间图 ----
GANTT_ROWS = [
    ("链0 MTE2→V", 0, 2, "EVENT_ID0", "#3b82f6"),
    ("链1 MTE2→V", 1, 3, "EVENT_ID1", "#8b5cf6"),
    ("链0 V→MTE3", 2, 4, "EVENT_ID0", "#3b82f6"),
    ("链1 V→MTE3", 3, 5, "EVENT_ID1", "#8b5cf6"),
]
OP_LABELS = ["load0", "load1", "vadd0", "vadd1", "store0", "store1"]
ROW_LABEL_W, UNIT_W, ROW_H, GANTT_TOP = 160, 92, 34, TOP + 40
gantt_w = UNIT_W * len(OP_LABELS)
gantt_h = ROW_H * len(GANTT_ROWS)

# ---- 下半:表格(池/生命周期/冲突/event id + widen 对照) ----
COLS = ["同步对", "池 (src,dst)", "生命周期 [set,wait]", "同池冲突?", "分得 event id"]
ROWS = [
    ["链0 MTE2→V", "(MTE2, V)", "[0, 2]", "与链1 [1,3] 重叠", "EVENT_ID0"],
    ["链1 MTE2→V", "(MTE2, V)", "[1, 3]", "与链0 [0,2] 重叠", "EVENT_ID1"],
    ["链0 V→MTE3", "(V, MTE3)", "[2, 4]", "异池,互不冲突", "EVENT_ID0(异池)"],
    ["链1 V→MTE3", "(V, MTE3)", "[3, 5]", "与链0(V,MTE3) [2,4] 重叠", "EVENT_ID1"],
]
COL_W = [150, 130, 160, 210, 160]
TABLE_TOP = GANTT_TOP + gantt_h + 90
HEADER_H = 34
ROW_H2 = 40
table_w = sum(COL_W)

w = PAD * 2 + max(ROW_LABEL_W + gantt_w, table_w) + 40
h = TABLE_TOP + HEADER_H + ROW_H2 * len(ROWS) + 110

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="15.5" '
     f'fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 上半标题
L.append(f'<text x="{PAD}" y="{GANTT_TOP-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">生命周期区间(横轴=指令序,同色=同一 event id)</text>')
# op 刻度
for i, name in enumerate(OP_LABELS):
    x = PAD + ROW_LABEL_W + i * UNIT_W
    L.append(f'<text x="{x+UNIT_W/2}" y="{GANTT_TOP-2}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#64748b">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{GANTT_TOP}" x2="{x}" y2="{GANTT_TOP+gantt_h}" '
              'stroke="#e2e8f0" stroke-width="1"/>')
L.append(f'<line x1="{PAD+ROW_LABEL_W+gantt_w}" y1="{GANTT_TOP}" '
          f'x2="{PAD+ROW_LABEL_W+gantt_w}" y2="{GANTT_TOP+gantt_h}" stroke="#e2e8f0"/>')

for i, (label, s, e, evid, color) in enumerate(GANTT_ROWS):
    y = GANTT_TOP + i * ROW_H
    L.append(f'<text x="{PAD+ROW_LABEL_W-12}" y="{y+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="11.5" fill="#0f172a">{esc(label)}</text>')
    bx = PAD + ROW_LABEL_W + s * UNIT_W
    bw = (e - s) * UNIT_W
    L.append(f'<rect x="{bx}" y="{y+6}" width="{bw}" height="{ROW_H-12}" rx="6" '
              f'fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="2"/>')
    L.append(f'<text x="{bx+bw/2}" y="{y+ROW_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="{color}">{esc(evid)}</text>')

# ---- 表格 ----
L.append(f'<text x="{PAD}" y="{TABLE_TOP-16}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">两个独立池对照:(MTE2,V) 与 (V,MTE3) 都能从 EVENT_ID0 起</text>')
col_x = [PAD]
for cw in COL_W[:-1]:
    col_x.append(col_x[-1] + cw)
for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TABLE_TOP}" width="{COL_W[j]-4}" height="{HEADER_H}" '
              'fill="#3b82f6" stroke="#1e3a5f"/>')
    L.append(f'<text x="{x+(COL_W[j]-4)/2}" y="{TABLE_TOP+HEADER_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
              f'fill="white">{esc(name)}</text>')
for i, row in enumerate(ROWS):
    ry = TABLE_TOP + HEADER_H + i * ROW_H2
    pool_color = "#3b82f6" if "MTE2, V" in row[1] else "#8b5cf6"
    for j, cell in enumerate(row):
        x = col_x[j]
        fill = "white" if j != 1 else pool_color
        L.append(f'<rect x="{x}" y="{ry}" width="{COL_W[j]-4}" height="{ROW_H2-4}" '
                  f'fill="{"#f8fafc" if j!=1 else pool_color}" '
                  f'fill-opacity="{1 if j!=1 else 0.18}" stroke="#cbd5e1"/>')
        tf = pool_color if (j == 1 or j == 4) else "#334155"
        L.append(f'<text x="{x+(COL_W[j]-4)/2}" y="{ry+ROW_H2/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{tf}">{esc(cell)}</text>')
L.append('</svg>')

# ---- 底部数字标注 ----
foot_y = TABLE_TOP + HEADER_H + ROW_H2 * len(ROWS) + 34
L[-1:-1] = [
    f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
    f'fill="#0f172a">(MTE2,V) 池最高 event id 编号 = 1(EVENT_ID1,inject-sync.mlir:L246/L249);'
    f'(V,MTE3) 池最低复用编号 = 0(EVENT_ID0,L254/L259)</text>',
    f'<text x="{PAD}" y="{foot_y+18}" font-family="sans-serif" font-size="11" '
    f'fill="#0f172a">单池 event id 硬上限 = 8(kTotalEventIdNum,SyncEventIdAllocation.h:L29);'
    f'widen 例占满时最高编号 = 7(EVENT_ID7,L446-L453/L512-L519)</text>',
]

out = Path(__file__).with_name('fig-m6-eventid-lifetime.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
