#!/usr/bin/env python3
"""fig-m7-six-stage-pipeline: AutoInjectSync 六阶段流水线,竖排主链 + 早退侧支。
取自 InjectSync.cpp:L62-L107。全坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "核内同步 pass AutoInjectSync:六阶段流水线,同一份 SyncIR 依次加工"
SUBTITLE = "InjectSync.cpp:L62-L107;开头一道 size≤1 早退闸"

STAGES = [
    ("① IRTranslator", "裸 IR → 线性 SyncIR:每个 hivm op 变 1 个 CompoundInstanceElement", "trans.Build()"),
    ("② SyncAnalyzer.Plan", "顺序扫 SyncIR,依赖判据+已同步剪枝,决定插 barrier / flag", "Plan()"),
    ("③ MoveSyncState.StateOptimize", "把循环不变的同步点外提到循环前/后", "StateOptimize()"),
    ("④ RemoveRedundantSync.Plan", "删被传递覆盖的冗余同步", "Plan()"),
    ("⑤ SyncEventIdAllocation.Allocate", "按 (srcPipe,dstPipe) 分池,生命周期不冲突复用 event id", "Allocate()"),
    ("⑥ SyncCodegen.Build", "把抽象同步落成真的 set_flag/wait_flag/pipe_barrier op", "Build()"),
]

BOX_W, BOX_H, GAP, PAD, TOP = 560, 56, 30, 46, 130
main_cx = PAD + BOX_W / 2

# 早退侧支
SIDE_W = 260
side_x = main_cx + BOX_W / 2 + 80
side_cx = side_x + SIDE_W / 2

n = len(STAGES)
last_stage_bottom_pre = TOP + (n - 1) * (BOX_H + GAP) + BOX_H
h = last_stage_bottom_pre + GAP + 10 + 28 + 60
w = side_x + SIDE_W + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="26" font-family="sans-serif" font-size="15.5" '
     f'fill="#1e40af" font-weight="bold">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="48" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 入口
entry_y = TOP - 46
L.append(f'<rect x="{main_cx-90}" y="{entry_y}" width="180" height="28" rx="14" '
          'fill="#dcfce7" stroke="#22c55e" stroke-width="1.5"/>')
L.append(f'<text x="{main_cx}" y="{entry_y+19}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#166534">syncIR 构建完成</text>')
L.append(f'<line x1="{main_cx}" y1="{entry_y+28}" x2="{main_cx}" y2="{TOP-4}" '
          'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')

last_stage_bottom = TOP
for i, (name, detail, loc) in enumerate(STAGES):
    y = TOP + i * (BOX_H + GAP)
    L.append(f'<rect x="{PAD}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>')
    L.append(f'<text x="{main_cx}" y="{y+21}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#1e3a8a">{esc(name)}</text>')
    L.append(f'<text x="{main_cx}" y="{y+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="11" fill="#334155">{esc(detail)}</text>')
    L.append(f'<text x="{PAD+BOX_W-10}" y="{y+BOX_H-8}" text-anchor="end" '
              f'font-family="sans-serif" font-size="9.5" fill="#94a3b8">{esc(loc)}</text>')
    if i < len(STAGES) - 1:
        L.append(f'<line x1="{main_cx}" y1="{y+BOX_H}" x2="{main_cx}" y2="{y+BOX_H+GAP-4}" '
                  'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    else:
        last_stage_bottom = y + BOX_H

# 出口(留出足够垂直箭头间距,避免与前一框/文字贴靠)
exit_y = last_stage_bottom + GAP + 10
L.append(f'<line x1="{main_cx}" y1="{last_stage_bottom}" x2="{main_cx}" y2="{exit_y-4}" '
          'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{main_cx-110}" y="{exit_y}" width="220" height="28" rx="14" '
          'fill="#ffedd5" stroke="#f97316" stroke-width="1.5"/>')
L.append(f'<text x="{main_cx}" y="{exit_y+19}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#9a3412">同步 op 已插入 IR</text>')

# 早退侧支:从入口右侧分岔
branch_y = entry_y + 14
L.append(f'<line x1="{main_cx+90}" y1="{branch_y}" x2="{side_cx}" y2="{branch_y}" '
          'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#g)"/>')
side_box_y = branch_y + 30
L.append(f'<rect x="{side_x}" y="{side_box_y}" width="{SIDE_W}" height="70" rx="8" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4"/>')
L.append(f'<text x="{side_cx}" y="{side_box_y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#475569">早退:直接 return</text>')
L.append(f'<text x="{side_cx}" y="{side_box_y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">syncIR.size() ≤ 1</text>')
L.append(f'<text x="{side_cx}" y="{side_box_y+56}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">(单条指令,没人可等)</text>')

foot_y = h - 20
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#0f172a">六个阶段(InjectSync.cpp:L62-L107);早退阈值 syncIR.size() ≤ 1(L74);'
          f'IRTranslator 每个 op → 1 个 CompoundInstanceElement</text>')
L.append('</svg>')

out = Path(__file__).with_name('fig-m7-six-stage-pipeline.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out}')
