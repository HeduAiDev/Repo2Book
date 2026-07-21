#!/usr/bin/env python3
"""swimlane 模板（并行泳道，非跨道消息）：sync_block_set 的两代下降路径。
两条独立生命线（旧代 aux_ops / 新代 core.py），每条各自往下走过 4 个阶段，
底部一条跨两道的共享结论条。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LANES = ["旧代：aux_ops.sync_block_set", "新代：core.sync_block_set"]
LANE_W = 620
PAD, TOP, STEP = 50, 108, 118
w = PAD * 2 + LANE_W + 140
h = TOP + STEP * 4 + 210

X = {LANES[0]: PAD + LANE_W / 2 - 40, LANES[1]: PAD + LANE_W + 140 - LANE_W / 2 + 40}
# reposition: put two lanes with generous separation
LX = PAD + 150
RX = w - PAD - 150

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="32" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("同名 API、两条下降路径：custom_op 通用外挂 vs HIVM 专用 op")}</text>',
     f'<text x="{PAD}" y="54" font-family="sans-serif" font-size="12.5" fill="#64748b">'
     f'{esc("两代都在 ast_to_ttir 阶段由 builder 直接 emit——不是一边 ttir 一边 ttadapter")}</text>']

# lane headers + lifelines
for name, x, color in [(LANES[0], LX, "#b45309"), (LANES[1], RX, "#1d4ed8")]:
    L.append(f'<rect x="{x-190}" y="{TOP-46}" width="380" height="34" rx="7" '
              f'fill="{"#fef3c7" if x==LX else "#dbeafe"}" stroke="{color}" stroke-width="1.5"/>')
    L.append(f'<text x="{x}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13.5" font-weight="bold" fill="{color}">{esc(name)}</text>')

def stage(x, y, lines, fill, stroke, w_box=380):
    h_box = 18 + 17 * len(lines)
    L.append(f'<rect x="{x-w_box/2}" y="{y}" width="{w_box}" height="{h_box}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    y0 = y + 20
    for i, (line, small) in enumerate(lines):
        fs = 10.5 if small else 12
        fw = '' if small else 'font-weight="bold" '
        L.append(f'<text x="{x}" y="{y0+i*17}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{fs}" {fw}fill="{stroke}">{esc(line)}</text>')
    return h_box

# Row 1: entry signature
y = TOP
h1 = stage(LX, y, [("aux_ops.sync_block_set(sender, receiver, event_id)", False)],
           "#fffbeb", "#b45309")
h1b = stage(RX, y, [("core.sync_block_set(sender, receiver, event_id,", False),
                    ("sender_pipe=None, receiver_pipe=None)", False),
                    ("比旧代多 2 个参数：sender_pipe / receiver_pipe", True)],
            "#eff6ff", "#1d4ed8")
row1_bot = y + max(h1, h1b)
L.append(f'<line x1="{LX}" y1="{y+h1}" x2="{LX}" y2="{row1_bot+18}" stroke="#94a3b8" marker-end="url(#a)"/>')
L.append(f'<line x1="{RX}" y1="{y+h1b}" x2="{RX}" y2="{row1_bot+18}" stroke="#94a3b8" marker-end="url(#a)"/>')

# Row 2: deprecation warning (old only); new lane shows "无此步"
y2 = row1_bot + 30
h2 = stage(LX, y2, [("DeprecationWarning：", False),
                    ("“Use al.sync_block_set instead”", True)], "#fef2f2", "#b91c1c")
h2b = stage(RX, y2, [("（无此步——新代无弃用告警）", True)], "#f8fafc", "#94a3b8")
row2_bot = y2 + max(h2, h2b)
L.append(f'<line x1="{LX}" y1="{y2+h2}" x2="{LX}" y2="{row2_bot+18}" stroke="#94a3b8" marker-end="url(#a)"/>')
L.append(f'<line x1="{RX}" y1="{y2+h2b}" x2="{RX}" y2="{row2_bot+18}" stroke="#94a3b8" marker-end="url(#a)"/>')

# Row 3: what reaches builder / semantic
y3 = row2_bot + 30
h3 = stage(LX, y3, [("落到 builder 的实参：", False),
                    ("('sync_block_set', 'cube', 3)", True),
                    ("—— receiver 被丢弃", True)], "#fffbeb", "#b45309")
h3b = stage(RX, y3, [("create_sync_block 校验后", False),
                     ("四条校验：核名/同核/范围/pipe 类型", True)], "#eff6ff", "#1d4ed8")
row3_bot = y3 + max(h3, h3b)
L.append(f'<line x1="{LX}" y1="{y3+h3}" x2="{LX}" y2="{row3_bot+18}" stroke="#94a3b8" marker-end="url(#a)"/>')
L.append(f'<line x1="{RX}" y1="{y3+h3b}" x2="{RX}" y2="{row3_bot+18}" stroke="#94a3b8" marker-end="url(#a)"/>')

# Row 4: final IR shape
y4 = row3_bot + 30
h4 = stage(LX, y4, [("ascend.custom", False),
                    ('op_name = "sync_block_set"', True),
                    ("str_args = [StringAttr(sender), I32IntegerAttr(id)]", True)],
           "#fef2f2", "#b91c1c")
h4b = stage(RX, y4, [("hivm.sync_block_set", False),
                     ("(coreAttr, prodPipe, consPipe, idI64)", True)], "#ecfdf5", "#047857")
row4_bot = y4 + max(h4, h4b)

# shared bottom bar
sb_y = row4_bot + 34
sb_h = 56
L.append(f'<rect x="{PAD}" y="{sb_y}" width="{w-2*PAD}" height="{sb_h}" rx="9" '
          'fill="#f1f5f9" stroke="#475569" stroke-width="1.5"/>')
L.append(f'<text x="{w/2}" y="{sb_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#334155">'
          f'{esc("两代共同的拒绝：sender == receiver（同核对）→ ValueError")}</text>')
L.append(f'<text x="{w/2}" y="{sb_y+42}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#64748b">'
          f'{esc("旧代先 warn 后抛；新代直接抛——判定逻辑一致")}</text>')
for x in (LX, RX):
    L.append(f'<line x1="{x}" y1="{row4_bot+2}" x2="{x}" y2="{sb_y}" '
              'stroke="#94a3b8" stroke-dasharray="3,3" marker-end="url(#a)"/>')

foot_y = sb_y + sb_h + 30
foot_lines = [
    "旧代经 _utils.py 里那个只认三个 op 名的手写分发函数（与自定义算子注册表同名不同物）",
    "落成通用外挂 op，receiver 与 pipe 信息在语言层就丢了；",
    "新代直接建 HIVM 专用 op，落核与两侧 pipe 一并写进 IR——同一个 API 名，下降链上却是两种东西。",
]
for i, line in enumerate(foot_lines):
    L.append(f'<text x="{PAD}" y="{foot_y+i*20}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(line)}</text>')
h = foot_y + len(foot_lines) * 20
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
L[2] = f'<rect width="{w}" height="{h}" fill="white"/>'
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m6-two-generations.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
