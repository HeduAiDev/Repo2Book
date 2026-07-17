#!/usr/bin/env python3
"""fig-dot-dispatch (flow 模板,决策树)
tt.dot 降级派单:只读结果 tensor 的 NvidiaMma 布局 versionMajor,不看 GPU 型号。
根 -> 两条分支标签(NvidiaMma / 非 mma)-> 5 个叶子(Volta/Turing/Ampere/Hopper/FMA兜底)。
守卫条件另起一条注记框,side 注记挂在 Ampere 叶子旁。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

LEAVES = [
    ("isVolta", "versionMajor = 1", "convertMMA884", "mma.884", "#dbeafe", "#2563eb"),
    ("isTuring", "versionMajor = 2", "convertMMA1688", "mma.1688", "#dbeafe", "#2563eb"),
    ("isAmpere", "versionMajor = 3", "convertMMA16816", "mma.16816", "#dbeafe", "#2563eb"),
    ("isHopper", "WarpGroupDotOp", "convertWGMMA", "wgmma", "#dbeafe", "#2563eb"),
    ("非 mma", "BlockedEncodingAttr", "convertFMADot", "逐元素 FMA 兜底", "#fee2e2", "#b91c1c"),
]

LEAF_W, LEAF_H, GAP = 190, 92, 22
PAD, TOP = 40, 40
ROOT_H = 56
BRANCH_H = 40
BRANCH_Y_GAP = 46
LEAF_Y_GAP = 40

n = len(LEAVES)
leaves_w = n * LEAF_W + (n - 1) * GAP
w = PAD * 2 + leaves_w
root_y = TOP + 30
split_y = root_y + ROOT_H + 34
branch_y = split_y + 30
leaf_y = branch_y + BRANCH_H + LEAF_Y_GAP
h = leaf_y + LEAF_H + 26 + 34 + 20 + 58 + 48

leaf_x = [PAD + i * (LEAF_W + GAP) for i in range(n)]
leaf_cx = [x + LEAF_W / 2 for x in leaf_x]

# 分组:前 4 个(NvidiaMma 分支) / 第 5 个(非 mma 分支)
group_a = leaf_cx[:4]
group_b = leaf_cx[4:]
branch_a_cx = sum(group_a) / len(group_a)
branch_b_cx = sum(group_b) / len(group_b)
BRANCH_W_A, BRANCH_W_B = 300, 190

root_cx = w / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="24" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("tt.dot 降级派单:只读结果布局的版本,不看 GPU 型号")}</text>']

# 根节点
ROOT_W = 420
root_x = root_cx - ROOT_W / 2
L.append(f'<rect x="{root_x}" y="{root_y}" width="{ROOT_W}" height="{ROOT_H}" rx="10" '
          'fill="#e2e8f0" stroke="#334155" stroke-width="1.5"/>')
L.append(f'<text x="{root_cx}" y="{root_y+22}" text-anchor="middle" font-family="monospace" '
          f'font-size="13" font-weight="bold" fill="#0f172a">{esc("结果 tensor 的 encoding")}</text>')
L.append(f'<text x="{root_cx}" y="{root_y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#475569">{esc("(降级期已由上游 pass 烙进 layout,不查算力)")}</text>')

# 根 -> 分裂点(T 字)
root_bottom = root_y + ROOT_H
L.append(f'<line x1="{root_cx}" y1="{root_bottom}" x2="{root_cx}" y2="{split_y}" '
          'stroke="#334155" stroke-width="1.5"/>')
L.append(f'<line x1="{branch_a_cx}" y1="{split_y}" x2="{branch_b_cx}" y2="{split_y}" '
          'stroke="#334155" stroke-width="1.5"/>')

# 两个分支标签
BRANCHES = [
    (branch_a_cx, BRANCH_W_A, "NvidiaMma(versionMajor = v)", "#dbeafe", "#2563eb"),
    (branch_b_cx, BRANCH_W_B, "非 mma(如 Blocked)", "#fee2e2", "#b91c1c"),
]
for bcx, bw, label, fill, stroke in BRANCHES:
    bx = bcx - bw / 2
    L.append(f'<line x1="{bcx}" y1="{split_y}" x2="{bcx}" y2="{branch_y}" '
              f'stroke="{stroke}" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<rect x="{bx}" y="{branch_y}" width="{bw}" height="{BRANCH_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{bcx}" y="{branch_y+BRANCH_H/2+5}" text-anchor="middle" '
              f'font-family="monospace" font-size="12.5" font-weight="bold" '
              f'fill="{stroke}">{esc(label)}</text>')

# 分支 -> 叶子 fan-out
branch_bottom = branch_y + BRANCH_H
for i, cx in enumerate(leaf_cx):
    src_cx = branch_a_cx if i < 4 else branch_b_cx
    stroke = "#2563eb" if i < 4 else "#b91c1c"
    L.append(f'<line x1="{src_cx}" y1="{branch_bottom}" x2="{cx}" y2="{leaf_y}" '
              f'stroke="{stroke}" stroke-width="1.3" marker-end="url(#a)"/>')

# 叶子框
for i, (guard, ver, fn, ptx, fill, stroke) in enumerate(LEAVES):
    x = leaf_x[i]
    L.append(f'<rect x="{x}" y="{leaf_y}" width="{LEAF_W}" height="{LEAF_H}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    cx = leaf_cx[i]
    L.append(f'<text x="{cx}" y="{leaf_y+22}" text-anchor="middle" font-family="monospace" '
              f'font-size="12.5" font-weight="bold" fill="{stroke}">{esc(guard)}</text>')
    L.append(f'<text x="{cx}" y="{leaf_y+40}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#334155">{esc(ver)}</text>')
    L.append(f'<text x="{cx}" y="{leaf_y+58}" text-anchor="middle" font-family="monospace" '
              f'font-size="11" fill="#334155">{esc(fn)}</text>')
    L.append(f'<rect x="{x+14}" y="{leaf_y+68}" width="{LEAF_W-28}" height="18" rx="4" '
              f'fill="white" stroke="{stroke}" stroke-width="1"/>')
    L.append(f'<text x="{cx}" y="{leaf_y+81}" text-anchor="middle" font-family="monospace" '
              f'font-size="11.5" font-weight="bold" fill="{stroke}">{esc(ptx)}</text>')

# 算力换算注记(独立满宽条,不与叶子/箭头相撞)
note_y = leaf_y + LEAF_H + 26
note_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{note_y}" width="{note_w}" height="34" rx="7" '
          'fill="#fef9c3" stroke="#a16207" stroke-width="1.2"/>')
L.append(f'<text x="{PAD+16}" y="{note_y+22}" font-family="sans-serif" font-size="12" '
          f'fill="#713f12">{esc("算力换算(isAmpere → mma.16816 一条):")} '
          f'<tspan font-family="monospace" font-weight="bold">m16n8k16 = 2048 次乘加 / 一条 mma.sync</tspan></text>')

# 守卫条件框(底部,回指根节点)
guard_y = note_y + 34 + 20
guard_w = w - PAD * 2
L.append(f'<rect x="{PAD}" y="{guard_y}" width="{guard_w}" height="58" rx="8" '
          'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.3"/>')
L.append(f'<text x="{PAD+16}" y="{guard_y+22}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#0f172a">{esc("守卫条件(走到 mma 叶子前必须成立):")}</text>')
L.append(f'<text x="{PAD+16}" y="{guard_y+42}" font-family="monospace" font-size="12" '
          f'fill="#334155">{esc("!isOuter (K != 1)  且  supportMMA(versionMajor)  —— 否则 report_fatal_error")}</text>')

L.append(f'<text x="{PAD}" y="{h-16}" font-family="sans-serif" font-size="12" fill="#64748b">'
          f'{esc("决策点在结果布局的版本,不是芯片型号:版本已被上游 pass 依算力烙进 encoding,降级期照 encoding 派单。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-dot-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  {w}x{h}")
