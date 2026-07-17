#!/usr/bin/env python3
"""before-after 模板:xor swizzle 消 bank 冲突(bank-bucket 版)。

2026-07-17 修订(评审 #1):旧版把 0..31 摆在同一张固定网格里只换色,数字位置
两个面板完全不动,和正文 §2『xor 按行重排』的手算例对不上。改版按
explainer/traces/swizzle_trace.json 的 bank_conflict 数据,把两个面板都画成
『8 个 bank 桶,每个地址实际落进哪一桶』的占用图——
左面板(无 swizzle,maxPhase=1):32 个地址全挤进 bank 0 这一个桶,其余 7 桶空;
右面板(xor swizzle,maxPhase=8):32 个地址按 r mod 8 均分到 8 个桶,每桶 4 个。
同一组地址,在两个面板里**真的换了桶**(除了本来就是 8 的倍数的几个),
『打散』第一次变成读者能直接看见的位置变化,而不是纯换色。
数字与桶分配逐一等于 traces/swizzle_trace.json 的 bank_conflict.no_swizzle_maxPhase1
/ swizzle_maxPhase8,零手算、零杜撰。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

NUM_BANKS = 8
BANK_COLORS = ["#ef4444", "#f97316", "#eab308", "#16a34a",
               "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899"]

# ---- 与 explainer/traces/swizzle_trace.json 的 bank_conflict 完全一致 ----
# no_swizzle_maxPhase1: distinct_banks=1, way_conflict=32, banks={"0": 0..31}
NO_SWIZZLE_BANKS = {0: list(range(32))}
for b in range(1, NUM_BANKS):
    NO_SWIZZLE_BANKS[b] = []
# swizzle_maxPhase8: distinct_banks=8, way_conflict=4, banks={b: [b,b+8,b+16,b+24]}
SWIZZLE_BANKS = {b: [addr for addr in range(32) if addr % NUM_BANKS == b]
                 for b in range(NUM_BANKS)}
assert SWIZZLE_BANKS[0] == [0, 8, 16, 24]
assert SWIZZLE_BANKS[7] == [7, 15, 23, 31]

# ---- 桶内小格子几何 ----
MCHIP_W, MCHIP_H, MGAP = 18, 13, 2          # 左桶 0 内 4x8 打包小格
RCHIP_W, RCHIP_H, RGAP = 34, 20, 4          # 右侧每桶 2x2 打包小格
LEFT_COLS = 4
left_grid_w = LEFT_COLS * MCHIP_W + (LEFT_COLS - 1) * MGAP
left_grid_h = 8 * MCHIP_H + 7 * MGAP
right_grid_w = 2 * RCHIP_W + RGAP
right_grid_h = 2 * RCHIP_H + RGAP
EMPTY_W, EMPTY_H = 46, 22                    # 空桶占位框

BUCKET_SLOT_W = max(left_grid_w, right_grid_w) + 10
BUCKET_GAP = 10
GRID_W = NUM_BANKS * BUCKET_SLOT_W + (NUM_BANKS - 1) * BUCKET_GAP
MAX_CONTENT_H = max(left_grid_h, right_grid_h)

LABEL_H = 22
LABEL_GAP = 8
CONTENT_GAP_BOTTOM = 16
BADGE_H = 46
FOOT_H = 76
PANEL_GAP = 130
PAD, TOP = 40, 96

content_top = TOP + LABEL_H + LABEL_GAP
badge_y = content_top + MAX_CONTENT_H + CONTENT_GAP_BOTTOM

w = PAD * 2 + GRID_W * 2 + PANEL_GAP
h = badge_y + BADGE_H + 16 + FOOT_H + PAD

panel_x = [PAD, PAD + GRID_W + PANEL_GAP]

SUBTITLE = ("共享内存 = 32 个 bank(numBanks=32,TritonGPUAttrDefs.td:L288);"
            "本图只画这次访问实际落到的 8 个桶(其余 24 个未涉及);"
            "同一 bank 的并发访问被硬件拆成多拍串行(回指第 2 章)")

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{28}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">'
     f'{esc("同一个 warp 反复读逻辑列 0:32 次访问落进哪些 bank")}</text>',
     f'<text x="{PAD}" y="{48}" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc(SUBTITLE)}</text>']

PANELS = [
    (0, "无 swizzle(maxPhase=1)", "32 次访问全部落 bank 0,其余 7 个桶空",
     NO_SWIZZLE_BANKS, "distinct banks = 1 · 32-way 冲突"),
    (1, "xor swizzle(maxPhase=8)", "32 次访问按 r mod 8 均分到 8 个桶,每桶 4 次",
     SWIZZLE_BANKS, "distinct banks = 8 · 4-way 冲突"),
]

for idx, title, caption, banks, badge in PANELS:
    px = panel_x[idx]
    L.append(f'<text x="{px + GRID_W/2}" y="{TOP - 34}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#0f172a">{esc(title)}</text>')
    L.append(f'<text x="{px + GRID_W/2}" y="{TOP - 16}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#64748b">'
              f'{esc(caption)}</text>')

    for b in range(NUM_BANKS):
        bx = px + b * (BUCKET_SLOT_W + BUCKET_GAP)
        slot_cx = bx + BUCKET_SLOT_W / 2
        # bank 标签 chip
        label_w = 30
        L.append(f'<rect x="{slot_cx - label_w/2}" y="{TOP}" width="{label_w}" '
                  f'height="{LABEL_H}" rx="5" fill="{BANK_COLORS[b]}"/>')
        L.append(f'<text x="{slot_cx}" y="{TOP + LABEL_H/2 + 4}" text-anchor="middle" '
                  f'font-family="monospace" font-size="11" font-weight="bold" '
                  f'fill="white">B{b}</text>')

        members = banks[b]
        if not members:
            # 空桶占位
            ey = content_top + (MAX_CONTENT_H - EMPTY_H) / 2
            L.append(f'<rect x="{slot_cx - EMPTY_W/2}" y="{ey}" width="{EMPTY_W}" '
                      f'height="{EMPTY_H}" rx="4" fill="none" stroke="#cbd5e1" '
                      f'stroke-width="1.5" stroke-dasharray="3,3"/>')
            L.append(f'<text x="{slot_cx}" y="{ey + EMPTY_H/2 + 4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" fill="#94a3b8">0</text>')
            continue

        if len(members) > 4:
            # 左面板 bank0:32 个,打包成 4 列 x 8 行
            gx0 = slot_cx - left_grid_w / 2
            gy0 = content_top  # 顶对齐,正好占满 MAX_CONTENT_H
            for i, addr in enumerate(members):
                col, row = i % LEFT_COLS, i // LEFT_COLS
                cx = gx0 + col * (MCHIP_W + MGAP)
                cy = gy0 + row * (MCHIP_H + MGAP)
                L.append(f'<rect x="{cx}" y="{cy}" width="{MCHIP_W}" height="{MCHIP_H}" '
                          f'rx="2" fill="{BANK_COLORS[b]}" stroke="white" stroke-width="0.6"/>')
                L.append(f'<text x="{cx + MCHIP_W/2}" y="{cy + MCHIP_H/2 + 3}" '
                          f'text-anchor="middle" font-family="monospace" font-size="8" '
                          f'font-weight="bold" fill="white">{addr}</text>')
        else:
            # 右面板:每桶 4 个,打包成 2x2
            gx0 = slot_cx - right_grid_w / 2
            gy0 = content_top + (MAX_CONTENT_H - right_grid_h) / 2
            for i, addr in enumerate(members):
                col, row = i % 2, i // 2
                cx = gx0 + col * (RCHIP_W + RGAP)
                cy = gy0 + row * (RCHIP_H + RGAP)
                L.append(f'<rect x="{cx}" y="{cy}" width="{RCHIP_W}" height="{RCHIP_H}" '
                          f'rx="3" fill="{BANK_COLORS[b]}" stroke="white" stroke-width="0.8"/>')
                L.append(f'<text x="{cx + RCHIP_W/2}" y="{cy + RCHIP_H/2 + 4}" '
                          f'text-anchor="middle" font-family="monospace" font-size="11" '
                          f'font-weight="bold" fill="white">{addr}</text>')

    L.append(f'<rect x="{px}" y="{badge_y}" width="{GRID_W}" height="{BADGE_H}" rx="8" '
              f'fill="{"#dbeafe" if idx == 0 else "#dcfce7"}" '
              f'stroke="{"#1d4ed8" if idx == 0 else "#166534"}" stroke-width="1.5"/>')
    L.append(f'<text x="{px + GRID_W/2}" y="{badge_y + BADGE_H/2 + 5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{"#1e3a5f" if idx == 0 else "#14532d"}">{esc(badge)}</text>')

# 中间箭头:同一批地址,换了桶
ax1 = panel_x[0] + GRID_W + 14
ax2 = panel_x[1] - 14
amidy = content_top + MAX_CONTENT_H / 2
L.append(f'<line x1="{ax1}" y1="{amidy}" x2="{ax2}" y2="{amidy}" '
          'stroke="#d97706" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{amidy - 12}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#92400e">'
          f'{esc("xor 按行号")}</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{amidy + 20}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#92400e">'
          f'{esc("换桶打散")}</text>')

# 底部旁注:.td 例 1 的逐位 xor(换一组更小参数演示同一条规则,避免和上方 maxPhase=8 混)
fy0 = badge_y + BADGE_H + 16
foot_box_h = FOOT_H - 10
L.append(f'<rect x="{PAD}" y="{fy0}" width="{GRID_W*2 + PANEL_GAP}" height="{foot_box_h}" '
          f'rx="6" fill="#f8fafc" stroke="#cbd5e1"/>')
tag_w = 132
L.append(f'<rect x="{PAD + 14}" y="{fy0 + 10}" width="{tag_w}" height="18" rx="4" '
          f'fill="#e2e8f0" stroke="#94a3b8"/>')
L.append(f'<text x="{PAD + 14 + tag_w/2}" y="{fy0 + 10 + 13}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#475569">'
          f'{esc("换一组更小参数,同一条规则")}</text>')
foot = ("例 1,vec=1,perPhase=1,maxPhase=4(与上方 maxPhase=8 是同一条 xor 公式的不同取值):"
        "第 1 行 phase=1,逻辑列 c=[0,1,2,3] 经 c⊕1 变成物理列 [1,0,3,2]"
        "(TritonGPUAttrDefs.td 例 1)")
L.append(f'<text x="{PAD + 14}" y="{fy0 + 46}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc(foot)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-xor-before-after.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
