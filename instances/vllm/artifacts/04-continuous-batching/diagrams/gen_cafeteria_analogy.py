"""
Generate SVG: 排队打饭类比 — Static Batching vs Continuous Batching
"""
import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 900, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>')
L.append('<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>')
L.append('<filter id="sh"><feDropShadow dx="2" dy="2" stdDeviation="2" flood-opacity="0.15"/></filter>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="#fafaf9"/>')

# ── Title ──
L.append(f'<text x="{w//2}" y="36" text-anchor="middle" font-size="20" font-weight="bold" fill="#1e293b">{esc("排队打饭类比：为什么 Continuous Batching 远胜 Static Batching？")}</text>')

# ── Divider ──
mx = 450
L.append(f'<line x1="{mx}" y1="55" x2="{mx}" y2="{h-15}" stroke="#cbd5e1" stroke-width="2" stroke-dasharray="6,4"/>')

# ═══════════════════════════════════════════════════
# LEFT: Static Batching (传统食堂)
# ═══════════════════════════════════════════════════
left_x = 30
left_w = 390
L.append(f'<text x="{left_x + left_w//2}" y="72" text-anchor="middle" font-size="16" font-weight="bold" fill="#b91c1c">{esc("传统食堂 (Static Batching)")}</text>')

# Phase labels y positions
phases = [
    ("米饭区", 95, "#ef4444"),
    ("菜区",   230, "#f59e0b"),
    ("汤区",   365, "#3b82f6"),
]
person_names = ["r1", "r2", "r3", "r4"]
person_sizes = [800, 200, 128, 64]  # prompt lengths
max_size = max(person_sizes)

for phase_name, phase_y, phase_color in phases:
    # Phase label box
    L.append(f'<rect x="{left_x + 5}" y="{phase_y}" width="60" height="24" rx="4" fill="{phase_color}" opacity="0.15"/>')
    L.append(f'<text x="{left_x + 35}" y="{phase_y + 16}" text-anchor="middle" font-size="11" font-weight="bold" fill="{phase_color}">{esc(phase_name)}</text>')

    # Queue illustration — horizontal bars representing waiting time
    bar_y = phase_y + 55
    for i, (name, sz) in enumerate(zip(person_names, person_sizes)):
        py = bar_y + i * 32
        # Person label
        L.append(f'<text x="{left_x + 8}" y="{py + 14}" font-size="10" fill="#64748b" text-anchor="start">{esc(name)}</text>')
        # Work bar (blue = actual work)
        bar_w = int(60 + sz * 150 / max_size)
        L.append(f'<rect x="{left_x + 30}" y="{py}" width="{bar_w}" height="20" rx="3" fill="#3b82f6" opacity="0.8"/>')
        L.append(f'<text x="{left_x + 34}" y="{py + 14}" font-size="9" fill="white" font-weight="bold">{esc(str(sz))}</text>')
        # Red idle bubble after work
        idle_w = int(320 - bar_w)
        if idle_w > 5:
            L.append(f'<rect x="{left_x + 30 + bar_w}" y="{py}" width="{idle_w}" height="20" rx="3" fill="#ef4444" opacity="0.2"/>')
            if i > 0:
                L.append(f'<text x="{left_x + 30 + bar_w + idle_w//2}" y="{py + 14}" text-anchor="middle" font-size="8" fill="#ef4444">{esc("空闲等待")}</text>')

    # Arrow between phases
    if phase_name != "汤区":
        arrow_x = left_x + 200
        L.append(f'<line x1="{arrow_x}" y1="{bar_y + len(person_names)*32}" x2="{arrow_x}" y2="{bar_y + len(person_names)*32 + 40}" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ar)"/>')
        L.append(f'<text x="{arrow_x + 10}" y="{bar_y + len(person_names)*32 + 20}" font-size="9" fill="#94a3b8">{esc("所有人都完成才能进入下一区")}</text>')

# Bubble zone annotation
L.append(f'<rect x="{left_x + 30}" y="bar_y + len(person_names)*32 + 45" width="330" height="30" rx="6" fill="#ef4444" opacity="0.12"/>')
L.append(f'<text x="{left_x + 195}" y="bar_y + len(person_names)*32 + 64" text-anchor="middle" font-size="10" fill="#b91c1c">{esc("BUBBLE = GPU 大量空闲，利用率不到 50%")}</text>')

# Static stats
s_y = 540
L.append(f'<rect x="{left_x + 10}" y="{s_y}" width="370" height="65" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1"/>')
L.append(f'<text x="{left_x + 195}" y="{s_y + 24}" text-anchor="middle" font-size="12" font-weight="bold" fill="#991b1b">{esc("Static Batching 效率")}</text>')
L.append(f'<text x="{left_x + 195}" y="{s_y + 44}" text-anchor="middle" font-size="11" fill="#b91c1c">{esc("步数 = max(米饭) + max(汤) = 800 + 100 = 900")}</text>')
L.append(f'<text x="{left_x + 195}" y="{s_y + 60}" text-anchor="middle" font-size="11" fill="#b91c1c">{esc("GPU 利用率 ≈ 45%  (一半以上空闲！)")}</text>')

# ═══════════════════════════════════════════════════
# RIGHT: Continuous Batching (自助餐)
# ═══════════════════════════════════════════════════
right_x = 480
right_w = 390
L.append(f'<text x="{right_x + right_w//2}" y="72" text-anchor="middle" font-size="16" font-weight="bold" fill="#16a34a">{esc("自助餐 (Continuous Batching)")}</text>')

# Single buffet line
buffet_x = right_x + 40
buffet_w = 310
buffet_y = 100

# Buffet table
L.append(f'<rect x="{buffet_x}" y="{buffet_y}" width="{buffet_w}" height="40" rx="6" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5"/>')
L.append(f'<text x="{buffet_x + buffet_w//2}" y="{buffet_y + 16}" text-anchor="middle" font-size="11" font-weight="bold" fill="#92400e">{esc("自助餐台  (菜品种类不限，随便拿)")}</text>')
L.append(f'<text x="{buffet_x + buffet_w//2}" y="{buffet_y + 33}" text-anchor="middle" font-size="9" fill="#a16207">{esc("米饭 | 炒菜 | 汤 | 水果  —  不分区，混着装")}</text>')

# Tray metaphor
tray_requests = [
    ("r1", 800, [(0, 512, "#3b82f6"), (512, 800, "#22c55e")], "先装 512 个 → 下次来"),
    ("r2", 200, [(0, 200, "#22c55e")], "200 个一次装完 ✓"),
    ("r3", 50,  [(0, 24, "#f59e0b")], "装 24 个 (餐盘满了)"),
]

tray_y = 165
for pi, (name, total, segments, note) in enumerate(tray_requests):
    ty = tray_y + pi * 75
    # Person label
    L.append(f'<text x="{right_x + 5}" y="{ty + 18}" font-size="10" fill="#64748b">{esc(name)}</text>')
    # Tray (餐盘)
    tray_w = 280
    L.append(f'<rect x="{right_x + 35}" y="{ty}" width="{tray_w}" height="28" rx="14" fill="white" stroke="#94a3b8" stroke-width="2"/>')
    # Segments on tray
    cx = right_x + 38
    for seg_start, seg_end, seg_color in segments:
        seg_w = int((seg_end - seg_start) * tray_w / 800)
        L.append(f'<rect x="{cx}" y="{ty + 4}" width="{seg_w}" height="20" rx="4" fill="{seg_color}" opacity="0.7"/>')
        if seg_end - seg_start >= 50:
            L.append(f'<text x="{cx + seg_w//2}" y="{ty + 18}" text-anchor="middle" font-size="8" fill="white" font-weight="bold">{esc(str(seg_end - seg_start))}</text>')
        cx += seg_w

    # Remaining empty space on tray
    if cx < right_x + 35 + tray_w - 5:
        empty_w = right_x + 35 + tray_w - cx
        L.append(f'<rect x="{cx}" y="{ty + 4}" width="{empty_w}" height="20" rx="4" fill="#e2e8f0"/>')

    # Note
    L.append(f'<text x="{right_x + 40}" y="{ty + 46}" font-size="9" fill="#64748b">{esc(note)}</text>')
    # Checkmark or return arrow
    if "✓" in note:
        L.append(f'<text x="{right_x + 355}" y="{ty + 16}" font-size="14" fill="#16a34a" font-weight="bold">{esc("✓")}</text>')

# Token budget indicator
L.append(f'<rect x="{right_x + 10}" y="400" width="370" height="55" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="1"/>')
L.append(f'<text x="{right_x + 195}" y="422" text-anchor="middle" font-size="11" font-weight="bold" fill="#166534">{esc("规则：每个餐盘容量 = Token Budget B = 512")}</text>')
L.append(f'<text x="{right_x + 195}" y="442" text-anchor="middle" font-size="10" fill="#16a34a">{esc("想装多少装多少，餐盘满了就结账 → 吃完再来装")}</text>')

# Step illustration
step_y = 475
L.append(f'<text x="{right_x + 15}" y="{step_y}" font-size="10" font-weight="bold" fill="#166534">{esc("第一步 (Step 1):")}</text>')
L.append(f'<text x="{right_x + 25}" y="{step_y + 16}" font-size="10" fill="#334155">{esc("r1 装 512/800  (餐盘满 → 下次继续)")}</text>')
L.append(f'<text x="{right_x + 25}" y="{step_y + 32}" font-size="10" fill="#334155">{esc("餐盘剩余 0 →  第一步结束")}</text>')
step2_y = step_y + 52
L.append(f'<text x="{right_x + 15}" y="{step2_y}" font-size="10" font-weight="bold" fill="#166534">{esc("第二步 (Step 2):")}</text>')
L.append(f'<text x="{right_x + 25}" y="{step2_y + 16}" font-size="10" fill="#334155">{esc("r1 装 288/800  (288+200+24=512 → 三人都吃上)")}</text>')
L.append(f'<text x="{right_x + 25}" y="{step2_y + 32}" font-size="10" fill="#334155">{esc("r2 装 200/200 ✓   r3 装 24/50  (将在下一步装完)")}</text>')

# CB stats
cb_y = 540
L.append(f'<rect x="{right_x + 10}" y="{cb_y}" width="370" height="65" rx="8" fill="#f0fdf4" stroke="#86efac" stroke-width="1"/>')
L.append(f'<text x="{right_x + 195}" y="{cb_y + 24}" text-anchor="middle" font-size="12" font-weight="bold" fill="#166534">{esc("Continuous Batching 效率")}</text>')
L.append(f'<text x="{right_x + 195}" y="{cb_y + 44}" text-anchor="middle" font-size="11" fill="#16a34a">{esc("步数 ≈ ceil(总食物/餐盘) = ceil(1220/512) ≈ 3")}</text>')
L.append(f'<text x="{right_x + 195}" y="{cb_y + 60}" text-anchor="middle" font-size="11" fill="#16a34a">{esc("GPU 利用率 ≈ 100%  (餐盘几乎不可能空着)")}</text>')

# VS text
L.append(f'<text x="{w//2}" y="590" text-anchor="middle" font-size="14" font-weight="bold" fill="#6b7280">{esc("传统食堂 900 步 → 自助餐 3 步 → 快了 300 倍！")}</text>')

L.append('</svg>')
svg = '\n'.join(L)

import os
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(out_dir, 'cafeteria_analogy.svg')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(svg)
print(f'Wrote {out_path}')
