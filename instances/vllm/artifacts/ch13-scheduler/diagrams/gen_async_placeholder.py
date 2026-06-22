#!/usr/bin/env python3
"""AsyncScheduler 占位机制：schedule(N) 与 forward(N-1) 重叠 vs 同步必须等。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


w, h = 900, 500
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w//2}" y="32" text-anchor="middle" font-size="21" '
         'font-weight="bold" fill="#0f172a">同步 Scheduler vs 异步 AsyncScheduler：占位让调度/执行重叠</text>')

# time axis
t0, tw = 120, 680
ncols = 4
cw = tw / ncols
ticks = ["拍 0", "拍 1", "拍 2", "拍 3"]
for i, tk in enumerate(ticks):
    tx = t0 + i * cw
    L.append(f'<line x1="{tx}" y1="64" x2="{tx}" y2="470" stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{tx+cw/2:.0f}" y="80" text-anchor="middle" font-size="13" '
             f'fill="#64748b">{esc(tk)}</text>')

# --- Sync lane ---
sy = 110
L.append(f'<text x="20" y="{sy+30}" font-size="14" font-weight="bold" fill="#0f172a">同步</text>')
L.append(f'<text x="20" y="{sy+48}" font-size="11" fill="#64748b">Scheduler</text>')


def block(x, y, bw, bh, label, fill, stroke, fs=11.5):
    L.append(f'<rect x="{x:.0f}" y="{y}" width="{bw:.0f}" height="{bh}" rx="5" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+bw/2:.0f}" y="{y+bh/2+4:.0f}" text-anchor="middle" '
             f'font-size="{fs}" fill="#0f172a">{esc(label)}</text>')


# sync: schedule then forward, strictly serial within each tick
for i in range(3):
    x = t0 + i * cw
    block(x + 6, sy, cw / 2 - 10, 34, f"sched {i}", "#dbeafe", "#3b82f6")
    block(x + cw / 2 + 2, sy, cw / 2 - 8, 34, f"fwd {i}", "#fed7aa", "#f97316")
L.append(f'<text x="{t0:.0f}" y="{sy+62}" font-size="11.5" fill="#b91c1c">'
         'sched 必须等上一拍 fwd 出 token 才能算 num_computed_tokens → GPU 有气泡</text>')

# --- Async lane ---
ay = 250
L.append(f'<text x="20" y="{ay+30}" font-size="14" font-weight="bold" fill="#0f172a">异步</text>')
L.append(f'<text x="20" y="{ay+48}" font-size="11" fill="#64748b">AsyncScheduler</text>')

# async: schedule(N) overlaps forward(N-1)
for i in range(4):
    x = t0 + i * cw
    block(x + 6, ay, cw - 14, 30, f"sched {i}", "#dbeafe", "#3b82f6")
for i in range(3):
    x = t0 + (i + 1) * cw
    block(x + 6, ay + 36, cw - 14, 30, f"fwd {i}", "#fed7aa", "#f97316")
# overlap brace
L.append(f'<text x="{t0+cw:.0f}" y="{ay+90}" font-size="11.5" fill="#15803d">'
         'sched 1 与 fwd 0 同墙钟重叠 —— 靠 num_output_placeholders 占位预调度下一拍 decode 槽</text>')

# placeholder counter row
py = ay + 110
L.append(f'<text x="20" y="{py+18}" font-size="12.5" font-weight="bold" fill="#0f172a">'
         'num_output_placeholders</text>')
counts = [
    (0, "+1", "→1", "sched 0 后记 1 个占位"),
    (1, "+1", "→2", "sched 1 又记 1（token 还没回）"),
    (1, "−1", "→1", "fwd 0 出 token，兑现 1 个"),
]
# show a clean timeline of the counter
vals = ["0", "1", "2 → 1", "1 → ..."]
notes = ["", "sched0 +1", "sched1 +1 / fwd0 −1", "配平"]
for i in range(4):
    x = t0 + i * cw
    L.append(f'<rect x="{x+cw/2-26:.0f}" y="{py+28}" width="52" height="30" rx="5" '
             'fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>')
    L.append(f'<text x="{x+cw/2:.0f}" y="{py+48}" text-anchor="middle" font-size="13" '
             f'font-weight="bold" font-family="monospace" fill="#166534">{esc(vals[i])}</text>')
    if notes[i]:
        L.append(f'<text x="{x+cw/2:.0f}" y="{py+74}" text-anchor="middle" font-size="10" '
                 f'fill="#475569">{esc(notes[i])}</text>')

L.append(f'<text x="{w//2}" y="492" text-anchor="middle" font-size="12.5" '
         'fill="#475569" font-style="italic">'
         '占位 += 1+num_spec_tokens（调度时），−= len(new_token_ids)（token 回流时），始终配平且 ≥ 0</text>')

L.append('</svg>')
open("13-async-placeholder.svg", "w").write('\n'.join(L))
print("wrote 13-async-placeholder.svg")
