"""DP wave state machine: RUNNING <-> PAUSED with (current_wave, engines_running)."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


w, h = 920, 500
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
    'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append('<text x="20" y="34" font-size="20" font-weight="bold" fill="#0f172a" '
         'font-family="sans-serif">DP wave 状态机：全体运行 ⇄ 全体暂停</text>')

# two state circles
def state(cx, cy, r, title, sub, color, fill):
    L.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{color}" '
             'stroke-width="3"/>')
    L.append(f'<text x="{cx}" y="{cy - 8}" font-size="22" text-anchor="middle" '
             f'font-weight="bold" fill="{color}" font-family="sans-serif">{esc(title)}</text>')
    L.append(f'<text x="{cx}" y="{cy + 18}" font-size="13" text-anchor="middle" '
             f'fill="#475569" font-family="sans-serif">{esc(sub)}</text>')


run_cx, pause_cx, cy, r = 230, 690, 205, 110
state(run_cx, cy, r, "RUNNING", "engines_running=True", "#047857", "#ecfdf5")
state(pause_cx, cy, r, "PAUSED", "engines_running=False", "#b45309", "#fffbeb")

# RUNNING -> PAUSED (top arc)
L.append(f'<path d="M {run_cx + r - 8} {cy - 60} '
         f'C {(run_cx + pause_cx) // 2} {cy - 150}, '
         f'{(run_cx + pause_cx) // 2} {cy - 150}, '
         f'{pause_cx - r + 8} {cy - 60}" fill="none" stroke="#475569" '
         'stroke-width="2.5" marker-end="url(#a)"/>')
mid = (run_cx + pause_cx) // 2
L.append(f'<text x="{mid}" y="{cy - 158}" font-size="14" text-anchor="middle" '
         f'fill="#0f172a" font-weight="bold" font-family="sans-serif">'
         'rank0 all-reduce 判定全体无未完成</text>')
L.append(f'<text x="{mid}" y="{cy - 138}" font-size="13" text-anchor="middle" '
         f'fill="#475569" font-family="sans-serif">→ 报 wave_complete，current_wave++</text>')

# PAUSED -> RUNNING (bottom arc)
L.append(f'<path d="M {pause_cx - r + 8} {cy + 60} '
         f'C {mid} {cy + 150}, {mid} {cy + 150}, '
         f'{run_cx + r - 8} {cy + 60}" fill="none" stroke="#475569" '
         'stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="{mid}" y="{cy + 150}" font-size="14" text-anchor="middle" '
         f'fill="#0f172a" font-weight="bold" font-family="sans-serif">'
         'FIRST_REQ / START_DP_WAVE / 收到 stale-wave 请求</text>')
L.append(f'<text x="{mid}" y="{cy + 170}" font-size="13" text-anchor="middle" '
         f'fill="#475569" font-family="sans-serif">→ engines_running=True</text>')

# self loop on RUNNING (dummy batch)
L.append(f'<path d="M {run_cx - 40} {cy - r + 12} '
         f'C {run_cx - 130} {cy - r - 40}, {run_cx - 130} {cy + r + 40}, '
         f'{run_cx - 40} {cy + r - 12}" fill="none" stroke="#94a3b8" '
         'stroke-width="2" stroke-dasharray="5 4" marker-end="url(#a)"/>')
L.append(f'<text x="{run_cx - 145}" y="{cy - 4}" font-size="12" text-anchor="end" '
         f'fill="#64748b" font-family="sans-serif">无 ready</text>')
L.append(f'<text x="{run_cx - 145}" y="{cy + 14}" font-size="12" text-anchor="end" '
         f'fill="#64748b" font-family="sans-serif">→ dummy batch</text>')

# bottom note: two-phase pause
ny = h - 56
L.append(f'<rect x="20" y="{ny}" width="{w - 40}" height="44" rx="8" '
         'fill="#fef2f2" stroke="#fecaca"/>')
L.append(f'<text x="34" y="{ny + 20}" font-size="13" fill="#b91c1c" '
         'font-weight="bold" font-family="sans-serif">两阶段暂停：</text>')
L.append(f'<text x="120" y="{ny + 20}" font-size="13" fill="#7f1d1d" '
         'font-family="sans-serif">各 rank 设 pending_pause 继续空转 → all-reduce 确认全员同意（SUM==dp_size）</text>')
L.append(f'<text x="34" y="{ny + 38}" font-size="13" fill="#7f1d1d" '
         'font-family="sans-serif">→ 共识达成置 ignore_start_dp_wave，丢弃管道里迟到的 START_DP_WAVE，避免「刚停又被旧唤醒拉起」</text>')

L.append('</svg>')
open("instances/vllm/artifacts/ch21-async-engine/diagrams/dp-wave-state-machine.svg",
     "w", encoding="utf-8").write('\n'.join(L))
print("wrote dp-wave-state-machine.svg")
