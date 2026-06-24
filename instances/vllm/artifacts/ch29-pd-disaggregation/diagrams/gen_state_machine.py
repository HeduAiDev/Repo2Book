import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 1040, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>')
L.append('<marker id="af" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# swimlane: skipped_waiting region
slx, sly, slw, slh = 280, 140, 700, 300
L.append(f'<rect x="{slx}" y="{sly}" width="{slw}" height="{slh}" rx="12" fill="#fff7ed" stroke="#fb923c" stroke-width="2" stroke-dasharray="8 5"/>')
L.append(f'<text x="{slx+16}" y="{sly+26}" font-size="14" font-weight="bold" fill="#c2410c">skipped_waiting 泳道（隔离区，避队头阻塞 HoL）</text>')

def node(x, y, nw, nh, text, fill, stroke, tcol, fs=14, mono=False):
    L.append(f'<rect x="{x}" y="{y}" width="{nw}" height="{nh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    fam = ' font-family="monospace"' if mono else ''
    L.append(f'<text x="{x+nw/2}" y="{y+nh/2+5}" text-anchor="middle" font-size="{fs}"{fam} font-weight="bold" fill="{tcol}">{esc(text)}</text>')

# WAITING node
node(60, 250, 150, 56, "WAITING", "#dbeafe", "#3b82f6", "#1e3a8a", 15)
# WAITING_FOR_REMOTE_KVS (in swimlane)
node(330, 250, 230, 56, "WAITING_FOR_REMOTE_KVS", "#ffe4e6", "#f43f5e", "#9f1239", 13, True)
# finished_recving set (in swimlane)
node(620, 250, 240, 56, "finished_recving_kv_req_ids", "#fef9c3", "#eab308", "#854d0e", 12, True)
# promote outcome -> WAITING/PREEMPTED (in swimlane upper-right)
node(620, 360, 240, 50, "WAITING / PREEMPTED", "#dcfce7", "#22c55e", "#166534", 14)
# RUNNING (outside, right)
node(620, 470, 240, 56, "RUNNING（开始 decode）", "#d1fae5", "#10b981", "#065f46", 14)

# Arrow WAITING -> WAITING_FOR_REMOTE_KVS
L.append(f'<line x1="210" y1="278" x2="330" y2="278" stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="270" y="232" text-anchor="middle" font-size="12" fill="#334155">命中且</text>')
L.append(f'<text x="270" y="248" text-anchor="middle" font-size="12" font-family="monospace" fill="#334155">load_kv_async</text>')

# Arrow WAITING_FOR_REMOTE_KVS -> finished_recving set
L.append(f'<line x1="560" y1="278" x2="620" y2="278" stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="590" y="230" text-anchor="middle" font-size="11" fill="#334155">worker 报</text>')
L.append(f'<text x="590" y="244" text-anchor="middle" font-size="11" font-family="monospace" fill="#334155">finished_recving</text>')

# Arrow finished_recving -> promote -> WAITING/PREEMPTED
L.append(f'<line x1="740" y1="306" x2="740" y2="360" stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="752" y="332" font-size="11" fill="#334155">_try_promote_blocked_waiting_request</text>')
L.append(f'<text x="752" y="347" font-size="11" fill="#334155">+ _update_waiting_for_remote_kv（缓存 block）</text>')

# Arrow WAITING/PREEMPTED -> RUNNING
L.append(f'<line x1="740" y1="410" x2="740" y2="470" stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<text x="752" y="445" font-size="11" fill="#334155">重新进入正常调度 / allocate_slots</text>')

# failure path: WAITING_FOR_REMOTE_KVS -> 回退
node(330, 470, 230, 56, "failed_recving_kv_req_ids", "#fee2e2", "#ef4444", "#991b1b", 12, True)
L.append(f'<path d="M 420 306 Q 360 390 420 470" fill="none" stroke="#b91c1c" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#af)"/>')
L.append(f'<text x="300" y="392" font-size="11" fill="#b91c1c">传输失败</text>')
L.append(f'<text x="345" y="556" font-size="11" fill="#b91c1c">重设 num_computed_tokens 后重试</text>')

# Side note: waiting queue keeps flowing
node(60, 470, 150, 56, "waiting 队列", "#eff6ff", "#93c5fd", "#1e40af", 13)
L.append(f'<text x="135" y="552" text-anchor="middle" font-size="11" fill="#1e40af">继续流动 · 不被阻塞</text>')
L.append(f'<path d="M 135 470 Q 135 400 135 306" fill="none" stroke="#93c5fd" stroke-width="2" stroke-dasharray="4 4"/>')

# Title
L.append(f'<text x="{w/2}" y="40" text-anchor="middle" font-size="18" font-weight="bold" fill="#111827">WAITING_FOR_REMOTE_KVS 阻塞态：远程 KV 加载与提升路径</text>')
L.append(f'<text x="{w/2}" y="64" text-anchor="middle" font-size="13" fill="#64748b">被阻塞请求隔离进 skipped_waiting，主 waiting 队列不受影响</text>')

L.append('</svg>')
open("ch29-remote-kv-state-machine.svg","w",encoding="utf-8").write('\n'.join(L))
print("wrote ch29-remote-kv-state-machine.svg")
