import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 980, 680
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="aThick" viewBox="0 0 12 8" refX="11" refY="4" markerWidth="9" markerHeight="6" orient="auto"><path d="M0,0 L12,4 L0,8 Z" fill="#1d4ed8"/></marker>')
L.append('<marker id="aThin" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# Top factory box
fx, fy, fw, fh = 240, 24, 500, 56
L.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" rx="8" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>')
L.append(f'<text x="{fx+fw/2}" y="{fy+24}" text-anchor="middle" font-size="15" font-weight="bold" fill="#5b21b6">KVConnectorFactory.create_connector(role=...)</text>')
L.append(f'<text x="{fx+fw/2}" y="{fy+44}" text-anchor="middle" font-size="13" fill="#6d28d9">同一 connector 类 · 按 role 分别实例化两份</text>')

# Two columns
colY = 130
colW = 380
colH = 360
leftX = 40
rightX = w - 40 - colW

cols = [
    (leftX, "SCHEDULER role", "调度器进程 · 决策侧", "#1e3a8a", "#dbeafe", "#3b82f6",
     ["get_num_new_matched_tokens", "update_state_after_alloc",
      "build_connector_meta", "update_connector_output", "request_finished"]),
    (rightX, "WORKER role", "worker 进程 · 搬运侧", "#7c2d12", "#ffedd5", "#ea580c",
     ["start_load_kv", "wait_for_layer_load", "save_kv_layer",
      "wait_for_save", "get_finished"]),
]

for cx, title, sub, tcol, fillc, strokec, methods in cols:
    L.append(f'<rect x="{cx}" y="{colY}" width="{colW}" height="{colH}" rx="10" fill="{fillc}" stroke="{strokec}" stroke-width="2.5"/>')
    L.append(f'<text x="{cx+colW/2}" y="{colY+30}" text-anchor="middle" font-size="18" font-weight="bold" fill="{tcol}">{esc(title)}</text>')
    L.append(f'<text x="{cx+colW/2}" y="{colY+52}" text-anchor="middle" font-size="13" fill="{tcol}">{esc(sub)}</text>')
    my = colY + 78
    for m in methods:
        L.append(f'<rect x="{cx+24}" y="{my}" width="{colW-48}" height="42" rx="6" fill="white" stroke="{strokec}" stroke-width="1.5"/>')
        L.append(f'<text x="{cx+colW/2}" y="{my+26}" text-anchor="middle" font-size="14" font-family="monospace" fill="#111827">{esc(m)}</text>')
        my += 52

# Thick arrow left -> right (metadata down-flow)
ay1 = colY + colH + 50
L.append(f'<line x1="{leftX+colW}" y1="{ay1}" x2="{rightX}" y2="{ay1}" stroke="#1d4ed8" stroke-width="5" marker-end="url(#aThick)"/>')
L.append(f'<text x="{w/2}" y="{ay1-14}" text-anchor="middle" font-size="14" font-weight="bold" fill="#1d4ed8">KVConnectorMetadata</text>')
L.append(f'<text x="{w/2}" y="{ay1+24}" text-anchor="middle" font-size="12" fill="#1d4ed8">随 SchedulerOutput 下发 · 不透明信使</text>')

# Thin arrow right -> left (output回传)
ay2 = ay1 + 64
L.append(f'<line x1="{rightX}" y1="{ay2}" x2="{leftX+colW}" y2="{ay2}" stroke="#b45309" stroke-width="2" marker-end="url(#aThin)"/>')
L.append(f'<text x="{w/2}" y="{ay2+22}" text-anchor="middle" font-size="12" fill="#b45309">KVConnectorOutput.finished_recving / finished_sending（回传）</text>')

L.append('</svg>')
open("ch29-role-split.svg","w",encoding="utf-8").write('\n'.join(L))
print("wrote ch29-role-split.svg")
