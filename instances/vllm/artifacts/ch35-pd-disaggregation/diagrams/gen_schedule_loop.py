import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 920, 1080
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>')
L.append('<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#c2410c"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

cx = 300   # main column center
bw = 360   # box width

def box(y, bh, text2, fill="#f1f5f9", stroke="#64748b", tcol="#0f172a", fs=14, mono=None, x=None, bwl=None):
    bx = (cx - bw/2) if x is None else x
    bbw = bw if bwl is None else bwl
    L.append(f'<rect x="{bx}" y="{y}" width="{bbw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lines = text2 if isinstance(text2, list) else [text2]
    n = len(lines)
    for i, (t, m) in enumerate(lines):
        ty = y + bh/2 + (i - (n-1)/2)*18 + 5
        fam = ' font-family="monospace"' if m else ''
        fw = ' font-weight="bold"' if not m else ' font-weight="bold"'
        L.append(f'<text x="{bx+bbw/2}" y="{ty}" text-anchor="middle" font-size="{fs}"{fam}{fw} fill="{tcol}">{esc(t)}</text>')

def diamond(cy, text2, half=180, vh=42, fill="#fef3c7", stroke="#d97706", tcol="#92400e"):
    pts = f"{cx},{cy-vh} {cx+half},{cy} {cx},{cy+vh} {cx-half},{cy}"
    L.append(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lines = text2 if isinstance(text2, list) else [text2]
    n=len(lines)
    for i,(t,m) in enumerate(lines):
        ty = cy + (i-(n-1)/2)*16 + 5
        fam = ' font-family="monospace"' if m else ''
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="12"{fam} font-weight="bold" fill="{tcol}">{esc(t)}</text>')

def varrow(y1, y2, label=None):
    L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="#475569" stroke-width="2.5" marker-end="url(#a)"/>')
    if label:
        L.append(f'<text x="{cx+8}" y="{(y1+y2)/2+4}" font-size="11" fill="#334155">{esc(label)}</text>')

# isolate box on the right side
isox = cx + 250
def isolate(cy, text="隔离进 step_skipped_waiting"):
    L.append(f'<rect x="{isox}" y="{cy-22}" width="300" height="44" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/>')
    L.append(f'<text x="{isox+150}" y="{cy-2}" text-anchor="middle" font-size="12" font-weight="bold" fill="#9a3412">{esc(text)}</text>')
    L.append(f'<text x="{isox+150}" y="{cy+15}" text-anchor="middle" font-size="11" fill="#9a3412">pop + prepend，continue</text>')

y = 30
box(y, 44, [("_select_waiting_queue_for_scheduling()", True)], "#e0e7ff", "#6366f1", "#3730a3", 13)
varrow(y+44, y+74, "peek_request")

y=104
diamond(y+24, [("_is_blocked_waiting_status?", False)])
# yes branch -> right
L.append(f'<line x1="{cx+180}" y1="{y+24}" x2="{isox}" y2="{y+24}" stroke="#c2410c" stroke-width="2" marker-end="url(#ao)"/>')
L.append(f'<text x="{cx+190}" y="{y+16}" text-anchor="middle" font-size="11" fill="#c2410c">是 → try_promote</text>')
L.append(f'<rect x="{isox}" y="{y+2}" width="300" height="44" rx="8" fill="#fef9c3" stroke="#eab308" stroke-width="2"/>')
L.append(f'<text x="{isox+150}" y="{y+22}" text-anchor="middle" font-size="12" font-weight="bold" fill="#854d0e">_try_promote_blocked_waiting_request</text>')
L.append(f'<text x="{isox+150}" y="{y+39}" text-anchor="middle" font-size="11" fill="#854d0e">成功→提升；失败→隔离</text>')
varrow(y+66, y+96, "否")

y=200
diamond(y+24, [("num_computed_tokens == 0 ?", True)])
varrow(y+66, y+96, "是（新请求）")

y=296
box(y, 60, [("get_computed_blocks（本地命中）", False),
            ("connector.get_num_new_matched_tokens", True)], "#dbeafe", "#3b82f6", "#1e3a8a", 12)
varrow(y+60, y+90)

y=386
diamond(y+24, [("ext_tokens is None ?", True)])
# yes -> isolate
L.append(f'<line x1="{cx+180}" y1="{y+24}" x2="{isox}" y2="{y+24}" stroke="#c2410c" stroke-width="2" marker-end="url(#ao)"/>')
L.append(f'<text x="{cx+190}" y="{y+16}" text-anchor="middle" font-size="11" fill="#c2410c">是 → 隔离，下步再问</text>')
isolate(y+24)
varrow(y+66, y+96, "否")

y=482
diamond(y+24, [("load_kv_async ?", True)])
varrow(y+66, y+96, "（两支都继续）")
L.append(f'<text x="{cx+186}" y="{y+24}" font-size="11" fill="#1e40af">是 → num_new_tokens = 0</text>')

y=578
box(y, 60, [("allocate_slots(", True),
            ("  num_external_computed_tokens, delay_cache_blocks=load_kv_async)", True)], "#dcfce7", "#22c55e", "#166534", 11)
varrow(y+60, y+90)

y=668
box(y, 50, [("connector.update_state_after_alloc(...)", True)], "#dcfce7", "#22c55e", "#166534", 12)
varrow(y+50, y+80)

y=748
diamond(y+24, [("load_kv_async ?", True)])
# yes -> WAITING_FOR_REMOTE_KVS + isolate
L.append(f'<line x1="{cx+180}" y1="{y+24}" x2="{isox}" y2="{y+24}" stroke="#c2410c" stroke-width="2" marker-end="url(#ao)"/>')
L.append(f'<rect x="{isox}" y="{y+2}" width="300" height="44" rx="8" fill="#ffe4e6" stroke="#f43f5e" stroke-width="2"/>')
L.append(f'<text x="{isox+150}" y="{y+20}" text-anchor="middle" font-size="11" font-weight="bold" fill="#9f1239">status=WAITING_FOR_REMOTE_KVS</text>')
L.append(f'<text x="{isox+150}" y="{y+37}" text-anchor="middle" font-size="11" fill="#9f1239">隔离进 skipped，continue</text>')
varrow(y+66, y+96, "否")

y=844
box(y, 46, [("进入 running（正常 RUNNING）", False)], "#d1fae5", "#10b981", "#065f46", 13)
varrow(y+46, y+82, "（循环结束）")

y=928
box(y, 58, [("循环末尾：build_connector_meta(scheduler_output)", False),
            ("→ scheduler_output.kv_connector_metadata", True)], "#ede9fe", "#7c3aed", "#5b21b6", 12, bwl=560, x=cx-280)

L.append(f'<text x="{cx}" y="1010" text-anchor="middle" font-size="13" fill="#64748b">橙色出口 = 隔离避队头阻塞；红色 = 远程 KV 阻塞态</text>')

L.append('</svg>')
open("ch29-schedule-loop-flow.svg","w",encoding="utf-8").write('\n'.join(L))
print("wrote ch29-schedule-loop-flow.svg")
