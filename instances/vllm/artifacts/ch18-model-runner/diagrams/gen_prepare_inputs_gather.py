#!/usr/bin/env python3
"""ch18-prepare-inputs-gather: 从持久批次到 input_ids 的扁平收集."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1240, 760
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>')
L.append('<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#0e7490"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def text(x,y,t,size=14,anchor="start",fill="#0f172a",weight="normal",mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(t)}</text>')

# === LEFT: token_ids_cpu 2D grid ===
text(40, 50, "token_ids_cpu  (max_num_reqs × max_model_len M)", 17, weight="bold")
cell = 30
gx, gy = 60, 80
ncols = 9   # show first 9 cols of M
# per req: num_computed (already done, grey), num_scheduled window (colored), rest empty
reqs = [
    # (label, computed_count, scheduled_count)
    ("req0", 4, 2),
    ("req1", 1, 5),
    ("req2", 6, 3),
]
win_color = ["#bae6fd", "#bbf7d0", "#fde68a"]
done_color = "#e2e8f0"
for r,(lab,nc,ns) in enumerate(reqs):
    ry = gy + r*(cell+18)
    text(50, ry+cell//2+5, lab, 14, anchor="end")
    for c in range(ncols):
        x = gx + c*cell
        if c < nc:
            fc = done_color; tc="#94a3b8"; t=str(c)
        elif c < nc+ns:
            fc = win_color[r]; tc="#0f172a"; t=str(c)
        else:
            fc = "#f8fafc"; tc="#cbd5e1"; t="·"
        L.append(f'<rect x="{x}" y="{ry}" width="{cell}" height="{cell}" fill="{fc}" stroke="#cbd5e1"/>')
        text(x+cell//2, ry+cell//2+5, t, 12, anchor="middle", fill=tc)
    text(gx+ncols*cell+10, ry+cell//2+5, "…", 16)
# legend for grid
lgy = gy + len(reqs)*(cell+18) + 6
L.append(f'<rect x="{gx}" y="{lgy}" width="22" height="18" fill="{done_color}" stroke="#cbd5e1"/>')
text(gx+28, lgy+14, "已算 (num_computed_tokens)", 12.5)
L.append(f'<rect x="{gx+220}" y="{lgy}" width="22" height="18" fill="#bae6fd" stroke="#cbd5e1"/>')
text(gx+248, lgy+14, "本拍要算 (num_scheduled_tokens 窗口)", 12.5)

# === MIDDLE: index arithmetic ===
mx = 60
my = lgy + 70
text(mx, my, "num_scheduled_tokens = [2, 5, 3]", 15, mono=True, weight="bold")
text(mx, my+34, "np.repeat → req_indices = [0,0, 1,1,1,1,1, 2,2,2]", 14.5, mono=True, fill="#0369a1")
text(mx, my+62, "query_pos          = [0,1, 0,1,2,3,4, 0,1,2]", 14.5, mono=True, fill="#15803d")
text(mx, my+90, "positions = num_computed[req_indices] + query_pos", 14.5, mono=True, fill="#b45309")
text(mx+10, my+114, "= [4,5, 1,2,3,4,5, 6,7,8]", 14.5, mono=True, fill="#b45309")
text(mx, my+148, "token_indices = positions + req_indices · M", 15, mono=True, weight="bold", fill="#7c2d12")
text(mx+10, my+172, "(二维坐标 (r,p) → 一维扁平偏移 r·M + p)", 13.5, fill="#475569")

# === RIGHT: index_select → input_ids ===
rx = 820
ry0 = 95
text(rx, ry0-20, "torch.index_select(token_ids_cpu.flatten())", 14.5, weight="bold", mono=True)
text(rx, ry0-2, "→ 连续 input_ids", 14, fill="#334155")
flat = [(0,"#bae6fd"),(1,"#bae6fd"),(0,"#bbf7d0"),(1,"#bbf7d0"),(2,"#bbf7d0"),(3,"#bbf7d0"),(4,"#bbf7d0"),(0,"#fde68a"),(1,"#fde68a"),(2,"#fde68a")]
fc_w = 34
for i,(v,col) in enumerate(flat):
    x = rx + i*fc_w
    L.append(f'<rect x="{x}" y="{ry0}" width="{fc_w}" height="{fc_w}" fill="{col}" stroke="#94a3b8"/>')
    text(x+fc_w//2, ry0+fc_w//2+5, str(i), 12, anchor="middle")
text(rx, ry0+fc_w+22, "total_num_scheduled_tokens = 10 个 token 收成一行", 12.5, fill="#475569")

# arrow from arithmetic to input_ids
L.append(f'<path d="M {mx+430} {my+150} C {rx-60} {my+150}, {rx-60} {ry0+20}, {rx-8} {ry0+20}" fill="none" stroke="#0e7490" stroke-width="2" marker-end="url(#arr)"/>')

# === BOTTOM: same req_indices/positions → slot_mapping ===
sy = my + 230
text(mx, sy, "同一套 req_indices / positions，经 compute_slot_mapping → slot_mapping", 15, weight="bold", fill="#7c2d12")
sl = [(7,"#bae6fd"),(8,"#bae6fd"),(3,"#bbf7d0"),(4,"#bbf7d0"),(5,"#bbf7d0"),(6,"#bbf7d0"),(7,"#bbf7d0"),(9,"#fde68a"),(10,"#fde68a"),(11,"#fde68a")]
for i,(v,col) in enumerate(sl):
    x = rx + i*fc_w
    L.append(f'<rect x="{x}" y="{sy-20}" width="{fc_w}" height="{fc_w}" fill="{col}" stroke="#94a3b8"/>')
    text(x+fc_w//2, sy-20+fc_w//2+5, str(v), 12, anchor="middle")
text(rx, sy+fc_w+2, "每个 token 的物理 KV slot_id", 12.5, fill="#475569")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch18-model-runner/diagrams/ch18-prepare-inputs-gather.svg","w").write('\n'.join(L))
print("ok")
