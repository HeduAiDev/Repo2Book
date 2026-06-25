#!/usr/bin/env python3
"""ch18-slot-mapping: position → 物理 KV slot, block_table 镜像 + Triton 映射."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1120, 700
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>')
L.append('<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>')
L.append('<marker id="arr2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def text(x,y,t,size=14,anchor="start",fill="#0f172a",weight="normal",mono=False):
    fam="monospace" if mono else "sans-serif"
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(t)}</text>')

# === TOP: block_table row, CPU mirror -> GPU mirror ===
text(40, 45, "block_table[req] —— 逻辑块号 → 物理块号", 17, weight="bold")
cell_w, cell_h = 110, 50
bx, by = 80, 70
logical = [("逻辑块0","7"),("逻辑块1","3"),("逻辑块2","9")]
# CPU mirror
text(bx-12, by+cell_h//2+5, "CPU", 14, anchor="end", fill="#64748b")
for i,(lg,phys) in enumerate(logical):
    x = bx + i*cell_w
    L.append(f'<rect x="{x}" y="{by}" width="{cell_w}" height="{cell_h}" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>')
    text(x+cell_w//2, by+20, lg, 13, anchor="middle", fill="#5b21b6")
    text(x+cell_w//2, by+40, "→ phys "+phys, 13.5, anchor="middle", weight="bold", fill="#4c1d95")
# GPU mirror (below) via commit_block_table copy
gy = by + 90
text(bx-12, gy+cell_h//2+5, "GPU", 14, anchor="end", fill="#64748b")
for i,(lg,phys) in enumerate(logical):
    x = bx + i*cell_w
    L.append(f'<rect x="{x}" y="{gy}" width="{cell_w}" height="{cell_h}" fill="#f5f3ff" stroke="#a78bfa" stroke-width="1.5"/>')
    text(x+cell_w//2, gy+20, lg, 13, anchor="middle", fill="#5b21b6")
    text(x+cell_w//2, gy+40, "→ phys "+phys, 13.5, anchor="middle", weight="bold", fill="#4c1d95")
# copy arrow — start at bottom edge of CPU row, end at top edge of GPU row
cax = bx + 3*cell_w + 30
L.append(f'<path d="M {cax} {by+cell_h} L {cax} {gy}" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#arr2)"/>')
text(cax+12, (by+gy)//2+30, "commit_block_table", 13, fill="#475569")
text(cax+12, (by+gy)//2+48, "CPU→GPU 拷贝", 12.5, fill="#94a3b8")

# === MIDDLE: worked example pos=33 ===
ey = gy + 110
text(40, ey, "映射一例 (block_size = 16, pos = 33)", 16, weight="bold", fill="#7c2d12")
steps = [
    "block_index = pos // block_size = 33 // 16 = 2",
    "block_numbers = block_table[req, 2] = 9",
    "block_offset  = pos % block_size = 33 % 16 = 1",
    "slot_id = block_numbers · block_size + offset = 9·16 + 1 = 145",
]
for i,s in enumerate(steps):
    text(70, ey+28+i*26, s, 14.5, mono=True, fill=("#7c2d12" if i==3 else "#334155"))
text(40, ey+28+4*26+18, "非 CP 公式：  slot = block_table[req, pos // bs] · bs + pos % bs", 14.5, mono=True, weight="bold", fill="#be123c")

# === BOTTOM: token row -> slot_mapping with PAD tail ===
ty = ey + 28 + 4*26 + 70
text(40, ty, "一行 token 的位置 → slot_mapping（尾部 PAD_SLOT_ID 供 CUDA graph 固定形状）", 15, weight="bold")
positions = [16,17,18,19,20,33]
slots = ["48","49","50","51","52","145"]
pads = ["PAD","PAD"]
sw = 78
sty = ty + 20
# positions row
for i,p in enumerate(positions):
    x = 70 + i*sw
    L.append(f'<rect x="{x}" y="{sty}" width="{sw}" height="40" fill="#fef3c7" stroke="#d97706"/>')
    text(x+sw//2, sty+25, f"pos {p}", 13, anchor="middle", fill="#92400e")
xpad0 = 70 + len(positions)*sw
for j,p in enumerate(pads):
    x = xpad0 + j*sw
    L.append(f'<rect x="{x}" y="{sty}" width="{sw}" height="40" fill="#f1f5f9" stroke="#cbd5e1"/>')
    text(x+sw//2, sty+25, "—", 13, anchor="middle", fill="#94a3b8")
# slot row below
sty2 = sty + 70
for i,s in enumerate(slots):
    x = 70 + i*sw
    L.append(f'<rect x="{x}" y="{sty2}" width="{sw}" height="40" fill="#ddd6fe" stroke="#7c3aed"/>')
    text(x+sw//2, sty2+25, f"slot {s}", 13, anchor="middle", fill="#4c1d95")
    L.append(f'<path d="M {x+sw//2} {sty+40} L {x+sw//2} {sty2}" fill="none" stroke="#7c3aed" stroke-width="1.5" marker-end="url(#arr)"/>')
for j,p in enumerate(pads):
    x = xpad0 + j*sw
    L.append(f'<rect x="{x}" y="{sty2}" width="{sw}" height="40" fill="#fee2e2" stroke="#ef4444"/>')
    text(x+sw//2, sty2+25, "PAD_ID", 12, anchor="middle", fill="#b91c1c")
text(70, sty2+62, "slot_mapping[t] 是第 t 个 token 的 KV 物理槽位；前向就照它把 K、V 写进 paged cache。", 13, fill="#475569")

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch18-model-runner/diagrams/ch18-slot-mapping.svg","w").write('\n'.join(L))
print("ok")
