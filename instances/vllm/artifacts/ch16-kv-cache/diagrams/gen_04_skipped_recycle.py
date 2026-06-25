#!/usr/bin/env python3
"""04-skipped-blocks-recycle: before/after sliding-window block recycle."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1000, 580
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def txt(x, y, t, fs=14, anchor="start", fill="#334155", tw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{tw}" fill="{fill}">{esc(t)}</text>')

def blk(x, y, bw, bh, fill, stroke, label, sub="", fs=18):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    if sub:
        L.append(f'<text x="{x+bw/2}" y="{y+bh/2-4}" text-anchor="middle" font-size="{fs}" font-weight="bold" fill="#0f172a">{esc(label)}</text>')
        L.append(f'<text x="{x+bw/2}" y="{y+bh/2+18}" text-anchor="middle" font-size="12" fill="#64748b">{esc(sub)}</text>')
    else:
        L.append(f'<text x="{x+bw/2}" y="{y+bh/2+6}" text-anchor="middle" font-size="{fs}" font-weight="bold" fill="#0f172a">{esc(label)}</text>')

txt(40, 40, "remove_skipped_blocks —— 滑窗外块回收（append-only，下标不变）", 20, "start", "#0f172a", "bold")
txt(40, 68, "sliding_window = 8，block_size = 4，num_computed = 11", 14, "start", "#64748b")
txt(40, 90, "get_num_skipped_tokens(11) = max(0, 11 − 8 + 1) = 4 → num_skipped_blocks = 4 // 4 = 1", 13.5, "start", "#475569")

bw, bh = 150, 70
gap = 16

# BEFORE
txt(60, 150, "前　req_to_blocks", 16, "start", "#0f172a", "bold")
bx = 60
by = 170
for i, (lab, sub, fill, stroke) in enumerate([
        ("B0", "token 0~3", "#fecaca", "#ef4444"),
        ("B1", "token 4~7", "#bbf7d0", "#22c55e"),
        ("B2", "token 8~11", "#bbf7d0", "#22c55e")]):
    blk(bx + i*(bw+gap), by, bw, bh, fill, stroke, lab, sub)
    txt(bx + i*(bw+gap)+bw/2, by-12, f"下标 {i}", 12, "middle", "#94a3b8")
txt(bx, by+bh+30, "B0 落在滑窗之外（逆序扫描第一个待回收块）", 13, "start", "#b91c1c")

# arrow down
ax = 60 + bw/2
L.append(f'<line x1="{ax}" y1="{by+bh+50}" x2="{ax}" y2="{by+bh+95}" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>')
txt(ax+18, by+bh+80, "B0 换成 null_block，真实显存归还 free queue", 13.5, "start", "#475569", "bold")

# AFTER
ay = 385
txt(60, ay-24, "后　req_to_blocks", 16, "start", "#0f172a", "bold")
for i, (lab, sub, fill, stroke) in enumerate([
        ("NULL", "占位 / 不缓存", "#e2e8f0", "#94a3b8"),
        ("B1", "token 4~7", "#bbf7d0", "#22c55e"),
        ("B2", "token 8~11", "#bbf7d0", "#22c55e")]):
    blk(bx + i*(bw+gap), ay, bw, bh, fill, stroke, lab, sub)
    txt(bx + i*(bw+gap)+bw/2, ay-10, f"下标 {i}", 12, "middle", "#94a3b8")

# annotations
ny = ay + bh + 36
txt(60, ny, "• 下标 0 仍占位 → block table 是 append-only，逻辑偏移不乱", 13.5, "start", "#334155")
txt(60, ny+24, "• B0 的真实块进 free queue，可被别的请求取用 → 真显存被释放", 13.5, "start", "#334155")
txt(60, ny+48, "• 逆序扫描，遇已是 null 的块立即早停 → 幂等，重复调用无副作用", 13.5, "start", "#334155")

svg = '\n'.join(L) + '\n</svg>\n'
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch16-kv-cache/diagrams/04-skipped-blocks-recycle.svg", "w", encoding="utf-8").write(svg)
print("wrote 04")
