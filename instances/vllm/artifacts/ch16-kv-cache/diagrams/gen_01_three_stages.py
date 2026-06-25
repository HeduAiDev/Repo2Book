#!/usr/bin/env python3
"""01-allocate-slots-three-stages: horizontal pipeline of allocate_slots."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1180, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
         '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def box(x, y, bw, bh, fill, stroke, lines, fs=15, tw="normal", tc="#0f172a"):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    cy = y + bh/2 - (n-1)*(fs+4)/2 + fs/2 - 2
    for i, t in enumerate(lines):
        L.append(f'<text x="{x+bw/2}" y="{cy+i*(fs+4)}" text-anchor="middle" font-size="{fs}" '
                 f'font-weight="{tw}" fill="{tc}">{esc(t)}</text>')

def stagebody(x, y, bw, lines, fs=13, y_top=52):
    """Top-anchored body text starting below the stage title."""
    for i, t in enumerate(lines):
        if not t:
            continue
        L.append(f'<text x="{x+bw/2}" y="{y+y_top+i*(fs+6)}" text-anchor="middle" '
                 f'font-size="{fs}" fill="#0f172a">{esc(t)}</text>')

def txt(x, y, t, fs=14, anchor="start", fill="#334155", tw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{tw}" fill="{fill}">{esc(t)}</text>')

# Title
txt(40, 40, "KVCacheManager.allocate_slots —— 三阶段分配流水线", 22, "start", "#0f172a", "bold")

# Block layout strip (top)
segs = [("comp", "已算", "#dbeafe", 150), ("new_comp", "前缀命中", "#bfdbfe", 150),
        ("ext_comp", "外部命中", "#bae6fd", 150), ("new", "本次新算", "#fde68a", 160),
        ("lookahead", "投机草稿", "#fecaca", 150)]
lx = 90
ly = 72
txt(40, ly+24, "块布局", 14, "start", "#64748b", "bold")
for name, cn, fill, sw in segs:
    box(lx, ly, sw, 46, fill, "#94a3b8", [f"{name}", cn], 13)
    lx += sw
# bracket annotations
txt(90+150+150, ly+108, "↑ 已缓存（vLLM 或 connector）", 12, "start", "#2563eb")
txt(90+150+150+150, ly+108, "↑ 待计算（new + lookahead）", 12, "start", "#b45309")

# inputs box
iy = 215
box(40, iy, 200, 130, "#f1f5f9", "#94a3b8",
    ["输入参数", "num_new_tokens", "num_new_computed_tokens", "new_computed_blocks",
     "num_external_computed_tokens", "num_lookahead_tokens"], 12.5, "normal")

# three stage boxes
sx = 285
sw = 270
gap = 18
sy = 200
sh = 165

# Stage 1
L.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="8" fill="#ecfdf5" stroke="#10b981" stroke-width="1.5"/>')
txt(sx+sw/2, sy+30, "阶段一　释放 + 预算检查", 15, "middle", "#065f46", "bold")
stagebody(sx, sy, sw, ["remove_skipped_blocks", "窗外块换 null → 归还 free queue", "",
                       "get_num_blocks_to_allocate", "需块 > free → return None"])
# return None escape
box(sx+40, sy+sh+30, sw-80, 40, "#fef2f2", "#b91c1c", ["return None（不足 → 触发抢占）"], 13, "normal", "#b91c1c")
L.append(f'<line x1="{sx+sw/2}" y1="{sy+sh}" x2="{sx+sw/2}" y2="{sy+sh+30}" stroke="#b91c1c" stroke-width="1.6" marker-end="url(#ar)"/>')

# Stage 2
sx2 = sx + sw + gap
L.append(f'<rect x="{sx2}" y="{sy}" width="{sw}" height="{sh}" rx="8" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5"/>')
txt(sx2+sw/2, sy+30, "阶段二　挂命中块", 15, "middle", "#1e40af", "bold")
stagebody(sx2, sy, sw, ["touch 命中块 (ref_cnt+1)", "null_block 填 skipped 段",
                        "external → get_new_blocks", "", "（挂进 req_to_blocks）"])

# Stage 3
sx3 = sx2 + sw + gap
L.append(f'<rect x="{sx3}" y="{sy}" width="{sw}" height="{sh}" rx="8" fill="#fefce8" stroke="#ca8a04" stroke-width="1.5"/>')
txt(sx3+sw/2, sy+30, "阶段三　新建块", 15, "middle", "#854d0e", "bold")
stagebody(sx3, sy, sw, ["allocate_new_blocks", "新建 new + lookahead 槽位", "",
                        "get_new_blocks 取真实块", "（含投机草稿预留）"])

# arrows between
L.append(f'<line x1="240" y1="{iy+65}" x2="{sx}" y2="{sy+sh/2}" stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{sx+sw}" y1="{sy+sh/2}" x2="{sx2}" y2="{sy+sh/2}" stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{sx2+sw}" y1="{sy+sh/2}" x2="{sx3}" y2="{sy+sh/2}" stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')

# finalize box (bottom)
fy = 470
fw2 = sw+40
L.append(f'<rect x="{sx3-40}" y="{fy}" width="{fw2}" height="100" rx="8" fill="#f5f3ff" stroke="#8b5cf6" stroke-width="1.5"/>')
txt(sx3-40+fw2/2, fy+26, "收尾　缓存满块", 15, "middle", "#5b21b6", "bold")
stagebody(sx3-40, fy, fw2, ["num_tokens_to_cache =",
                            "min(total_computed+new, request.num_tokens)",
                            "→ cache_full_blocks 写哈希入表"], fs=12.5, y_top=48)
L.append(f'<line x1="{sx3+sw/2}" y1="{sy+sh}" x2="{sx3+sw/2-20}" y2="{fy}" stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')
txt(sx3-40, fy+118, "未定稿草稿 token 被 min() 封掉，不污染前缀缓存", 12, "start", "#6d28d9")

svg = '\n'.join(L) + '\n</svg>\n'
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch16-kv-cache/diagrams/01-allocate-slots-three-stages.svg", "w", encoding="utf-8").write(svg)
print("wrote 01")
