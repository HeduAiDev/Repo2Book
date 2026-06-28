#!/usr/bin/env python3
"""fig-ch06-310p-fallback: 310P 用 all_gather 模拟 broadcast / int64 all_reduce。
上半 broadcast310p（all_gather 收齐→取 tensor_list[src]）；
下半 all_reduce_310p int64（all_gather→stack→sum/max）。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1080, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs>')
L.append('<marker id="ar" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def chip(x, y, w, h, text, fill, stroke, tcol="#1e293b", fs=14, bold=False, rx=6):
    fw = "bold" if bold else "normal"
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="1.6" rx="{rx}"/>')
    L.append(f'<text x="{x+w/2}" y="{y+h/2+fs*0.36}" text-anchor="middle" font-size="{fs}" font-weight="{fw}" fill="{tcol}">{esc(text)}</text>')

# ============ 上半：broadcast310p ============
L.append(f'<text x="40" y="40" font-size="17" font-weight="bold" fill="#0f172a">上 · broadcast310p（src=0，world_size=4）</text>')
L.append(f'<text x="40" y="62" font-size="13" fill="#64748b">device tensor 才走这条；CPU tensor 直通原 broadcast</text>')

# 四个 rank，每个手里只有自己那份（src=0 有真值 v0）
rank_y = 84
bx, bw, gap = 56, 150, 40
for r in range(4):
    x = bx + r*(bw+gap)
    val = "v0（真值）" if r == 0 else f"空 v{r}"
    fill = "#dcfce7" if r == 0 else "#f1f5f9"
    stroke = "#16a34a" if r == 0 else "#cbd5e1"
    chip(x, rank_y, bw, 50, f"rank{r}", fill, stroke, fs=14, bold=True)
    L.append(f'<text x="{x+bw/2}" y="{rank_y+44}" text-anchor="middle" font-size="12" fill="#475569">{esc(val)}</text>')

# all_gather 横条
ag_y = rank_y + 86
L.append(f'<rect x="{bx}" y="{ag_y}" width="{4*bw+3*gap}" height="40" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.6" rx="8"/>')
L.append(f'<text x="{bx+(4*bw+3*gap)/2}" y="{ag_y+25}" text-anchor="middle" font-size="14" font-weight="bold" fill="#92400e">all_gather → 每个 rank 都拿到 [v0, v1, v2, v3]</text>')
for r in range(4):
    x = bx + r*(bw+gap) + bw/2
    L.append(f'<line x1="{x}" y1="{rank_y+50}" x2="{x}" y2="{ag_y-2}" stroke="#475569" stroke-width="1.5" marker-end="url(#ar)"/>')

# 取 tensor_list[src]
take_y = ag_y + 70
for r in range(4):
    x = bx + r*(bw+gap)
    chip(x, take_y, bw, 46, f"rank{r}: tensor[0]=v0", "#dcfce7", "#16a34a", tcol="#065f46", fs=12.5, bold=True)
    cx = x + bw/2
    L.append(f'<line x1="{cx}" y1="{ag_y+40}" x2="{cx}" y2="{take_y-2}" stroke="#475569" stroke-width="1.5" marker-end="url(#ar)"/>')
L.append(f'<text x="{bx+(4*bw+3*gap)/2}" y="{take_y+72}" text-anchor="middle" font-size="13" fill="#475569">各 rank 取 tensor_list[src] → 等价 broadcast；通信量 O(numel) → O(world_size·numel)</text>')

# 分隔线
sep_y = take_y + 96
L.append(f'<line x1="40" y1="{sep_y}" x2="{W-40}" y2="{sep_y}" stroke="#e2e8f0" stroke-width="1.5"/>')

# ============ 下半：all_reduce_310p int64 ============
L.append(f'<text x="40" y="{sep_y+34}" font-size="17" font-weight="bold" fill="#0f172a">下 · all_reduce_310p（仅 int64）</text>')
L.append(f'<text x="40" y="{sep_y+56}" font-size="13" fill="#64748b">非 int64 直通原 all_reduce（310P 浮点 all_reduce 原生可用）</text>')

fy = sep_y + 76
# all_gather 收齐
chip(bx, fy, 250, 46, "all_gather → [t0, t1, t2, t3]", "#fef3c7", "#f59e0b", tcol="#92400e", fs=13.5, bold=True)
# stack
chip(bx+300, fy, 200, 46, "torch.stack(...)  叠成新维", "#eff6ff", "#3b82f6", tcol="#1e3a8a", fs=13.5)
L.append(f'<line x1="{bx+250}" y1="{fy+23}" x2="{bx+298}" y2="{fy+23}" stroke="#475569" stroke-width="1.6" marker-end="url(#ar)"/>')
# 归约分支
chip(bx+560, fy-28, 220, 44, "SUM → .sum(0)", "#dcfce7", "#16a34a", tcol="#065f46", fs=13.5, bold=True)
chip(bx+560, fy+30, 220, 44, "MAX → numpy().max(0)", "#dcfce7", "#16a34a", tcol="#065f46", fs=13.5, bold=True)
L.append(f'<line x1="{bx+500}" y1="{fy+23}" x2="{bx+558}" y2="{fy-6}" stroke="#475569" stroke-width="1.6" marker-end="url(#ar)"/>')
L.append(f'<line x1="{bx+500}" y1="{fy+23}" x2="{bx+558}" y2="{fy+52}" stroke="#475569" stroke-width="1.6" marker-end="url(#ar)"/>')
L.append(f'<text x="{bx+560}" y="{fy+96}" font-size="12.5" fill="#991b1b">其余 op → raise（只实现 SUM/MAX）</text>')

L.append('</svg>')
open("p310_fallback.svg", "w").write('\n'.join(L))
print("wrote p310_fallback.svg", W, H)
