#!/usr/bin/env python3
"""Generate Diagram 2: Tiled Attention with Online Softmax state evolution."""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

def build_svg():
    w, h = 960, 680
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>')
    L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>')
    L.append('</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="#f8fafc"/>')

    # Title
    L.append(f'<text x="{w//2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#1e3a5f" font-weight="bold">{esc("Tiled Attention + Online Softmax 算法")}</text>')
    L.append(f'<text x="{w//2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#64748b">{esc("Q 分成 3 个 Block，每个 Block 遍历 4 个 KV Block — S [4x4] 永远不写入 HBM")}</text>')

    # ---------- Top: HBM / SRAM boundary ----------
    L.append(f'<line x1="10" y1="70" x2="{w-10}" y2="70" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4,4"/>')
    L.append(f'<text x="{w-20}" y="64" text-anchor="end" font-family="sans-serif" font-size="11" fill="#94a3b8">{esc("HBM (显存)")}</text>')
    L.append(f'<text x="{w-20}" y="86" text-anchor="end" font-family="sans-serif" font-size="11" fill="#3b82f6" font-weight="bold">{esc("SRAM (on-chip)")}</text>')

    # Q blocks in HBM
    q_colors = ['#2563eb', '#7c3aed', '#059669']
    q_labels = ['Q₀ [64×128]', 'Q₁ [64×128]', 'Q₂ [64×128]']
    qbx, qby = 20, 76
    for i, (c, lab) in enumerate(zip(q_colors, q_labels)):
        x = qbx + i * 110
        L.append(f'<rect x="{x}" y="{qby}" width="100" height="28" rx="4" fill="{c}" opacity="0.2" stroke="{c}" stroke-width="1.5"/>')
        L.append(f'<text x="{x+50}" y="{qby+19}" text-anchor="middle" font-family="monospace" font-size="11" fill="{c}" font-weight="bold">{esc(lab)}</text>')

    # KV blocks in HBM (right side)
    kvb_x = qbx + 3 * 110 + 60
    kv_colors = ['#b45309'] * 4
    kv_labels = ['KV₀ [64×128]', 'KV₁ [64×128]', 'KV₂ [64×128]', 'KV₃ [64×128]']
    for i, (c, lab) in enumerate(zip(kv_colors, kv_labels)):
        x = kvb_x
        y = qby + i * 32
        L.append(f'<rect x="{x}" y="{y}" width="120" height="28" rx="4" fill="#fef3c7" stroke="#b45309" stroke-width="1.5"/>')
        L.append(f'<text x="{x+60}" y="{y+19}" text-anchor="middle" font-family="monospace" font-size="11" fill="#78350f" font-weight="bold">{esc(lab)}</text>')

    # K,V labels
    L.append(f'<text x="{x+15}" y="{qby-5}" font-family="monospace" font-size="10" fill="#b45309">{esc("K/V blocks [4×64×128]")}</text>')

    # ---------- SRAM region: Three Q block iterations ----------
    sram_y = 210
    L.append(f'<text x="20" y="{sram_y-10}" font-family="sans-serif" font-size="14" fill="#1e40af" font-weight="bold">{esc("SRAM: 每个 Q Block 的外层循环")}</text>')

    col_w = 290
    col_gap = 15
    col_h = 410
    for qi in range(3):
        cx = 20 + qi * (col_w + col_gap)
        cy = sram_y

        # Column background
        L.append(f'<rect x="{cx}" y="{cy}" width="{col_w}" height="{col_h}" rx="6" fill="{q_colors[qi]}" opacity="0.06" stroke="{q_colors[qi]}" stroke-width="1" stroke-dasharray="4,2"/>')
        L.append(f'<text x="{cx+col_w//2}" y="{cy+20}" text-anchor="middle" font-family="monospace" font-size="12" fill="{q_colors[qi]}" font-weight="bold">{esc(f"Q Block {qi} (q_start={qi*64})")}</text>')

        # Inner loop header
        L.append(f'<text x="{cx+10}" y="{cy+45}" font-family="monospace" font-size="10" fill="#64748b">{esc("for kv in [0,1,2,3]:")}</text>')

        # State initial values
        init_y = cy + 55
        L.append(f'<text x="{cx+10}" y="{init_y}" font-family="monospace" font-size="10" fill="#64748b">{esc(f"m = -inf, l = 0")}</text>')
        L.append(f'<text x="{cx+10}" y="{init_y+14}" font-family="monospace" font-size="10" fill="#64748b">{esc(f"O_acc = [64,128] = 0")}</text>')

        # Show 4 KV iterations as mini-boxes
        for kvi in range(4):
            ki_y = init_y + 28 + kvi * 80
            # KV iteration box
            L.append(f'<rect x="{cx+5}" y="{ki_y}" width="{col_w-10}" height="74" rx="3" fill="white" stroke="#e2e8f0" stroke-width="1"/>')

            # Step labels
            L.append(f'<text x="{cx+12}" y="{ki_y+14}" font-family="monospace" font-size="9" fill="#64748b">{esc(f"kv={kvi}:")}</text>')

            # S computation
            L.append(f'<text x="{cx+55}" y="{ki_y+14}" font-family="monospace" font-size="9" fill="#3b82f6">{esc(f"S = Q{qi} @ K{kvi}^T [64,64]")}</text>')

            # m_new
            L.append(f'<text x="{cx+55}" y="{ki_y+27}" font-family="monospace" font-size="9" fill="#059669">{esc(f"m_new = max(m, row_max(S))")}</text>')

            # P and correction
            L.append(f'<text x="{cx+55}" y="{ki_y+40}" font-family="monospace" font-size="9" fill="#7c3aed">{esc(f"P = exp(S - m_new)")}</text>')

            # O update
            L.append(f'<text x="{cx+55}" y="{ki_y+53}" font-family="monospace" font-size="9" fill="#dc2626">{esc(f"corr = exp(m - m_new), l += ...")}</text>')

            # Highlight correction convergence
            if kvi >= 2:
                L.append(f'<rect x="{cx+5}" y="{ki_y}" width="{col_w-10}" height="74" rx="3" fill="#f0fdf4" stroke="#22c55e" stroke-width="1.5"/>')
                L.append(f'<text x="{cx+col_w-30}" y="{ki_y+14}" font-family="monospace" font-size="8" fill="#22c55e">{esc("corr=1✓")}</text>')

        # Final normalization
        fin_y = init_y + 28 + 4 * 80 + 4
        L.append(f'<rect x="{cx+5}" y="{fin_y}" width="{col_w-10}" height="24" rx="4" fill="{q_colors[qi]}" opacity="0.15" stroke="{q_colors[qi]}" stroke-width="1.5"/>')
        L.append(f'<text x="{cx+col_w//2}" y="{fin_y+17}" text-anchor="middle" font-family="monospace" font-size="10" fill="{q_colors[qi]}" font-weight="bold">{esc(f"O{qi} = O_acc / l → Write HBM")}</text>')

    # ---------- Bottom: Memory analysis ----------
    mem_y = sram_y + col_h + 20
    L.append(f'<text x="20" y="{mem_y}" font-family="sans-serif" font-size="13" fill="#1e40af" font-weight="bold">{esc("SRAM 用量分析 (BLOCK_Q=64, BLOCK_KV=64, HEAD_DIM=128, fp16)")}</text>')

    mem_items = [("Q_block", "16 KB", "#dbeafe", "#2563eb"),
                 ("K_block", "16 KB", "#d1fae5", "#059669"),
                 ("V_block", "16 KB", "#fce4ec", "#dc2626"),
                 ("S (fp32)", "16 KB", "#fef3c7", "#b45309"),
                 ("P (fp32)", "16 KB", "#ede9fe", "#7c3aed"),
                 ("O_acc (fp32)", "32 KB", "#fce7f3", "#db2777")]

    bar_y = mem_y + 20
    bar_h = 18
    bar_x = 20
    max_bar_w = 500
    for i, (name, size, bg, fg) in enumerate(mem_items):
        y = bar_y + i * (bar_h + 3)
        kb = int(size.replace(" KB", ""))
        bw = int(kb / 40 * max_bar_w)
        L.append(f'<rect x="{bar_x}" y="{y}" width="{bw}" height="{bar_h}" rx="3" fill="{bg}" stroke="{fg}" stroke-width="1"/>')
        L.append(f'<text x="{bar_x + bw + 8}" y="{y + 13}" font-family="monospace" font-size="10" fill="#334155">{esc(f"{name}: {size}")}</text>')

    total_y = bar_y + len(mem_items) * (bar_h + 3) + 5
    total_w = int(112 / 40 * max_bar_w)
    L.append(f'<rect x="{bar_x}" y="{total_y}" width="{total_w}" height="{bar_h}" rx="3" fill="#1e40af" stroke="#1e3a5f" stroke-width="2"/>')
    L.append(f'<text x="{bar_x + total_w + 8}" y="{total_y + 13}" font-family="monospace" font-size="11" fill="#1e40af" font-weight="bold">{esc("Total: 112 KB")}</text>')

    # H100 limit
    L.append(f'<line x1="{bar_x}" y1="{total_y + 28}" x2="{bar_x + max_bar_w}" y2="{total_y + 28}" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="6,3"/>')
    L.append(f'<text x="{bar_x + max_bar_w + 5}" y="{total_y + 32}" font-family="monospace" font-size="10" fill="#ef4444" font-weight="bold">{esc("H100 L1/SMEM: 228 KB")}</text>')

    # Key insight
    ins_y = total_y + 55
    L.append(f'<text x="20" y="{ins_y}" font-family="sans-serif" font-size="12" fill="#dc2626" font-weight="bold">{esc("关键约束: BLOCK_Q × BLOCK_KV × HEAD_DIM 不能超过 L1 cache 大小 (H100: 228KB)")}</text>')
    L.append(f'<text x="20" y="{ins_y+18}" font-family="sans-serif" font-size="11" fill="#64748b">{esc("128×128×128×2B = 4MB ≫ 228KB → 必须用更小的 tile → BLOCK_Q=64, BLOCK_KV=64")}</text>')

    L.append('</svg>')
    return '\n'.join(L)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_02_tiled_attention.py <output_path_without_ext>")
        sys.exit(1)

    base = sys.argv[1]
    svg_path = base + '.svg'
    png_path = base + '.png'
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)

    svg = build_svg()
    with open(svg_path, 'w') as f:
        f.write(svg)
    print(f"SVG: {svg_path} ({len(svg)} bytes)")

    r = subprocess.run(['xmllint', '--noout', svg_path], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"XML INVALID: {r.stderr[:300]}")
        sys.exit(1)
    print("xmllint: VALID")

    val_script = Path(__file__).parent.parent.parent.parent.parent.parent / '.claude/skills/svg-diagram/scripts/validate_svg.py'
    r2 = subprocess.run([sys.executable, str(val_script), svg_path], capture_output=True, text=True)
    if r2.returncode != 0:
        print(r2.stdout)
        sys.exit(1)
    print("semantics: CLEAN")

    r3 = subprocess.run(['convert', '-density', '150', svg_path, png_path], capture_output=True, text=True)
    if r3.returncode != 0:
        print(f"PNG failed: {r3.stderr[:300]}")
        sys.exit(1)
    print(f"PNG: {png_path} ({os.path.getsize(png_path)//1024} KB)")
