#!/usr/bin/env python3
"""Generate Diagram 1: Self-Attention computation pipeline."""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

def build_svg():
    w, h = 920, 620
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>')
    L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>')
    L.append('<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#3b82f6"/></marker>')
    L.append('</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="#f8fafc"/>')

    # Title
    L.append(f'<text x="{w//2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#1e3a5f" font-weight="bold">{esc("Self-Attention 计算管线")}</text>')
    L.append(f'<text x="{w//2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#64748b">{esc("从输入序列到注意力输出 — 五步全流程")}</text>')

    # ---------- Step 1: Input + QKV Projections ----------
    y0 = 78
    # Input box
    L.append(f'<rect x="20" y="{y0}" width="140" height="48" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>')
    L.append(f'<text x="90" y="{y0+20}" text-anchor="middle" font-family="monospace" font-size="13" fill="#1e40af" font-weight="bold">{esc("Input X")}</text>')
    L.append(f'<text x="90" y="{y0+38}" text-anchor="middle" font-family="monospace" font-size="11" fill="#3b82f6">{esc("[3, d_model]")}</text>')

    # Three projection arrows
    proj_colors = [('#2563eb', 'W^Q', '[3, d_k]'), ('#059669', 'W^K', '[3, d_k]'), ('#dc2626', 'W^V', '[3, d_k]')]
    qkv_y = [y0 - 28, y0 + 12, y0 + 56]
    qkv_boxes = []
    for i, (color, label, shape) in enumerate(proj_colors):
        cy = qkv_y[i]
        L.append(f'<line x1="160" y1="{y0+24}" x2="210" y2="{cy+24}" stroke="{color}" stroke-width="1.5" marker-end="url(#a)"/>')
        bx = 220
        L.append(f'<rect x="{bx}" y="{cy}" width="100" height="48" rx="6" fill="white" stroke="{color}" stroke-width="2"/>')
        L.append(f'<text x="{bx+50}" y="{cy+20}" text-anchor="middle" font-family="monospace" font-size="13" fill="{color}" font-weight="bold">{esc(label)}</text>')
        L.append(f'<text x="{bx+50}" y="{cy+38}" text-anchor="middle" font-family="monospace" font-size="11" fill="#64748b">{esc(shape)}</text>')
        qkv_boxes.append((bx+100, cy+24, color, label))

    # ---------- Step 2: Q @ K^T ----------
    y1 = 170
    L.append(f'<rect x="20" y="{y1}" width="880" height="130" rx="4" fill="#eff6ff" stroke="#93c5fd" stroke-width="1" stroke-dasharray="6,3"/>')
    L.append(f'<text x="35" y="{y1+20}" font-family="sans-serif" font-size="14" fill="#1e40af" font-weight="bold">{esc("Step 2: 计算注意力分数 S = Q @ K^T")}</text>')

    # Q matrix
    qx, qy = 40, y1 + 35
    L.append(f'<rect x="{qx}" y="{qy}" width="80" height="80" rx="3" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>')
    L.append(f'<text x="{qx+40}" y="{qy-5}" text-anchor="middle" font-family="monospace" font-size="11" fill="#2563eb">{esc("Q [3×4]")}</text>')
    L.append(f'<text x="{qx+40}" y="{qy+45}" text-anchor="middle" font-family="monospace" font-size="9" fill="#64748b">{esc("q₀·")}</text>')
    L.append(f'<text x="{qx+40}" y="{qy+58}" text-anchor="middle" font-family="monospace" font-size="9" fill="#64748b">{esc("q₁·")}</text>')
    L.append(f'<text x="{qx+40}" y="{qy+71}" text-anchor="middle" font-family="monospace" font-size="9" fill="#64748b">{esc("q₂·")}</text>')

    # × symbol
    L.append(f'<text x="{qx+95}" y="{qy+45}" font-family="monospace" font-size="16" fill="#64748b">{esc("@")}</text>')

    # K^T matrix
    kx = qx + 115
    L.append(f'<rect x="{kx}" y="{qy}" width="80" height="80" rx="3" fill="#d1fae5" stroke="#059669" stroke-width="1.5"/>')
    L.append(f'<text x="{kx+40}" y="{qy-5}" text-anchor="middle" font-family="monospace" font-size="11" fill="#059669">{esc("K^T [4×3]")}</text>')
    L.append(f'<text x="{kx+15}" y="{qy+30}" font-family="monospace" font-size="9" fill="#64748b">{esc("k₀")}</text>')
    L.append(f'<text x="{kx+42}" y="{qy+30}" font-family="monospace" font-size="9" fill="#64748b">{esc("k₁")}</text>')
    L.append(f'<text x="{kx+69}" y="{qy+30}" font-family="monospace" font-size="9" fill="#64748b">{esc("k₂")}</text>')
    for r in range(4):
        L.append(f'<text x="{kx+5}" y="{qy+45+r*13}" font-family="monospace" font-size="9" fill="#64748b">{esc(f"d{r}")}</text>')

    # = arrow
    L.append(f'<line x1="{kx+85}" y1="{qy+40}" x2="{kx+115}" y2="{qy+40}" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

    # Scores matrix
    sx = kx + 125
    L.append(f'<rect x="{sx}" y="{qy}" width="120" height="80" rx="3" fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
    L.append(f'<text x="{sx+60}" y="{qy-5}" text-anchor="middle" font-family="monospace" font-size="11" fill="#b45309">{esc("Scores S [3×3]")}</text>')

    # Show actual dots
    dots = [("q₀·k₀=0.8", "q₀·k₁=0.3", "q₀·k₂=0.1"),
            ("q₁·k₀=0.5", "q₁·k₁=1.2", "q₁·k₂=0.4"),
            ("q₂·k₀=0.2", "q₂·k₁=0.7", "q₂·k₂=1.5")]
    cell_w, cell_h = 38, 18
    for ri, row in enumerate(dots):
        for ci, d in enumerate(row):
            cx = sx + 3 + ci * cell_w
            cy = qy + 3 + ri * cell_h
            fill = "#fef9c3" if ri >= ci else "#fee2e2"
            L.append(f'<rect x="{cx}" y="{cy}" width="{cell_w}" height="{cell_h}" rx="2" fill="{fill}" stroke="#d4d4d4" stroke-width="0.5"/>')
            L.append(f'<text x="{cx+cell_w//2}" y="{cy+13}" text-anchor="middle" font-family="monospace" font-size="8" fill="#78350f">{esc(d)}</text>')

    # ---------- Step 3: Scale ----------
    y2 = 330
    L.append(f'<line x1="{sx+60}" y1="{qy+80}" x2="{sx+60}" y2="{y2}" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

    scale_cx = sx + 60
    L.append(f'<rect x="{scale_cx-100}" y="{y2}" width="200" height="40" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>')
    L.append(f'<text x="{scale_cx}" y="{y2+25}" text-anchor="middle" font-family="monospace" font-size="13" fill="#5b21b6" font-weight="bold">{esc("÷ √d_k  (scale)")}</text>')

    # ---------- Step 4: Softmax ----------
    y3 = 400
    L.append(f'<line x1="{scale_cx}" y1="{y2+40}" x2="{scale_cx}" y2="{y3}" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

    soft_cx = scale_cx
    L.append(f'<rect x="{soft_cx-100}" y="{y3}" width="200" height="40" rx="6" fill="#fce7f3" stroke="#db2777" stroke-width="2"/>')
    L.append(f'<text x="{soft_cx}" y="{y3+25}" text-anchor="middle" font-family="monospace" font-size="13" fill="#9d174d" font-weight="bold">{esc("Softmax (per row)")}</text>')

    # Attention weights result
    aw_y = y3 + 65
    L.append(f'<line x1="{soft_cx}" y1="{y3+40}" x2="{soft_cx}" y2="{aw_y}" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

    awx = soft_cx - 110
    L.append(f'<rect x="{awx}" y="{aw_y}" width="220" height="60" rx="3" fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
    L.append(f'<text x="{awx+110}" y="{aw_y-5}" text-anchor="middle" font-family="monospace" font-size="11" fill="#b45309">{esc("Attention Weights A [3×3]")}</text>')

    # Show sample weights
    weights = [("0.52", "0.31", "0.17"), ("0.28", "0.55", "0.17"), ("0.15", "0.25", "0.60")]
    for ri, row in enumerate(weights):
        for ci, w in enumerate(row):
            cx2 = awx + 5 + ci * 70
            cy2 = aw_y + 20 + ri * 13
            alpha = float(w)
            bg = f"rgba(37,99,235,{alpha*0.5:.2f})"
            L.append(f'<text x="{cx2}" y="{cy2}" font-family="monospace" font-size="11" fill="#1e40af">{esc(w)}</text>')
    L.append(f'<text x="{awx+110}" y="{aw_y+58}" text-anchor="middle" font-family="monospace" font-size="9" fill="#64748b">{esc("每行之和 = 1.0")}</text>')

    # ---------- Step 5: @ V ----------
    y5 = aw_y + 90
    L.append(f'<line x1="{awx+110}" y1="{aw_y+60}" x2="{awx+110}" y2="{y5-2}" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

    # V matrix display
    vx = awx - 30
    vy = y5
    L.append(f'<rect x="{vx}" y="{vy}" width="80" height="50" rx="3" fill="#fce4ec" stroke="#dc2626" stroke-width="1.5"/>')
    L.append(f'<text x="{vx+40}" y="{vy-5}" text-anchor="middle" font-family="monospace" font-size="11" fill="#dc2626">{esc("V [3×4]")}</text>')
    L.append(f'<text x="{vx+40}" y="{vy+32}" text-anchor="middle" font-family="monospace" font-size="9" fill="#64748b">{esc("v₀, v₁, v₂")}</text>')

    # @ symbol
    L.append(f'<text x="{awx+110}" y="{vy+30}" text-anchor="middle" font-family="monospace" font-size="16" fill="#64748b">{esc("×")}</text>')

    # Output box
    ox = awx + 180
    oy = vy
    L.append(f'<line x1="{awx+110}" y1="{vy+50}" x2="{awx+110}" y2="{vy+50}" stroke="#64748b" stroke-width="0"/>')
    # arrow from combined operation
    L.append(f'<line x1="{awx+140}" y1="{vy+25}" x2="{ox-10}" y2="{vy+25}" stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')

    L.append(f'<rect x="{ox}" y="{oy}" width="140" height="50" rx="6" fill="#dbeafe" stroke="#1e40af" stroke-width="2.5"/>')
    L.append(f'<text x="{ox+70}" y="{oy-5}" text-anchor="middle" font-family="monospace" font-size="11" fill="#1e40af">{esc("Output O [3×d_k]")}</text>')
    L.append(f'<text x="{ox+70}" y="{oy+32}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1e3a5f" font-weight="bold">{esc("← 注意力输出!")}</text>')

    # Bottom legend
    ly = vy + 75
    L.append(f'<text x="20" y="{ly}" font-family="sans-serif" font-size="11" fill="#64748b">{esc("核心公式: Attention(Q,K,V) = softmax(QK^T / √d_k) V")}</text>')
    L.append(f'<text x="20" y="{ly+18}" font-family="sans-serif" font-size="11" fill="#64748b">{esc("Q: 我要找什么 · K: 我能提供什么 · V: 我的实际内容 · √d_k: 防止方差爆炸")}</text>')

    L.append('</svg>')
    return '\n'.join(L)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_01_attention_pipeline.py <output_path_without_ext>")
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
