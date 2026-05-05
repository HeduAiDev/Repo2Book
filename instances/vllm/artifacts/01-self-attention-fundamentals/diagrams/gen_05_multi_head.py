#!/usr/bin/env python3
"""Generate Diagram: Multi-Head Attention architecture — h parallel heads."""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

def build_svg():
    w, h = 940, 560
    num_heads = 4
    head_colors = ['#2563eb', '#059669', '#dc2626', '#7c3aed']
    # y positions for the 3 QKV boxes
    qkv_ys = [85, 115, 145]

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>')
    L.append('<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>')
    L.append('</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="#f8fafc"/>')

    # Title
    L.append(f'<text x="{w//2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#1e3a5f" font-weight="bold">{esc("Multi-Head Attention: h 个并行的注意力")}</text>')
    L.append(f'<text x="{w//2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#64748b">{esc("每个头在独立的 d_k 维子空间里计算 Attention，所有头拼接后通过 W^O 投影回全空间")}</text>')

    # ===== Left: Input X =====
    inp_x, inp_y, inp_w, inp_h = 25, 100, 120, 48
    L.append(f'<rect x="{inp_x}" y="{inp_y}" width="{inp_w}" height="{inp_h}" rx="8" fill="#dbeafe" stroke="#3b82f6" stroke-width="2.5"/>')
    L.append(f'<text x="{inp_x+inp_w//2}" y="{inp_y+20}" text-anchor="middle" font-family="monospace" font-size="14" fill="#1e40af" font-weight="bold">{esc("Input X")}</text>')
    L.append(f'<text x="{inp_x+inp_w//2}" y="{inp_y+38}" text-anchor="middle" font-family="monospace" font-size="10" fill="#3b82f6">{esc("[B, L, d_model]")}</text>')

    # ===== QKV Projections (vertical stack) =====
    proj_x = inp_x + inp_w + 35
    proj_w, proj_h = 90, 32
    proj_names = [("W¹", "#2563eb"), ("Wᴏ", "#059669"), ("Wᵛ", "#dc2626")]

    # Dashed box around projections
    L.append(f'<rect x="{proj_x-8}" y="72" width="{proj_w+16}" height="140" rx="4" fill="#eff6ff" stroke="#93c5fd" stroke-width="1" stroke-dasharray="6,3"/>')
    L.append(f'<text x="{proj_x+proj_w//2}" y="84" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#1e40af" font-weight="bold">{esc("QKV 投影")}</text>')

    for pi, (pname, pcolor) in enumerate(proj_names):
        py = qkv_ys[pi]
        L.append(f'<rect x="{proj_x}" y="{py}" width="{proj_w}" height="{proj_h}" rx="5" fill="white" stroke="{pcolor}" stroke-width="2"/>')
        L.append(f'<text x="{proj_x+proj_w//2}" y="{py+21}" text-anchor="middle" font-family="monospace" font-size="13" fill="{pcolor}" font-weight="bold">{esc(pname)}</text>')

    # Arrow from input → three projections (branching)
    branch_x = inp_x + inp_w + 10
    L.append(f'<line x1="{inp_x+inp_w}" y1="{inp_y+inp_h//2}" x2="{branch_x}" y2="{inp_y+inp_h//2}" stroke="#64748b" stroke-width="1.5"/>')
    L.append(f'<line x1="{branch_x}" y1="{qkv_ys[0]+proj_h//2}" x2="{branch_x}" y2="{qkv_ys[-1]+proj_h//2}" stroke="#64748b" stroke-width="1.5"/>')
    for pi in range(3):
        L.append(f'<line x1="{branch_x}" y1="{qkv_ys[pi]+proj_h//2}" x2="{proj_x}" y2="{qkv_ys[pi]+proj_h//2}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

    # ===== Per-Head Attention (horizontal after projections) =====
    head_x = proj_x + proj_w + 55
    head_w, head_h = 130, 38
    head_gap = 50  # vertical gap between heads
    head_tops = [95 + i * head_gap for i in range(num_heads)]

    L.append(f'<text x="{head_x-10}" y="74" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#64748b">{esc("reshape → h heads")}</text>')

    head_rights = []
    head_mids = []
    for hi in range(num_heads):
        hy = head_tops[hi]
        hc = head_colors[hi]
        L.append(f'<rect x="{head_x}" y="{hy}" width="{head_w}" height="{head_h}" rx="6" fill="white" stroke="{hc}" stroke-width="2"/>')
        L.append(f'<text x="{head_x+head_w//2}" y="{hy+14}" text-anchor="middle" font-family="monospace" font-size="12" fill="{hc}" font-weight="bold">{esc(f"Head {hi+1}")}</text>')
        L.append(f'<text x="{head_x+head_w//2}" y="{hy+31}" text-anchor="middle" font-family="monospace" font-size="9" fill="#64748b">{esc("QKV_i → Attention → hd_i")}</text>')
        head_rights.append(head_x + head_w)
        head_mids.append(hy + head_h//2)

    # Dashed arrows from projections to each head (show data flow splitting)
    for hi in range(num_heads):
        hc = head_colors[hi]
        # From each QKV projection to this head
        for pi in range(3):
            L.append(f'<line x1="{proj_x+proj_w}" y1="{qkv_ys[pi]+proj_h//2}" x2="{head_x}" y2="{head_mids[hi]}" stroke="{hc}" stroke-width="0.8" stroke-opacity="0.25"/>')

    # ===== Concat section =====
    concat_x = head_x + head_w + 30
    concat_w = 28
    first_y = head_tops[0]
    last_y_bottom = head_tops[-1] + head_h
    concat_h = last_y_bottom - first_y

    L.append(f'<rect x="{concat_x-5}" y="{first_y-8}" width="{concat_w+10}" height="{concat_h+16}" rx="4" fill="#fef3c7" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="6,3"/>')
    L.append(f'<text x="{concat_x+concat_w//2}" y="{first_y-12}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#b45309" font-weight="bold">{esc("Concat")}</text>')

    for hi in range(num_heads):
        L.append(f'<line x1="{head_rights[hi]}" y1="{head_mids[hi]}" x2="{concat_x}" y2="{head_mids[hi]}" stroke="{head_colors[hi]}" stroke-width="1.2" stroke-opacity="0.5" marker-end="url(#a)"/>')

    # ===== W^O Projection =====
    wo_x = concat_x + concat_w + 25
    wo_y = 170
    wo_w, wo_h = 110, 56
    mid_head_center = (head_tops[0] + head_tops[-1] + head_h) / 2

    L.append(f'<rect x="{wo_x}" y="{wo_y}" width="{wo_w}" height="{wo_h}" rx="8" fill="white" stroke="#b45309" stroke-width="2.5"/>')
    L.append(f'<text x="{wo_x+wo_w//2}" y="{wo_y+20}" text-anchor="middle" font-family="monospace" font-size="14" fill="#78350f" font-weight="bold">{esc("W⁰")}</text>')
    L.append(f'<text x="{wo_x+wo_w//2}" y="{wo_y+40}" text-anchor="middle" font-family="monospace" font-size="10" fill="#b45309">{esc("Output Proj")}</text>')

    # Connect concat to W^O (from middle of concat range to W^O)
    L.append(f'<line x1="{concat_x+concat_w}" y1="{mid_head_center}" x2="{wo_x}" y2="{wo_y+wo_h//2}" stroke="#b45309" stroke-width="1.5" marker-end="url(#a)"/>')

    # ===== Output =====
    out_x = wo_x + wo_w + 40
    out_y = wo_y
    out_w, out_h = 130, 56
    L.append(f'<line x1="{wo_x+wo_w}" y1="{out_y+out_h//2}" x2="{out_x}" y2="{out_y+out_h//2}" stroke="#1e40af" stroke-width="2" marker-end="url(#a)"/>')

    L.append(f'<rect x="{out_x}" y="{out_y}" width="{out_w}" height="{out_h}" rx="8" fill="#dbeafe" stroke="#1e40af" stroke-width="2.5"/>')
    L.append(f'<text x="{out_x+out_w//2}" y="{out_y+20}" text-anchor="middle" font-family="monospace" font-size="14" fill="#1e40af" font-weight="bold">{esc("Output")}</text>')
    L.append(f'<text x="{out_x+out_w//2}" y="{out_y+40}" text-anchor="middle" font-family="monospace" font-size="10" fill="#3b82f6">{esc("[B, L, d_model]")}</text>')

    # ===== Bottom: Formula box =====
    fy = last_y_bottom + 45
    L.append(f'<rect x="20" y="{fy}" width="{w-40}" height="120" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1"/>')

    L.append(f'<text x="40" y="{fy+22}" font-family="sans-serif" font-size="13" fill="#1e3a5f" font-weight="bold">{esc("核心公式")}</text>')
    L.append(f'<text x="40" y="{fy+45}" font-family="monospace" font-size="12" fill="#334155">{esc("head_i = Attention(X · W_i^Q,  X · W_i^K,  X · W_i^V)   ← 每个头独立计算")}</text>')
    L.append(f'<text x="40" y="{fy+67}" font-family="monospace" font-size="12" fill="#334155">{esc("MHA(X)   = Concat(head_1, head_2, ..., head_h) · W^O          ← 拼接后投影")}</text>')
    L.append(f'<text x="40" y="{fy+92}" font-family="sans-serif" font-size="11" fill="#64748b">{esc("d_k = d_model / h   (Llama 8B: 4096 / 32 = 128)  |  参数量 = 4×d_model²  (与 L 无关)")}</text>')
    L.append(f'<text x="40" y="{fy+112}" font-family="sans-serif" font-size="11" fill="#93c5fd">{esc("vLLM 源码: attention.py:L455-L460 — reshape 是 Multi-Head 切分子空间的关键一步")}</text>')

    L.append('</svg>')
    return '\n'.join(L)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_05_multi_head.py <output_path_without_ext>")
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

    val_script = Path('/mnt/e/Laboratory/vllm-from-scratch/.claude/skills/svg-diagram/scripts/validate_svg.py')
    if val_script.exists():
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
