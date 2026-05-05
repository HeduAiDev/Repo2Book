#!/usr/bin/env python3
"""
FlashAttention Tiling Diagram — Chapter 01
Shows Q→KV block iteration with online softmax, SRAM/HBM boundary.
"""
import sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

def validate(svg_path):
    r = subprocess.run(['xmllint', '--noout', svg_path], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"XML INVALID: {r.stderr[:300]}")
        return False
    return True

def convert_to_png(svg_path, png_path):
    r = subprocess.run(['convert', '-density', '150', svg_path, png_path], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"PNG conversion failed: {r.stderr[:300]}")
        return False
    return True

BLOCK_Q = 4
BLOCK_KV = 4
SEQ = 12
HEAD_DIM = 8
NUM_Q = SEQ // BLOCK_Q   # 3
NUM_KV = SEQ // BLOCK_KV  # 3

def build_svg():
    W, H = 1140, 820
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']

    # Arrowhead markers
    L.append('<defs>')
    L.append('<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#3b82f6"/></marker>')
    L.append('<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>')
    L.append('<marker id="ap" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#9333ea"/></marker>')
    L.append('<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>')
    L.append('<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#ea580c"/></marker>')
    L.append('<marker id="ag2" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>')
    L.append('</defs>')

    L.append(f'<rect width="{W}" height="{H}" fill="#f8fafc"/>')

    # --- Title ---
    L.append(f'<text x="570" y="36" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#1e293b" font-weight="bold">{esc("FlashAttention Tiling — BLOCK_Q=4, BLOCK_KV=4 (演示用 L=12)")}</text>')
    L.append(f'<text x="570" y="58" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#64748b">{esc("每个 Q block 加载一次，遍历所有 KV block；S [4x4] 只在 SRAM 存在，永不写回 HBM")}</text>')

    # --- Layout constants ---
    q_left = 60
    q_w = 100
    q_block_h = 60
    q_h = NUM_Q * q_block_h
    q_top = 110

    kv_left = 270
    kv_w = 100
    kv_block_h = 50
    kv_h = NUM_KV * kv_block_h
    kv_top = 110

    v_left = 390
    v_w = 100

    sram_left = 550
    sram_right = 790

    o_left = 870
    o_w = 100
    o_top = 110

    # --- HBM region label ---
    L.append(f'<rect x="30" y="85" width="1030" height="350" rx="6" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4"/>')
    L.append(f'<text x="1045" y="100" font-family="sans-serif" font-size="11" fill="#94a3b8">{esc("HBM (GPU 显存)")}</text>')

    # --- SRAM region (around the S matrices) ---
    sram_y = 330
    sram_h = 180
    L.append(f'<rect x="{sram_left - 15}" y="{sram_y - 5}" width="{sram_right - sram_left + 30}" height="{sram_h + 10}" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="2" stroke-dasharray="8,4"/>')
    L.append(f'<text x="{sram_left}" y="{sram_y + 16}" font-family="sans-serif" font-size="12" fill="#dc2626" font-weight="bold">{esc("SRAM (片上缓存 ~112KB) — S 矩阵永不写回 HBM")}</text>')

    # --- Q Matrix ---
    q_title_y = q_top - 10
    L.append(f'<text x="{q_left + q_w//2}" y="{q_title_y}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#1e40af" font-weight="bold">{esc("Q [12 x 8]")}</text>')

    for i in range(NUM_Q):
        by = q_top + i * q_block_h
        color = '#3b82f6' if i == 0 else '#93c5fd'
        fill = color if i == 0 else '#eff6ff'
        text_c = 'white' if i == 0 else '#1e40af'
        L.append(f'<rect x="{q_left}" y="{by}" width="{q_w}" height="{q_block_h - 4}" rx="4" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        L.append(f'<text x="{q_left + q_w//2}" y="{by + q_block_h//2 + 5}" text-anchor="middle" font-family="monospace" font-size="13" fill="{text_c}" font-weight="bold">{esc(f"Q{i} [4x8]")}</text>')

    # --- K Matrix ---
    kv_title_y = kv_top - 10
    L.append(f'<text x="{kv_left + kv_w//2}" y="{kv_title_y}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#166534" font-weight="bold">{esc("K [12 x 8]")}</text>')

    for i in range(NUM_KV):
        by = kv_top + i * kv_block_h
        shade = 0.3 + 0.3 * i
        color = '#16a34a'
        L.append(f'<rect x="{kv_left}" y="{by}" width="{kv_w}" height="{kv_block_h - 4}" rx="4" fill="#f0fdf4" stroke="{color}" stroke-width="2"/>')
        L.append(f'<text x="{kv_left + kv_w//2}" y="{by + kv_block_h//2 + 5}" text-anchor="middle" font-family="monospace" font-size="13" fill="{color}" font-weight="bold">{esc(f"K{i} [4x8]")}</text>')

    # --- V Matrix ---
    L.append(f'<text x="{v_left + v_w//2}" y="{kv_title_y}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#6b21a8" font-weight="bold">{esc("V [12 x 8]")}</text>')

    for i in range(NUM_KV):
        by = kv_top + i * kv_block_h
        color = '#9333ea'
        L.append(f'<rect x="{v_left}" y="{by}" width="{v_w}" height="{kv_block_h - 4}" rx="4" fill="#faf5ff" stroke="{color}" stroke-width="2"/>')
        L.append(f'<text x="{v_left + v_w//2}" y="{by + kv_block_h//2 + 5}" text-anchor="middle" font-family="monospace" font-size="13" fill="{color}" font-weight="bold">{esc(f"V{i} [4x8]")}</text>')

    # --- O Matrix ---
    L.append(f'<text x="{o_left + o_w//2}" y="{q_title_y}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#9a3412" font-weight="bold">{esc("O [12 x 8]")}</text>')

    for i in range(NUM_Q):
        by = o_top + i * q_block_h
        color = '#ea580c'
        fill = '#fff7ed'
        L.append(f'<rect x="{o_left}" y="{by}" width="{o_w}" height="{q_block_h - 4}" rx="4" fill="{fill}" stroke="{color}" stroke-width="2" stroke-dasharray="4,2"/>')
        L.append(f'<text x="{o_left + o_w//2}" y="{by + q_block_h//2 + 5}" text-anchor="middle" font-family="monospace" font-size="13" fill="{color}" font-weight="bold">{esc(f"O{i} [4x8]")}</text>')

    # --- Arrows: Q0 → K0,V0 (iteration 1) ---
    q0_right = q_left + q_w
    q0_midy = q_top + q_block_h // 2

    k0_left = kv_left
    k0_midy = kv_top + kv_block_h // 2
    v0_midy = kv_top + kv_block_h // 2
    v0_right = v_left + v_w

    # Q0 → K0
    L.append(f'<path d="M{q0_right} {q0_midy} C{q0_right+40} {q0_midy}, {k0_left-40} {k0_midy}, {k0_left} {k0_midy}" stroke="#3b82f6" stroke-width="2.5" fill="none" marker-end="url(#ab)"/>')

    # K0 → SRAM block S₀
    s1_left = sram_left
    s1_top = sram_y + 30
    s1_w = 70
    s1_h = 40

    L.append(f'<rect x="{s1_left}" y="{s1_top}" width="{s1_w}" height="{s1_h}" rx="4" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>')
    L.append(f'<text x="{s1_left + s1_w//2}" y="{s1_top + s1_h//2 + 5}" text-anchor="middle" font-family="monospace" font-size="12" fill="#991b1b" font-weight="bold">{esc("S₀ [4x4]")}</text>')
    L.append(f'<text x="{s1_left + s1_w//2}" y="{s1_top + s1_h + 14}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#dc2626">{esc("Q0@K0^T")}</text>')

    k0_right = kv_left + kv_w
    L.append(f'<path d="M{k0_right} {k0_midy} C{k0_right+30} {k0_midy}, {s1_left-30} {s1_top+s1_h//2}, {s1_left} {s1_top+s1_h//2}" stroke="#16a34a" stroke-width="2" fill="none" marker-end="url(#ar)"/>')

    v0_right = v_left + v_w
    L.append(f'<path d="M{v0_right} {v0_midy} C{v0_right+30} {v0_midy}, {s1_left+s1_w-30} {s1_top+s1_h//2+10}, {s1_left+s1_w} {s1_top+s1_h//2+10}" stroke="#9333ea" stroke-width="2" fill="none" marker-end="url(#ar)"/>')
    L.append(f'<text x="{s1_left + s1_w + 5}" y="{s1_top + s1_h//2 + 6}" font-family="monospace" font-size="9" fill="#9333ea">{esc("V0")}</text>')

    # --- Iteration 2: Q0 → K1,V1 → S₁ ---
    k1_midy = kv_top + kv_block_h + kv_block_h // 2
    s2_left = s1_left + 90
    s2_top = s1_top
    s2_h = 40

    L.append(f'<rect x="{s2_left}" y="{s2_top}" width="{s1_w}" height="{s2_h}" rx="4" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>')
    L.append(f'<text x="{s2_left + s1_w//2}" y="{s2_top + s2_h//2 + 5}" text-anchor="middle" font-family="monospace" font-size="12" fill="#991b1b" font-weight="bold">{esc("S₁ [4x4]")}</text>')
    L.append(f'<text x="{s2_left + s1_w//2}" y="{s2_top + s2_h + 14}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#dc2626">{esc("Q0@K1^T")}</text>')

    L.append(f'<path d="M{k0_right} {k1_midy} C{k0_right+30} {k1_midy}, {s2_left-30} {s2_top+s2_h//2}, {s2_left} {s2_top+s2_h//2}" stroke="#16a34a" stroke-width="2" fill="none" marker-end="url(#ar)"/>')

    v1_midy = kv_top + kv_block_h + kv_block_h // 2
    L.append(f'<path d="M{v0_right} {v1_midy} C{v0_right+30} {v1_midy}, {s2_left+s1_w-30} {s2_top+s2_h//2+10}, {s2_left+s1_w} {s2_top+s2_h//2+10}" stroke="#9333ea" stroke-width="2" fill="none" marker-end="url(#ar)"/>')
    L.append(f'<text x="{s2_left + s1_w + 5}" y="{s2_top + s2_h//2 + 6}" font-family="monospace" font-size="9" fill="#9333ea">{esc("V1")}</text>')

    # --- Iteration 3: Q0 → K2,V2 → S₂ ---
    k2_midy = kv_top + 2*kv_block_h + kv_block_h // 2
    s3_left = s2_left + 90
    s3_top = s1_top

    L.append(f'<rect x="{s3_left}" y="{s3_top}" width="{s1_w}" height="{s2_h}" rx="4" fill="#fecaca" stroke="#dc2626" stroke-width="2"/>')
    L.append(f'<text x="{s3_left + s1_w//2}" y="{s3_top + s2_h//2 + 5}" text-anchor="middle" font-family="monospace" font-size="12" fill="#991b1b" font-weight="bold">{esc("S₂ [4x4]")}</text>')
    L.append(f'<text x="{s3_left + s1_w//2}" y="{s3_top + s2_h + 14}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#dc2626">{esc("Q0@K2^T")}</text>')

    L.append(f'<path d="M{k0_right} {k2_midy} C{k0_right+30} {k2_midy}, {s3_left-30} {s3_top+s2_h//2}, {s3_left} {s3_top+s2_h//2}" stroke="#16a34a" stroke-width="2" fill="none" marker-end="url(#ar)"/>')

    v2_midy = kv_top + 2*kv_block_h + kv_block_h // 2
    L.append(f'<path d="M{v0_right} {v2_midy} C{v0_right+30} {v2_midy}, {s3_left+s1_w-30} {s3_top+s2_h//2+10}, {s3_left+s1_w} {s3_top+s2_h//2+10}" stroke="#9333ea" stroke-width="2" fill="none" marker-end="url(#ar)"/>')

    # --- Online Softmax annotation ---
    os_y = s1_top + s2_h + 40
    L.append(f'<text x="{sram_left + (sram_right-sram_left)//2}" y="{os_y}" text-anchor="middle" font-family="monospace" font-size="11" fill="#991b1b" font-weight="bold">{esc("Online Softmax: m, l, O_acc 在三次迭代中逐步更新")}</text>')
    L.append(f'<text x="{sram_left + (sram_right-sram_left)//2}" y="{os_y + 18}" text-anchor="middle" font-family="monospace" font-size="10" fill="#64748b">{esc("O_new = correction x O_old + P @ V_block     correction = exp(m_old - m_new)")}</text>')

    # --- Arrow: SRAM → O accumulator ---
    o0_midy = o_top + q_block_h // 2
    sram_across_mid = s3_left + s1_w
    acc_arrow_start = sram_across_mid + 30
    L.append(f'<line x1="{acc_arrow_start}" y1="{s3_top + s2_h//2}" x2="{o_left}" y2="{o0_midy}" stroke="#ea580c" stroke-width="2.5" marker-end="url(#ao)"/>')
    L.append(f'<text x="{acc_arrow_start + 10}" y="{s3_top + s2_h//2 - 8}" font-family="monospace" font-size="9" fill="#ea580c">{esc("O_acc/l")}</text>')

    # --- Outer loop annotation (bottom) ---
    loop_y = 550
    L.append(f'<rect x="60" y="{loop_y}" width="1020" height="220" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="6,4"/>')
    L.append(f'<text x="80" y="{loop_y + 22}" font-family="sans-serif" font-size="13" fill="#1e40af" font-weight="bold">{esc("外循环 (Outer Loop): 三个 Q block 各做一次")}</text>')

    # Show 3 outer iterations
    for outer_i in range(NUM_Q):
        bx = 120 + outer_i * 300
        by = loop_y + 40
        bw = 270
        bh = 165

        L.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="6" fill="white" stroke="#93c5fd" stroke-width="1.5"/>')

        q_idx = outer_i
        L.append(f'<text x="{bx + bw//2}" y="{by + 20}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#1e40af" font-weight="bold">{esc(f"Q block {q_idx}: rows {q_idx*4}-{(q_idx+1)*4-1}")}</text>')

        # Inner loop description
        inner_y = by + 38
        L.append(f'<text x="{bx + 10}" y="{inner_y}" font-family="monospace" font-size="10" fill="#334155">{esc(f"Initialize: m = -inf,  l = 0,  O_acc = 0")}</text>')
        L.append(f'<text x="{bx + 10}" y="{inner_y + 20}" font-family="monospace" font-size="10" fill="#334155">{esc(f"for kv in [0,1,2]:")}</text>')

        # Three inner iterations
        arrow_char = '→'
        for kv_i in range(NUM_KV):
            li_y = inner_y + 38 + kv_i * 28
            L.append(f'<text x="{bx + 22}" y="{li_y}" font-family="monospace" font-size="10" fill="#64748b">'
                     f'{esc(f"  kv={kv_i}: load K{kv_i},V{kv_i} {arrow_char} S=Q{q_idx}@K{kv_i}^T {arrow_char} online softmax {arrow_char} O_acc += P@V{kv_i}")}</text>')

        # Final
        fin_y = inner_y + 38 + 3*28 + 8
        L.append(f'<text x="{bx + 10}" y="{fin_y}" font-family="monospace" font-size="10" fill="#ea580c" font-weight="bold">{esc(f"O{q_idx} = O_acc / l   ← 最终归一化，写回 HBM")}</text>')

    # HBM bandwidth annotation
    hbm_y = loop_y + 195
    L.append(f'<text x="570" y="{hbm_y}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">'
             f'{esc("HBM traffic: O(L·d) bytes  |  SRAM compute: O(L²·d) FLOPs  |  Attention matrix [L×L] NEVER stored")}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 gen_fa_tiling.py <output_path_without_ext>")
        sys.exit(1)

    base = sys.argv[1]
    svg_path = base + '.svg'
    png_path = base + '.png'

    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)

    svg = build_svg()
    with open(svg_path, 'w') as f:
        f.write(svg)
    print(f"SVG: {svg_path} ({len(svg)} bytes)")

    if not validate(svg_path):
        sys.exit(1)
    print("xmllint: VALID")

    val_script = Path(__file__).parent.parent.parent.parent / '.claude/skills/svg-diagram/scripts/validate_svg.py'
    if val_script.exists():
        r2 = subprocess.run([sys.executable, str(val_script), svg_path], capture_output=True, text=True)
        if r2.returncode != 0:
            print(r2.stdout)
            sys.exit(1)
        print("semantics: CLEAN")

    if not convert_to_png(svg_path, png_path):
        sys.exit(1)
    print(f"PNG: {png_path} ({os.path.getsize(png_path)//1024} KB)")
