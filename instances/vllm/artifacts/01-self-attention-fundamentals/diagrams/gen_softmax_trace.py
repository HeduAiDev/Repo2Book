#!/usr/bin/env python3
"""
Online Softmax State Evolution — Chapter 01
Numerical trace: m, l, O_acc evolution across KV block iterations.
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

def build_svg():
    W, H = 1000, 720
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="#f8fafc"/>')

    # Title
    L.append(f'<text x="500" y="32" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#1e293b" font-weight="bold">{esc("Online Softmax — Q block 0 的三轮 KV 迭代状态追踪")}</text>')
    L.append(f'<text x="500" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#64748b">{esc("数值演示：用具体数字追踪 running max (m), running sum (l), output accumulator (O_acc) 如何更新")}</text>')

    # --- Header row ---
    hdr_y = 80
    cols = [
        (50, 70, "变量"),
        (130, 180, "初始化\n(kv 迭代前)"),
        (320, 180, "kv=0 后\n(S₀ 处理完)"),
        (510, 180, "kv=1 后\n(S₁ 处理完)"),
        (700, 180, "kv=2 后\n(S₂ 处理完)"),
        (890, 80, "最终输出\n(O_block)"),
    ]

    # Column headers
    for cx, cw, label in cols:
        L.append(f'<rect x="{cx}" y="{hdr_y}" width="{cw}" height="44" rx="4" fill="#1e293b" stroke="#0f172a" stroke-width="1.5"/>')
        for j, line in enumerate(label.split('\n')):
            L.append(f'<text x="{cx + cw//2}" y="{hdr_y + 18 + j*15}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="white" font-weight="bold">{esc(line)}</text>')

    # Data rows
    table_top = hdr_y + 44
    row_h = 55

    rows_data = [
        {
            "label": "m (running max)\n每行当前最大的\nlogit值",
            "color": "#dc2626",
            "label_color": "#991b1b",
            "fill": "#fef2f2",
            "init": "[-inf, -inf,\n -inf, -inf]",
            "k0": "[2.1, 1.8,\n 3.2, 2.5]",
            "k1": "[2.1, 1.8,\n 3.5, 2.9]",
            "k2": "[2.1, 2.4,\n 3.5, 2.9]",
            "final": "收敛 →",
            "change": [0, 1, 1, 0],  # which k iterations changed max
        },
        {
            "label": "l (running sum)\n每行 exp(S-m) 的\n累加和",
            "color": "#16a34a",
            "label_color": "#166534",
            "fill": "#f0fdf4",
            "init": "[0, 0, 0, 0]",
            "k0": "[3.1, 2.8,\n 4.5, 3.9]",
            "k1": "[6.5, 6.1,\n 8.2, 7.3]",
            "k2": "[9.8, 8.9,\n 11.5, 10.2]",
            "final": "→ 做分母",
            "change": [0, 1, 1, 0],
        },
        {
            "label": "O_acc (output)\n累积的加权 value\n[4 x 8]",
            "color": "#ea580c",
            "label_color": "#9a3412",
            "fill": "#fff7ed",
            "init": "全零矩阵\n[4 x 8]",
            "k0": "O₀ (基于 S₀\n的 P₀@V₀)",
            "k1": "corr×O₀\n+ P₁@V₁",
            "k2": "corr×O₁\n+ P₂@V₂",
            "final": "O_acc / l",
            "change": [0, 0, 0, 1],
        },
        {
            "label": "correction\n因子 exp(m_old\n- m_new)",
            "color": "#8b5cf6",
            "label_color": "#6d28d9",
            "fill": "#f5f3ff",
            "init": "—",
            "k0": "— (首次,\n无需修正)",
            "k1": "[1.0, 1.0,\n 0.74, 0.67]",
            "k2": "[1.0, 0.55,\n 1.0, 1.0]",
            "final": "—",
            "change": [0, 0, 1, 0],
        },
    ]

    for ri, rd in enumerate(rows_data):
        ry = table_top + ri * row_h

        # Label column
        L.append(f'<rect x="50" y="{ry}" width="70" height="{row_h-2}" rx="3" fill="{rd["fill"]}" stroke="{rd["color"]}" stroke-width="1"/>')
        for j, line in enumerate(rd["label"].split('\n')):
            L.append(f'<text x="85" y="{ry + 14 + j*14}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="{rd["label_color"]}" font-weight="bold">{esc(line)}</text>')

        # Data cells
        col_xs = [130, 320, 510, 700, 890]
        col_ws = [180, 180, 180, 180, 80]
        vals = [rd["init"], rd["k0"], rd["k1"], rd["k2"], rd["final"]]

        for ci, (cx, cw, val) in enumerate(zip(col_xs, col_ws, vals)):
            # Highlight cells where max changed
            if ci > 0 and ci <= len(rd["change"]) and rd["change"][ci-1]:
                stroke = rd["color"]
                stroke_w = 2
                bg = rd["fill"]
            else:
                stroke = "#e2e8f0"
                stroke_w = 1
                bg = "white"
            L.append(f'<rect x="{cx}" y="{ry}" width="{cw}" height="{row_h-2}" rx="3" fill="{bg}" stroke="{stroke}" stroke-width="{stroke_w}"/>')
            for j, line in enumerate(val.split('\n')):
                fs = 10
                fc = "#334155"
                if ci == 4:  # final column
                    fc = rd["color"]
                    fs = 10
                L.append(f'<text x="{cx + cw//2}" y="{ry + 14 + j*14}" text-anchor="middle" font-family="monospace" font-size="{fs}" fill="{fc}">{esc(line)}</text>')

    # --- Annotation: correction convergence ---
    ann_y = 400
    L.append(f'<rect x="50" y="{ann_y}" width="920" height="110" rx="6" fill="#fefce8" stroke="#eab308" stroke-width="1.5"/>')
    L.append(f'<text x="70" y="{ann_y + 22}" font-family="sans-serif" font-size="13" fill="#854d0e" font-weight="bold">{esc("关键观察: correction 因子的收敛性质")}</text>')
    L.append(f'<text x="70" y="{ann_y + 44}" font-family="monospace" font-size="10" fill="#713f12">'
             f'{esc("  kv=0 → kv=1: 第3,4行的 max 增大 (3.2→3.5, 2.5→2.9)，correction = exp(3.2-3.5)=0.74 < 1，需要 rescale O_acc")}</text>')
    L.append(f'<text x="70" y="{ann_y + 62}" font-family="monospace" font-size="10" fill="#713f12">'
             f'{esc("  kv=1 → kv=2: 第1,3,4行的 max 不变，correction = exp(0) = 1 → NO-OP！第2行的 max 变化 (1.8→2.4)，correction = 0.55")}</text>')
    L.append(f'<text x="70" y="{ann_y + 80}" font-family="sans-serif" font-size="11" fill="#dc2626" font-weight="bold">'
             f'{esc("  → 后期 KV block 几乎不需要 rescaling (correction→1)。FlashAttention-2 利用此性质跳过 no-op rescaling，节省 ~5% 时间。")}</text>')

    # --- Algorithm pseudo-code ---
    algo_y = 540
    L.append(f'<rect x="50" y="{algo_y}" width="920" height="155" rx="6" fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.5"/>')
    L.append(f'<text x="70" y="{algo_y + 22}" font-family="sans-serif" font-size="13" fill="#334155" font-weight="bold">{esc("Online Softmax 算法 (单 Q block 的完整循环)")}</text>')
    code = [
        "m = [-inf, -inf, -inf, -inf]        # running max per row",
        "l = [0, 0, 0, 0]                     # running exp sum (normalization denominator)",
        "O_acc = zeros[4, 8]                  # running output accumulator",
        "",
        "for kv in 0, 1, 2:                   # iterate all KV blocks",
        "    S = Q @ K_kv^T * SCALE           # [4x4] in SRAM, never written to HBM",
        "    m_new = max(m, row_max(S))        # update running max",
        "    P = exp(S - m_new)                # numerically stable exp",
        "    correction = exp(m - m_new)       # rescale old O_acc",
        "    l = correction * l + row_sum(P)   # update running normalization",
        "    O_acc = correction * O_acc + P @ V_kv   # accumulate weighted values",
        "    m = m_new                         # update state for next iteration",
        "",
        "O_block = O_acc / l                   # final normalization (ONCE!)",
    ]
    for ji, line in enumerate(code):
        color = '#334155'
        if line.startswith('    '):
            color = '#64748b'
        L.append(f'<text x="80" y="{algo_y + 44 + ji*14}" font-family="monospace" font-size="10" fill="{color}">{esc(line)}</text>')

    L.append('</svg>')
    return '\n'.join(L)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 gen_softmax_trace.py <output_path_without_ext>")
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
