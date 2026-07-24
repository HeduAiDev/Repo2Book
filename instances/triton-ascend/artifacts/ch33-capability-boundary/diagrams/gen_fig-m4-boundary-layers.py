#!/usr/bin/env python3
"""fig-m4-boundary-layers — layout 模板:未支持面按技术栈分层的自顶向下堆栈。
每层一个横条,宽度 ∝ 命中数,右侧挂代表文件;标题标注不同应对预期。
全部坐标由循环/常量计算,文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "未支持面按技术栈分层:卡在哪一层,决定怎么等"
SUBTITLE = "同为 skip,归因落在五个不同的层——应对预期完全不同(等上游/等编译器/等版本/换 shape/尚未纳管)"

LAYERS = [
    ("上游软件层 · waiting for TA to support", 13,
     "test_device_print_script.py:L50", "等 triton-ascend 上游支持", "#1d4ed8"),
    ("闭源编译器层 · 编译器(bishengir)", 9,
     "test_pow.py:L47/L76/L105(6 条 reason 逐字含 bishengir-compile,另 3 条只写 compiler to support)",
     "等编译器本体迭代", "#0369a1"),
    ("版本回退层 · NPUIR updated in April", 5,
     "test_dot.py:L128 / test_compile_hint.py:L48",
     "曾经能、现在暂不能——等后续修复", "#0891b2"),
    ("硬件资源层 · UB overflow", 3,
     "test_03_matrix_multiplication.py:L215(leaky_relu_custom) / test_11_rab_time.py:L390",
     "片上 Unified Buffer 容量物理边界——换 shape 绕开", "#15803d"),
    ("整块未纳管 · attn_cp", 3,
     "test_attn_cp.py:L486-L496(三个 test_prove_*)",
     "reason 仅 'attn_cp'——最含糊,整批划到 CI 线外", "#78716c"),
]

MAX_VAL = 13
PAD = 40
W = 1360
TOP = 100
LAYER_H = 78
LAYER_GAP = 14
BAR_MAX_W = 420
LABEL_COL_W = 430

H = TOP + len(LAYERS) * (LAYER_H + LAYER_GAP) - LAYER_GAP + 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="19" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

y = TOP
for name, val, cite, response, color in LAYERS:
    L.append(f'<rect x="{PAD}" y="{y}" width="{W-2*PAD}" height="{LAYER_H}" rx="8" '
              f'fill="{color}" fill-opacity="0.07" stroke="{color}" stroke-width="1.3"/>')
    L.append(f'<text x="{PAD+16}" y="{y+22}" font-family="sans-serif" font-size="14" '
              f'font-weight="bold" fill="{color}">{esc(name)}</text>')
    bar_w = BAR_MAX_W * val / MAX_VAL
    bar_x = PAD + 16
    bar_y = y + 32
    L.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="16" rx="3" '
              f'fill="{color}"/>')
    L.append(f'<text x="{bar_x+bar_w+10}" y="{bar_y+13}" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{color}">{esc(str(val)+" 处")}</text>')
    L.append(f'<text x="{bar_x}" y="{y+LAYER_H-10}" font-family="sans-serif" font-size="10.5" '
              f'fill="#64748b">{esc("代表:"+cite)}</text>')
    resp_x = PAD + LABEL_COL_W + 480
    L.append(f'<text x="{resp_x}" y="{y+LAYER_H/2+4}" font-family="sans-serif" '
              f'font-size="12.5" fill="#334155">{esc("→ "+response)}</text>')
    y += LAYER_H + LAYER_GAP

foot_y = H - 22
L.append(f'<line x1="{PAD}" y1="{foot_y-16}" x2="{W-PAD}" y2="{foot_y-16}" stroke="#e2e8f0"/>')
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("五层合计 13+9+5+3+3=33 处——即反面清单五主类;条长 ∝ 命中数,同一比例尺(最大 13)")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m4-boundary-layers.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
