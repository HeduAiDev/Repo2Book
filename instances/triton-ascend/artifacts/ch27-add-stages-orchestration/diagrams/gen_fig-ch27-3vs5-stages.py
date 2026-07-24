#!/usr/bin/env python3
"""fig-ch27-3vs5-stages：昇腾三段比基座 CUDA 五段少一整层 TTGIR——因为昇腾无真实 warp
（warp_size=0），不需 GPU 那层显式 layout/warp/CTA 指派，Triton-MLIR 经 triton_adapter
直降 Linalg。坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)


def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


TITLE = "三段 vs 五段：省掉的是 TTGIR 这一整层"
SUBTITLE = "昇腾 third_party/ascend/backend/compiler.py:L941/949/953 ｜ 对照基座 ch36 CUDABackend.add_stages"

PAD = 44
PANEL_W = 380
GUTTER = 90
TOP = 96
w = PAD * 2 + PANEL_W * 2 + GUTTER

elems = []


def add(s):
    elems.append(s)


def box(cx, y, lines, w=PANEL_W, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e",
        bold=False, fs=13, dashed=False):
    n = len(lines)
    box_h = 26 + 19 * (n - 1) + 30
    bx = cx - w / 2
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{w:.0f}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dash}/>')
    y0 = y + box_h / 2 - (n - 1) * 9.5 + 5
    fw = 'font-weight="bold" ' if bold else ''
    for k, line in enumerate(lines):
        add(f'<text x="{cx:.0f}" y="{y0+k*19:.0f}" text-anchor="middle" '
            f'font-family="monospace" font-size="{fs}" {fw}fill="{text_fill}">{esc(line)}</text>')
    return box_h


def varrow(x, y1, y2, color="#334155"):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')


x_left = PAD + PANEL_W / 2
x_right = PAD + PANEL_W + GUTTER + PANEL_W / 2
GAP = 22

titles = [
    (x_left, "昇腾：add_stages 登记 3 段"),
    (x_right, "基座 CUDABackend：add_stages 登记 5 段"),
]
for cx, t in titles:
    add(f'<text x="{cx:.0f}" y="{TOP-12:.0f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13.5" font-weight="bold" fill="#0f172a">{esc(t)}</text>')

# --- 左：昇腾 3 段 ---
ly = TOP + 14
lbh = box(x_left, ly, ["ttir（make_ttir，L941）"])
ly += lbh
varrow(x_left, ly, ly + GAP)
ly += GAP
lbh2 = box(x_left, ly, ["ttadapter（ttir_to_linalg，L949）", "11 个 ascend.passes.ttir.add_*"],
           fill="#dcfce7", stroke="#15803d", text_fill="#14532d")
ly += lbh2
varrow(x_left, ly, ly + GAP)
ly += GAP
lbh3 = box(x_left, ly, ["npubin（L953，二选一）"])
ly += lbh3
left_stage_bottom = ly

# 昇腾侧对应"缺失的 TTGIR"占位（虚线、灰底），与右侧 TTGIR 位置对齐，直观显示"少一层"
ly2 = ly + 30
ghost_h = box(x_left, ly2, ["（无 TTGIR 这一层）", "warp_size = 0 · driver.py:L173"],
              fill="#f8fafc", stroke="#94a3b8", text_fill="#475569", dashed=True, fs=11.5)
left_bottom = ly2 + ghost_h

# --- 右：基座 5 段 ---
ry = TOP + 14
GPU_STEPS = ["ttir", "ttgir", "llir", "ptx", "cubin"]
gpu_boxes_bottom = []
for i, s in enumerate(GPU_STEPS):
    if s == "ttgir":
        rbh = box(x_right, ry, ["ttgir：layout/warp/CTA 指派", "warp_size = 32（对照，CUDA 典型值）"],
                   fill="#fef3c7", stroke="#b45309", text_fill="#78350f", fs=11.5)
    else:
        rbh = box(x_right, ry, [s])
    gpu_boxes_bottom.append(ry + rbh)
    ry += rbh
    if i < len(GPU_STEPS) - 1:
        varrow(x_right, ry, ry + GAP)
        ry += GAP
right_bottom = ry

content_bottom = max(left_bottom, right_bottom)

# 中间横向连接线，标出"省掉"的对应关系
mid_x = (x_left + PANEL_W / 2 + x_right - PANEL_W / 2) / 2
ghost_mid_y = ly2 + ghost_h / 2
ttgir_mid_y = gpu_boxes_bottom[0]  # placeholder, recomputed below
# 找到 ttgir 框的纵向中点：它是第 2 个框（index 1）
ttgir_top = TOP + 14 + (26 + 19 * 0 + 30) + GAP  # 第 1 个框(ttir)高度 + GAP
ttgir_h = 26 + 19 * 1 + 30
ttgir_mid_y = ttgir_top + ttgir_h / 2
add(f'<line x1="{x_left+PANEL_W/2:.0f}" y1="{ghost_mid_y:.0f}" x2="{x_right-PANEL_W/2:.0f}" y2="{ttgir_mid_y:.0f}" '
    'stroke="#b45309" stroke-width="1.6" stroke-dasharray="4,4"/>')
add(f'<text x="{mid_x:.0f}" y="{(ghost_mid_y+ttgir_mid_y)/2-8:.0f}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" font-weight="bold" fill="#b45309">省掉的 1 层</text>')

# --- 底部注解 ---
note_lines = [
    "昇腾没有真实 warp（硬件描述符 warp_size=0），故不需 GPU 在 TTGIR 层做的 layout/warp/CTA 指派；",
    "Triton-MLIR 经 triton_adapter 直降 Linalg（11 个 add_*，L118-L157），并行/内存/流水决策全推给闭源 bishengir。",
]
note_top = content_bottom + 30
note_h = 22 * len(note_lines) + 22
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*22:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-ch27-3vs5-stages.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
