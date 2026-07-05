#!/usr/bin/env python3
"""ch30 §30.3 LoRA 算子分发全链路。
照 vllm_ascend/lora/punica_npu.py 源码分支绘制：
① __init__ 按『device==310P 或 max_lora_rank>=128』二选一绑算子——真则回退
   vLLM 通用 torch_ops，假则绑昇腾 NPU kernel lora_ops；两条路都收敛到『6 个算子
   绑成实例属性』。② add_lora_linear 走 shrink（降到 r 维 buffer）→ expand（升回
   输出维）两步，调用哪副算子由 __init__ 期绑定的属性决定，运行期不再判断。
   旁注 FLOPs 对比：r=16 两步 262144（省 128×）vs r=128 两步 2097152（省 16×），
   满秩基线 33554432。风格对齐同章 fig30-1 / netloader-flow。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# palette（对齐同章 gen_netloader_flow.py）
P_FILL, P_STROKE, P_TC = "#f3e8ff", "#7c3aed", "#5b21b6"   # 处理框（紫，主干/NPU 快路径）
D_FILL, D_STROKE, D_TC = "#fef3c7", "#d97706", "#92400e"   # 判定菱形（琥珀）
R_FILL, R_STROKE, R_TC = "#f1f5f9", "#94a3b8", "#475569"   # 回退框（灰，PyTorch 通用）
G_FILL, G_STROKE, G_TC = "#dcfce7", "#16a34a", "#166534"   # 收敛/成功框（绿）
S_FILL, S_STROKE, S_TC = "#e0f2fe", "#0284c7", "#075985"   # FLOPs 旁注（蓝）

W, H = 1120, 900
cx = 300            # 主干中轴
bw = 400            # 主干框宽

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="ap" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker>'
    '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# title
L.append(f'<text x="{W/2}" y="42" text-anchor="middle" font-family="sans-serif" '
         f'font-size="26" font-weight="bold" fill="#1e293b">'
         f'{esc("PunicaWrapperNPU：构造期二选一绑算子，运行期 shrink→expand 落地")}</text>')
L.append(f'<text x="{W/2}" y="70" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" fill="#64748b">'
         f'{esc("绑定一次、处处复用——add_lora_linear 的调用点不含任何 device / rank 判断")}</text>')


def rbox(x, y, w, h, fill, stroke, tc, lines, fs=15, mono=False, bold=True, rx=10):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    ff = "monospace" if mono else "sans-serif"
    fw = "bold" if bold else "normal"
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * (fs + 5) / 2 + fs / 2 - 2
    for i, ln in enumerate(lines):
        L.append(f'<text x="{x+w/2}" y="{y0+i*(fs+5)}" text-anchor="middle" '
                 f'font-family="{ff}" font-size="{fs}" font-weight="{fw}" '
                 f'fill="{tc}">{esc(ln)}</text>')


def diamond(dcx, dcy, hw, hh, lines):
    pts = f'{dcx},{dcy-hh} {dcx+hw},{dcy} {dcx},{dcy+hh} {dcx-hw},{dcy}'
    L.append(f'<polygon points="{pts}" fill="{D_FILL}" stroke="{D_STROKE}" stroke-width="2"/>')
    n = len(lines)
    y0 = dcy - (n - 1) * 20 / 2 + 5
    for i, (ln, fs) in enumerate(lines):
        L.append(f'<text x="{dcx}" y="{y0+i*20}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" font-weight="bold" '
                 f'fill="{D_TC}">{esc(ln)}</text>')


def arrow(x1, y1, x2, y2, marker="ap", stroke="#7c3aed", w=2):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
             f'stroke-width="{w}" marker-end="url(#{marker})"/>')


def alabel(x, y, lines, fill="#7c3aed", fs=14):
    n = len(lines)
    y0 = y - (n - 1) * (fs + 3) / 2
    for i, ln in enumerate(lines):
        L.append(f'<text x="{x}" y="{y0+i*(fs+3)}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" font-weight="bold" '
                 f'fill="{fill}">{esc(ln)}</text>')


# ============ 主干：构造期二选一绑算子 ============
# A 入口
rbox(cx - bw / 2, 84, bw, 52, P_FILL, P_STROKE, P_TC,
     ["PunicaWrapperNPU.__init__(vllm_config)"], fs=15, mono=True)

# D1 决策菱形
diamond(cx, 224, 195, 78,
        [("device == 310P(310)", 15), ("或 max_lora_rank >= 128 ?", 15)])

# B False 分支：继续走主干，绑昇腾 NPU kernel
rbox(cx - bw / 2, 344, bw, 86, P_FILL, P_STROKE, P_TC,
     ["绑 vllm_ascend.lora.lora_ops", "→ torch.ops._C_ascend.*（NPU kernel）",
      "假：服务小 rank（如 r=16）时省 128×"], fs=15)

# 右列 True 分支：回退 PyTorch 通用实现
fx, fw2 = 700, 380
rbox(fx, 310, fw2, 70, R_FILL, R_STROKE, R_TC,
     ["绑 vllm.lora.ops.torch_ops（PyTorch 通用）", "310P / rank≥128 保底，r=128 时仅省 16×"], fs=14)

# C 收敛：6 个算子绑成实例属性
rbox(cx - bw / 2, 452, bw, 60, G_FILL, G_STROKE, G_TC,
     ["6 个算子绑成实例属性：", "self.bgmv_* / self.sgmv_*"], fs=15)

# ============ 运行期：add_lora_linear 走 shrink→expand ============
rbox(cx - bw / 2, 552, bw, 48, P_FILL, P_STROKE, P_TC,
     ["add_lora_linear(y, x, lora_a_stacked, ...)"], fs=14, mono=True)
rbox(cx - bw / 2, 624, bw, 44, R_FILL, R_STROKE, R_TC,
     ["开一个 r 维 buffer"], fs=14)
bw2 = bw + 40  # shrink/expand 这两行签名偏长，单独放宽宽度避免贴边
rbox(cx - bw2 / 2, 692, bw2, 74, P_FILL, P_STROKE, P_TC,
     ["add_shrink：is_prefill? sgmv_shrink : bgmv_shrink",
      "x · A  →  降到 r 维，写入 buffer"], fs=13)
rbox(cx - bw2 / 2, 800, bw2, 74, P_FILL, P_STROKE, P_TC,
     ["add_expand：is_prefill? sgmv_expand : bgmv_expand",
      "buffer · B · s  →  升回 o 维，累加进 y"], fs=13)

# ============ 右侧旁注：FLOPs 对比条 ============
sx, sw = 700, 380
rbox(sx, 692, sw, 74, S_FILL, S_STROKE, S_TC,
     ["r = 16：两步 262144 FLOPs", "满秩直算 33554432 FLOPs  →  省 128×"], fs=14)
rbox(sx, 800, sw, 74, S_FILL, S_STROKE, S_TC,
     ["r = 128：两步 2097152 FLOPs", "满秩直算 33554432 FLOPs  →  省 16×"], fs=14)

# ============ 箭头 ============
arrow(cx, 136, cx, 146, marker="ap")                       # A -> D1
arrow(cx, 302, cx, 344, marker="ap")                        # D1 -> B（假）
alabel(cx + 34, 326, ["假"], fill=P_TC, fs=13)
arrow(cx + 195, 224, fx - 6, 224, marker="ag", stroke="#94a3b8")   # D1 -> 右列（真）
L.append(f'<line x1="{fx-6}" y1="224" x2="{fx-6}" y2="310" stroke="#94a3b8" '
         f'stroke-width="2" marker-end="url(#ag)"/>')
alabel((cx + 195 + fx) / 2, 214, ["真（任一成立）"], fill="#475569", fs=13)

arrow(cx, 430, cx, 452, marker="ap")                        # B -> C
# 右列 -> C：先竖直下行到 C 中线高度，再水平接入 C 右边缘（正交折线，避免斜线穿越空白）
L.append(f'<line x1="{fx+fw2/2}" y1="380" x2="{fx+fw2/2}" y2="482" '
         f'stroke="#94a3b8" stroke-width="2"/>')
L.append(f'<line x1="{fx+fw2/2}" y1="482" x2="{cx+bw/2+6}" y2="482" '
         f'stroke="#94a3b8" stroke-width="2" marker-end="url(#ag)"/>')

arrow(cx, 512, cx, 552, marker="ah", stroke="#16a34a")       # C -> add_lora_linear
arrow(cx, 600, cx, 624, marker="ap")                         # -> buffer
arrow(cx, 668, cx, 692, marker="ap")                         # -> add_shrink
arrow(cx, 766, cx, 800, marker="ap")                         # -> add_expand

L.append('</svg>')
svg = '\n'.join(L)
with open('ch30-lora-ops-dispatch.svg', 'w', encoding='utf-8') as f:
    f.write(svg)
print("wrote ch30-lora-ops-dispatch.svg", W, H)
