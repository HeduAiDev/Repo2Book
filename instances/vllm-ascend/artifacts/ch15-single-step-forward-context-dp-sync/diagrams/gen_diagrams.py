#!/usr/bin/env python3
"""ch15 三张图：forward-spine / forward-context-injection / dp-sync。
全部坐标由 Python 计算，无手填魔法数。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

FONT = 'font-family="sans-serif"'
INK = "#1e293b"
SUB = "#64748b"
BLUE = "#2563eb"
BLUEBG = "#eff6ff"
PURP = "#7c3aed"
PURPBG = "#f5f3ff"
GREEN = "#16a34a"
GREENBG = "#f0fdf4"
RED = "#dc2626"
GREY = "#94a3b8"
GREYBG = "#f8fafc"
AMBER = "#d97706"


def svg_open(w, h):
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>'
             '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
             '<marker id="arp" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
             '<marker id="arg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
             '</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    return L


def box(L, x, y, w, h, fill, stroke, rx=10, sw=2):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(L, x, y, s, size=15, fill=INK, anchor="middle", weight="normal", mono=False):
    f = 'font-family="monospace"' if mono else FONT
    wt = f' font-weight="{weight}"' if weight != "normal" else ""
    L.append(f'<text x="{x}" y="{y}" {f} font-size="{size}" fill="{fill}" text-anchor="{anchor}"{wt}>{esc(s)}</text>')


def arrow(L, x1, y1, x2, y2, marker="ar", dash=False, color="#475569"):
    d = ' stroke-dasharray="6 4"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"{d} marker-end="url(#{marker})"/>')


# ============================================================
# 图 1: forward-spine —— 一拍前向的主干竖链
# ============================================================
def gen_spine():
    W, H = 1180, 760
    L = svg_open(W, H)
    text(L, W/2, 38, "execute_model 一拍前向的主干七步", size=22, weight="bold")
    text(L, W/2, 62, "横切实体（注意力后端 / MoE 通信原语 / 采样器内部）只点名，分别留 ch18 / ch26 / 采样章", size=13, fill=SUB)

    cx = 360          # 主链中心
    bw, bh = 360, 60
    gap = 26
    y = 92
    steps = [
        ("execute_model(scheduler_output)", "更新持久 batch · 早返回判定", BLUE, BLUEBG),
        ("_prepare_inputs", "整形输入 → input_ids / positions / logits_indices", BLUE, BLUEBG),
        ("_determine_batch_execution_and_padding", "dispatch cudagraph + DP 协调", BLUE, BLUEBG),
        ("_build_attention_metadata", "建 attn_metadata（后端实体）", GREY, GREYBG),
        ("set_ascend_forward_context", "包基座 + 注入昇腾上下文（本章重点）", PURP, PURPBG),
        ("_model_forward", "跑 self.model + flash_comm_v1 回收", BLUE, BLUEBG),
        ("_sample", "sampler / rejection_sampler 派发", GREY, GREYBG),
    ]
    centers = []
    for i, (t, sub, st, bg) in enumerate(steps):
        bx = cx - bw/2
        sw = 2.5 if "set_ascend" in t else 2
        box(L, bx, y, bw, bh, bg, st, sw=sw)
        if "set_ascend" in t:
            # 双层框暗示「包基座」
            box(L, bx+8, y+8, bw-16, bh-16, "white", st, rx=7, sw=1.4)
        tsize = 13 if len(t) > 26 else 15
        text(L, cx, y+26, t, size=tsize, fill=INK, weight="bold", mono=True)
        text(L, cx, y+46, sub, size=12, fill=SUB)
        centers.append((cx, y, y+bh))
        if i > 0:
            arrow(L, cx, centers[i-1][2], cx, y)
        y += bh + gap

    # DP 同步分叉（第 3 步右侧）
    det = centers[2]
    rx0 = cx + bw/2
    sx = 830
    sw_ = 250
    sy = det[1]
    box(L, sx, sy, sw_, bh, GREENBG, GREEN, sw=2)
    text(L, sx+sw_/2, sy+24, "_sync_metadata_across_dp", size=12, fill=INK, weight="bold", mono=True)
    text(L, sx+sw_/2, sy+44, "DP 各卡一次 all_reduce", size=12, fill=GREEN)
    arrow(L, rx0, sy+bh/2, sx, sy+bh/2, marker="arg", dash=True, color=GREEN)
    text(L, (rx0+sx)/2, sy+bh/2-8, "dp_size>1", size=11, fill=GREEN)

    # 横切旁注（右侧，anchor=start，避开 dp-sync 那行）
    annos = [
        (centers[3], "AscendAttentionBackend / MLA → ch18"),
        (centers[4], "MoE 通信原语 all_to_all / mc2 → ch26"),
        (centers[6], "采样器内部留采样章"),
    ]
    for c, txt in annos:
        ay = (c[1]+c[2])/2
        text(L, rx0+20, ay+4, txt, size=12, fill=AMBER, anchor="start")

    L.append('</svg>')
    return W, H, '\n'.join(L)


# ============================================================
# 图 2: forward-context-injection —— 包基座再注入
# ============================================================
def gen_inject():
    W, H = 1020, 620
    L = svg_open(W, H)
    text(L, W/2, 38, "set_ascend_forward_context：包住基座，再往同一个上下文注入昇腾字段", size=20, weight="bold")

    # 外框：基座 ForwardContext
    ox, oy, ow, oh = 330, 70, 660, 510
    box(L, ox, oy, ow, oh, "#f8fafc", BLUE, rx=14, sw=2.5)
    text(L, ox+ow/2, oy+30, "vLLM ForwardContext  ·  with set_forward_context(**kwargs) 建", size=15, fill=BLUE, weight="bold")
    base_fields = ["attn_metadata", "dp_metadata", "cudagraph_runtime_mode", "batch_descriptor"]
    fw = (ow - 60) / 2
    for i, f in enumerate(base_fields):
        col = i % 2
        row = i // 2
        fx = ox + 20 + col*(fw+20)
        fy = oy + 48 + row*40
        box(L, fx, fy, fw, 30, "white", BLUE, rx=6, sw=1.2)
        text(L, fx+fw/2, fy+20, f, size=13, fill=INK, mono=True)

    # 内框：昇腾注入
    ix, iy, iw, ih = ox+20, oy+140, ow-40, oh-160
    box(L, ix, iy, iw, ih, PURPBG, PURP, rx=12, sw=2.5)
    text(L, ix+iw/2, iy+28, "昇腾额外注入  ·  按本拍 token 形状 / 并行模式动态选", size=15, fill=PURP, weight="bold")
    inj = [
        "moe_comm_type", "moe_comm_method",
        "flash_comm_v1_enabled", "flashcomm_v2_enabled",
        "mmrs_fusion", "pad_size",
        "padded_num_tokens", "mc2_mask",
        "max_tokens_across_dp", "max_tokens_across_pcp",
        "layer_idx", "is_first_layer",
    ]
    ifw = (iw - 60) / 2
    for i, f in enumerate(inj):
        col = i % 2
        row = i // 2
        fx = ix + 20 + col*(ifw+20)
        fy = iy + 46 + row*42
        hot = f in ("moe_comm_type", "mc2_mask")
        box(L, fx, fy, ifw, 32, "white", RED if hot else PURP, rx=6, sw=1.6 if hot else 1.2)
        text(L, fx+ifw/2, fy+21, f, size=13, fill=INK, mono=True)

    # 左侧 select_moe_comm_method 三输入
    sx, sy, sw_, sh = 30, 200, 260, 200
    box(L, sx, sy, sw_, sh, GREENBG, GREEN, rx=12, sw=2)
    text(L, sx+sw_/2, sy+26, "select_moe_comm_method", size=14, fill=GREEN, weight="bold", mono=True)
    ins = [
        "soc 版本  A2 / A3 / 310P / A5",
        "num_tokens  vs  mc2_tokens_capacity",
        "是否 EP · 量化类型",
    ]
    for i, t in enumerate(ins):
        iyy = sy + 50 + i*42
        box(L, sx+16, iyy, sw_-32, 32, "white", GREEN, rx=6, sw=1.2)
        text(L, sx+sw_/2, iyy+21, t, size=11.5, fill=INK)
    text(L, sx+sw_/2, sy+sh-12, "三因素 → moe_comm_type", size=12, fill=GREEN)
    # 箭头指向 moe_comm_type 字段（内框第一格）
    target_x = ix + 20
    target_y = iy + 46 + 16
    arrow(L, sx+sw_, sy+sh/2, target_x, target_y, marker="arp", color=PURP)

    text(L, W/2, H-18, "外蓝框 = 基座通用字段；内紫框 = 昇腾就地挂的额外字段，模型各层 forward 时直接取用（红框为本章主追的两项）", size=12, fill=SUB)
    L.append('</svg>')
    return W, H, '\n'.join(L)


# ============================================================
# 图 3: dp-sync —— 打包一次 all_reduce
# ============================================================
def gen_dpsync():
    W, H = 1000, 600
    L = svg_open(W, H)
    text(L, W/2, 36, "DP 跨卡同步：把 num_tokens + cudagraph_mode 打包进一次 all_reduce", size=19, weight="bold")
    text(L, W/2, 58, "packed_tensor 形状 [2, dp_size]，各卡只填自己那一列、其余为 0 → sum-allreduce 即汇齐广播", size=12.5, fill=SUB)

    # 两卡：各持己列
    card_w, card_h = 300, 150
    c0x, c1x = 80, 620
    cy = 90
    cards = [
        (c0x, "DP rank 0", 40, "FULL (2)", [["40", "0"], ["2", "0"]]),
        (c1x, "DP rank 1", 24, "NONE (0)", [["0", "24"], ["0", "0"]]),
    ]
    for cx, name, nt, mode, mat in cards:
        box(L, cx, cy, card_w, card_h, BLUEBG, BLUE, rx=12, sw=2)
        text(L, cx+card_w/2, cy+24, name, size=15, fill=BLUE, weight="bold")
        text(L, cx+card_w/2, cy+46, f"num_tokens = {nt}   ·   cudagraph_mode = {mode}", size=12, fill=INK)
        # 2x2 packed 子张量
        gx, gy = cx+70, cy+62
        cell = 38
        text(L, gx-26, gy+cell, "行0", size=10, fill=SUB, anchor="end")
        text(L, gx-26, gy+cell*2, "行1", size=10, fill=SUB, anchor="end")
        text(L, gx+cell/2, gy-4, "列0", size=10, fill=SUB)
        text(L, gx+cell*1.5, gy-4, "列1", size=10, fill=SUB)
        for r in range(2):
            for c in range(2):
                v = mat[r][c]
                own = (v != "0")
                box(L, gx+c*cell, gy+cell*r, cell, cell, "#fef9c3" if own else "white", AMBER if own else GREY, rx=4, sw=1.6 if own else 1)
                text(L, gx+c*cell+cell/2, gy+cell*r+cell*0.65, v, size=14, fill=INK, mono=True, weight="bold" if own else "normal")

    # all_reduce 汇聚
    arx, ary, arw, arh = 400, 290, 200, 46
    box(L, arx, ary, arw, arh, GREEN, GREEN, rx=10, sw=2)
    text(L, arx+arw/2, ary+29, "all_reduce (SUM)", size=16, fill="white", weight="bold")
    arrow(L, c0x+card_w/2, cy+card_h, arx+arw*0.3, ary, marker="arg", color=GREEN)
    arrow(L, c1x+card_w/2, cy+card_h, arx+arw*0.7, ary, marker="arg", color=GREEN)
    text(L, arx+arw/2-80, ary-6, "dp_allreduce_on_npu → 走 NPU group 规避 CPU 脏数据", size=11, fill=GREEN, anchor="middle")

    # 结果表（两卡同得）
    ry = 384
    text(L, W/2 - 150, ry-6, "两卡都拿到完整 packed_tensor ↓", size=12.5, fill=GREEN, anchor="end")
    rgx = W/2 - 40
    rgy = ry + 10
    cell = 44
    res = [["40", "24"], ["2", "0"]]
    text(L, rgx-26, rgy+cell*0.65, "tokens", size=11, fill=SUB, anchor="end")
    text(L, rgx-26, rgy+cell*1.65, "mode", size=11, fill=SUB, anchor="end")
    for r in range(2):
        for c in range(2):
            box(L, rgx+c*cell, rgy+cell*r, cell, cell, "#dcfce7", GREEN, rx=4, sw=1.4)
            text(L, rgx+c*cell+cell/2, rgy+cell*r+cell*0.65, res[r][c], size=15, fill=INK, mono=True, weight="bold")
    arrow(L, arx+arw/2, ary+arh, rgx+cell, rgy, marker="arg", color=GREEN)

    # 解包结论
    conc_y = rgy + 6
    cxx = rgx + cell*2 + 40
    box(L, cxx, conc_y, 320, 44, "#fff7ed", AMBER, rx=8, sw=1.6)
    text(L, cxx+160, conc_y+19, "tokens：进图卡补到 max=40，eager 卡保留原值", size=12, fill=INK)
    text(L, cxx+160, conc_y+36, "mode 取 min(2,0)=0  → 任一 NONE 则全卡 NONE", size=12, fill=INK)
    box(L, cxx, conc_y+54, 320, 36, GREENBG, GREEN, rx=8, sw=1.6)
    text(L, cxx+160, conc_y+77, "两卡就这两个数达成一致 → 一起进/不进同一张图", size=12, fill=GREEN, weight="bold")

    L.append('</svg>')
    return W, H, '\n'.join(L)


import os
here = os.path.dirname(os.path.abspath(__file__))
for name, gen in [("forward-spine", gen_spine), ("forward-context-injection", gen_inject), ("dp-sync", gen_dpsync)]:
    w, h, svg = gen()
    p = os.path.join(here, name + ".svg")
    with open(p, "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", p, w, h)
