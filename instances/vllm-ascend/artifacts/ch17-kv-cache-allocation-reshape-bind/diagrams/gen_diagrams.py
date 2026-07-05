#!/usr/bin/env python3
"""ch16 三张图：alloc-geometry / adjust-kv-layout / bind-dispatch。
全部坐标由 Python 计算，无手填魔法数。"""
import os
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
REDBG = "#fef2f2"
GREY = "#94a3b8"
GREYBG = "#f8fafc"
AMBER = "#d97706"
AMBERBG = "#fff7ed"


def svg_open(w, h):
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>'
             '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
             '<marker id="arp" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
             '<marker id="arg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
             '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
             '</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    return L


def box(L, x, y, w, h, fill, stroke, rx=10, sw=2, dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(L, x, y, s, size=15, fill=INK, anchor="middle", weight="normal", mono=False):
    f = 'font-family="monospace"' if mono else FONT
    wt = f' font-weight="{weight}"' if weight != "normal" else ""
    L.append(f'<text x="{x}" y="{y}" {f} font-size="{size}" fill="{fill}" text-anchor="{anchor}"{wt}>{esc(s)}</text>')


def arrow(L, x1, y1, x2, y2, marker="ar", dash=False, color="#475569"):
    d = ' stroke-dasharray="6 4"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"{d} marker-end="url(#{marker})"/>')


# ============================================================
# 图 1: alloc-geometry —— 基座一整块 vs 昇腾拆 K/V + 2MB 对齐
# ============================================================
def gen_alloc():
    W, H = 1200, 660
    L = svg_open(W, H)
    text(L, W/2, 38, "同一份 KV 字节，基座摆一条、昇腾摆两条", size=22, weight="bold")
    text(L, W/2, 62, "_allocate_kv_cache_tensors：基座 torch.zeros 单块 int8；昇腾按 split_factor 拆独立 K/V，各自首地址 2MB 对齐", size=13, fill=SUB)

    band_x = 120
    band_w = 760
    bh = 56

    # ---- 上排：基座 ----
    by = 110
    text(L, band_x, by - 12, "基座 gpu_model_runner.py", size=15, fill=BLUE, weight="bold", anchor="start")
    box(L, band_x, by, band_w, bh, BLUEBG, BLUE, sw=2)
    text(L, band_x + band_w/2, by + bh/2 - 4, "一整块 int8  torch.zeros(size)", size=15, fill=INK, weight="bold", mono=True)
    text(L, band_x + band_w/2, by + bh/2 + 16, "K 与 V 同住一张量（含 size-2 的 K/V 维），整块 shared_by 所有层", size=12, fill=SUB)
    text(L, band_x + band_w + 16, by + bh/2 + 4, "reshape：一次 view + permute", size=12.5, fill=BLUE, anchor="start")

    # ---- 下排：昇腾 ----
    ay = 250
    text(L, band_x, ay - 34, "昇腾 model_runner_v1.py", size=15, fill=PURP, weight="bold", anchor="start")
    gap = 28
    half = (band_w - gap) / 2
    # K band
    box(L, band_x, ay, half, bh, PURPBG, PURP, sw=2)
    text(L, band_x + half/2, ay + bh/2 - 4, "K_tensor (int8)", size=15, fill=INK, weight="bold", mono=True)
    text(L, band_x + half/2, ay + bh/2 + 16, "size // ((k+v)/k)", size=11.5, fill=SUB, mono=True)
    # V band
    vbx = band_x + half + gap
    box(L, vbx, ay, half, bh, PURPBG, PURP, sw=2)
    text(L, vbx + half/2, ay + bh/2 - 4, "V_tensor (int8)", size=15, fill=INK, weight="bold", mono=True)
    text(L, vbx + half/2, ay + bh/2 + 16, "size // ((k+v)/v)", size=11.5, fill=SUB, mono=True)
    # 2MB align markers on each band start
    for sxx in (band_x, vbx):
        text(L, sxx, ay - 8, "▼ 2MB 对齐", size=11, fill=RED, anchor="middle", weight="bold")
    # gap label
    text(L, band_x + half + gap/2, ay + bh + 22, "独立分配", size=10.5, fill=SUB)
    text(L, band_x + band_w + 16, ay + bh/2 + 4, "calc_split_factor 定比例", size=12.5, fill=PURP, anchor="start")

    # ---- sparse-c8 追加条带 ----
    sy = 370
    text(L, band_x, sy - 12, "DeepSeek-V3.2 sparse-c8 再追加（同一块对齐 int8 的两个视图）", size=13.5, fill=AMBER, weight="bold", anchor="start")
    sk = band_w * 0.62
    ss = band_w - sk - gap
    box(L, band_x, sy, sk, bh, AMBERBG, AMBER, sw=2)
    text(L, band_x + sk/2, sy + bh/2 - 4, "dsa_k (int8)", size=14, fill=INK, weight="bold", mono=True)
    text(L, band_x + sk/2, sy + bh/2 + 16, "indexer key", size=11.5, fill=SUB)
    sbx = band_x + sk
    # contiguous (no gap) to show shared storage
    box(L, sbx, sy, ss, bh, "#fde68a", AMBER, sw=2)
    text(L, sbx + ss/2, sy + bh/2 - 4, "dsa_k_scale", size=13, fill=INK, weight="bold", mono=True)
    text(L, sbx + ss/2, sy + bh/2 + 16, "scale_offset=_align_up", size=10.5, fill=SUB, mono=True)
    text(L, band_x + sk, sy - 8, "↕ 同一裸张量", size=11, fill=AMBER, anchor="middle")
    text(L, band_x + band_w + 16, sy + bh/2 + 4, "一次 register_buffer", size=12.5, fill=AMBER, anchor="start")
    text(L, band_x + band_w + 16, sy + bh/2 + 22, "省 HCCL/Mooncake 注册条目", size=11, fill=SUB, anchor="start")

    # ---- 底部要点 ----
    ky = 480
    box(L, band_x, ky, band_w + 200, 130, GREYBG, GREY, rx=12, sw=1.5)
    text(L, band_x + 20, ky + 28, "为什么非拆不可：", size=14, fill=INK, weight="bold", anchor="start")
    pts = [
        "① PD 分离要求 kv_cache 的 0 维是 num_blocks、K/V 各自连续 —— 单块整张量做不到",
        "② Mooncake/ADXL 的 RDMA 注册要求首地址按 2MB 大页对齐 —— 多分配 alignment 字节再 _align_memory 切到边界",
        "③ 一律先 int8 裸字节分配，dtype 留到 reshape 决定 —— 分配器只跟字节数打交道，bf16/fp8/int8-quant 同一条路",
    ]
    for i, p in enumerate(pts):
        text(L, band_x + 20, ky + 56 + i*26, p, size=12.5, fill=INK, anchor="start")

    L.append('</svg>')
    return W, H, '\n'.join(L)


# ============================================================
# 图 2: adjust-kv-layout —— as_strided 把 block 维 stride 钉成一页
# ============================================================
def gen_layout():
    W, H = 1180, 660
    L = svg_open(W, H)
    text(L, W/2, 38, "_adjust_kv_layout：把 block 维的 stride 钉成「一整页」", size=21, weight="bold")
    text(L, W/2, 62, "as_strided 把 block 维 stride 钉成 num_element_per_page —— page 被 padding 抬大时，块间露出页对齐空隙", size=12.5, fill=SUB)

    colors = [BLUE, PURP, GREEN]
    ew = 7              # 每个元素的像素宽
    bx = 130
    GAPFILL = "#e5e7eb"

    # ── 上带：page == 自然块（退化，as_strided 等于恒等）──
    yA = 110
    text(L, bx, yA - 14, "情形一  page == 自然块（16）：target_stride[0]=16=自然 stride[0] → 三块紧挨、无空隙（as_strided 退化成恒等）",
         size=13, fill=INK, anchor="start", weight="bold")
    for k in range(3):
        x0 = bx + k*16*ew
        box(L, x0, yA, 16*ew, 44, colors[k], colors[k], rx=3, sw=2)
        L[-1] = L[-1].replace(f'fill="{colors[k]}"', f'fill="{colors[k]}" fill-opacity="0.20"')
        text(L, x0 + 8*ew, yA + 27, f"block {k}", size=12.5, fill=colors[k], weight="bold", mono=True)
        text(L, x0 + 8*ew, yA + 62, f"offset {k}×16={k*16}", size=10.5, fill=SUB, mono=True)

    # ── 下带：page=24 > 自然块 16（padding，真正用上 as_strided）──
    yB = 250
    text(L, bx, yB - 14, "情形二  page=24 > 自然块 16（page_size_bytes 被 padding）：target_stride[0]=24 → 每块 16 元素数据 + 8 元素页对齐空隙",
         size=13, fill=INK, anchor="start", weight="bold")
    page_el = 24
    for k in range(3):
        x0 = bx + k*page_el*ew
        # 数据段（16 元素，彩色）
        box(L, x0, yB, 16*ew, 44, colors[k], colors[k], rx=3, sw=2)
        L[-1] = L[-1].replace(f'fill="{colors[k]}"', f'fill="{colors[k]}" fill-opacity="0.20"')
        text(L, x0 + 8*ew, yB + 27, f"block {k}", size=12.5, fill=colors[k], weight="bold", mono=True)
        # 空隙段（8 元素，灰）
        box(L, x0 + 16*ew, yB, 8*ew, 44, GAPFILL, GREY, rx=3, sw=1)
        text(L, x0 + 20*ew, yB + 28, "gap", size=10.5, fill=SUB, mono=True)
        text(L, x0 + 8*ew, yB + 62, f"offset {k}×24={k*24}", size=10.5, fill=SUB, mono=True)
    # 元素刻度
    for tick in (0, 16, 24, 40, 48, 64):
        text(L, bx + tick*ew, yB + 80, str(tick), size=9.5, fill=SUB)

    # 公式条
    fy = 400
    box(L, 90, fy, W - 180, 96, AMBERBG, AMBER, rx=12, sw=1.6)
    text(L, 110, fy + 28, "target_stride = (num_element_per_page, *stride[1:])", size=15, fill=INK, anchor="start", mono=True, weight="bold")
    text(L, 110, fy + 52, "block(第0)维强制成「一页」步长 num_element_per_page，其余维保持自然 stride", size=12.5, fill=SUB, anchor="start")
    text(L, 110, fy + 74, "对照基座：整块 view 自然 contiguous；仅 page_size 被 padding（情形二）时 as_strided 才露出页对齐空隙——多段视图据此在同一裸张量按页 overlap", size=12, fill=SUB, anchor="start")

    L.append('</svg>')
    return W, H, '\n'.join(L)


# ============================================================
# 图 3: bind-dispatch —— 绑定按模型分三路
# ============================================================
def gen_bind():
    W, H = 1120, 640
    L = svg_open(W, H)
    text(L, W/2, 38, "bind：把 reshape 好的 KV 挂回各层 —— 按模型走三条路", size=21, weight="bold")
    text(L, W/2, 62, "普通模型直接复用基座 bind_kv_cache；少数模型的层序/层数不规整，得特化", size=12.5, fill=SUB)

    # 顶部入口
    ex, ey, ew, eh = W/2 - 150, 92, 300, 50
    box(L, ex, ey, ew, eh, BLUEBG, BLUE, sw=2.5)
    text(L, W/2, ey + 22, "kv_caches  (reshape 产物)", size=14, fill=INK, weight="bold", mono=True)
    text(L, W/2, ey + 40, "model_config.hf_text_config.model_type ?", size=11.5, fill=SUB, mono=True)

    # 三条分支
    cols_x = [200, W/2, W - 200]
    by = 230
    bw, bh = 300, 130
    branches = [
        (PURP, PURPBG, "deepseek_v4", [
            "extract_dsv4_layer_index 排序",
            "MTP 层排到主模型层之后",
            "手填 self.kv_caches",
            "+ static_forward_context[l].kv_cache",
        ]),
        (BLUE, BLUEBG, "其它（普通模型）", [
            "bind_kv_cache(",
            "  num_attn_module = 1)",
            "直接复用基座绑定",
            "按 layer_index 默认排序",
        ]),
        (GREEN, GREENBG, "longcat_flash", [
            "bind_kv_cache(",
            "  num_attn_module = 2)",
            "每解码层含 2 个",
            "attention module",
        ]),
    ]
    conds = ["== deepseek_v4", "else", "== longcat_flash"]
    for i, (col, bg, title, lines) in enumerate(branches):
        cx = cols_x[i]
        bx = cx - bw/2
        box(L, bx, by, bw, bh, bg, col, sw=2)
        text(L, cx, by + 26, title, size=15, fill=col, weight="bold", mono=True)
        for j, ln in enumerate(lines):
            text(L, cx, by + 50 + j*20, ln, size=12, fill=INK, mono=True)
        arrow(L, W/2, ey + eh, cx, by, color=col)
        text(L, (W/2 + cx)/2, (ey + eh + by)/2 - 4, conds[i], size=11, fill=col)

    # 汇合：hamming sparse 追加
    hy = 410
    hx, hw, hh = W/2 - 240, 480, 56
    box(L, hx, hy, hw, hh, AMBERBG, AMBER, rx=12, sw=2, dash=True)
    text(L, W/2, hy + 24, "if enable_hamming_sparse:", size=13.5, fill=AMBER, weight="bold", mono=True)
    text(L, W/2, hy + 44, "init_and_bind_hashk_cache  额外初始化 hashk cache", size=12.5, fill=INK, mono=True)
    for cx in cols_x:
        arrow(L, cx, by + bh, W/2 if cx == W/2 else (hx if cx < W/2 else hx + hw), hy, color=AMBER, dash=True)

    # 出口
    oy = 510
    ox, ow, oh = W/2 - 170, 340, 50
    box(L, ox, oy, ow, oh, GREENBG, GREEN, sw=2)
    text(L, W/2, oy + 22, "self.kv_caches  +  forward_context", size=13.5, fill=INK, weight="bold", mono=True)
    text(L, W/2, oy + 40, "每层 forward 时就地取用", size=11.5, fill=GREEN)
    arrow(L, W/2, hy + hh, W/2, oy, color=GREEN)

    L.append('</svg>')
    return W, H, '\n'.join(L)


here = os.path.dirname(os.path.abspath(__file__))
for name, gen in [("alloc-geometry", gen_alloc), ("adjust-kv-layout", gen_layout), ("bind-dispatch", gen_bind)]:
    w, h, svg = gen()
    p = os.path.join(here, name + ".svg")
    with open(p, "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote", p, w, h)
