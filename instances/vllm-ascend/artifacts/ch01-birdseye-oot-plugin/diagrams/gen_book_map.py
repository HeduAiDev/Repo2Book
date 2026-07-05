#!/usr/bin/env python3
"""全书地图（无高亮）：7 Part / 36 章。每 Part 一栏组，列出本 Part 各章短中文标题。
原理篇（primer 模式，前置数学/算法根基章）用浅色底+虚线框标出，图例见底部。
"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# (Part 标题, 副标, [ (ch_id, 短中文标题, is_primer) ... ])
# 章序/Part 归属唯一真相源：instances/vllm-ascend/book/cartography/outline-final.json
PARTS = [
    ("Part I", "插件如何挂进 vLLM（三支柱）", [
        ("ch01", "鸟瞰：OOT 插件三支柱", False),
        ("ch02", "entry points 与 NPUPlatform", False),
        ("ch03", "两段式 monkey-patch", False),
        ("ch04", "引擎核心 KV-cache patch", False),
        ("ch05", "check_and_update_config", False),
    ]),
    ("Part II", "通信器与 sleep-mode 分配器", [
        ("ch06", "NPUCommunicator 通信器", False),
        ("ch07", "sleep-mode 与 camem 分配器", False),
    ]),
    ("Part III", "组·热迁移·PD 分离", [
        ("ch08", "昇腾并行组：MC2/TP/CP", False),
        ("ch09", "EPLB 均衡算法本体", True),
        ("ch10", "EPLB：子进程规划+D2D迁移", False),
        ("ch11", "PD 分离：mooncake P2P", False),
        ("ch12", "KV 池化与 ascend_store", False),
        ("ch13", "KV 卸载：host/CPU", False),
    ]),
    ("Part IV", "NPUWorker 与 NPUModelRunner", [
        ("ch14", "NPUWorker 重写", False),
        ("ch15", "NPUModelRunner 继承+猴补", False),
        ("ch16", "单步前向 execute_model", False),
        ("ch17", "KV cache 落地：分配/绑定", False),
        ("ch18", "310P 推理芯片特化", False),
    ]),
    ("Part V", "NPU 后端特化", [
        ("ch19", "注意力后端选择", False),
        ("ch20", "AscendAttention 标准 MHA", False),
        ("ch21", "MLA：低秩压缩/解耦 RoPE", True),
        ("ch22", "MLA 在 NPU 上：权重吸收", False),
        ("ch23", "稀疏注意力谱系 NSA→DSA", True),
        ("ch24", "稀疏注意力 SFA/DSA", False),
        ("ch25", "KV 管理与调度器", False),
        ("ch26", "V4 CSA/HCA 压缩注意力", True),
    ]),
    ("Part VI", "换头不换身", [
        ("ch27", "CustomOp OOT 顶替", False),
        ("ch28", "torch.library 与 meta 注册", False),
        ("ch29", "AscendCompiler / ACLGraph", False),
        ("ch30", "FusedMoE / batch-invariant", False),
    ]),
    ("Part VII", "扩展点的注册范式", [
        ("ch31", "量化数学：scale/GPTQ/AWQ", True),
        ("ch32", "昇腾量化框架", False),
        ("ch33", "采样的 NPU 对位", False),
        ("ch34", "投机采样：拒绝采样定理", True),
        ("ch35", "投机解码 proposer 工厂", False),
        ("ch36", "模型/LoRA/netloader 注册", False),
    ]),
]

# 布局：两列 Part，竖排
COL_W = 490
COL_GAP = 30
PAD = 24
PART_HEAD_H = 40
CH_H = 26
PART_GAP = 16
TITLE_H = 56
LEGEND_H = 40


def part_h(p):
    return PART_HEAD_H + len(p[2]) * CH_H + 14


# 7 Part 分两列：左 I-IV（5+2+6+5=18 章），右 V-VII（8+4+6=18 章）——两列章数均衡。
col_assign = [0, 0, 0, 0, 1, 1, 1]
cols = {0: [], 1: []}
for i, p in enumerate(PARTS):
    cols[col_assign[i]].append(p)


def col_height(parts):
    return sum(part_h(p) + PART_GAP for p in parts) - PART_GAP


H = TITLE_H + PAD + max(col_height(cols[0]), col_height(cols[1])) + PAD + LEGEND_H
W = PAD + COL_W + COL_GAP + COL_W + PAD

ACCENT = {0: "#2563eb", 1: "#0891b2", 2: "#7c3aed", 3: "#059669",
          4: "#d97706", 5: "#db2777", 6: "#4f46e5"}
PRIMER_BG = "#fffbeb"
PRIMER_BD = "#f59e0b"
PRIMER_TXT = "#b45309"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="22" font-weight="bold" '
         f'fill="#0f172a">vLLM-Ascend 源码解读 · 全书地图（7 Part / 36 章）</text>')

part_index = {id(p): i for i, p in enumerate(PARTS)}
for c in (0, 1):
    x0 = PAD + c * (COL_W + COL_GAP)
    y = TITLE_H + PAD
    for p in cols[c]:
        idx = part_index[id(p)]
        accent = ACCENT[idx]
        ph = part_h(p)
        # Part 容器
        L.append(f'<rect x="{x0}" y="{y}" width="{COL_W}" height="{ph}" rx="10" '
                 f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>')
        # Part 标题条
        L.append(f'<rect x="{x0}" y="{y}" width="6" height="{ph}" rx="3" fill="{accent}"/>')
        L.append(f'<text x="{x0+18}" y="{y+25}" font-size="16" font-weight="bold" '
                 f'fill="{accent}">{esc(p[0])}</text>')
        L.append(f'<text x="{x0+104}" y="{y+25}" font-size="13" fill="#475569">{esc(p[1])}</text>')
        # 章列表
        cy = y + PART_HEAD_H
        for ch_id, title, is_primer in p[2]:
            if is_primer:
                L.append(f'<rect x="{x0+14}" y="{cy+2}" width="{COL_W-28}" height="{CH_H-5}" '
                         f'rx="5" fill="{PRIMER_BG}" stroke="{PRIMER_BD}" stroke-width="1.3" '
                         f'stroke-dasharray="3,2"/>')
            id_col = PRIMER_TXT if is_primer else accent
            ttl_col = PRIMER_TXT if is_primer else "#1e293b"
            L.append(f'<text x="{x0+20}" y="{cy+17}" font-size="12.5" font-weight="bold" '
                     f'fill="{id_col}">{esc(ch_id)}</text>')
            L.append(f'<text x="{x0+62}" y="{cy+17}" font-size="12.5" '
                     f'fill="{ttl_col}">{esc(title)}</text>')
            if is_primer:
                badge_w = 34
                bx = x0 + COL_W - 14 - badge_w
                L.append(f'<rect x="{bx}" y="{cy+4}" width="{badge_w}" height="16" rx="8" '
                         f'fill="{PRIMER_BD}"/>')
                L.append(f'<text x="{bx+badge_w/2}" y="{cy+16}" text-anchor="middle" '
                         f'font-size="10" font-weight="bold" fill="white">原理</text>')
            cy += CH_H
        y += ph + PART_GAP

# 图例（原理篇标记说明）
leg_y = H - LEGEND_H + 8
L.append(f'<rect x="{PAD}" y="{leg_y}" width="30" height="16" rx="4" fill="{PRIMER_BG}" '
         f'stroke="{PRIMER_BD}" stroke-width="1.3" stroke-dasharray="3,2"/>')
L.append(f'<text x="{PAD+38}" y="{leg_y+13}" font-size="12.5" fill="#475569">'
         f'= 原理篇（primer；前置数学/算法根基，回指后续落地章）——'
         f'共 6 章：ch09 / ch21 / ch23 / ch26 / ch31 / ch34</text>')

L.append('</svg>')
with open("book-map.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote book-map.svg", f"({W}x{H})")
