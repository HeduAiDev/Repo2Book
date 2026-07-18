#!/usr/bin/env python3
"""fig-ch01-book-map — 全书地图（详细目录图，只 ch01 画一次）。
7 Part × 33 章沿昇腾结构化下降链顺序展开；primer 原理篇（ch02/ch09）加徽标。
数据取自 book/cartography/outline-final.json（章号/标题/part/kind）。
两列排布，每 Part 一个容器（配主题色），每章一行 chNN + 短标题。
"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

# (Part 标题, 副标, [ (ch_id, 短标题, is_primer) ... ]) —— 唯一真相源：outline-final.json
PARTS = [
    ("Part I", "鸟瞰与达芬奇硬件模型", [
        ("ch01", "鸟瞰：三支柱与全书地图", False),
        ("ch02", "达芬奇硬件模型", True),
        ("ch03", "上手：vector-add GPU→NPU", False),
    ]),
    ("Part II", "语言层：CANN 扩展", [
        ("ch04", "双 builder 与内建路由", False),
        ("ch05", "显式内存层级 UB/GM/L1/L0C", False),
        ("ch06", "昇腾内建算子", False),
        ("ch07", "自定义算子框架 + libdevice", False),
        ("ch08", "scope/核间同步/流水提示", False),
    ]),
    ("Part III", "分水岭：Triton→Linalg", [
        ("ch09", "MLIR 与 Linalg", True),
        ("ch10", "分水岭：triton_adapter 总览", False),
        ("ch11", "PtrAnalysis：指针逆向工程", False),
        ("ch12", "BlockPtr → memref", False),
        ("ch13", "MaskAnalysis：边界语义", False),
        ("ch14", "Unstructure 兜底路径", False),
    ]),
    ("Part IV", "昇腾优化 pass：异构双核编排", [
        ("ch15", "AutoBlockify：网格折叠", False),
        ("ch16", "核亲和：Cube 还是 Vector", False),
        ("ch17", "Scope 切分与核间同步搬运", False),
        ("ch18", "DAGSSBuffer：UB 多缓冲流水", False),
        ("ch19", "离散访存的驯服", False),
    ]),
    ("Part V", "HFusion/HIVM 硬件 IR 与下降", [
        ("ch20", "TritonAscend 方言 + 逃生舱", False),
        ("ch21", "HFusion 方言：张量级融合", False),
        ("ch22", "FusionKind 与自动调度", False),
        ("ch23", "HIVM 方言：达芬奇硬件 IR", False),
        ("ch24", "HIVM 显式同步", False),
        ("ch25", "下降收官：→ AscendC 库调用", False),
    ]),
    ("Part VI", "后端与运行时", [
        ("ch26", "AscendBackend 契约", False),
        ("ch27", "三段下降链：add_stages", False),
        ("ch28", "闭源边界 bishengir-compile", False),
        ("ch29", "NPU 驱动与二进制装载", False),
        ("ch30", "动态发射器：rtKernelLaunch", False),
        ("ch31", "一套后端，两个框架", False),
    ]),
    ("Part VII", "度量与实战", [
        ("ch32", "实战：flash-attention CV 融合", False),
        ("ch33", "能力边界：测试套件谱系", False),
    ]),
]

TOTAL_PARTS = len(PARTS)
TOTAL_CHAPTERS = sum(len(p[2]) for p in PARTS)
PRIMER_IDS = [ch for p in PARTS for ch, _, is_p in p[2] if is_p]

COL_W = 500
COL_GAP = 30
PAD = 24
PART_HEAD_H = 42
CH_H = 27
PART_GAP = 16
TITLE_H = 66
LEGEND_H = 42

def part_h(p):
    return PART_HEAD_H + len(p[2]) * CH_H + 14

# 两列：左 I-IV（3+5+6+5=19 章），右 V-VII（6+6+2=14 章）
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
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-size="21" font-weight="bold" '
         f'fill="#0f172a">{esc(f"Triton-Ascend 源码解读 · 全书地图（{TOTAL_PARTS} Part / {TOTAL_CHAPTERS} 章）")}</text>')
L.append(f'<text x="{W/2}" y="52" text-anchor="middle" font-size="13" '
         f'fill="#64748b">{esc("同一 Triton 前端沿结构化下降链走到达芬奇 cube/vector 双核，每 Part 放大链上一段")}</text>')

part_index = {id(p): i for i, p in enumerate(PARTS)}
for c in (0, 1):
    x0 = PAD + c * (COL_W + COL_GAP)
    y = TITLE_H + PAD
    for p in cols[c]:
        idx = part_index[id(p)]
        accent = ACCENT[idx]
        ph = part_h(p)
        L.append(f'<rect x="{x0}" y="{y}" width="{COL_W}" height="{ph}" rx="10" '
                 f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>')
        L.append(f'<rect x="{x0}" y="{y}" width="6" height="{ph}" rx="3" fill="{accent}"/>')
        L.append(f'<text x="{x0+18}" y="{y+26}" font-size="16" font-weight="bold" '
                 f'fill="{accent}">{esc(p[0])}</text>')
        L.append(f'<text x="{x0+108}" y="{y+26}" font-size="13" fill="#475569">{esc(p[1])}</text>')
        cy = y + PART_HEAD_H
        for ch_id, title, is_primer in p[2]:
            if is_primer:
                L.append(f'<rect x="{x0+14}" y="{cy+2}" width="{COL_W-28}" height="{CH_H-5}" '
                         f'rx="5" fill="{PRIMER_BG}" stroke="{PRIMER_BD}" stroke-width="1.3" '
                         f'stroke-dasharray="3,2"/>')
            id_col = PRIMER_TXT if is_primer else accent
            ttl_col = PRIMER_TXT if is_primer else "#1e293b"
            L.append(f'<text x="{x0+20}" y="{cy+18}" font-size="12.5" font-weight="bold" '
                     f'fill="{id_col}">{esc(ch_id)}</text>')
            L.append(f'<text x="{x0+62}" y="{cy+18}" font-size="12.5" '
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

leg_y = H - LEGEND_H + 10
L.append(f'<rect x="{PAD}" y="{leg_y}" width="30" height="16" rx="4" fill="{PRIMER_BG}" '
         f'stroke="{PRIMER_BD}" stroke-width="1.3" stroke-dasharray="3,2"/>')
L.append(f'<text x="{PAD+38}" y="{leg_y+13}" font-size="12.5" fill="#475569">'
         f'{esc(f"= 原理篇（primer；先修数学/基础设施根基）——共 {len(PRIMER_IDS)} 章：" + "、".join(PRIMER_IDS))}</text>')

L.append('</svg>')
with open("fig-ch01-book-map.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print(f"wrote fig-ch01-book-map.svg ({W}x{H}) parts={TOTAL_PARTS} chapters={TOTAL_CHAPTERS} primers={PRIMER_IDS}")
