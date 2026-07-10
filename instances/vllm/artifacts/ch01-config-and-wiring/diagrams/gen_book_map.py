#!/usr/bin/env python3
"""全书地图（无高亮）：8 Part / 37 章。每 Part 一栏组，列出本 Part 各章短中文标题。
原理篇（primer 模式，交错插入 Part VI 的论文精读章）用浅色底+虚线框标出，图例见底部。
章序/Part 归属/标题唯一真相源：instances/vllm/book/cartography/outline-final.json
（短标题据各章 narrative/chapter.md 的 H1 提炼，比 outline.title 更贴近定稿）。
"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# (Part 标题, 副标, [ (ch_id, 短中文标题, is_primer) ... ])
PARTS = [
    ("Part I", "全局图景：一个请求从头到尾", [
        ("ch01", "导读：v1 心智模型 + 本书读法", False),
        ("ch02", "请求的一生（鸟瞰追踪）", False),
        ("ch03", "EngineArgs → VllmConfig 装配", False),
    ]),
    ("Part II", "异步三段式解耦（旗舰）", [
        ("ch04", "AsyncLLM 三段式异步解耦", False),
        ("ch05", "Stage 1 · 输入处理", False),
        ("ch06", "Parallel Sampling 扇出 n>1", False),
        ("ch07", "IPC 边界：ZMQ/msgpack", False),
        ("ch08", "Stage 3 · 输出处理", False),
        ("ch09", "增量去 token 化与 stop string", False),
        ("ch10", "Logprobs 装配与字节回退", False),
    ]),
    ("Part III", "EngineCore 内部：忙循环", [
        ("ch11", "EngineCore 与忙循环", False),
        ("ch12", "batch queue：流水线并行", False),
    ]),
    ("Part IV", "调度与 KV Cache", [
        ("ch13", "Token 为中心的连续批处理", False),
        ("ch14", "抢占与请求生命周期回流", False),
        ("ch15", "分页 KV 缓存：块池/前缀缓存", False),
        ("ch16", "KV 块分配与多注意力协调", False),
    ]),
    ("Part V", "执行：Worker/Runner/算子", [
        ("ch17", "Executor 与 Worker 生命周期", False),
        ("ch18", "持久化批次与输入准备", False),
        ("ch19", "前向与采样解耦", False),
        ("ch20", "分布式并行：组与集合通信", False),
        ("ch21", "异步通信与数据并行", False),
    ]),
    ("Part VI", "模型/算子/注意力/采样", [
        ("ch22", "模型契约与权重装载（Llama）", False),
        ("ch23", "CustomOp 与 torch.compile", False),
        ("ch24", "FlashAttention 原理：online-softmax→IO-aware", True),
        ("ch25", "注意力后端抽象与元数据", False),
        ("ch26", "量化数学：scale/GPTQ/AWQ/SmoothQuant", True),
        ("ch27", "Lightning Indexer 原理：敢扫全历史的打分器", True),
        ("ch28", "读整模型：DeepSeek-V4", False),
        ("ch29", "从模型代码到架构图", False),
        ("ch30", "Sampler 九步采样流水线", False),
        ("ch31", "EAGLE：特征自回归与树验证", True),
        ("ch32", "投机解码：提议与拒绝采样", False),
    ]),
    ("Part VII", "Prefill/Decode 分离", [
        ("ch33", "PD 分离 I：KV Connector 契约", False),
        ("ch34", "PD 分离 II：Worker 执行与后端", False),
    ]),
    ("Part VIII", "服务接口", [
        ("ch35", "离线 LLM API", False),
        ("ch36", "OpenAI 兼容服务器", False),
        ("ch37", "高级引擎运维：弹性扩缩与多轮", False),
    ]),
]

# 布局：两列 Part，竖排
COL_W = 500
COL_GAP = 30
PAD = 24
PART_HEAD_H = 40
CH_H = 26
PART_GAP = 16
TITLE_H = 56
LEGEND_H = 40


def part_h(p):
    return PART_HEAD_H + len(p[2]) * CH_H + 14


# 8 Part 分两列：左 I-IV（3+7+2+4=16 章），右 V-VIII（5+11+2+3=21 章）——按 Part 顺序切列，
# 阅读顺序仍是先左列从上到下再右列从上到下。
col_assign = [0, 0, 0, 0, 1, 1, 1, 1]
cols = {0: [], 1: []}
for i, p in enumerate(PARTS):
    cols[col_assign[i]].append(p)


def col_height(parts):
    return sum(part_h(p) + PART_GAP for p in parts) - PART_GAP


H = TITLE_H + PAD + max(col_height(cols[0]), col_height(cols[1])) + PAD + LEGEND_H
W = PAD + COL_W + COL_GAP + COL_W + PAD

ACCENT = {0: "#2563eb", 1: "#0891b2", 2: "#7c3aed", 3: "#059669",
          4: "#d97706", 5: "#db2777", 6: "#4f46e5", 7: "#0d9488"}
PRIMER_BG = "#fffbeb"
PRIMER_BD = "#f59e0b"
PRIMER_TXT = "#b45309"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="22" font-weight="bold" '
         f'fill="#0f172a">vLLM v1 源码解读 · 全书地图（8 Part / 37 章）</text>')

part_index = {id(p): i for i, p in enumerate(PARTS)}
for c in (0, 1):
    x0 = PAD + c * (COL_W + COL_GAP)
    y = TITLE_H + PAD
    for p in cols[c]:
        idx = part_index[id(p)]
        accent = ACCENT[idx]
        ph = part_h(p)
        flagship = "旗舰" in p[1]
        # Part 容器
        bg = "#fffbeb" if flagship else "#f8fafc"
        bd = "#f59e0b" if flagship else "#e2e8f0"
        sw = 2.5 if flagship else 1
        L.append(f'<rect x="{x0}" y="{y}" width="{COL_W}" height="{ph}" rx="10" '
                 f'fill="{bg}" stroke="{bd}" stroke-width="{sw}"/>')
        # Part 标题条
        L.append(f'<rect x="{x0}" y="{y}" width="6" height="{ph}" rx="3" fill="{accent}"/>')
        L.append(f'<text x="{x0+18}" y="{y+25}" font-size="16" font-weight="bold" '
                 f'fill="{accent}">{esc(p[0])}</text>')
        sub = p[1]
        L.append(f'<text x="{x0+104}" y="{y+25}" font-size="13" fill="#475569">{esc(sub)}</text>')
        if flagship:
            L.append(f'<rect x="{x0+COL_W-58}" y="{y+9}" width="46" height="20" rx="10" fill="#f59e0b"/>')
            L.append(f'<text x="{x0+COL_W-35}" y="{y+23}" text-anchor="middle" font-size="12" '
                     f'font-weight="bold" fill="white">旗舰</text>')
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
         f'= 原理篇（primer；论文精读，交错插入 Part VI，回指落地章）——'
         f'共 4 章：ch24 / ch26 / ch27 / ch31</text>')

L.append('</svg>')
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "book-map.svg"
open(out, "w", encoding="utf-8").write("\n".join(L))
print("wrote", out, f"({W}x{H})")
