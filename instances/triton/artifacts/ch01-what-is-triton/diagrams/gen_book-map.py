#!/usr/bin/env python3
"""全书地图（详细目录图，非开篇窄长条 roadmap）：9 Part / 43 章。
每 Part 一张圆角卡片（主题色左边条 + Part 号 + 副标题），块内每章一行
「chNN + 短标题」，原理篇（kind=primer）浅色底+虚线框+右侧「原理」黄徽标。
底部图例列出共几章 primer 及章号。
章序/Part 归属/kind 唯一真相源：instances/triton/book/cartography/outline-final.json
"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# (Part 号, 副标, [ (ch_id, 短中文标题, is_primer) ... ])
PARTS = [
    ("Part I", "起步：一门 DSL 与它的目标机器", [
        ("ch01", "Triton 是什么 · 本书怎么读", False),
        ("ch02", "GPU 执行模型：SIMT 与内存层级", True),
        ("ch03", "一个 kernel 的一生（鸟瞰）", False),
    ]),
    ("Part II", "领域语言 tl.*：追踪期翻成 IR", [
        ("ch04", "tl.* 两层结构、@builtin 与 constexpr", False),
        ("ch05", "类型系统与 tensor", False),
        ("ch06", "类型提升与隐式广播", False),
        ("ch07", "造块、形状、访存与原子", False),
        ("ch08", "dot、归约与扫描", False),
        ("ch09", "自举标准库、extern 与随机数", False),
    ]),
    ("Part III", "宿主运行时：从 fn[grid] 到发射", [
        ("ch10", "@triton.jit、JITFunction 与缓存键", False),
        ("ch11", "run()：一次完整 launch", False),
        ("ch12", "driver、autotune 与磁盘缓存", False),
        ("ch13", "TRITON_INTERPRET：无 GPU 替身", False),
    ]),
    ("Part IV", "编译前端：把 AST 翻译成 TTIR", [
        ("ch14", "compile() 主循环与后端契约", False),
        ("ch15", "SSA 与结构化控制流", True),
        ("ch16", "CodeGenerator：AST→TTIR", False),
        ("ch17", "控制流下降到 scf", False),
        ("ch18", "双语桥：pybind11 绑定层", False),
    ]),
    ("Part V", "IR 与布局：编译器在摆弄什么", [
        ("ch19", "tt.* 词汇表与方言黏合层", False),
        ("ch20", "布局即函数", True),
        ("ch21", "Distributed 布局", False),
        ("ch22", "Shared 编码与 swizzle", False),
        ("ch23", "LinearLayout：统一所有布局", True),
        ("ch24", "ttg.* 与 ttng.* 算子", False),
    ]),
    ("Part VI", "优化 pass：朴素 IR 变高性能", [
        ("ch25", "AxisInfo 与 Coalesce", False),
        ("ch26", "共享内存分配与屏障", False),
        ("ch27", "Tensor Core 与 MMA 布局", True),
        ("ch28", "AccelerateMatmul 布局优化", False),
        ("ch29", "软件流水线与模调度", True),
        ("ch30", "软件流水线落地：建模与展开", False),
        ("ch31", "Prefetch 与 Warp Specialization", False),
    ]),
    ("Part VII", "降级：带布局的张量 IR 到 PTX", [
        ("ch32", "第一跳 TTIR→TTGIR 贴布局", False),
        ("ch33", "类型塌缩与 convert_layout 三路", False),
        ("ch34", "共享内存降级与访存向量化", False),
        ("ch35", "dot 指令选择与 PTX 出口", False),
    ]),
    ("Part VIII", "硬件后端：一块新卡怎么接进来", [
        ("ch36", "CUDABackend：五段 stages 注入", False),
        ("ch37", "PTX→cubin→装载→launcher", False),
        ("ch38", "AMD HIP 后端对照", False),
    ]),
    ("Part IX", "工具生态：量它读它部署它学它", [
        ("ch39", "proton、roofline 与 do_bench", False),
        ("ch40", "AOT compile/link 与反汇编", False),
        ("ch41", "triton-opt、tensor-layout、tutorials", False),
        ("ch42", "FlashAttention：在线 softmax", True),
        ("ch43", "收官实战：fused-attention 端到端", False),
    ]),
]

# 布局：两列 Part，竖排
COL_W = 530
COL_GAP = 32
PAD = 26
PART_HEAD_H = 40
CH_H = 26
PART_GAP = 16
TITLE_H = 60
LEGEND_H = 44


def part_h(p):
    return PART_HEAD_H + len(p[2]) * CH_H + 14


# 9 Part 分两列（保持顺序阅读）：左 I-V（3+6+4+5+6=24 章），右 VI-IX（7+4+3+5=19 章）。
col_assign = [0, 0, 0, 0, 0, 1, 1, 1, 1]
cols = {0: [], 1: []}
for i, p in enumerate(PARTS):
    cols[col_assign[i]].append(p)


def col_height(parts):
    return sum(part_h(p) + PART_GAP for p in parts) - PART_GAP


H = TITLE_H + PAD + max(col_height(cols[0]), col_height(cols[1])) + PAD + LEGEND_H
W = PAD + COL_W + COL_GAP + COL_W + PAD

# 9 Part 主题色（沿降级阶梯冷→暖渐变，色相区分相邻 Part）
ACCENT = {0: "#2563eb", 1: "#0891b2", 2: "#0d9488", 3: "#059669",
          4: "#7c3aed", 5: "#db2777", 6: "#e11d48", 7: "#d97706", 8: "#4f46e5"}
PRIMER_BG = "#fffbeb"
PRIMER_BD = "#f59e0b"
PRIMER_TXT = "#b45309"

primer_ids = [ch for p in PARTS for (ch, _t, isp) in p[2] if isp]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="38" text-anchor="middle" font-size="23" font-weight="bold" '
         f'fill="#0f172a">Triton 源码解读 · 全书地图（9 Part / 43 章）</text>')

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
        L.append(f'<text x="{x0+110}" y="{y+25}" font-size="13" fill="#475569">{esc(p[1])}</text>')
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
leg_y = H - LEGEND_H + 10
L.append(f'<rect x="{PAD}" y="{leg_y}" width="30" height="16" rx="4" fill="{PRIMER_BG}" '
         f'stroke="{PRIMER_BD}" stroke-width="1.3" stroke-dasharray="3,2"/>')
L.append(f'<text x="{PAD+38}" y="{leg_y+13}" font-size="12.5" fill="#475569">'
         f'= 原理篇（primer；前置数学/算法根基，回指后续落地章）——'
         f'共 {len(primer_ids)} 章：{" / ".join(primer_ids)}</text>')

L.append('</svg>')
with open("book-map.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote book-map.svg", f"({W}x{H})", "primers:", primer_ids)
