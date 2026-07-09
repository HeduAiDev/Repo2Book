#!/usr/bin/env python3
"""paper-fig-1: 重绘自 deepseek-ai/EPLB README「example.png」(arXiv:2412.19437 附属
参考实现,非 arXiv 正文图)。原图已抓到:
https://raw.githubusercontent.com/deepseek-ai/EPLB/main/example.png —— 用 Read 亲眼
核对过版式:2 个 Node 方框(橙),每个 Node 内 4 个 GPU 方框(蓝),每个 GPU 方框内
Layer 0 / Layer 1 两行、每行 2 个物理槽位格子,格内数字 = 该槽位对应的逻辑专家 id。

数据来源 = README `Interface and Example` 一节 print(phy2log) 的真实输出(逐格核对
与原图完全一致,见下方 PHY2LOG 两行注释旁的核对记录),不是杜撰示意数字:
tensor([[ 5,  6,  5,  7,  8,  4,  3,  4, 10,  9, 10,  2,  0,  1, 11,  1],
        [ 7, 10,  6,  8,  6, 11,  8,  9,  2,  4,  5,  1,  5,  0,  3,  1]])

版式改动(非像素复制,信息结构对齐原图):原图 2 个 Node 左右并排、画布宽高比
6.5:1,超出本书画布预算(≤2.6:1);这里保留左右并排(更贴近原图“同一层跨节点看
布局”的阅读方式),改用加高的标题/图例/底部策略说明区把宽高比收进预算内。配色
套本书语言:Node=琥珀 #fef3c7/#d97706,GPU=靛蓝 #dbeafe/#1d4ed8(与本章其他图的
"核心机制蓝"同色系),槽位格=浅灰白/深色数字,复制样例高亮=琥珀。文字译中
(Node→节点,Layer→层;GPU 保留原词,是通用硬件术语不译)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(与 example-chapter-map.py 同口径):全角按 1.0x size,
    半角按 0.58x size,求和——中文字符是方块字,不能按半角系数估算否则算少导致
    图例文字被下一色块压住。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---- 数据(README 示例真实输出,逐格已核对与原图 example.png 一致) ----
NUM_NODES = 2
GPUS_PER_NODE = 4
SLOTS_PER_GPU = 2
NUM_REPLICAS = GPUS_PER_NODE * SLOTS_PER_GPU * NUM_NODES  # 16
NUM_LOGICAL_EXPERTS = 12
NUM_REDUNDANT = 4

PHY2LOG = [
    [5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],   # Layer 0
    [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1],    # Layer 1
]
LAYER_LABELS = ["层 0", "层 1"]

# 高亮样例:同一逻辑专家的两份冗余副本落在不同 GPU 上(专家 5 → node0/gpu0/slot0
# 与 node0/gpu1/slot0,均在 Layer 0 行,flat index 0 与 2)
HIGHLIGHT_LAYER = 0
HIGHLIGHT_EXPERT = 5
HIGHLIGHT_FLAT = [0, 2]


def flat_to_pos(flat):
    node = flat // (GPUS_PER_NODE * SLOTS_PER_GPU)
    rem = flat % (GPUS_PER_NODE * SLOTS_PER_GPU)
    gpu = rem // SLOTS_PER_GPU
    slot = rem % SLOTS_PER_GPU
    return node, gpu, slot


# ---- 尺寸常量 ----
SLOT_W, SLOT_H = 34, 32
SLOT_GAP = 6
GPU_PAD = 8
NODE_PAD = 12
NODE_GAP_X = 50
LEFT_LABEL_W = 56
PAD = 30
TOP_PAD = 20

GPU_W = 2 * SLOT_W + SLOT_GAP + 2 * GPU_PAD
NODE_W = GPUS_PER_NODE * GPU_W + 2 * NODE_PAD

TITLE_Y = TOP_PAD + 18
SUBTITLE_Y = TITLE_Y + 22
SUBTITLE2_Y = SUBTITLE_Y + 18
NODE_TOP = SUBTITLE2_Y + 30

NODE_TOP_PAD = 10
LAYER_GAP = 4
GPU_BAR_H = SLOTS_PER_GPU // 1 * 0 + 2 * SLOT_H + LAYER_GAP  # 2 层 * SLOT_H + 层间距
GPU_LABEL_GAP = 16
GPU_LABEL_H = 16
NODE_LABEL_GAP = 10
NODE_LABEL_H = 20
NODE_BOTTOM_PAD = 10
NODE_INNER_H = (NODE_TOP_PAD + GPU_BAR_H + GPU_LABEL_GAP + GPU_LABEL_H
                + NODE_LABEL_GAP + NODE_LABEL_H + NODE_BOTTOM_PAD)
NODE_BOTTOM = NODE_TOP + NODE_INNER_H

LEGEND_Y = NODE_BOTTOM + 30
SUMMARY_TOP = LEGEND_Y + 22
SUMMARY_H = 78
BOTTOM_PAD = 20

W = PAD * 2 + LEFT_LABEL_W + NODE_W * 2 + NODE_GAP_X
H = SUMMARY_TOP + SUMMARY_H + BOTTOM_PAD

INK = "#0f172a"
SUB = "#64748b"
NODE_FILL, NODE_STROKE = "#fef3c7", "#d97706"
GPU_FILL, GPU_STROKE = "#dbeafe", "#1d4ed8"
SLOT_FILL, SLOT_STROKE = "#f8fafc", "#334155"
HL_FILL, HL_STROKE = "#fde68a", "#b45309"


def node_x0(node):
    return PAD + LEFT_LABEL_W + node * (NODE_W + NODE_GAP_X)


def gpu_x0(node, gpu):
    return node_x0(node) + NODE_PAD + gpu * GPU_W


def cell_x0(node, gpu, slot):
    return gpu_x0(node, gpu) + GPU_PAD + slot * (SLOT_W + SLOT_GAP)


def cell_y0(layer):
    return NODE_TOP + NODE_TOP_PAD + layer * (SLOT_H + LAYER_GAP)


DEFS = ['<defs></defs>']
BODY = []

BODY.append(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="17" '
            f'font-weight="bold" fill="{INK}">'
            f'{esc("重绘参考图:EPLB README 示例——16 个物理副本的分层放置网格")}</text>')
BODY.append(f'<text x="{PAD}" y="{SUBTITLE_Y}" font-family="sans-serif" font-size="12" '
            f'fill="{SUB}">'
            f'{esc(f"2 层 MoE,每层 {NUM_LOGICAL_EXPERTS} 个逻辑专家 + {NUM_REDUNDANT} 个冗余副本 = {NUM_REPLICAS} 个物理副本;")}'
            '</text>')
BODY.append(f'<text x="{PAD}" y="{SUBTITLE2_Y}" font-family="sans-serif" font-size="12" '
            f'fill="{SUB}">'
            f'{esc(f"格内数字 = phy2log 张量给出的逻辑专家 id,按分层策略摊到 {NUM_NODES} 节点 x {GPUS_PER_NODE} GPU(每 GPU {SLOTS_PER_GPU} 个槽位)")}'
            '</text>')

# ---- 行标签(层 0 / 层 1),只画一次,靠画布最左 ----
for layer, label in enumerate(LAYER_LABELS):
    ly = cell_y0(layer) + SLOT_H / 2 + 4
    BODY.append(f'<text x="{PAD + LEFT_LABEL_W - 10:.1f}" y="{ly:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="13" font-weight="bold" '
                f'fill="{INK}">{esc(label)}</text>')

# ---- 3 条横向虚线导视,贯穿两个 Node(呼应原图的跨节点对齐线) ----
guide_x0 = PAD + LEFT_LABEL_W - 4
guide_x1 = node_x0(NUM_NODES - 1) + NODE_W
for gy in (NODE_TOP + NODE_TOP_PAD,
           NODE_TOP + NODE_TOP_PAD + SLOT_H + LAYER_GAP / 2,
           NODE_TOP + NODE_TOP_PAD + GPU_BAR_H):
    BODY.append(f'<line x1="{guide_x0:.1f}" y1="{gy:.1f}" x2="{guide_x1:.1f}" y2="{gy:.1f}" '
                f'stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4,4"/>')

highlight_centers = {}

for node in range(NUM_NODES):
    nx0 = node_x0(node)
    BODY.append(f'<rect x="{nx0:.1f}" y="{NODE_TOP:.1f}" width="{NODE_W:.1f}" '
                f'height="{NODE_INNER_H:.1f}" rx="8" fill="{NODE_FILL}" '
                f'stroke="{NODE_STROKE}" stroke-width="1.6"/>')

    for gpu in range(GPUS_PER_NODE):
        gx0 = gpu_x0(node, gpu)
        gpu_bar_top = NODE_TOP + NODE_TOP_PAD
        BODY.append(f'<rect x="{gx0:.1f}" y="{gpu_bar_top:.1f}" width="{GPU_W:.1f}" '
                    f'height="{GPU_BAR_H:.1f}" fill="{GPU_FILL}" stroke="{GPU_STROKE}" '
                    f'stroke-width="1.3"/>')

        for layer in range(len(PHY2LOG)):
            for slot in range(SLOTS_PER_GPU):
                flat = node * GPUS_PER_NODE * SLOTS_PER_GPU + gpu * SLOTS_PER_GPU + slot
                expert_id = PHY2LOG[layer][flat]
                cx0 = cell_x0(node, gpu, slot)
                cy0 = cell_y0(layer)
                is_hl = (layer == HIGHLIGHT_LAYER and flat in HIGHLIGHT_FLAT)
                fill = HL_FILL if is_hl else SLOT_FILL
                stroke = HL_STROKE if is_hl else SLOT_STROKE
                sw = 2.2 if is_hl else 1.2
                BODY.append(f'<rect x="{cx0:.1f}" y="{cy0:.1f}" width="{SLOT_W:.1f}" '
                            f'height="{SLOT_H:.1f}" rx="5" fill="{fill}" stroke="{stroke}" '
                            f'stroke-width="{sw}"/>')
                text_color = HL_STROKE if is_hl else INK
                BODY.append(f'<text x="{cx0+SLOT_W/2:.1f}" y="{cy0+SLOT_H/2+4:.1f}" '
                            f'text-anchor="middle" font-family="sans-serif" font-size="13" '
                            f'font-weight="bold" fill="{text_color}">{expert_id}</text>')
                if is_hl:
                    highlight_centers[flat] = (cx0 + SLOT_W / 2, cy0)

        gpu_label_y = gpu_bar_top + GPU_BAR_H + GPU_LABEL_GAP
        BODY.append(f'<text x="{gx0+GPU_W/2:.1f}" y="{gpu_label_y:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                    f'fill="{INK}">{esc(f"GPU {gpu}")}</text>')

    node_label_y = NODE_TOP + NODE_TOP_PAD + GPU_BAR_H + GPU_LABEL_GAP + GPU_LABEL_H + NODE_LABEL_GAP + NODE_LABEL_H - 4
    BODY.append(f'<text x="{nx0+NODE_W/2:.1f}" y="{node_label_y:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
                f'fill="#92400e">{esc(f"节点 {node}")}</text>')

# ---- 高亮样例的连接弧 + 标注(专家 5 的两份副本) ----
if len(highlight_centers) == 2:
    (x0, y0), (x1, y1) = highlight_centers[HIGHLIGHT_FLAT[0]], highlight_centers[HIGHLIGHT_FLAT[1]]
    arc_y = min(y0, y1) - 14
    mid_x = (x0 + x1) / 2
    BODY.append(f'<path d="M{x0:.1f},{y0:.1f} Q{mid_x:.1f},{arc_y:.1f} {x1:.1f},{y1:.1f}" '
                f'fill="none" stroke="{HL_STROKE}" stroke-width="1.6" stroke-dasharray="3,3"/>')
    BODY.append(f'<text x="{mid_x:.1f}" y="{arc_y-6:.1f}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
                f'fill="{HL_STROKE}">{esc(f"专家 {HIGHLIGHT_EXPERT} 的两份冗余副本")}</text>')

# ---- 图例 ----
LEGEND = [
    (NODE_FILL, NODE_STROKE, "节点(Node)"),
    (GPU_FILL, GPU_STROKE, "GPU 显存"),
    (SLOT_FILL, SLOT_STROKE, "物理槽位(数字=逻辑专家 id)"),
    (HL_FILL, HL_STROKE, "同一逻辑专家的冗余副本"),
]
LEGEND_FONT = 12
legend_x = PAD + LEFT_LABEL_W
for key_fill, key_stroke, label in LEGEND:
    BODY.append(f'<rect x="{legend_x:.1f}" y="{LEGEND_Y-13:.1f}" width="16" height="16" '
                f'rx="3" fill="{key_fill}" stroke="{key_stroke}" stroke-width="1.3"/>')
    BODY.append(f'<text x="{legend_x+22:.1f}" y="{LEGEND_Y:.1f}" font-family="sans-serif" '
                f'font-size="{LEGEND_FONT}" fill="{INK}">{esc(label)}</text>')
    legend_x += 22 + cjk_text_width(label, LEGEND_FONT) + 30

# ---- 底部策略说明条(区分分层策略示例 vs 本章 DefaultEplb 走的全局策略) ----
BODY.append(f'<rect x="{PAD}" y="{SUMMARY_TOP:.1f}" width="{W-2*PAD:.1f}" '
            f'height="{SUMMARY_H}" rx="10" fill="#eef2ff" stroke="#6366f1" stroke-width="1.8"/>')
BODY.append(f'<text x="{W/2:.1f}" y="{SUMMARY_TOP+30:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#3730a3">'
            f'{esc("此图是分层策略(Hierarchical)的示例:先按专家组把专家摊到节点,再节点内复制、按 GPU 装箱")}</text>')
BODY.append(f'<text x="{W/2:.1f}" y="{SUMMARY_TOP+54:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#4338ca">'
            f'{esc("本章 DefaultEplb 走的是另一支——全局策略(Global):跳过节点分组,把全部专家副本直接摊到所有 GPU 上")}</text>')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}">']
L.append(f'<rect width="{W:.0f}" height="{H:.0f}" fill="white"/>')
L += DEFS
L += BODY
L.append('</svg>')

out = Path(__file__).with_name("paper-fig-1.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  W={W:.0f} H={H:.0f} ratio={W/H:.2f}")
