#!/usr/bin/env python3
"""ch32 机制图 · 掩码行序不变式：ids 随包裹走（figure_spec ch32-fig-row-order，模板 tensor-flow）

放大自 L0 采样列·结构化输出组『表跨进程』段的展开——GrammarOutput 从调度列过 IPC 到
采样列 worker 侧重排（L2 站 2→12 的行算术放大），架构归属回指 L2 章图，
不另立第二种架构画法（FIGURE-SYSTEM §3）。

claim：一张 5 行紧凑掩码（调度序 A,B）过进程后按 worker 批序（B,A）+spec 偏移重排成
sorted_bitmask：A 的行 0 落到 logits 行 4、B 的行 1..4 落到行 0..3——行序权威是随掩码
同传的 ids 列表。

数字全部取自 figure_spec.numbers（紧凑掩码 5 行 / 批序走表 bi·offset→logit_index /
全覆盖 5==5 skip indices=None / 部分覆盖 out_indices=[5,0,1,2,3] / C 行 -1 全允许 /
玩具词表 V=64 行 i 只允许 token i*3 / GrammarOutput 定义 output.py:L286-L291）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 880
MX, BXR = 60, 1440

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>'
    f'<marker id="zmq" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
    f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ZMQ_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '行序权威随包裹走：ids 列表与掩码同传，worker 按自家批序 + spec 偏移重排', 16.5,
        lc.C_TXT, 'start', True, maxw=1150, tag='title')
lc.text(MX, 58, '调度序 [A,B] 装配紧凑掩码 5 行（只含结构化请求）→ GrammarOutput 双件套过进程 → worker 批序 [B,A]：'
               'B(bi 0, off 0) 起行 0 占 4 行、A(bi 1, off 3) 起行 4——玩具词表 V=64，行 i 只允许 token i*3，每行一个可分辨指纹',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列·结构化输出组 · L2 站 2→12 的行算术放大'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 上段：GrammarOutput 双件套过进程 =================
JY, JH = 92, 140
JW = 470
JX = (W - JW) / 2
lc.rect(JX, JY, JW, JH, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=10, sw=2.0)
lc.text(JX + 16, JY + 20, 'GrammarOutput —— 过进程的双件套（源码逐字）', 10.5, lc.C_ZMQ_S,
        'start', True, maxw=JW - 32, tag='j:t')
for j, ln in enumerate(['    # ids of structured output requests.',
                        '    structured_output_request_ids: list[str]',
                        '    # Bitmask ordered as structured_output_request_ids.',
                        '    grammar_bitmask: "npt.NDArray[np.int32]"']):
    lc.text(JX + 16, JY + 38 + j * 15, ln, 8.0, '#334155', 'start', maxw=JW - 30,
            tag='j:l' + str(j))
lc.text(JX + 16, JY + 104, '本例 ids=[A, B] · grammar_bitmask=ndarray 5×2 int32（V=64 → 2 int32/行）',
        8.2, lc.C_ZMQ_S, 'start', True, maxw=JW - 30, tag='j:l3')
lc.text(JX + 16, JY + 124, 'vllm/v1/core/sched/output.py:L286-L291', 7.8, lc.C_FAINT, 'start',
        maxw=JW - 30, tag='j:file')
# 左右两条汇聚箭头：调度侧 → 双件套 → worker 侧
lc.parrow([(JX - 130, JY + JH / 2), (JX, JY + JH / 2)], lc.C_ZMQ_S, 2.0, 'zmq')
lc.text(JX - 130 + 65, JY + JH / 2 - 24, '装配完成打包', 8.4, lc.C_MUTE, 'middle', maxw=130, tag='j:in')
lc.parrow([(JX + JW, JY + JH / 2), (JX + JW + 130, JY + JH / 2)], lc.C_ZMQ_S, 2.0, 'zmq')
lc.text(JX + JW + 65, JY + JH / 2 - 24, '搭 sample_tokens', 8.4, lc.C_MUTE, 'middle', maxw=130, tag='j:out')
lc.text(JX + JW + 65, JY + JH / 2 - 10, 'RPC 的车过进程', 8.4, lc.C_MUTE, 'middle', maxw=130, tag='j:out2')

# ================= 下段：5 行掩码 → sorted 5 行的重排映射 =================
LY0 = 260
ROW_H, ROW_GAP = 62, 8
LX, LW = 140, 330
RX, RW = 1030, 330

lc.text(LX + LW / 2, LY0 - 26, '调度侧 · 紧凑掩码（行序 = 调度序）', 11, lc.C_SAM_S, 'middle',
        True, maxw=LW, tag='lh')
lc.text(LX + LW / 2, LY0 - 10, '行 k 属于 ids[k] · 只含结构化请求的行', 8, lc.C_MUTE, 'middle',
        maxw=LW, tag='lh2')
lc.text(RX + RW / 2, LY0 - 26, 'worker 侧 · sorted_bitmask（行序 = worker 批序）', 11, lc.C_GPU_S,
        'middle', True, maxw=RW, tag='rh')
lc.text(RX + RW / 2, LY0 - 10, 'B(bi 0, off 0)→行 0 · A(bi 1, off 3)→行 4', 8, lc.C_MUTE,
        'middle', maxw=RW, tag='rh2')

# 行块：调度侧行 0..4（A 1 行 + B 4 行）；指纹 token = i*3
SRC = [(0, 'A', 0), (1, 'B', 3), (2, 'B', 6), (3, 'B', 9), (4, 'B', 12)]
# worker 侧 sorted 行 0..4：行 0..3 ← 掩码行 1..4（B），行 4 ← 掩码行 0（A）
DST = [(0, 1, 'B', 3), (1, 2, 'B', 6), (2, 3, 'B', 9), (3, 4, 'B', 12), (4, 0, 'A', 0)]


def row_block(x, y, w, who, src_label, fp, side):
    hot = who == 'A'
    f_, s_ = (lc.C_BADGE_F, lc.C_ENG_S) if hot else ('#ffffff', lc.C_SAM_S if side == 'src' else lc.C_GPU_S)
    lc.rect(x, y, w, ROW_H, f_, s_, rx=7, sw=1.4)
    lc.text(x + 14, y + 20, src_label, 9.6, lc.C_TXT, 'start', True, maxw=w - 150, tag='rb:' + src_label)
    lc.text(x + w - 14, y + 20, who, 9.6, s_, 'end', True, maxw=40, tag='rb:w' + src_label)
    lc.text(x + 14, y + 42, f'允许 token {fp}（指纹 = 行号×3，全表仅此一位亮）', 8, '#334155',
            'start', maxw=w - 28, tag='rb:f' + src_label)


for i, (r_, who, fp) in enumerate(SRC):
    row_block(LX, LY0 + i * (ROW_H + ROW_GAP), LW, who, f'掩码行 {r_}', fp, 'src')
for i, (r_, from_m, who, fp) in enumerate(DST):
    row_block(RX, LY0 + i * (ROW_H + ROW_GAP), RW, who, f'logits 行 {r_}（← 掩码行 {from_m}）', fp, 'dst')

# 重排连线（交叉束：A 的线跨过四条 B 线——这就是重排本身）
for i, (r_, who, fp) in enumerate(SRC):
    dst_i = next(j for j, d in enumerate(DST) if d[1] == r_)
    y1 = LY0 + i * (ROW_H + ROW_GAP) + ROW_H / 2
    y2 = LY0 + dst_i * (ROW_H + ROW_GAP) + ROW_H / 2
    color = lc.C_ENG_S if who == 'A' else lc.C_GPU_S
    lc.seg(LX + LW, y1, RX, y2, color, 1.8, 'sam')
lc.text((LX + LW + RX) / 2, LY0 + 5 * (ROW_H + ROW_GAP) + 18,
        '连线交叉 = 重排：A 的行 0 从调度序头名落到 worker 批序行 4（3 个 spec 草稿把 A 顶后 3 位）',
        8.4, lc.C_MUTE, 'middle', maxw=560, tag='cross')

# ================= 底部三注 =================
# 面板次序（左→右）：skip 快路径 | 错位后果 | 部分覆盖对照——「部分覆盖对照」必须落在最右：
# 图注按「右下对照部分覆盖」定位该面板（figure-integration 评审阻断项，2026-09-18 修）。
# 左→右顺带与正文 trace 表一致（出发→到达→重排→对照收尾）。
NY, NH = 664, 130
NW = (BXR - MX - 2 * 16) / 3
NOTES = [
    ('skip 快路径（全覆盖，本例）',
     ['行数恰好对满：掩码 5 行 == logits 5 行', '→ xgr 调用 indices=None 免传（张量都省了）',
      '对账机器可查：len(out_indices)==logits 行数'], lc.C_SAM_S),
    ('错位不重排的后果：静默出错',
     ['按批序直填：A 的行 0（只允许 token 0）会戴到', 'B 的第一个采样位头上——xgr 只管逐行写 -inf，',
      '无从知道行主是谁，且不报任何错'], lc.C_ABORT),
    ('部分覆盖对照（差一行都得传）',
     ['批里混入非语法请求 C：logits 6 行、掩码 5 行', '→ indices=[5,0,1,2,3] 显式传（int32 上卡）',
      'C 所在行 4 保持初始 -1 = 全允许（实测未被动过）'], lc.C_ZMQ_S),
]
for k, (t_, lines, c_) in enumerate(NOTES):
    x = MX + k * (NW + 16)
    lc.rect(x, NY, NW, NH, '#ffffff', c_, rx=8, sw=1.2, dash=(c_ == lc.C_ABORT))
    lc.text(x + 14, NY + 20, t_, 9.4, c_, 'start', True, maxw=NW - 28, tag='n' + str(k) + 't')
    for j, ln in enumerate(lines):
        lc.text(x + 14, NY + 40 + j * 17, ln, 8.2, '#334155', 'start', maxw=NW - 26,
                tag='n' + str(k) + 'l' + str(j))

# ================= 图例 + 页脚 =================
LY = 824
lx0 = MX
for c_, name in [(lc.C_SAM_S, '调度侧行（品红描边）'), (lc.C_GPU_S, 'worker 侧行（绿描边）'),
                 (lc.C_ENG_S, '请求 A 的行'), (lc.C_ZMQ_S, '过进程载体（GrammarOutput）')]:
    lc.seg(lx0 + 4, LY - 3, lx0 + 30, LY - 3, c_, 2.0, 'sam')
    lc.text(lx0 + 36, LY + 1, name, 8.8, lc.C_TXT, 'start', maxw=190, tag='lg:' + name[:6])
    lx0 += 36 + lc.tw(name, 8.8) + 18
lc.text(lx0, LY + 1, '指纹 toy 词表 V=64（真路径：真 apply_grammar_bitmask + 真 xgr.apply_token_bitmask_inplace）',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx0, tag='lg:tail')

lc.text(MX, 852, 'vllm/v1/core/sched/scheduler.py:L1646-L1668（行序账本：按 num_scheduled_tokens 迭代序收集 ids）'
                 '· vllm/v1/core/sched/output.py:L286-L291（GrammarOutput 定义）· vllm/v1/structured_output/utils.py:L113-L141（worker 重排循环）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 870, '5 行 / 批序走表 / 5==5 skip / out_indices=[5,0,1,2,3] / C 行全允许 / 指纹 0,3,6,9,12 ＝ 本章驱动脚本实测（真 utils + 真 xgr，V=64）'
                 '· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-row-order.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
