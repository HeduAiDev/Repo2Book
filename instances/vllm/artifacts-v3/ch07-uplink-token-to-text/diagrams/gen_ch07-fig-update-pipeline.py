#!/usr/bin/env python3
"""ch07 机制图 2 · update() 四步流水与两本账（figure_spec ch07-fig-update-pipeline，模板 flow）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程上行泳道』）的去 token 化
工位——即本章 L2 章图 center 拍片 ④ 『增量去 token』的机制展开。架构归属回指 L2/L0。

claim：update() 的四步流水（跳 stop token → 逐 token decode_next 累积 → min_tokens 推进
stop_check_offset → check_stop_strings 窗口查找截断）维持两本账：id 账收下每个 token、
文本账只收被解码字符，命中停止串时同一调用内截断文本账——场景 A 实测 2+2+1 个 token
后文本账 2 字符、id 账 5 个。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 856
MX = 60
BXR = 1440
LX1 = 920            # 左列右缘
CUT = lc.C_ABORT     # 截断位
GUARD_F, GUARD_S = lc.C_ENG_F, lc.C_ENG_S   # 守卫区（暖色，图例声明）


def cell(x, y, w, h, txt, fill, stroke, tcol, tag, dashed=False, fs=9.5):
    lc.rect(x, y, w, h, fill, stroke, rx=4, sw=1.1, dash=dashed)
    lc.text(x + w / 2, y + h / 2 + 3.5, txt, fs, tcol, 'middle', True, maxw=w - 3, tag=tag)


def cells(x0, y, w, h, items, fill, stroke, tcol, dashed=False, gap=6):
    """items = [str]；返回右缘 x。"""
    x = x0
    for i, t in enumerate(items):
        cell(x, y, w, h, t, fill, stroke, tcol, 'c:' + t + str(i), dashed)
        x += w + gap
    return x - gap


# ---------------- 标题区 ----------------
lc.text(MX, 34, 'update() 的四步流水与两本账：id 账全收，文本账只收被解码字符',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '引擎每步送来一串 token id——弹掉整单退回的停止 token、逐 token 累积、守卫期推安全线、'
        '新增窗口找停止串，命中同拍截断文本账', 10.5, lc.C_MUTE, 'start', maxw=1020, tag='subtitle')
_ch = '放大自 L2 拍片 ④ 增量去 token · L0：API 进程上行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

CX = 490          # 左列中轴（箭头走线）

# ---------------- 输入条 ----------------
lc.rect(MX, 88, LX1 - MX, 54, lc.C_API_F, lc.C_API_S, rx=8, sw=1.5)
lc.text(MX + 16, 110, '输入：new_token_ids + stop_terminated 旗标（update 是每轮唯一入口，空输入早退）',
        10.5, lc.C_TXT, 'start', True, maxw=800, tag='in:t')
lc.text(MX + 16, 130, '场景 A 主线（stop=["END"] · min_tokens=0）：[65,66] → [69,78] → [68]'
        '（65="A" 66="B" 69="E" 78="N" 68="D"，byte 级 tokenizer 可心算）', 8.5, lc.C_MUTE,
        'start', maxw=800, tag='in:s')
lc.seg(CX, 142, CX, 166, lc.C_API_S, 2.0, 'dn')
lc.text(CX + 10, 160, 'update(new_token_ids, stop_terminated)', 8.5, lc.C_API_S, 'start',
        maxw=250, tag='a:in')

# ---------------- ① 弹 stop token ----------------
S1_Y, S1_H = 166, 102
lc.rect(MX, S1_Y, LX1 - MX, S1_H, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
lc.text(MX + 16, S1_Y + 22, '① 弹 stop token —— stop_terminated 且不含停止串：末 token 不进文本账（场景 B）',
        10.5, lc.C_TXT, 'start', True, maxw=770, tag='s1:t')
for i, ln in enumerate([
        'skipped_stop_token_id = new_token_ids[-1]（=69）→ new_token_ids = new_token_ids[:-1]（弹出不解码）',
        'id 账事后补登：循环后 token_ids.append(69)——外部 API 承诺的 token 序列包含它，账实相符',
        '实测场景 B：文本账 "C" 1 字符 vs id 账 [67,69] 2 个——两本账从此各走各的账本']):
    lc.text(MX + 16, S1_Y + 42 + i * 17, ln, 8.8, '#334155', 'start', maxw=830, tag='s1:l' + str(i))
lc.text(LX1 - 10, S1_Y + 22, 'detokenizer.py:L108-L127', 8.5, lc.C_FAINT, 'end', tag='s1:f')
lc.seg(CX, S1_Y + S1_H, CX, S1_Y + S1_H + 22, lc.C_API_S, 2.0, 'dn')

# ---------------- ② 逐 token decode_next：两本账 ----------------
S2_Y, S2_H = 290, 122
lc.rect(MX, S2_Y, LX1 - MX, S2_H, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
lc.text(MX + 16, S2_Y + 22, '② 逐 token decode_next 累积 —— 两本账分道（场景 A 终态）', 10.5,
        lc.C_TXT, 'start', True, maxw=700, tag='s2:t')
CELL_W, CELL_H, CELL_Y0 = 46, 28, S2_Y + 32
lc.text(232, CELL_Y0 + 18, 'id 账 token_ids', 9, lc.C_TXT, 'end', True, maxw=150, tag='s2:idl')
cells(240, CELL_Y0, CELL_W, CELL_H, ['65', '66', '69', '78', '68'], lc.C_API_S, lc.C_API_S,
      '#ffffff')
lc.text(530, CELL_Y0 + 18, '全收 5 个（含 68=D——命中那轮照登）· 实测 num_output_tokens=5',
        8.5, '#334155', 'start', maxw=380, tag='s2:idn')
TXT_Y = CELL_Y0 + 38
lc.text(232, TXT_Y + 18, '文本账 output_text', 9, lc.C_TXT, 'end', True, maxw=150, tag='s2:txl')
cells(240, TXT_Y, 40, CELL_H, ['A', 'B'], lc.C_API_F, lc.C_API_S, lc.C_TXT)
cut_x = 240 + 2 * (40 + 6)      # B 与 E 之间的截断位 = 332
cells(cut_x + 6, TXT_Y, 40, CELL_H, ['E', 'N', 'D'], '#f8fafc', lc.C_FAINT, lc.C_MUTE, dashed=True)
lc.seg(cut_x + 3, TXT_Y - 5, cut_x + 3, TXT_Y + CELL_H + 5, CUT, 1.6, dash=True)
lc.text(cut_x - 5, TXT_Y - 9, '✂', 11, CUT, 'end', tag='sc1')
lc.text(530, TXT_Y + 18, '只收被解码字符：A B 保留；E N D 命中停止串被同拍剪掉（→ ④）',
        8.5, '#334155', 'start', maxw=380, tag='s2:txn')
lc.seg(CX, S2_Y + S2_H, CX, S2_Y + S2_H + 22, lc.C_API_S, 2.0, 'dn')

# ---------------- ③ min_tokens 守卫 ----------------
S3_Y, S3_H = 434, 100
lc.rect(MX, S3_Y, LX1 - MX, S3_H, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
lc.text(MX + 16, S3_Y + 22, '③ min_tokens 守卫：安全线 stop_check_offset 一路推到文末（场景 C：min_tokens=3）',
        10.5, lc.C_TXT, 'start', True, maxw=790, tag='s3:t')
GB_Y, GB_H = S3_Y + 36, 30
lc.rect(236, GB_Y - 4, 3 * 40 + 2 * 6 + 8, GB_H + 8, GUARD_F, GUARD_F, rx=5, sw=1.0)
cells(240, GB_Y, 40, GB_H, ['A', 'B', 'C'], '#ffffff', GUARD_S, '#334155')
cells(240 + 3 * 46, GB_Y, 40, GB_H, ['D'], '#ffffff', lc.C_API_S, '#334155')
safe_x = 240 + 3 * 46 + 40 + 8
lc.seg(safe_x, GB_Y - 8, safe_x, GB_Y + GB_H + 8, GUARD_S, 2.0)
lc.text(safe_x + 8, GB_Y + 10, 'stop_check_offset = len(output_text)', 8.5, GUARD_S, 'start',
        True, maxw=300, tag='s3:safe')
lc.text(safe_x + 8, GB_Y + 25, '守卫期每个 token 都把安全线推到文末', 8, lc.C_MUTE, 'start',
        maxw=300, tag='s3:safes')
lc.text(240, GB_Y + GB_H + 20, '守卫区（num ≤ 3）：此区不查停——守卫内完成的停止串等于没长', 8.5,
        GUARD_S, 'start', True, maxw=460, tag='s3:gz')
lc.text(700, GB_Y + GB_H + 20, '第二道闸：num > min_tokens 才查（L892）', 8.5,
        lc.C_MUTE, 'start', maxw=215, tag='s3:gate')
lc.seg(CX, S3_Y + S3_H, CX, S3_Y + S3_H + 22, lc.C_API_S, 2.0, 'dn')

# ---------------- ④ check_stop_strings ----------------
S4_Y, S4_H = 556, 132
lc.rect(MX, S4_Y, LX1 - MX, S4_H, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
lc.text(MX + 16, S4_Y + 22, '④ check_stop_strings：只在新增窗口找，命中当场剪掉文本账（轮 3：新增 "D"）',
        10.5, lc.C_TXT, 'start', True, maxw=790, tag='s4:t')
WB_Y, WB_H = S4_Y + 34, 30
cells(240, WB_Y, 40, WB_H, ['A', 'B'], lc.C_API_F, lc.C_API_S, lc.C_TXT)
wcells_x0 = 240 + 2 * 46                      # E 起点
win_cells = cells(wcells_x0, WB_Y, 40, WB_H, ['E', 'N', 'D'], lc.C_API_F, lc.C_API_S, lc.C_TXT)
win_x1 = win_cells + 6
lc.rect(wcells_x0 - 3, WB_Y - 6, win_x1 - wcells_x0 + 6, WB_H + 12, 'none', lc.C_API_S,
        rx=6, sw=1.5, dash=True)
lc.text((wcells_x0 + win_x1) / 2, WB_Y + WB_H + 16, 'find 窗口 = 新增 1 格 + 回看 L−1 = 2 格（盖住 E N D）',
        8.5, lc.C_API_S, 'middle', True, maxw=300, tag='s4:win')
lc.seg(cut_x + 3, WB_Y - 8, cut_x + 3, WB_Y + WB_H + 8, CUT, 1.6, dash=True)
lc.text(MX + 16, S4_Y + 106, 'find 起点 = 1 − 新增 − 串长（负索引回看，绝不重扫全文）：新增 1 + 串长 3 → −3'
        '（另一实测例：新增 2 + 串长 2 → −3）', 8.8, '#334155', 'start', maxw=540, tag='s4:f')
lc.text(700, S4_Y + 98, '命中同拍截断：output_text[:2] → "AB"', 8.8, CUT, 'start', True,
        maxw=215, tag='s4:r')
lc.text(700, S4_Y + 116, '返回 stop_string = "END"', 8.8, CUT, 'start', True, maxw=215, tag='s4:r2')

# ---------------- 终态条 ----------------
RS_Y = S4_Y + S4_H + 20
lc.rect(MX, RS_Y, LX1 - MX, 54, lc.C_API_F, lc.C_API_S, rx=8, sw=1.4)
lc.text(MX + 16, RS_Y + 22, '两本账终态（场景 A 实测）：文本账 2 字符 vs id 账 5 个 · 场景 B：1 字符 / 2 个',
        10, lc.C_TXT, 'start', True, maxw=780, tag='rs:t')
lc.text(MX + 16, RS_Y + 42, 'stop_buffer_length = max(len(stop)) − 1 = 2（流式出口扣留——见取文本 holdback 图）',
        8.8, lc.C_MUTE, 'start', maxw=780, tag='rs:s')

# ---------------- 右栏：场景 C 小时间线 ----------------
RP_X, RP_W = 960, 480
RP_Y, RP_H = 88, 660
lc.rect(RP_X, RP_Y, RP_W, RP_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(RP_X + 16, RP_Y + 24, '场景 C 时间线：stop=["AB"] · min_tokens=3', 10.5, lc.C_TXT,
        'start', True, maxw=420, tag='rp:t')
lc.text(RP_X + 16, RP_Y + 42, '守卫吞掉守卫内完成的命中；守卫后新出现的立即剪', 8.5, lc.C_MUTE,
        'start', maxw=420, tag='rp:s')
ROWS = [
    dict(hdr='轮 5 · num=2 ≤ 3 —— 不查', cs=['A', 'B'], guard=2,
         v1='AB 完整躺进文本——安全线推到文末，守卫吞掉命中', v2='（stop_check_offset 推到 "AB" 末尾）'),
    dict(hdr='轮 6 · num=3 ≤ 3 —— 不查（守卫末轮）', cs=['A', 'B', 'C'], guard=3,
         v1='第二道闸（num > min_tokens）也未开', v2='文本照常产出，只是不查停'),
    dict(hdr='轮 7 · num=4 > 3 —— 门开了，但看不见', cs=['A', 'B', 'C', 'D'], guard=0, win=2,
         v1='窗口 new_char_count=1：find 起点 −2，回看 1 格只到 C——旧 AB（前 2 位）在窗外',
         v2='守卫内完成的停止串永久不可见'),
    dict(hdr='轮 8 · num=6 —— 新 AB 立即命中', cs=['A', 'B', 'C', 'D', 'A', 'B'], guard=0, win=2, cut=True,
         v1='新出现的 AB（新增 2）落进窗口：当场截回 "ABCD"、返回 "AB"', v2='守卫后新出现的停止串立即可见'),
]
ry = RP_Y + 58
C_RY = []
for r in ROWS:
    C_RY.append(ry)
    lc.text(RP_X + 16, ry + 14, r['hdr'], 9.5, lc.C_TXT, 'start', True, maxw=440,
            tag='r:h' + r['hdr'][:6])
    cy, ch_ = ry + 24, 26
    cw_ = 34
    if r['guard']:
        gw = r['guard'] * cw_ + (r['guard'] - 1) * 6 + 8
        lc.rect(RP_X + 12, cy - 4, gw, ch_ + 8, GUARD_F, GUARD_F, rx=5, sw=1.0)
    n = len(r['cs'])
    for i, t in enumerate(r['cs']):
        new = i >= n - r.get('win', 0)
        cell(RP_X + 16 + i * (cw_ + 6), cy, cw_, ch_, t,
             lc.C_API_F if not new else '#ffffff', lc.C_API_S, lc.C_TXT, 'rc:' + t + str(i))
    if r.get('win'):
        wx0 = RP_X + 16 + (n - r['win']) * (cw_ + 6) - 3
        wx1 = RP_X + 16 + (n - 1) * (cw_ + 6) + cw_ + 3
        lc.rect(wx0, cy - 7, wx1 - wx0, ch_ + 14, 'none', lc.C_API_S, rx=6, sw=1.4, dash=True)
        if r.get('cut'):       # 轮 8：截断位在新 AB 串首
            lc.seg(wx0 - 3, cy - 9, wx0 - 3, cy + ch_ + 9, CUT, 1.5, dash=True)
    lc.text(RP_X + 16, ry + 72, r['v1'], 8.3, '#334155', 'start', maxw=450, tag='r:v1' + r['hdr'][:6])
    lc.text(RP_X + 16, ry + 88, r['v2'], 8.3, lc.C_MUTE, 'start', maxw=450, tag='r:v2' + r['hdr'][:6])
    ry += 112
# 双生效注 + 终态
lc.text(RP_X + 16, ry + 6, '双生效：循环内推安全线（L882-L884）+ 门外过线才查（L892）', 8.5,
        GUARD_S, 'start', True, maxw=440, tag='rp:dbl')
lc.text(RP_X + 16, ry + 24, 'min_tokens 的优先级高于停止串：守卫期文本是强制产出', 8.3,
        lc.C_MUTE, 'start', maxw=440, tag='rp:pr')
lc.text(RP_X + 16, ry + 42, '终态实测：output_text="ABCD" · num_output_tokens=6', 8.3,
        lc.C_MUTE, 'start', maxw=440, tag='rp:fin')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = RP_Y + RP_H + 30
lx = MX
items = [('idcell', 'id 账收下'), ('txtcell', '文本账保留'), ('cutcell', '命中后被剪字符'),
         ('guardbox', 'min_tokens 守卫区'), ('cutline', '截断位'), ('winbox', 'find 窗口')]
for kind, name in items:
    if kind == 'idcell':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_API_S, lc.C_API_S, rx=3, sw=1.0)
    elif kind == 'txtcell':
        lc.rect(lx, LEG_Y - 8, 20, 13, lc.C_API_F, lc.C_API_S, rx=3, sw=1.0)
    elif kind == 'cutcell':
        lc.rect(lx, LEG_Y - 8, 20, 13, '#f8fafc', lc.C_FAINT, rx=3, sw=1.0, dash=True)
    elif kind == 'guardbox':
        lc.rect(lx, LEG_Y - 8, 20, 13, GUARD_F, GUARD_F, rx=3, sw=1.0)
    elif kind == 'cutline':
        lc.seg(lx, LEG_Y - 6, lx + 20, LEG_Y - 6, CUT, 1.5, dash=True)
    else:
        lc.rect(lx, LEG_Y - 10, 20, 15, 'none', lc.C_API_S, rx=4, sw=1.2, dash=True)
    lc.text(lx + 26, LEG_Y + 2, name, 9, lc.C_TXT, 'start', maxw=170, tag='leg' + name)
    lx += 26 + lc.tw(name, 9) + 20
lc.text(MX, LEG_Y + 28, '四步控制流 verbatim vllm/v1/engine/detokenizer.py:L96-L143 · 窗口起点 L337 · 安全线推进 PR #22014 · '
        '判定数字 host 实测（byte 级 tokenizer 为可心算 seam，判定逻辑为 pin 代码）', 9, lc.C_FAINT,
        'start', maxw=BXR - MX, tag='foot1')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch07-fig-update-pipeline.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
