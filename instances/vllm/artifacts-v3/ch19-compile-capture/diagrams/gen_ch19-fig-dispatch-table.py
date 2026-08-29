#!/usr/bin/env python3
"""ch19 机制图 7 · dispatcher 查表三出口（figure_spec ch19-fig-dispatch-table，模板 state-table）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ③ 拍片
『dispatcher keys 预生成』（站 5）与 center ⑦ 拍片『一拍裁决·查表·padding』
（站 11，查表 dispatch）的机制展开。架构归属回指 L0/L2：右上角指北小签。

claim：查表三出口六拍实测：uniform decode 命 FULL（1 行白算）、非均匀落
PIECEWISE（key 放宽 num_reqs=None）、cascade 禁 FULL 同形状降级 PIECEWISE、
超界与 force_eager 落 NONE；FULL-only 档 num_reqs 精确到 padded 值；默认刻度
51+35=86 个 key 启动期预生成。

数字全部取自 figure_spec.numbers（六拍查表/两级 key 集/86 key 默认刻度/
NONE 早退五条件——精简版 companion host 实跑 + 源码逐字锚）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX = 60
BXR = 1440
MODE_C = {'FULL': lc.C_GPU_S, 'PIECEWISE': lc.C_API_S, 'NONE': lc.C_MUTE}

# ---------------- 标题区 ----------------
lc.text(MX, 34, '查表三出口：先 FULL 精确、再 PIECEWISE 放宽、都 miss 落 NONE',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '运行期每拍只做两次 set 查询（cudagraph_dispatcher.py:L235-L324）——key 是启动期一次性预生成的有限集，营业中一张不许手写',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ③⇢⑦ · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 两张 key 卡 + 中间查询流 ----------------
KY, KH_ = 92, 118
# FULL 卡
lc.rect(MX, KY, 430, KH_, '#ffffff', lc.C_GPU_S, rx=9, sw=1.6)
lc.text(MX + 14, KY + 22, 'FULL 精确档 keys（玩具刻度）', 9.8, lc.C_GPU_S, 'start', True,
        maxw=280, tag='kcF:t')
lc.rect(MX + 330, KY + 8, 88, 20, lc.C_BADGE_F, lc.C_GPU_S, rx=9, sw=1.0)
lc.text(MX + 374, KY + 22, '启动期预生成', 7.2, lc.C_GPU_S, 'middle', True, maxw=84, tag='kcF:b')
for i, k in enumerate(['(1, 1, True)', '(2, 2, True)', '(4, 4, True)']):
    kx = MX + 14 + i * 136
    lc.rect(kx, KY + 36, 128, 24, lc.C_GPU_F, lc.C_GPU_S, rx=5, sw=1.1)
    lc.text(kx + 64, KY + 52, k, 8.2, lc.C_GPU_S, 'middle', True, maxw=122, tag='kF' + k)
lc.text(MX + 14, KY + 80, 'num_reqs 精确——注释原话：FULL mode needs exact num_reqs', 7.8,
        '#334155', 'start', maxw=402, tag='kcF:l1')
lc.text(MX + 14, KY + 96, 'because FA3\'s scheduler_metadata computation depends on it（L189-L202）', 7.8,
        '#334155', 'start', maxw=402, tag='kcF:l2')
# PIECEWISE 卡
PX0 = 1010
lc.rect(PX0, KY, BXR - PX0, KH_, '#ffffff', lc.C_API_S, rx=9, sw=1.6)
lc.text(PX0 + 14, KY + 22, 'PIECEWISE 放宽档 keys', 9.8, lc.C_API_S, 'start', True,
        maxw=240, tag='kcP:t')
lc.rect(PX0 + 322, KY + 8, 88, 20, '#eff6ff', lc.C_API_S, rx=9, sw=1.0)
lc.text(PX0 + 366, KY + 22, '启动期预生成', 7.2, lc.C_API_S, 'middle', True, maxw=84, tag='kcP:b')
for i, k in enumerate(['(1, None, False)', '(2, None, False)', '(4, None, False)']):
    ky = KY + 34 + i * 26
    lc.rect(PX0 + 14, ky, 128, 22, '#eff6ff', lc.C_API_S, rx=5, sw=1.1)
    lc.text(PX0 + 78, ky + 15, k, 7.8, lc.C_API_S, 'middle', True, maxw=122, tag='kP' + k)
lc.text(PX0 + 156, KY + 48, 'num_reqs=None——图不在乎几个请求，', 7.8, '#334155', 'start',
        maxw=270, tag='kcP:l1')
lc.text(PX0 + 156, KY + 64, '一份放宽 key 服务所有同形状批；', 7.8, '#334155', 'start',
        maxw=270, tag='kcP:l2')
lc.text(PX0 + 156, KY + 80, 'uniform 一并放宽为 False。', 7.8, '#334155', 'start',
        maxw=270, tag='kcP:l3')
# 中间查询流
FX0, FW = 520, 460
steps = [('构 padded key', lc.C_MUTE), ('① 查 FULL 精确', lc.C_GPU_S),
         ('② 查 PIECEWISE 放宽', lc.C_API_S), ('NONE 兜底', lc.C_MUTE)]
sw_ = (FW - 3 * 14) / 4
for i, (t, c) in enumerate(steps):
    sx = FX0 + i * (sw_ + 14)
    lc.rect(sx, KY + 40, sw_, 38, '#ffffff', c, rx=7, sw=1.3)
    lc.text(sx + sw_ / 2, KY + 63, t, 8.2, c, 'middle', True, maxw=sw_ - 8, tag='fs' + t[:6])
    if i < 3:
        lc.seg(sx + sw_, KY + 59, sx + sw_ + 14, KY + 59, lc.C_MUTE, 1.3, marker='std')
lc.text(FX0 + FW / 2, KY + KH_ - 8, 'miss 才走下一步；白算行数 = padded − num_tokens ≥ 0', 8,
        lc.C_MUTE, 'middle', maxw=FW - 20, tag='fmid')

# ---------------- 六拍查表 ----------------
TY0 = 236
COLS = [(MX, 76, '拍'), (MX + 84, 292, 'dispatch 输入'),
        (MX + 384, 200, '构 key（tokens, reqs, uniform）'), (MX + 590, 232, 'FULL 精确查'),
        (MX + 828, 224, 'PIECEWISE 放宽查'), (MX + 1060, 130, '判定 mode'),
        (MX + 1196, 84, '白算行数')]
lc.seg(MX, TY0, MX + 1288, TY0, lc.C_MUTE, 1.4)
for cx, cwd, t in COLS:
    lc.text(cx, TY0 + 17, t, 8.3, lc.C_MUTE, 'start' if cwd > 90 else 'middle', True,
            maxw=cwd - 6, tag='col:' + t[:6])
lc.seg(MX, TY0 + 26, MX + 1288, TY0 + 26, lc.C_MUTE, 1.0)

ROWS = [
    ('拍1', 'num_tokens=3 · uniform decode（每请求 1 token）', '(4, 4, True)——3 上取整到 4、num_reqs=4',
     ('hit', '命中 (4,4,True)'), ('skip', '——'), 'FULL', '1'),
    ('拍2', 'num_tokens=3 · 非均匀 mixed 批', '(4, 4, False)',
     ('miss', 'miss——FULL keys 只收 uniform 档'), ('hit', '命中 (4,None,False)'), 'PIECEWISE', '1'),
    ('拍3', 'num_tokens=2 · uniform · cascade attention 禁 FULL', '(2, 2, True)',
     ('ban', '被 invalid_modes={FULL} 排除'), ('hit', '命中 (2,None,False)'), 'PIECEWISE', '0'),
    ('拍4', 'num_tokens=9 超 max_size=4（force_eager 同理）', '不构 key，早退——原样 9',
     ('skip', '—'), ('skip', '—'), 'NONE', '0'),
    ('FULL-only 档补充', 'num_tokens=3 · 非均匀', '(4, 4, False)——num_reqs 精确',
     ('hit', '命中 (4,4,False)'), ('none2', '该档无 PIECEWISE keys'), 'FULL', '1'),
    ('默认刻度', 'num_tokens=9（默认 51 档捕获表）', '(16, None, False)——9 上取整到 16',
     ('skip', '——'), ('hit', '命中'), 'PIECEWISE', '7'),
]
RY0, RH_ = TY0 + 26, 56
for ri, (beat, inp, key, fq, pq, mode, waste) in enumerate(ROWS):
    ry = RY0 + ri * RH_
    if ri == 4:
        lc.seg(MX, ry, MX + 1288, ry, lc.C_MUTE, 1.0, dash=True)
    elif ri > 0:
        lc.seg(MX, ry, MX + 1288, ry, '#e2e8f0', 1.0)
    cy = ry + RH_ / 2
    # 拍
    lc.text(COLS[0][0] + COLS[0][1] / 2, cy, beat, 7.6 if len(beat) > 3 else 8.2, lc.C_TXT,
            'middle', True, maxw=COLS[0][1] - 4, tag='b' + beat)
    # 输入 / 构 key
    lc.text(COLS[1][0] + 4, cy, inp, 8.0, '#334155', 'start', maxw=COLS[1][1] - 10,
            tag='in' + beat)
    lc.text(COLS[2][0] + 4, cy, key, 7.8, '#334155', 'start', maxw=COLS[2][1] - 10,
            tag='ky' + beat)
    # FULL / PW 查询结果
    for col, (st, txt) in ((3, fq), (4, pq)):
        cc = MODE_C.get('FULL') if col == 3 else MODE_C.get('PIECEWISE')
        cx0 = COLS[col][0] + 4
        if st == 'hit':
            lc.text(cx0, cy, '✓ ' + txt, 8.0, cc, 'start', True, maxw=COLS[col][1] - 10,
                    tag='fq%d%d' % (ri, col))
        elif st == 'miss':
            lc.text(cx0, cy, txt, 8.0, lc.C_MUTE, 'start', maxw=COLS[col][1] - 10,
                    tag='fq%d%d' % (ri, col))
        elif st == 'ban':
            lc.text(cx0, cy, '× ' + txt, 8.0, lc.C_ABORT, 'start', True,
                    maxw=COLS[col][1] - 10, tag='fq%d%d' % (ri, col))
        elif st == 'none2':
            lc.text(cx0, cy, txt, 8.0, lc.C_MUTE, 'start', maxw=COLS[col][1] - 10,
                    tag='fq%d%d' % (ri, col))
        else:
            lc.text(cx0, cy, txt, 8.0, lc.C_FAINT, 'start', maxw=COLS[col][1] - 10,
                    tag='fq%d%d' % (ri, col))
    # mode 徽标
    mbx, mbw = COLS[5][0] + 10, COLS[5][1] - 20
    lc.rect(mbx, cy - 13, mbw, 26, '#ffffff', MODE_C[mode], rx=12, sw=1.5)
    lc.text(mbx + mbw / 2, cy + 4, mode, 8.6, MODE_C[mode], 'middle', True, maxw=mbw - 6,
            tag='md' + beat)
    # 白算
    hot = waste != '0'
    lc.text(COLS[6][0] + COLS[6][1] / 2, cy, waste, 12, lc.C_BEAT_T if hot else lc.C_FAINT,
            'middle', True, maxw=COLS[6][1] - 4, tag='ws' + beat)
lc.seg(MX, RY0 + 6 * RH_, MX + 1288, RY0 + 6 * RH_, lc.C_MUTE, 1.4)

# 图例（标题带下一行）
LY_ = 78
lx = MX
for m, lab in (('FULL', 'FULL'), ('PIECEWISE', 'PIECEWISE'), ('NONE', 'NONE（eager 原样）')):
    lc.rect(lx, LY_ - 10, 16, 12, '#ffffff', MODE_C[m], rx=3, sw=1.2)
    lc.text(lx + 22, LY_, lab, 8, lc.C_TXT, 'start', maxw=140, tag='lg' + m)
    lx += 22 + lc.tw(lab, 8) + 24
lc.text(lx + 8, LY_, '玩具刻度：capture_sizes=[1,2,4] · max_num_seqs=8 · 档位 FULL_AND_PIECEWISE', 8,
        lc.C_MUTE, 'start', maxw=520, tag='lgtoy')

# ---------------- 底部两卡 ----------------
CY = RY0 + 6 * RH_ + 22
CH2 = 104
lc.rect(MX, CY, 700, CH2, '#ffffff', lc.C_GPU_S, rx=8, sw=1.3)
lc.text(MX + 14, CY + 20, '默认刻度（config/compilation.py:L698-L706 文档模式 · max_num_seqs=256）', 9.3,
        lc.C_GPU_S, 'start', True, maxw=672, tag='dc:t')
DCL = ['51 个捕获档 → 51 PIECEWISE + 35 FULL = 86 个查表 key 全部启动期生成',
       '（decode FULL 只收 ≤ max_num_tokens=256 的档）；运行期每拍零显存分配零编译。',
       'capture descs 降序（largest-first）：512 / 496 / 480 …']
for i, ln in enumerate(DCL):
    lc.text(MX + 14, CY + 40 + i * 18, ln, 8.2, '#334155', 'start', maxw=672, tag='dcl' + str(i))
lc.rect(780, CY, BXR - 780, CH2, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(794, CY + 20, 'bs→padded 表（段首保形、段内上取整）与 NONE 早退', 9.3, lc.C_TXT,
        'start', True, maxw=BXR - 808, tag='pc:t')
PCL = ['实测 {1:1, 2:2, 3:4, 4:4}——padded ≥ bs 恒成立，白算行数非负；',
       'NONE 早退五条件：not keys_initialized / mode==NONE / max_size is None /',
       'num_tokens > max_size / allowed_modes 只剩 NONE——大 prefill 即由此落 eager（L274-L281）。']
for i, ln in enumerate(PCL):
    lc.text(794, CY + 40 + i * 18, ln, 8.2, '#334155', 'start', maxw=BXR - 808, tag='pcl' + str(i))

# ---------------- 页脚 ----------------
lc.text(MX, CY + CH2 + 24, '逐字锚 vllm/v1/cudagraph_dispatcher.py:L235-L324（dispatch 两级查询+早退）· L189-L203（FULL num_reqs 精确注释 · PIECEWISE 放宽）· L93-L102（分段填充）· 行号基线 vLLM v0.27.1',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, CY + CH2 + 42, '六拍查表、两级 key 集、bs→padded 表、默认刻度 86 key 取自精简版 companion host 实跑（真实 _compute_bs_to_padded_graph_size + initialize_cudagraph_keys + dispatch）',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-dispatch-table.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
