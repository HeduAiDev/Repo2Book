#!/usr/bin/env python3
"""ch05 机制图 · 线格式字节剖面（explainer m3 figure_spec ch05-fig-wire-format）

放大自 L0 紫色 ZMQ 边界带「一条 ADD 消息过线」的那一竖条——即本章 L2 站 5-6
「② ROUTER 过线 → ③ DEALER 判型」之间的线格式展开（帧字节自上而下堆叠剖面）。

claim：线格式三段式 (Identity, Type, *Payload)：标签字节 1B 免二次编码；张量 ≥256B
才多出一条 aux 零拷贝帧——发 4 帧、引擎实收 3 帧，identity 信封被投递层吃掉。

数字全部取自 explainer figure_spec.numbers（wire 实测帧长/帧数 + pin 锚点）；
帧高按对数刻度（真实字节数标注在帧内）；坐标由常量/循环计算；文本全 esc()。
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1280, 1000
MX = 96
BXR = 1224


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:12])
    return x


def fh(nbytes):
    """帧高：对数刻度（18 + 14·ln(nbytes)）。"""
    return 18 + 14 * math.log(nbytes)


def frame(cx, y, nbytes, label, stroke, fill, dash=False, narrow=False, label_in=True,
          tag=''):
    """一帧 = 纵向矩形块（窄条用于 1B 标签帧），帧内/旁标注实测长度。返回底边 y。"""
    w = 190 if not narrow else 26
    h = fh(nbytes)
    lc.rect(cx - w / 2, y, w, h, fill, stroke, rx=4, sw=1.5, dash=dash)
    if label_in and h >= 24:
        lc.text(cx, y + h / 2 + 3, label, 8.5, lc.C_TXT, 'middle', maxw=w - 8,
                tag=(tag or label[:10]))
    else:
        lc.text(cx + w / 2 + 8, y + h / 2 + 3, label, 8.5, lc.C_MUTE, 'start',
                maxw=140, tag=(tag or label[:10]))
    return y + h


C_TYPE_S, C_AUX_S, C_AUX_F = lc.C_ZMQ_S, lc.C_GPU_S, lc.C_GPU_F
C_ENV_S = lc.C_FAINT

# ---------------- 标题区 ----------------
lc.text(MX, 36, '一条 ADD 消息的字节剖面：三段式 (Identity, Type, *Payload)', 16.5,
        lc.C_TXT, 'start', True, maxw=820, tag='title')
lc.text(MX, 60,
        "首帧标签字节 = enum 字节值本身（'sent over sockets without separate encoding step'）；"
        'identity 信封被投递层吃掉——发送侧与引擎实收恰差这一条',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
chip(BXR, 12, '放大自 L2 站 5-6「② 过线 → ③ 判型」· L0：紫色 ZMQ 边界带', lc.C_ZMQ_S)

lc.text(MX, 94, '发送侧拼帧 (Identity, Type, *Payload) · _send_input → send_multipart(copy=False)'
        '（core_client.py:L1113-L1123）', 9.5, lc.C_API_S, 'start', maxw=1080, tag='snd')

# ---------------- 三列定义 ----------------
COLS = [
    dict(cx=270, head='① 无张量 req-plain', cnt='发 3 帧 → 收 2 帧',
         sender=[('env', 2, 'identity 信封 2B · 0000 小端'),
                 ('type', 1, 'type 0x00 · 1B'),
                 ('main', 94, '主帧 94B（msgpack）')],
         recv=[('type', 1, 'type 0x00 · 1B'),
               ('main', 94, '主帧 94B（msgpack）')],
         note=['无张量：载荷帧恰 1 条主帧',
               '解码还原 req-plain · token ids [1,2,3]']),
    dict(cx=640, head='② 小张量 32B（＜256B）', cnt='发 3 帧 → 收 2 帧',
         sender=[('env', 2, 'identity 信封 2B · 0000 小端'),
                 ('type', 1, 'type 0x00 · 1B'),
                 ('main', 139, '主帧 139B（32B 张量内联）')],
         recv=[('type', 1, 'type 0x00 · 1B'),
               ('main', 139, '主帧 139B（32B 张量内联）')],
         note=['32B ＜ 256B → 张量内联进主帧', '94B → 139B（+45B = 32B 张量 + 13B msgpack 开销）']),
    dict(cx=1010, head='③ 大张量 8192B（≥256B）', cnt='发 4 帧 → 收 3 帧',
         sender=[('env', 2, 'identity 信封 2B · 0000 小端'),
                 ('type', 1, 'type 0x00 · 1B'),
                 ('main', 105, '主帧 105B（dtype/shape/aux 索引）'),
                 ('aux', 8192, 'aux 帧 8192B（零拷贝 memoryview）')],
         recv=[('type', 1, 'type 0x00 · 1B'),
               ('main', 105, '主帧 105B（dtype/shape/aux 索引）'),
               ('aux', 8192, 'aux 帧 8192B（零拷贝 memoryview）')],
         note=['≥256B → 张量字节躺独立 aux 帧', '主帧只剩 (dtype, shape, aux 索引) 三元组']),
]
GAP = 6
STACK_Y0 = 150

# ---------------- 发送侧帧堆叠 ----------------
send_bottom = {}
for col in COLS:
    cx = col['cx']
    lc.text(cx, 116, col['head'], 11.5, lc.C_TXT, 'middle', True, maxw=330,
            tag='h:' + col['head'][:8])
    lc.text(cx, 134, col['cnt'], 9, lc.C_MUTE, 'middle', maxw=330,
            tag='c:' + col['head'][:8])
    y = STACK_Y0
    for kind, n, lab in col['sender']:
        if kind == 'env':
            y = frame(cx, y, n, lab, C_ENV_S, '#ffffff', dash=True, tag='env') + GAP
        elif kind == 'type':
            y = frame(cx, y, n, lab, C_TYPE_S, lc.C_ZMQ_F, narrow=True,
                      label_in=False, tag='ty') + GAP
        elif kind == 'main':
            y = frame(cx, y, n, lab, lc.C_API_S, '#ffffff', tag='mn') + GAP
        else:
            y = frame(cx, y, n, lab, C_AUX_S, C_AUX_F, tag='ax') + GAP
    send_bottom[cx] = y - GAP

# ---------------- 紫色投递带 ----------------
BAND_Y0 = max(send_bottom.values()) + 16
BAND_H = 70
lc.rect(MX, BAND_Y0, BXR - MX, BAND_H, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=10, sw=2.0)
lc.text(MX + 16, BAND_Y0 + 22, 'ZMQ 投递层', 10, lc.C_ZMQ_S, 'start', True, tag='bd')
lc.text(MX + 16, BAND_Y0 + 40, '信封用完即弃', 9, lc.C_MUTE, 'start', tag='bd2')

RECV_Y0 = BAND_Y0 + BAND_H + 16
recv_bottom = {}
for col in COLS:
    cx = col['cx']
    # 信封幽灵框（带内右侧，被吃掉 = 画 X）
    gw, gh, gx, gy = 66, 24, cx + 34, BAND_Y0 + 23
    lc.rect(gx, gy, gw, gh, 'none', C_ENV_S, rx=3, sw=1.2, dash=True)
    lc.seg(gx, gy, gx + gw, gy + gh, C_ENV_S, 1.2)
    lc.seg(gx, gy + gh, gx + gw, gy, C_ENV_S, 1.2)
    lc.text(gx + gw + 8, gy + 16, '信封 2B 被吃掉', 8, C_ENV_S, 'start', maxw=110,
            tag='ghost:' + str(cx))
    # 过线箭头：发送侧末帧底 → 接收侧首帧顶
    lc.seg(cx, send_bottom[cx], cx, RECV_Y0, lc.C_API_S, 2.6, 'dn')
    # 接收侧帧堆叠
    y = RECV_Y0
    for kind, n, lab in col['recv']:
        if kind == 'type':
            y = frame(cx, y, n, lab, C_TYPE_S, lc.C_ZMQ_F, narrow=True,
                      label_in=False, tag='rty') + GAP
        elif kind == 'main':
            y = frame(cx, y, n, lab, lc.C_API_S, '#ffffff', tag='rmn') + GAP
        else:
            y = frame(cx, y, n, lab, C_AUX_S, C_AUX_F, tag='rax') + GAP
    recv_bottom[cx] = y - GAP

lc.text(BXR, RECV_Y0 - 8, '引擎侧实收 · recv_multipart（core.py:L1705）：frames[0] = type_frame · '
        'frames[1:] = 载荷帧', 9, lc.C_ENG_S, 'end', maxw=640, tag='rcv')

# 列注（接收堆叠下方）
NOTE_Y = max(recv_bottom.values()) + 24
for col in COLS:
    for i, ln in enumerate(col['note']):
        lc.text(col['cx'], NOTE_Y + i * 16, ln, 9, '#334155', 'middle', maxw=330,
                tag='nt:' + ln[:8])

# ---------------- 底部两面板：阈值对照 + 标签字节表 ----------------
PNL_Y = NOTE_Y + 44
PNL_H = 66
lc.rect(MX, PNL_Y, 600, PNL_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(MX + 14, PNL_Y + 18, '内联阈值 256B：分界恰在 256——不小于阈值即上 aux 帧', 9.5,
        lc.C_TXT, 'start', True, maxw=572, tag='pnl1:t')
lc.text(MX + 14, PNL_Y + 38, 'obj.nbytes ＜ 256 内联（envs.py:L208）：252B（63×float32）→ 1 帧',
        9, '#334155', 'start', maxw=572, tag='pnl1:l1')
lc.text(MX + 14, PNL_Y + 55, '256B（64×float32）→ 2 帧（aux 独立帧）', 9, '#334155',
        'start', maxw=572, tag='pnl1:l2')
lc.rect(724, PNL_Y, 500, PNL_H, '#ffffff', lc.C_MUTE, rx=7, sw=1.3)
lc.text(738, PNL_Y + 18, '标签字节表 EngineCoreRequestType（__init__.py:L261-L274）', 9.5,
        lc.C_TXT, 'start', True, maxw=472, tag='pnl2:t')
lc.text(738, PNL_Y + 38, '0x00 ADD · 0x01 ABORT（ids 帧 9B）· 0x02 START_DP_WAVE（DP 控制面 · ch34）',
        8.5, '#334155', 'start', maxw=472, tag='pnl2:l1')
lc.text(738, PNL_Y + 55, '0x03 UTILITY（四元组帧 32B）· 0x04 / 0x05 引擎内部哨兵，不过线', 8.5,
        '#334155', 'start', maxw=472, tag='pnl2:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = PNL_Y + PNL_H + 26
items = [
    ('swatch', lc.C_ZMQ_S, lc.C_ZMQ_F, 'type 标签帧（enum 字节值，1B）'),
    ('swatch', lc.C_API_S, '#ffffff', 'msgpack 主帧'),
    ('swatch', C_AUX_S, C_AUX_F, 'aux 张量数据帧'),
    ('dash', C_ENV_S, None, 'identity 信封（仅发送侧存在）'),
]
lx = MX
for kind, s, f, name in items:
    if kind == 'swatch':
        lc.rect(lx, LEG_Y - 9, 16, 11, f, s, rx=3, sw=1.6)
    else:
        lc.rect(lx, LEG_Y - 9, 16, 11, 'none', s, rx=3, sw=1.2, dash=True)
    lc.text(lx + 21, LEG_Y + 1, name, 9.5, lc.C_TXT, 'start', maxw=240,
            tag='leg:' + name[:8])
    lx += 21 + lc.tw(name, 9.5) + 24
lc.text(MX, LEG_Y + 24,
        '帧高 = 对数刻度（真实字节数标注在帧旁）· 帧长与帧数为 host 实测（win32 回环 tcp；编码器与 pin 逐字同、'
        '请求字段为简化载荷）· 行号基线 vLLM v0.27.1',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch05-fig-wire-format.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
