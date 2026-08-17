#!/usr/bin/env python3
"""ch05 机制图 · 零拷贝的两侧两答案（explainer m5 figure_spec ch05-fig-zero-copy-two-sides）

放大自 L0 紫色 ZMQ 边界带的两侧发送/回程路径——即本章 L2 拍 ①「encode 多帧」
（下行）与 ⑤「PUSH 回程」（上行）+ south 注「零拷贝 · 两侧两答案」的机制展开。

claim：同一问题（copy=False 后 buffer 谁保活）两侧两种答案——客户端 buffer 归调用方
→zmq 引用链自动保活（#50053 删显式 tracker）；引擎输出侧 buffer 是自己复用的→
encode_into 复用 + 首帧 tracker + pending 回收。朴素单帧每条消息 2 次拷贝，
多帧零拷贝 0 次。

数字全部取自 explainer figure_spec.numbers（copy_accounting + rounds 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common 语系（同源强制）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1320, 948
BAND_X0, BAND_X1 = 620, 720
BAND_Y0, BAND_Y1 = 100, 880
LZ_R = 600                 # 左半区右缘
RZ_L = 740                 # 右半区左缘


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:12])
    return x


def box(x, y, w, h, title, lines, stroke, fill='#ffffff', dash=False, tfs=10.5,
        lfs=8.5, tag=''):
    lc.rect(x, y, w, h, fill, stroke, rx=7, sw=1.6, dash=dash)
    lc.text(x + 12, y + 20, title, tfs, lc.C_TXT, 'start', True, maxw=w - 22,
            tag=(tag or title))
    for i, ln in enumerate(lines):
        mut = ln.startswith('(') or ln.startswith('vllm/')
        lc.text(x + 12, y + 40 + i * 15.5, ln, lfs, lc.C_FAINT if mut else '#334155',
                'start', maxw=w - 22, tag=(tag or title) + ':' + ln[:10])


# ---------------- 标题区 ----------------
lc.text(96, 36, '零拷贝的两侧两答案：copy=False 之后，buffer 谁保活？', 16.5, lc.C_TXT,
        'start', True, maxw=860, tag='title')
lc.text(96, 60,
        '客户端 buffer 归调用方——zmq 沿 memoryview 引用链保活（#50053 删显式 tracker）；'
        '引擎输出 buffer 自己复用——首帧 tracker + pending 回收；拷贝账：朴素 2 次 vs 多帧 0 次',
        10.5, lc.C_MUTE, 'start', maxw=1130, tag='subtitle')
chip(1224, 12, '放大自 L2 拍 ①⑤「encode / PUSH 回程」· L0：紫色 ZMQ 边界带', lc.C_ZMQ_S)

# ---------------- 紫色竖带（带内含 PUSH→PULL 小徽 → 容器，箭头过带合法）----------------
lc.rect(BAND_X0, BAND_Y0, BAND_X1 - BAND_X0, BAND_Y1 - BAND_Y0, lc.C_ZMQ_F,
        lc.C_ZMQ_S, rx=10, sw=2.0)

# ---------------- 三栏题头：前端 | ZMQ 带 | 引擎（画在带矩形之后——中栏落在带上） ----------------
lc.text(96, 116, '前端进程（client 半边）', 10.5, lc.C_API_S, 'start', True, tag='z:l')
lc.text(670, 116, '进程边界 · ZMQ', 9.5, lc.C_ZMQ_S, 'middle', True, tag='z:m')
lc.text(1224, 116, 'EngineCore 进程（引擎半边）', 10.5, lc.C_ENG_S, 'end', True, tag='z:r')

lc.rect(632, 828, 76, 28, '#ffffff', lc.C_ZMQ_S, rx=6, sw=1.2)
lc.text(670, 846, 'PUSH→PULL', 8, lc.C_ZMQ_S, 'middle', True, maxw=70, tag='band:chip')

# ================= 上泳道 · 客户端下行 =================
lc.text(96, 140, '上泳道 · 客户端下行（ADD 过线）——答案一：buffer 归调用方，引用链自动保活',
        10.5, lc.C_API_S, 'start', True, maxw=520, tag='lane1')

# 调用方作用域（虚线）+ 张量存储（内框沉到两行说明文字之下，零相交）
box(90, 165, 230, 112, '调用方作用域（张量活着）',
    ['zmq 顺 memoryview 引用链', '就能找回它——保活自动成立'],
    lc.C_API_S, lc.C_API_F, dash=True, tag='scope')
lc.rect(108, 230, 194, 36, '#ffffff', lc.C_API_S, rx=4, sw=1.4)
lc.text(205, 251, '张量存储 8192B（2048×float32）', 8.5, lc.C_TXT, 'middle',
        maxw=182, tag='tensor')

# encode 框
box(350, 165, 210, 96, 'MsgpackEncoder.encode',
    ['主帧 bufs[0] + aux_buffers 追加', 'vllm/v1/serial_utils.py:L166-L178'],
    lc.C_API_S, tag='enc')
lc.seg(320, 213, 348, 213, lc.C_API_S, 1.8, 'std')

# 帧对：主帧 + aux 活视图窗
lc.parrow([(455, 261), (455, 274), (398, 274), (398, 283)], lc.C_API_S, 1.8, 'std')
lc.rect(350, 285, 96, 60, '#ffffff', lc.C_API_S, rx=4, sw=1.4)
lc.text(398, 319, '主帧', 9.5, lc.C_TXT, 'middle', True, tag='mf')
lc.rect(458, 285, 102, 60, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.5, dash=True)
lc.text(509, 312, 'aux 帧', 9.5, lc.C_GPU_S, 'middle', True, tag='aux')
lc.text(509, 328, '（活视图）', 8, lc.C_GPU_S, 'middle', tag='aux2')

# 引用链：aux 窗 → 张量存储（同一存储物证；尾端指向内框右缘中点）
lc.parrow([(509, 345), (509, 354), (330, 354), (330, 248), (304, 248)],
          lc.C_GPU_S, 1.4, 'std', dash=True)
lc.text(420, 368, '同一存储：编码后改张量 0 号 → 123.5，aux 首浮点读出 123.5', 8.5,
        lc.C_GPU_S, 'middle', maxw=360, tag='alias')

# 过线箭头（穿紫带）
lc.seg(560, 315, 738, 315, lc.C_API_S, 2.6, 'dn')
lc.text(652, 306, 'send_multipart(copy=False)', 8.5, lc.C_API_S, 'middle',
        maxw=170, tag='snd')

# 引擎接收 + #50053 注
box(740, 165, 250, 160, '引擎接收 · MsgpackDecoder.decode',
    ['recv_multipart → decode', 'torch.frombuffer 零拷贝视图：', 'data_ptr 相等——2048 个元素',
     '全程未拷贝（解码零拷贝）', '代价：视图锁住整条消息缓冲', 'vllm/v1/serial_utils.py:L389-L392'],
    lc.C_ENG_S, lc.C_ENG_F, tag='rcv')
box(1020, 165, 230, 160, '#50053 · 客户端显式 tracker 已删',
    ['旧三件套：pending_messages /', 'add_pending_message /', 'free_pending_messages',
     "'kept alive by zmq itself'", '（引用链替代显式管理）', 'vllm/v1/engine/core_client.py:L1116-L1120'],
    lc.C_MUTE, dash=True, tag='n50053')

# ================= 中间：拷贝账对比条（左右两段，中间让开紫色竖带） =================
BY0 = 455
lc.rect(96, BY0, 504, 95, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(112, BY0 + 21, '拷贝账（用户态）', 10.5, lc.C_TXT, 'start', True, maxw=300,
        tag='acc:t')
lc.rect(108, BY0 + 32, 230, 52, '#ffffff', lc.C_ABORT, rx=6, sw=1.3)
lc.text(120, BY0 + 48, '朴素单帧路径：2 次拷贝', 9.5, lc.C_ABORT, 'start', True,
        maxw=206, tag='p:naive')
lc.text(120, BY0 + 64, 'encode 拷 1 + zmq send 拷 1', 8, '#334155', 'start',
        maxw=206, tag='p:naive2')
lc.text(120, BY0 + 78, '100MB 张量 = 200MB 白拷（#13790 旧路径）', 8, '#334155',
        'start', maxw=206, tag='p:naive3')
lc.rect(354, BY0 + 32, 234, 52, '#ffffff', lc.C_ZMQ_S, rx=6, sw=1.3)
lc.text(366, BY0 + 48, '＜256B 内联：1 次小拷贝', 9.5, lc.C_ZMQ_S, 'start', True,
        maxw=210, tag='p:inl')
lc.text(366, BY0 + 64, 'host seam 1 次 / 真 msgspec 0 次', 8, '#334155', 'start',
        maxw=210, tag='p:inl2')
lc.text(366, BY0 + 78, '阈值 256B（envs.py:L208）', 8, '#334155', 'start',
        maxw=210, tag='p:inl3')
lc.rect(740, BY0, 484, 95, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(756, BY0 + 21, '多帧零拷贝：0 次用户态拷贝', 10.5, lc.C_GPU_S, 'start', True,
        maxw=300, tag='acc:r')
lc.text(756, BY0 + 45, 'aux 独立帧 + send_multipart(copy=False)', 9, '#334155',
        'start', maxw=452, tag='acc:r2')
lc.text(756, BY0 + 63, '跨进程搬运由内核完成，不占用户态带宽', 9, '#334155', 'start',
        maxw=452, tag='acc:r3')

# ================= 下泳道 · 引擎输出上行 =================
lc.text(96, 580, '下泳道 · 引擎输出上行（EngineCoreOutputs 回程）——答案二：buffer 自己复用，'
        '首帧 tracker 问『zmq 发完了吗』+ pending 回收', 10.5, lc.C_ENG_S, 'start', True,
        maxw=1130, tag='lane2')

# 引擎侧循环四框：pool → encode_into → 分帧 → pending → 还池
box(1000, 608, 250, 100, 'reuse_buffers 池',
    ['上限 max_reuse_bufs = sockets+1', 'vllm/v1/engine/core.py:L1776'],
    lc.C_ENG_S, lc.C_ENG_F, tag='pool')
box(740, 608, 230, 100, 'encode_into 原地重写',
    ['同一 bytearray 整体替换（截断语义', '对真 msgspec 0.21.1 容器实测一致）',
     'vllm/v1/engine/core.py:L1801-L1802'],
    lc.C_ENG_S, lc.C_ENG_F, tag='einto')
box(740, 738, 230, 100, '_send_msg_tracking_payload',
    ['首帧 send(track=True)：send_multipart', '只回最后一帧的 tracker',
     'vllm/v1/engine/core.py:L1813-L1827'],
    lc.C_ENG_S, lc.C_ENG_F, tag='split')
box(1000, 738, 250, 100, 'pending 队列',
    ['(tracker, buffer)：done=False 时', 'zmq 还攥着 buffer；done=True 才', '复用——连发 6 条零腐败',
     'vllm/v1/engine/core.py:L1795-L1810'],
    lc.C_ENG_S, lc.C_ENG_F, tag='pend')

lc.seg(1000, 658, 962, 658, lc.C_ENG_S, 1.8, 'std')                     # pool → encode_into
lc.text(981, 648, 'pop()', 8, lc.C_MUTE, 'middle', maxw=40, tag='a:pop')
lc.seg(855, 708, 855, 736, lc.C_ENG_S, 1.8, 'std')                      # encode_into → 分帧
lc.text(863, 726, 'encode_into 产出多帧', 8, lc.C_MUTE, 'start', maxw=120, tag='a:e')
lc.seg(970, 788, 998, 788, lc.C_ENG_S, 1.8, 'std')                      # 分帧 → pending
lc.seg(1125, 738, 1125, 710, lc.C_ENG_S, 1.8, 'std')                    # pending → 还池
lc.text(1133, 727, 'done=True → 还回池', 8, lc.C_ENG_S, 'start', maxw=110, tag='a:back')

# 回程总线：分帧底 → 下 → 左穿紫带 → 上 → PULL 底
lc.parrow([(850, 838), (850, 862), (240, 862), (240, 822)], lc.C_ENG_S, 2.6, 'up')
lc.text(560, 856, 'PUSH(linger=4000) 过线', 8.5, lc.C_ENG_S, 'end', maxw=170, tag='a:push')

# 前端接收 + linger 注
box(90, 700, 280, 120, '前端 PULL 收帧',
    ['validate_alive（单帧死讯哨兵）', 'decoder.decode(EngineCoreOutputs)',
     'vllm/v1/engine/core_client.py:L490-L493'],
    lc.C_API_S, lc.C_API_F, tag='pull2')
box(400, 700, 200, 120, 'linger=4000 · 死讯保镖',
    ['正常关 socket 立刻丢未发消息——', 'linger 撑 4 秒把 ENGINE_CORE_DEAD', '单帧死讯先冲出去',
     'vllm/v1/engine/core.py:L1758-L1763'],
    lc.C_MUTE, dash=True, tag='ling')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 908
items = [
    ('arrow', lc.C_API_S, 'dn', '下行（客户端 copy=False）'),
    ('arrow', lc.C_ENG_S, 'up', '上行（引擎 PUSH 回程）'),
    ('dash', lc.C_GPU_S, None, 'aux 活视图 / 引用链'),
    ('swatch', lc.C_ZMQ_S, lc.C_ZMQ_F, 'ZMQ 进程边界'),
    ('dash', lc.C_MUTE, None, '虚线框 = 注记'),
]
lx = 96
for kind, color, mk, name in items:
    if kind == 'arrow':
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, color, 2.2, mk)
    elif kind == 'dash':
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, color, 1.5, dash=True)
    else:
        lc.rect(lx, LEG_Y - 9, 16, 11, name == 'ZMQ 进程边界' and lc.C_ZMQ_F or '#ffffff',
                color, rx=3, sw=1.6)
    lc.text(lx + 40, LEG_Y + 1, name, 9.5, lc.C_TXT, 'start', maxw=220,
            tag='leg:' + name[:8])
    lx += 40 + lc.tw(name, 9.5) + 22
lc.text(96, LEG_Y + 24,
        '行号基线 vLLM v0.27.1 · 实测值标 host（win32 回环 tcp；encode_into 截断语义对真 msgspec 0.21.1 '
        '在 vllm 容器实测一致）· 框内灰字 = 规范源码路径',
        9, lc.C_MUTE, 'start', maxw=1128, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch05-fig-zero-copy-two-sides.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
