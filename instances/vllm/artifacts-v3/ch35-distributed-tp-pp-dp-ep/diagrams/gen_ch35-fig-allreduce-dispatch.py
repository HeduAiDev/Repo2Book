#!/usr/bin/env python3
"""ch35 机制图 · m5 TP all_reduce 三条出路（figure_spec ch35-fig-allreduce-dispatch）

放大自 L0 多实例视角 ③ TP·每层 all_reduce（L2 站 11）的派发路径展开。

claim：RowParallel 的部分和进 tensor_model_parallel_all_reduce 后有三条出路——
world_size==1 原样短路返回同一张量、use_custom_op_call 时走
torch.ops.vllm.all_reduce(group_name 字符串)（编译友好）、否则直接
_all_reduce_out_place——殊途同归于 device_communicator 的多级回退链。

数字全部取自 explainer figure_spec.numbers（traces/m01 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common（TP 数据流 GPU 绿、
判定/分发 ZMQ 紫辅）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1300, 1000
MX = 64
BXR = 1236
CX = 384                       # 派发主线（竖直脊）
RZ_X, RZ_W = 800, 436          # 右列：回退链优先级表

DEFS = lc.DEFS.replace('</defs>',
                       '<marker id="pu" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="6.5" markerHeight="4.6" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ZMQ_S}"/></marker>'
                       '</defs>')


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:10])
    return x


def diamond(cx, cy, hw, hh, label):
    lc.ELEMS.append(((cx - hw - 4, cy - hh - 4, cx + hw + 4, cy + hh + 4),
                     f'<path d="M{cx - hw},{cy} L{cx},{cy - hh} L{cx + hw},{cy} '
                     f'L{cx},{cy + hh} Z" fill="{lc.C_ZMQ_F}" stroke="{lc.C_ZMQ_S}" '
                     f'stroke-width="1.8"/>'))
    lc.text(cx, cy + 4, label, 10.5, lc.C_TXT, 'middle', True, maxw=hw * 2 - 20,
            tag='dia:' + label[:8])


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'tensor_model_parallel_all_reduce 的三条出路——殊途同归 device_communicator', 16.5,
        lc.C_TXT, 'start', True, maxw=1040, tag='title')
lc.text(MX, 62, '模型层唯一认识的函数：三出口对它完全透明——要不要短路、要不要编译进图，是下面两层的事；'
        '挑内核永远是 device_communicator 的事', 10.5, lc.C_MUTE, 'start', maxw=1150,
        tag='subtitle')
chip(BXR, 12, '放大自 L2 站 11 · L0：多实例视角', lc.C_GPU_S)

# ---------------- 左：派发流程 ----------------
lc.rect(MX - 20, 96, 720, 700, '#f8fafc', lc.C_FAINT, rx=10, sw=1.0)
# 入口
EY, EH = 116, 76
lc.rect(CX - 280, EY, 560, EH, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(CX, EY + 26, '模型层 RowParallelLinear 的部分和', 10, lc.C_MUTE, 'middle', tag='ent:s')
lc.text(CX, EY + 50, 'tensor_model_parallel_all_reduce(input)', 13, lc.C_TXT, 'middle',
        True, maxw=540, tag='ent:t')
lc.text(CX + 266, EY + 18, 'vllm/distributed/parallel_state.py', 8.5, lc.C_FAINT, 'end',
        tag='ent:f')
# 入口 → 菱形1
lc.seg(CX, EY + EH, CX, 218, lc.C_GPU_S, 2.2, 'std')
lc.alabel(CX + 8, 212, '每层 forward 各一次', 8.5, lc.C_GPU_S)
# 菱形1
diamond(CX, 252, 110, 34, 'world_size == 1 ?')
# 短路支（右）
SBX, SBY, SBW, SBH = 520, 190, 240, 100
lc.seg(CX + 110, 252, SBX, 252, lc.C_GPU_S, 2.2, 'std')
lc.alabel((CX + 110 + SBX) / 2, 244, '是', 9.5, lc.C_ZMQ_S)
lc.rect(SBX, SBY, SBW, SBH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(SBX + 12, SBY + 21, '原样短路返回', 11.5, lc.C_TXT, 'start', True, tag='sc:t')
lc.text(SBX + 12, SBY + 39, 'return input_（out is input=true）', 9,
        '#334155', 'start', maxw=SBW - 24, tag='sc:l1')
lc.text(SBX + 12, SBY + 58, '实测 in=[5.0,7.0]', 9.5, lc.C_TXT,
        'start', True, maxw=SBW - 24, tag='sc:n1')
lc.text(SBX + 12, SBY + 74, '→ out=[5.0,7.0] 原样', 9.5, lc.C_TXT,
        'start', True, maxw=SBW - 24, tag='sc:n2')
lc.text(SBX + 12, SBY + 91, 'DP=1 部署每层都走这条零成本旁路', 8.5, lc.C_MUTE, 'start',
        maxw=SBW - 24, tag='sc:l3')
# 否 → 菱形2
lc.seg(CX, 286, CX, 330, lc.C_GPU_S, 2.2, 'std')
lc.alabel(CX + 8, 316, '否', 9.5, lc.C_ZMQ_S)
diamond(CX, 364, 110, 34, 'use_custom_op_call ?')
# custom-op 支（右）
CBX, CBY, CBW, CBH = 520, 300, 240, 168
lc.seg(CX + 110, 364, CBX, 364, lc.C_GPU_S, 2.2, 'std')
lc.alabel((CX + 110 + CBX) / 2, 356, '是', 9.5, lc.C_ZMQ_S)
lc.rect(CBX, CBY, CBW, CBH, '#ffffff', lc.C_ZMQ_S, rx=8, sw=1.6)
lc.text(CBX + 12, CBY + 22, '编译进图（custom op）', 11.5, lc.C_TXT, 'start', True, tag='co:t')
lc.text(CBX + 12, CBY + 44, 'torch.ops.vllm.all_reduce(', 9, '#334155', 'start',
        maxw=CBW - 24, tag='co:l1')
lc.text(CBX + 24, CBY + 60, 'input_, group_name="tp:0")', 9, '#334155', 'start',
        maxw=CBW - 36, tag='co:l2')
lc.text(CBX + 12, CBY + 80, '· 字符串可符号化、进 torch.compile 图', 9, '#334155', 'start',
        maxw=CBW - 24, tag='co:l3')
lc.text(CBX + 12, CBY + 96, '· 配 fake 实现：编译期形状推断不真通信', 9, '#334155', 'start',
        maxw=CBW - 24, tag='co:l4')
lc.text(CBX + 12, CBY + 116, '· host 默认 false（CPU 平台 seam）', 9, lc.C_MUTE, 'start',
        maxw=CBW - 24, tag='co:l5')
lc.text(CBX + 12, CBY + 132, '  真 CUDA 平台 true；强开后同出 3.0', 9, lc.C_MUTE, 'start',
        maxw=CBW - 24, tag='co:l6')
lc.text(CBX + 12, CBY + 154, 'L525-L533', 8.5, lc.C_FAINT, 'start', tag='co:f')
# 直接路径（下）
DBX, DBY, DBW, DBH = CX - 140, 430, 280, 76
lc.seg(CX, 398, CX, DBY, lc.C_GPU_S, 2.2, 'std')
lc.alabel(CX + 8, 420, '否', 9.5, lc.C_ZMQ_S)
lc.rect(DBX, DBY, DBW, DBH, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
lc.text(DBX + 14, DBY + 26, '直接接线：_all_reduce_out_place(input_)', 10.5, lc.C_TXT,
        'start', True, maxw=DBW - 28, tag='dr:t')
lc.text(DBX + 14, DBY + 48, '不做符号化表示——推理期路径', 9, lc.C_MUTE, 'start',
        maxw=DBW - 28, tag='dr:l')
# 汇合 → device_communicator
HCY = 556
lc.parrow([(DBX + DBW / 2, DBY + DBH), (DBX + DBW / 2, HCY)], lc.C_GPU_S, 2.2, 'std')
lc.parrow([(CBX + CBW / 2, CBY + CBH), (CBX + CBW / 2, HCY - 22),
           (CX + 160, HCY - 22), (CX + 160, HCY)], lc.C_GPU_S, 2.0, 'std')
lc.text(CX + 230, HCY - 30, '殊途同归', 9, lc.C_MUTE, 'middle', tag='conv')
DCY, DCH = HCY, 92
lc.rect(CX - 200, DCY, 400, DCH, '#ffffff', lc.C_ENG_S, rx=8, sw=1.8)
lc.text(CX, DCY + 28, 'device_communicator.all_reduce(input_)', 12.5, lc.C_TXT, 'middle',
        True, maxw=380, tag='dc:t')
lc.text(CX, DCY + 52, '真正挑内核的下沉点（world_size>1 才挂——上一节解剖）', 9.5,
        '#334155', 'middle', maxw=380, tag='dc:l1')
lc.text(CX, DCY + 72, '回退链对调用方透明：挑哪个内核，模型层从不知道', 9.5, lc.C_MUTE,
        'middle', maxw=380, tag='dc:l2')
# device_communicator → 右列回退链
lc.seg(CX + 200, DCY + DCH / 2, RZ_X, DCY + DCH / 2, lc.C_ENG_S, 2.2, 'std')
lc.alabel((CX + 200 + RZ_X) / 2, DCY + DCH / 2 - 8, '逐级试探', 9, lc.C_ENG_S)

# why 注（docstring 两约束）
WY = DCY + DCH + 26
lc.rect(MX - 20 + 16, WY, 720 - 32, 96, 'none', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(MX, WY + 20, 'why · docstring 的两个约束（parallel_state.py:L662-L675 原文）', 10,
        lc.C_TXT, 'start', True, tag='why:t')
lc.text(MX, WY + 40, '· Dynamo 不收任意对象——self 不能进 custom op，只能传 group_name 字符串'
        '（按名查回 GroupCoordinator 再派发）', 9, '#334155', 'start', maxw=660, tag='why:l1')
lc.text(MX, WY + 58, '· PyTorch custom op 不许原地改、不许同 op 内既变又返——所以一律 out-of-place'
        '（out-of-place 是纪律，不是风格）', 9, '#334155', 'start', maxw=660, tag='why:l2')
lc.text(MX, WY + 80, '两个菱形都在 GroupCoordinator.all_reduce 里（L677-L687）：短路最先判，'
        '编译开关其次，直接路径殿后', 9, lc.C_MUTE, 'start', maxw=660, tag='why:l3')

# ---------------- 右：七级回退链优先级表 ----------------
lc.text(RZ_X, 116, '七级回退链 + 兜底（优先级自上而下）', 12.5, lc.C_ENG_S, 'start', True,
        maxw=RZ_W, tag='fb:t')
lc.text(RZ_X + RZ_W, 116, 'cuda_communicator.py:L275-L341', 8.5, lc.C_FAINT, 'end',
        tag='fb:f')
chain = [
    ('NCCL symm', 'torch.ops.vllm.all_reduce_symmetric_with_copy', False),
    ('quick reduce', '只配 ROCm MI3*（qr_comm）', False),
    ('flashinfer', 'fi_ar_comm.all_reduce', False),
    ('aiter', 'aiter_ar_comm.custom_all_reduce', False),
    ('CustomAllreduce', 'ca_comm.custom_all_reduce', False),
    ('symm_mem', 'symm_mem_comm.all_reduce', False),
    ('pynccl', 'pynccl_comm.all_reduce——seam：内部即 torch.distributed', True),
    ('torch.distributed', '兜底——源自述「this usually happens during testing」', False),
]
RY0, RROW, RPITCH = 138, 52, 62
for i, (name, note, hit) in enumerate(chain):
    y = RY0 + i * RPITCH
    fill = '#fff7ed' if hit else '#ffffff'
    sw = 2.0 if hit else 1.3
    lc.rect(RZ_X, y, RZ_W, RROW, fill, lc.C_ENG_S if hit else lc.C_MUTE, rx=6, sw=sw)
    lc.text(RZ_X + 16, y + 22, '%d' % (i + 1), 11, lc.C_ENG_S if hit else lc.C_MUTE,
            'middle', True, tag='fb%d:n' % i)
    lc.text(RZ_X + 32, y + 22, name, 11, lc.C_TXT, 'start', True, maxw=RZ_W - 44,
            tag='fb%d:t' % i)
    lc.text(RZ_X + 32, y + 40, note, 8.5, lc.C_MUTE, 'start', maxw=RZ_W - 44,
            tag='fb%d:l' % i)
    if hit:
        bw = lc.tw('host 实跑落点', 8.5, True) + 12
        lc.rect(RZ_X + RZ_W - bw - 8, y + 6, bw, 17, lc.C_BADGE_F, lc.C_ENG_S, rx=8, sw=1.0)
        lc.text(RZ_X + RZ_W - bw / 2 - 8, y + 18.5, 'host 实跑落点', 8.5, lc.C_ENG_S,
                'middle', True, maxw=bw - 2, tag='fb:badge')
# 连接链的小箭头（行间）
for i in range(len(chain) - 1):
    y = RY0 + i * RPITCH + RROW
    lc.seg(RZ_X + 16, y, RZ_X + 16, y + (RPITCH - RROW), lc.C_MUTE, 1.4, 'std')
# host 落点说明
HY = RY0 + len(chain) * RPITCH + 6
lc.rect(RZ_X, HY, RZ_W, 86, '#f8fafc', lc.C_FAINT, rx=6, sw=1.0)
lc.text(RZ_X + 12, HY + 19, 'host 实测（单机 gloo）', 9.5, lc.C_TXT, 'start', True,
        maxw=RZ_W - 24, tag='hy:t')
lc.text(RZ_X + 12, HY + 38, '前六级「构造面存在、判定面恒不启用」（qr/fi/aiter/symm', 9,
        '#334155', 'start', maxw=RZ_W - 24, tag='hy:l1')
lc.text(RZ_X + 12, HY + 54, '均未启用，ca 构造面存在但 disabled=True）→ 落到 pynccl', 9,
        '#334155', 'start', maxw=RZ_W - 24, tag='hy:l2')
lc.text(RZ_X + 12, HY + 72, 'seam——回退语义对调用方透明这一点不变', 9, '#334155', 'start',
        maxw=RZ_W - 24, tag='hy:l3')

# ---------------- 底部实测条 ----------------
BY = 836
lc.rect(MX, BY, BXR - MX, 78, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.1)
lc.text(MX + 16, BY + 22, 'TP=2 每层 all_reduce 实测：rank0 出部分和 1.0 · rank1 出 2.0 → '
        '直接路径 out=3.0 · 强开 custom-op 路径同出 3.0（torch.ops.vllm.all_reduce）', 10.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='num:l1')
lc.text(MX + 16, BY + 44, '两条 rank 双方同得 3.0（all_reduce 语义）；短路支实测单进程 world_size=1：'
        'in=[5.0,7.0] 原样返回且是同一对象', 9.5, '#334155', 'start', maxw=BXR - MX - 32,
        tag='num:l2')
lc.text(MX + 16, BY + 62, '三条路对模型层完全透明——RowLinear 只认 tensor_model_parallel_all_reduce '
        '这一个名字（model_executor/layers/linear.py:L1767 调用点）', 9.5, lc.C_MUTE,
        'start', maxw=BXR - MX - 32, tag='num:l3')

# ---------------- 图例 + 页脚 ----------------
LY = BY + 78 + 26
lx0 = MX
for color, name in [(lc.C_GPU_S, '数据流（部分和/结果张量）'), (lc.C_ZMQ_S, '判定/分发（菱形）'),
                    (lc.C_ENG_S, '内核下沉点')]:
    if name.startswith('数据流'):
        lc.seg(lx0 + 2, LY - 3, lx0 + 32, LY - 3, color, 2.2, 'std')
    elif name.startswith('判定'):
        lc.rect(lx0 + 4, LY - 10, 22, 15, lc.C_ZMQ_F, color, rx=3, sw=1.4)
    else:
        lc.rect(lx0 + 4, LY - 9, 16, 11, '#ffffff', color, rx=3, sw=1.6)
    off = 40 if name.startswith('数据流') else 36
    lc.text(lx0 + off, LY + 1, name, 9.5, lc.C_TXT, 'start', tag='leg:' + name[:6])
    lx0 += off + lc.tw(name, 9.5) + 22
lc.text(MX, LY + 24,
        '行号基线 vLLM v0.27.1（6e448d0ea）· 实测值标 host（gloo 单机退化——真 pin 多卡 NCCL，'
        '通信图语义等价）· 框内灰字 = 规范源码路径', 9, lc.C_MUTE, 'start', maxw=BXR - MX,
        tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch35-fig-allreduce-dispatch.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
