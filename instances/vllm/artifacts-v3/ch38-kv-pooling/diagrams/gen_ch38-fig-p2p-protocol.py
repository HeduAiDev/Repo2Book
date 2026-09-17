#!/usr/bin/env python3
"""ch38 机制图 m13 · p2p-protocol（figure_spec ch38-fig-p2p-protocol，模板 swimlane·UML 时序）

放大自 L2 站 12（P2P 层·实例间共享 CPU 池）· L0：双实例形态（ch37 已立 multi 布局）+外部池的交叉。
时序图严格 UML 文法：竖直生命线 + 共享时间轴 + 水平消息直线，禁一切折线/肘形。

claim：P2P 层五步线协议（Lookup→LookupResp→Fetch→NIXL WRITE→TransferDone）在无固定
P/D 角色的对称 peer 间搬运块，kv_transfer_params 三角色键由编排层驱动，
PYTHONHASHSEED 未设启动即拒。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · 三角色键：remote_kv_source → peer 10.0.0.2:5710 + do_probe=True；remote_prefiller →
    do_probe=False（P/D 跳过 Lookup 直接 Fetch）；remote_decoder → 目标键；无键 → 无源无宿
  · PYTHONHASHSEED 未设 → ValueError 拒启；设 0 后可建；握手期互验
  · 对称语义：PD consumer 查询 HIT、普通请求 MISS
  · 线协议消息五步与字段校验（keys/block_indexes 定长、负索引/类型错 ValueError）
  · watchdog：_UNBOUND_STORE_TIMEOUT_S=60s 收死对端缓冲
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1560
MX = 60
BXR = W - MX

# ---------------- 标题区 ----------------
lc.text(MX, 36, '两台引擎合用一间储藏室：查有什么 → 点名拉 → 单边写 → 回执', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 60, '没有固定房东房客——同一实例对不同请求既可当取货人也可当发货人；钥匙（块哈希）必须完全一样，开门前先对暗号',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L2 站 12（P2P 层·实例间共享 CPU 池）· L0：双实例+外部池交叉'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 时序面板 ----------------
PX0, PY0, PX1, PY1 = MX, 96, 1020, 664
lc.rect(PX0, PY0, PX1 - PX0, PY1 - PY0, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
# 启动门红条
lc.rect(PX0 + 12, PY0 + 10, PX1 - PX0 - 24, 40, '#fef2f2', lc.C_ABORT, rx=7, sw=1.3,
        dash=True)
lc.text(PX0 + 24, PY0 + 26, '启动门：PYTHONHASHSEED 未设 → ValueError 拒启（must be set … so that block hashes match across instances）', 8.8,
        lc.C_ABORT, 'start', True, maxw=PX1 - PX0 - 48, tag='gate:1')
lc.text(PX0 + 24, PY0 + 42, '设 0 后可建；握手期互验——哈希不同源的共享池 = 静默全 miss，硬门把静默错升级为拒启', 8,
        '#334155', 'start', maxw=PX1 - PX0 - 48, tag='gate:2')
A_X, B_X = 340, 760
NP_Y = PY0 + 66
for x, name in [(A_X, '实例 A（对称 peer）'), (B_X, '实例 B（对称 peer）')]:
    lc.rect(x - 110, NP_Y, 220, 26, '#ffffff', lc.C_ZMQ_S, rx=7, sw=1.4)
    lc.text(x, NP_Y + 17, name, 9.5, lc.C_ZMQ_S, 'middle', True, maxw=210, tag='life:' + name[:5])
lc.text((A_X + B_X) / 2, NP_Y + 42, '无固定 P/D 角色：同一实例对不同请求可当取货人 / 发货人', 8.3,
        lc.C_MUTE, 'middle', maxw=B_X - A_X - 20, tag='peer:note')
LF_T, LF_B = NP_Y + 30, PY1 - 96
for x in (A_X, B_X):
    lc.seg(x, LF_T, x, LF_B, lc.C_MUTE, 1.2, dash=True)
# 消息（全部水平直线；ZMQ 控制面细紫 / NIXL 数据面粗绿）
MSGS = [
    (1, 'LookupMsg', B_X, A_X, '① 池里有没有这些键？', lc.C_ZMQ_S, 1.6, False),
    (2, 'LookupRespMsg', A_X, B_X, '② hits 位图', lc.C_ZMQ_S, 1.6, False),
    (3, 'FetchMsg', B_X, A_X, '③ 点名拉（keys + block_indexes）', lc.C_ZMQ_S, 1.6, False),
    (4, 'NIXL WRITE', A_X, B_X, '④ 数据面单边写：A 池直写 B 池', lc.C_GPU_S, 4.0, False),
    (5, 'TransferDoneMsg', A_X, B_X, '⑤ 回执', lc.C_ZMQ_S, 1.6, False),
]
MY0, MSTEP = LF_T + 46, 54
for i, (n, mname, x1, x2, lab, color, sw, dash) in enumerate(MSGS):
    my = MY0 + i * MSTEP
    lc.seg(A_X - 170, my, B_X + 170, my, '#eef2f7', 1.0)   # 共享时间轴 gridline
    lc.seg(x1, my, x2, my, color, sw, 'std', dash)
    side = 'middle'
    lc.text((A_X + B_X) / 2, my - 7, lab, 8.8, color, side, True, maxw=B_X - A_X - 10,
            tag='m%d' % n)
# 图例
lg_y = MY0 + 5 * MSTEP + 6
lc.seg(PX0 + 24, lg_y, PX0 + 54, lg_y, lc.C_ZMQ_S, 1.6)
lc.text(PX0 + 60, lg_y + 3, 'ZMQ 控制面（细）', 8.2, lc.C_MUTE, 'start', maxw=140,
        tag='lg:1')
lc.seg(PX0 + 210, lg_y, PX0 + 240, lg_y, lc.C_GPU_S, 4.0)
lc.text(PX0 + 246, lg_y + 3, 'NIXL 数据面（粗·单边写）', 8.2, lc.C_MUTE, 'start', maxw=200,
        tag='lg:2')
# 脚下 CPU 池
POOL_Y = PY1 - 84
for x, name in [(A_X, 'A 的 CPU 池'), (B_X, 'B 的 CPU 池')]:
    lc.rect(x - 92, POOL_Y, 184, 64, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.5)
    lc.text(x, POOL_Y + 20, name, 9.5, lc.C_KV_S, 'middle', True, maxw=170, tag='pool:' + name[:3])
    lc.text(x, POOL_Y + 38, '对端池 = 一个 secondary tier', 7.8, '#334155', 'middle',
            maxw=170, tag='pools:' + name[:3])
    lc.text(x, POOL_Y + 54, '/dev/shm 共享 mmap', 7.5, lc.C_MUTE, 'middle', maxw=170,
            tag='poolf:' + name[:3])
    lc.seg(x, LF_B, x, POOL_Y, lc.C_KV_S, 1.4)

# ---------------- 右列 ----------------
RX, RW = 1060, BXR - 1060
rb = [
    ('三角色键（kv_transfer_params · 编排层驱动）', lc.C_ZMQ_S, [
        '· remote_kv_source → peer 10.0.0.2:5710',
        '   + do_probe=True（走满五步）',
        '· remote_prefiller → do_probe=False：',
        '   P/D 跳过 Lookup 直接 Fetch',
        '   （对端 prefiller 视为有全部块）',
        '· remote_decoder → 目标键',
        '· 无角色键 → 无源无宿（普通请求）',
    ], 'tiering/p2p/manager.py:L188-L231'),
    ('对称语义实测', lc.C_GPU_S, [
        '· PD consumer 查询 → HIT',
        '· 普通请求（无角色键）→ MISS',
        '   （无「对端退化」分支）',
    ], 'PYTHONHASHSEED=0 实测'),
    ('消息字段校验 + watchdog', lc.C_ABORT, [
        '· keys/block_indexes 定长校验：',
        '   fetch 2 vs 1 → ValueError',
        '· 负索引 / 类型错 → ValueError',
        '· watchdog：_UNBOUND_STORE_TIMEOUT_S',
        '   = 60s 收死对端缓冲',
    ], 'p2p/session/protocol.py:L178-L218'),
]
by = PY0
for t, color, lines, foot in rb:
    h = 30 + len(lines) * 15 + 20
    lc.rect(RX, by, RW, h, '#ffffff', color, rx=8, sw=1.4)
    lc.text(RX + 12, by + 19, t, 9.6, color, 'start', True, maxw=RW - 24, tag='rb:' + t[:6])
    for j, ln in enumerate(lines):
        lc.text(RX + 12, by + 38 + j * 15, ln, 8.2, '#334155', 'start', maxw=RW - 22,
                tag='rb:%s:%d' % (t[:4], j))
    lc.text(RX + 12, by + h - 8, foot, 7.4, lc.C_FAINT, 'start', maxw=RW - 22,
            tag='rb:%s:f' % t[:4])
    by += h + 16

# ---------------- 结论 + 页脚 ----------------
CONC_Y = PY1 + 24
lc.text(MX, CONC_Y,
        '图注结论：共享池的协议 = 查有什么 → 点名拉 → 单边写 → 回执；角色每请求可换，唯一不变的是两边的哈希必须同源。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 vllm/v1/kv_offload/tiering/p2p/manager.py:L188-L231（三角色键）· p2p/session/protocol.py:L178-L218（消息五步与字段校验）· docs/features/kv_offloading_usage.md:L189-L191（PYTHONHASHSEED 握手互验）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · 协议消息类 / 三角色键 / PYTHONHASHSEED 硬门为实测（ZMQ REQ 对脚本化 REP 服务）；NIXL session 机器不进精简版（正文散文交代）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-p2p-protocol.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
