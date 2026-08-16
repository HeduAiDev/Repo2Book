#!/usr/bin/env python3
"""ch04 机制图 · client 工厂二轴（explainer m2 figure_spec ch04-fig-client-factory）

放大自 L2 章图站 4「client 工厂 · make_client 二轴」（L0 区域 = API 进程带）——
FIGURE-SYSTEM §3.3 正文机制图：架构归属回指 L2/L0，不另立架构画法。

claim：make_client 只按 (multiprocess_mode × asyncio_mode) 二轴分发
SyncMPClient / AsyncMPClient / InprocClient 三实现、唯一拒绝 asyncio∧¬mp 组合；
「离线默认竟跨进程」的真因不在工厂——from_engine_args 被总闸
VLLM_ENABLE_V1_MULTIPROCESSING（默认 True）强翻。

数字全部取自 explainer figure_spec.numbers（trace m2_client_factory.json + pin 锚点）。
坐标由常量/循环计算；文本全 esc()；配色走 l0_common 语系（同源强制）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与配色 ----------------
W, H = 1240, 948
GRAY_S, GRAY_F = '#6b7280', '#f3f4f6'   # InprocClient 灰阶（进程内逃生舱）

MX = 60


def box(x, y, w, h, title, lines, stroke, fill='#ffffff', sw=1.6, dash=False,
        tfs=12, lfs=9, badge='', tag='', tcol=None):
    """内容框：标题(粗) + 若干行 + 右上角小徽标。行距 17。"""
    tcol = tcol or lc.C_TXT
    lc.rect(x, y, w, h, fill, stroke, rx=7, sw=sw, dash=dash)
    badge_w = (lc.tw(badge, 9, True) + 16) if badge else 0
    lc.text(x + 14, y + 21, title, tfs, tcol, 'start', True,
            maxw=w - 30 - badge_w, tag=(tag or title))
    if badge:
        lc.rect(x + w - badge_w - 9, y + 7, badge_w, 18, lc.C_BADGE_F, stroke,
                rx=8, sw=1.1)
        lc.text(x + w - badge_w / 2 - 9, y + 20, badge, 9, stroke, 'middle',
                True, maxw=badge_w - 4, tag='badge:' + badge)
    for i, row in enumerate(lines):
        ln, mut = row[0], row[1]
        fs = row[2] if len(row) > 2 else lfs
        lc.text(x + 14, y + 41 + i * 17, ln, fs, lc.C_MUTE if mut else '#334155',
                'start', maxw=w - 26, tag=(tag or title) + ':' + ln[:10])


def chip(x_right, y, label, color):
    """右上角回指小片（虚线）。返回 chip 左缘 x。"""
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:12])
    return x


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'client 工厂二轴：两问定运力', 17, lc.C_TXT, 'start', True)
lc.text(MX, 60,
        'make_client 只按 (multiprocess_mode × asyncio_mode) 分发 SyncMPClient / AsyncMPClient / InprocClient、'
        '唯一拒绝 asyncio∧¬mp；「离线默认竟跨进程」的真因不在工厂——入口总闸 envs 默认 True 强翻',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
chip_x0 = chip(1180, 12, '放大自 L2 站 4「client 工厂」· L0 区域：API 进程带', lc.C_API_S)

# ---------------- 行 A：两个使用面 ----------------
AY, AH = 96, 66
box(110, AY, 400, AH, '在线面 · AsyncLLM.__init__',
    [('调 make_async_mp_client（async_llm.py:L149-L156）', False),
     ('第三件 engine_core 的全部装配差异就在这一处', True)],
    lc.C_API_S, lc.C_API_F, tag='A1')
box(730, AY, 400, AH, '离线面 · LLMEngine.from_engine_args',
    [('先查总闸 envs → make_client(asyncio_mode=False)', False),
     ('（llm_engine.py:L105-L111 · L174-L176）', True)],
    lc.C_API_S, lc.C_API_F, tag='A2')

# 在线面直呼旁路（虚线，绕过工厂直达 AsyncMPClient 叶）
LY = 644
seg_x = 140
lc.seg(seg_x, AY + AH, seg_x, LY, lc.C_API_S, 1.5, 'dn', dash=True)
lc.text(seg_x + 12, AY + AH + 40, '直呼 make_async_mp_client', 9, lc.C_API_S,
        'start', maxw=230, tag='direct1')
lc.text(seg_x + 12, AY + AH + 56, '——不经 make_client 分发', 9, lc.C_MUTE,
        'start', maxw=230, tag='direct2')

# ---------------- 行 B：总闸（仅离线路径上） ----------------
GY, GH = 196, 64
box(420, GY, 400, GH, '总闸 envs.VLLM_ENABLE_V1_MULTIPROCESSING',
    [('默认 True（envs.py:L149）——加粗即真值', False),
     ('→ 强翻 multiprocess_mode=True 再进工厂', True)],
    lc.C_API_S, '#ffffff', sw=2.2, tfs=11.5, tag='gate')
# A2 → 总闸（肘形）
lc.parrow([(930, AY + AH), (930, AY + AH + 16), (620, AY + AH + 16), (620, GY)],
          lc.C_API_S, 1.8, 'dn')
lc.text(775, AY + AH + 12, '先过总闸', 9, lc.C_API_S, 'middle', tag='via-gate')
# 注①（虚线注）
lc.rect(860, GY, 340, GH, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text(874, GY + 20, '注① 形参默认 False 是误导', 9.5, lc.C_TXT, 'start', True,
        maxw=316, tag='note1a')
lc.text(874, GY + 38, '真值由 envs 强翻（llm_engine.py:L174-L176）', 9, lc.C_MUTE,
        'start', maxw=316, tag='note1b')
lc.text(874, GY + 54, 'enable_multiprocessing=False 只是签名默认', 8.5, lc.C_FAINT,
        'start', maxw=316, tag='note1c')
lc.seg(820, GY + 32, 860, GY + 32, lc.C_FAINT, 1.1, dash=True)

# ---------------- 行 C：工厂入口 ----------------
FY, FH = 296, 74
box(420, FY, 400, FH, 'EngineCoreClient.make_client',
    [('(multiprocess_mode, asyncio_mode) · core_client.py:L90-L112', False),
     ('2 轴 → 3 实现 + 1 拒绝 · 纯配置分发，O(1) 三分支', False)],
    lc.C_API_S, '#ffffff', sw=1.8, tfs=11.5, tag='factory')
# 总闸 → 工厂：True 加粗 / False 虚线，两路并行
lc.seg(560, GY + GH, 560, FY, lc.C_API_S, 3.0, 'dn')
lc.text(548, GY + GH + 22, 'True（默认）', 9, lc.C_API_S, 'end', True,
        maxw=100, tag='gate-true')
lc.seg(680, GY + GH, 680, FY, lc.C_MUTE, 1.5, 'std', dash=True)
lc.text(692, GY + GH + 22, '显式关 =False', 9, lc.C_MUTE, 'start', maxw=110,
        tag='gate-false')
# 工厂 → 第一问（离线路径延续加粗）
lc.seg(620, FY + FH, 620, 424, lc.C_API_S, 3.0, 'dn')

# ---------------- 行 D/E：两问 ----------------
Q1Y, QH = 424, 54
box(470, Q1Y, 300, QH, '第一问：跨进程吗？',
    [('multiprocess_mode = ?', True)], lc.C_API_S, lc.C_API_F, tfs=11.5, tag='Q1')
Q2Y = 528
box(180, Q2Y, 300, QH, '第二问：要 asyncio 吗？',
    [('asyncio_mode = ?', True)], lc.C_API_S, lc.C_API_F, tfs=11.5, tag='Q2L')
box(760, Q2Y, 300, QH, '第二问：要 asyncio 吗？',
    [('asyncio_mode = ?', True)], lc.C_API_S, lc.C_API_F, tfs=11.5, tag='Q2R')

# 第一问 → 两处第二问（mp=True 沿离线默认路径加粗）
MIDY = 504
lc.parrow([(545, Q1Y + QH), (545, MIDY), (330, MIDY), (330, Q2Y)],
          lc.C_API_S, 3.0, 'dn')
lc.text(437, MIDY - 6, 'mp=True（强翻后）', 9.5, lc.C_API_S, 'middle', True,
        maxw=160, tag='mp-true')
lc.parrow([(695, Q1Y + QH), (695, MIDY), (910, MIDY), (910, Q2Y)],
          lc.C_API_S, 1.8, 'dn')
lc.text(803, MIDY - 6, 'mp=False', 9.5, lc.C_MUTE, 'middle', maxw=100, tag='mp-false')

# ---------------- 行 F：四个叶 ----------------
LH = 116
box(110, LY, 280, LH, 'AsyncMPClient',
    [('asyncio 客户端 · ZMQ + 后台引擎进程', False),
     ('经 make_async_mp_client 装配（L116-L139）', False),
     ('出生参数：client_count=1 · client_index=0', False),
     ('（签名默认，core_client.py:L121-L122）', True)],
    lc.C_ENG_S, lc.C_ENG_F, badge='在线默认', tag='leaf-async')
box(440, LY, 260, LH, 'SyncMPClient',
    [('同步客户端 · step() 按拍阻塞拉 get_output', False),
     ('后台引擎进程——离线 LLM 默认也跨进程', False),
     ('同步 LLM ≠ 进程内引擎', False)],
    lc.C_ENG_S, lc.C_ENG_F, sw=3.0, badge='离线默认', tag='leaf-sync')
box(740, LY, 230, LH, 'NotImplementedError',
    [('asyncio ∧ ¬mp 当场拒绝', False),
     ('"TODO: support this for debugging purposes"', False, 8),
     ('core_client.py:L94-L98', True)],
    lc.C_ABORT, '#ffffff', sw=1.8, dash=True, tfs=11.5, tag='leaf-reject')
box(990, LY, 210, LH, 'InprocClient',
    [('V0-style 直连 · no busy loop（L306）', False),
     ('纯 CPU / 测试场景免税（无 IPC）', False),
     ('显式 envs=False 才走到', False)],
    GRAY_S, GRAY_F, badge='逃生舱', tag='leaf-inproc')

# 第二问 → 四叶（asyncio=False → SyncMPClient 沿离线默认路径加粗）
ELY = 616
lc.parrow([(300, Q2Y + QH), (300, ELY), (250, ELY), (250, LY)], lc.C_API_S, 1.8, 'dn')
lc.text(310, ELY - 5, 'asyncio=True', 9, '#334155', 'middle', maxw=90, tag='a-true1')
lc.parrow([(360, Q2Y + QH), (360, ELY), (570, ELY), (570, LY)], lc.C_API_S, 3.0, 'dn')
lc.text(470, ELY - 5, 'asyncio=False', 9, lc.C_API_S, 'middle', True, maxw=100,
        tag='a-false1')
lc.seg(850, Q2Y + QH, 850, LY, lc.C_API_S, 1.8, 'dn')
lc.text(862, ELY - 5, 'asyncio=True', 9, '#334155', 'start', maxw=90, tag='a-true2')
lc.parrow([(1000, Q2Y + QH), (1000, ELY), (1095, ELY), (1095, LY)], lc.C_API_S, 1.8, 'dn')
lc.text(1012, ELY - 5, 'asyncio=False', 9, '#334155', 'start', maxw=90, tag='a-false2')

# AsyncMPClient 叶旁 DP 分叉预告（虚线小注）
DPY = 774
lc.rect(110, DPY, 280, 46, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text(250, DPY + 19, 'DP>1 → DPAsyncMPClient / DPLBAsyncMPClient', 8.5,
        lc.C_MUTE, 'middle', maxw=268, tag='dp1')
lc.text(250, DPY + 36, '（ch34 预告 · core_client.py:L116-L139 再分流）', 8.5,
        lc.C_FAINT, 'middle', maxw=268, tag='dp2')
lc.seg(250, LY + LH, 250, DPY, lc.C_FAINT, 1.1, dash=True)

# ---------------- 图例 + 页脚 ----------------
LEG_Y = 852
items = [
    ('swatch', lc.C_API_S, '使用面与装配（API 进程侧）'),
    ('swatch', lc.C_ENG_S, '跨进程实现（引擎独立进程）'),
    ('swatch', GRAY_S, '进程内（无 IPC 逃生舱）'),
    ('dash', lc.C_ABORT, '拒绝组合'),
    ('thick', lc.C_API_S, '= 离线默认路径'),
]
lx = MX
for kind, color, name in items:
    if kind == 'swatch':
        lc.rect(lx, LEG_Y - 9, 16, 11, '#ffffff', color, rx=3, sw=1.6)
    elif kind == 'dash':
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, color, 1.6, dash=True)
    else:
        lc.seg(lx + 2, LEG_Y - 3, lx + 32, LEG_Y - 3, color, 3.2)
    lc.text(lx + 40, LEG_Y + 1, name, 9.5, lc.C_TXT, 'start', maxw=240,
            tag='leg:' + name[:8])
    lx += 40 + lc.tw(name, 9.5) + 24
lc.text(MX, LEG_Y + 26, '框内灰字 = 规范源码路径 · 行号基线 vLLM v0.27.1 · DP 分叉与 ch34 为预告（虚线）',
        9, lc.C_MUTE, 'start', maxw=1120, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch04-fig-client-factory.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
