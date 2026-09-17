#!/usr/bin/env python3
"""ch38 机制图 m14 · eco-three-fillings（figure_spec ch38-fig-eco-three-fillings，模板 layout）

放大自 L2 站 13（生态对照）· L0：外部池在生态位上的三张变体脸（同一 KV 边界的不同池后端）。

claim：同一份双面契约的三种填法对照：MooncakeStore=池出引擎进程（master 协调分布式仓+SSD 层、
查询 ZMQ RPC 异步 None=稍后再问、I/O 全押 get_finished）；LMCache=外部引擎薄壳
（use_native 双路 lazy import）；MultiConnector=组合器（P/D 与池化并存，列表顺序首个
命中数>0 者获加载权、全子收 store）。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · Mooncake 查询：未就绪 (None,False) → 就绪 (12,True)；本地已算对齐 12−12=0
  · Mooncake worker 双 no-op（I/O 全在 get_finished，docstring 原话 after model compute is launched）
  · LMCache use_native True/False 分别加载 native/latest 两实现
  · MultiConnector 首命中：A=0→B=5 获胜、分配 A 收 0/B 收 5；后续更长不覆盖（注释原话）
  · 注册表六面孔（本章树内可加载 3）
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
lc.text(MX, 36, '同一份双面契约的三种填法：分布式仓 · 外挂引擎 · 配电排', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 60, '契约不动、世界观随便换——MooncakeStore 把池搬出引擎进程，LMCache 是外挂引擎薄壳，MultiConnector 是 P/D 与池化并存的组合器',
        10.5, lc.C_MUTE, 'start', maxw=1100, tag='subtitle')
_ch = '放大自 L2 站 13（生态对照）· L0：外部池生态位的三张变体脸'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶梁：ch16 双面契约 ----------------
BM_Y, BM_H = 92, 44
lc.rect(MX, BM_Y, BXR - MX, BM_H, lc.C_ENG_F, lc.C_ENG_S, rx=8, sw=1.6)
lc.text((MX + BXR) / 2, BM_Y + 27, 'KVConnectorBase_V1 双面契约（ch16 立）——三栏装的都是同一道 KV 边界',
        11.5, lc.C_ENG_S, 'middle', True, maxw=BXR - MX - 40, tag='beam')

COL_W = (BXR - MX - 2 * 24) / 3
COL_Y, COL_H = 196, 388
cols = [
    ('MooncakeStoreConnector', '分布式仓：池出引擎进程', lc.C_ZMQ_S),
    ('LMCacheConnectorV1', '外挂引擎薄壳：双路 lazy import', lc.C_API_S),
    ('MultiConnector', '配电排：P/D 与池化并存', lc.C_GPU_S),
]
for i, (name, sub, color) in enumerate(cols):
    x = MX + i * (COL_W + 24)
    # 顶梁 → 栏 的下垂箭头
    lc.seg(x + COL_W / 2, BM_Y + BM_H, x + COL_W / 2, COL_Y, color, 1.8, 'std')
    lc.rect(x, COL_Y, COL_W, COL_H, '#ffffff', color, rx=8, sw=1.5)
    lc.text(x + 14, COL_Y + 22, name, 10, color, 'start', True, maxw=COL_W - 28,
            tag='c%d:t' % i)
    lc.text(x + 14, COL_Y + 40, sub, 8.6, lc.C_MUTE, 'start', maxw=COL_W - 28,
            tag='c%d:s' % i)

# ---- 栏 1：MooncakeStore ----
x1 = MX
EG_Y, EG_H = COL_Y + 52, 118
lc.rect(x1 + 14, EG_Y, COL_W - 28, EG_H, lc.C_ENG_F, lc.C_ENG_S, rx=7, sw=1.2, dash=True)
lc.text(x1 + 26, EG_Y + 18, '引擎进程内（薄半边）', 8.8, lc.C_ENG_S, 'start', True,
        maxw=COL_W - 52, tag='mc:eg')
for j, ln in enumerate(['· start_load_kv / wait_for_save 双 no-op',
                        '· I/O 全押 get_finished（docstring 原话：',
                        '   after model compute is launched）']):
    lc.text(x1 + 26, EG_Y + 36 + j * 15, ln, 8.2, '#334155', 'start', maxw=COL_W - 50,
            tag='mc:eg:l%d' % j)
ST_Y = EG_Y + EG_H + 30
lc.rect(x1 + 14, ST_Y, COL_W - 28, 92, '#f5f3ff', lc.C_ZMQ_S, rx=7, sw=1.5)
lc.text(x1 + 26, ST_Y + 18, '引擎进程外：MooncakeDistributedStore', 8.8, lc.C_ZMQ_S,
        'start', True, maxw=COL_W - 52, tag='mc:st:t')
for j, ln in enumerate(['· master 元数据协调 · 各 rank 出内存凑池',
                        '· standalone-store 外置进程持池 + SSD 层',
                        '· 池在引擎外——与本章正典的本质差异']):
    lc.text(x1 + 26, ST_Y + 36 + j * 15, ln, 8.2, '#334155', 'start', maxw=COL_W - 50,
            tag='mc:st:l%d' % j)
lc.seg(x1 + COL_W / 2, EG_Y + EG_H, x1 + COL_W / 2, ST_Y, lc.C_ZMQ_S, 1.6, 'std')
lc.text(x1 + COL_W / 2 + 6, EG_Y + EG_H + 19, 'ZMQ RPC', 8, lc.C_ZMQ_S, 'start', maxw=80,
        tag='mc:rpc')
QY = ST_Y + 92 + 20
lc.text(x1 + 14, QY, '查询异步（第三种 None=稍后再问，接 ch16 skipped）：', 8.6, lc.C_TXT,
        'start', True, maxw=COL_W - 24, tag='mc:q')
lc.text(x1 + 14, QY + 17, '· 未就绪 (None, False) → 就绪 (12, True)', 8.4, '#334155',
        'start', maxw=COL_W - 24, tag='mc:q1')
lc.text(x1 + 14, QY + 34, '· 差值对齐：外部 12 − 本地已算 12 = 0', 8.4, '#334155',
        'start', maxw=COL_W - 24, tag='mc:q2')
lc.text(x1 + 14, QY + 51, '   （need_to_allocate 与 OffloadingConnector 同构）', 8,
        lc.C_MUTE, 'start', maxw=COL_W - 24, tag='mc:q3')
lc.text(x1 + 14, COL_Y + COL_H - 10, 'mooncake/store/worker.py:L1538-L1562 · store/scheduler.py:L81-L134', 7.4,
        lc.C_FAINT, 'start', maxw=COL_W - 26, tag='mc:f')

# ---- 栏 2：LMCache ----
x2 = MX + COL_W + 24
SW_Y = COL_Y + 58
lc.rect(x2 + 14, SW_Y, COL_W - 28, 40, '#ffffff', lc.C_API_S, rx=7, sw=1.4)
lc.text(x2 + COL_W / 2, SW_Y + 24, '薄壳开关 use_native', 9.5, lc.C_API_S, 'middle', True,
        maxw=COL_W - 52, tag='lm:sw')
p_w = (COL_W - 28 - 16) / 2
P_Y = SW_Y + 88
paths = [('True → 内置适配器', 'vllm 内 lmcache_integration', '（原生 native）'),
         ('False → 外部包', 'lmcache 的 vllm_v1_adapter', '（latest dev）')]
for j, (t, l1, l2) in enumerate(paths):
    px = x2 + 14 + j * (p_w + 16)
    lc.rect(px, P_Y, p_w, 84, '#ffffff', lc.C_API_S, rx=7, sw=1.2)
    lc.text(px + p_w / 2, P_Y + 20, t, 8.8, lc.C_API_S, 'middle', True, maxw=p_w - 8,
            tag='lm:p%d:t' % j)
    lc.text(px + p_w / 2, P_Y + 40, l1, 7.8, '#334155', 'middle', maxw=p_w - 8,
            tag='lm:p%d:l1' % j)
    lc.text(px + p_w / 2, P_Y + 56, l2, 7.8, lc.C_MUTE, 'middle', maxw=p_w - 8,
            tag='lm:p%d:l2' % j)
    lc.text(px + p_w / 2, P_Y + 74, '引擎对象 A' if j == 0 else '引擎对象 B', 7.6, lc.C_MUTE,
            'middle', maxw=p_w - 8, tag='lm:p%d:e' % j)
    lc.seg(x2 + 14 + (COL_W - 28) / 2 if False else x2 + 14 + (0 if j == 0 else 1) * (p_w + 16) + p_w / 2,
           SW_Y + 40, px + p_w / 2, P_Y, lc.C_API_S, 1.6, 'std')
lc.text(x2 + 14, P_Y + 108, '· 两路加载的两个引擎对象互异（实测）', 8.4, '#334155',
        'start', maxw=COL_W - 24, tag='lm:n1')
lc.text(x2 + 14, P_Y + 125, '· 同一注册名、两种工程形态：内置', 8.4, '#334155', 'start',
        maxw=COL_W - 24, tag='lm:n2')
lc.text(x2 + 14, P_Y + 142, '   适配器 vs 外部包——装谁由开关定', 8.4, '#334155', 'start',
        maxw=COL_W - 24, tag='lm:n3')
lc.text(x2 + 14, COL_Y + COL_H - 10, 'lmcache_connector.py:L83-L113（双路 lazy import）', 7.4,
        lc.C_FAINT, 'start', maxw=COL_W - 26, tag='lm:f')

# ---- 栏 3：MultiConnector ----
x3 = MX + 2 * (COL_W + 24)
sub_rows = [('子连接器 A', '查得 0', False), ('子连接器 B', '查得 5', True),
            ('子连接器 C', '…', False)]
ry0 = COL_Y + 58
for j, (nm, val, win) in enumerate(sub_rows):
    ry = ry0 + j * 44
    lc.rect(x3 + 14, ry, COL_W - 28, 36, '#f0fdf4' if win else '#ffffff',
            lc.C_GPU_S if win else lc.C_MUTE, rx=6, sw=1.5 if win else 1.1)
    lc.text(x3 + 26, ry + 23, nm, 9, lc.C_GPU_S if win else lc.C_TXT, 'start', True,
            maxw=140, tag='mu:r%d' % j)
    lc.text(x3 + COL_W - 90, ry + 23, val, 9, lc.C_GPU_S if win else lc.C_MUTE, 'end',
            maxw=60, tag='mu:v%d' % j)
    if win:
        bw = lc.tw('首个>0 获加载权', 7.8, True) + 12
        lc.rect(x3 + 150, ry + 9, bw, 18, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.1)
        lc.text(x3 + 150 + bw / 2, ry + 22, '首个>0 获加载权', 7.8, lc.C_GPU_S, 'middle',
                True, maxw=bw - 6, tag='mu:w')
mu_y = ry0 + 3 * 44 + 8
mu_lines = [
    '· 列表顺序：第一个命中数>0 者获加载权',
    '· 分配路由实测：A 收 (r1, 0) · B 收 (r1, 5)',
    '· 下一请求 A=20 先非零 → A 获胜——每请求',
    '   独立判定；后续更长命中不覆盖（注释原话）',
    '· 全部子连接器都收 store（P/D 腿也能挂）',
]
for j, ln in enumerate(mu_lines):
    lc.text(x3 + 14, mu_y + j * 15.5, ln, 8.4, '#334155', 'start', maxw=COL_W - 24,
            tag='mu:l%d' % j)
lc.text(x3 + 14, COL_Y + COL_H - 10, 'multi_connector.py:L399-L404（首命中赋值）', 7.4,
        lc.C_FAINT, 'start', maxw=COL_W - 26, tag='mu:f')

# ---------------- 底条：注册表六面孔 ----------------
RG_Y = COL_Y + COL_H + 22
lc.rect(MX, RG_Y, BXR - MX, 64, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 16, RG_Y + 20, '生态注册表六面孔（kv_connector/factory.py:L152-L243）——本章树内可直接加载 3 个（实线）', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='rg:t')
reg_names = ['OffloadingConnector', 'MultiConnector', 'MooncakeConnector',
             'MooncakeStoreConnector', 'LMCacheConnectorV1', 'NixlConnector']
rx0 = MX + 16
for nm in reg_names:
    loadable = nm in ('OffloadingConnector', 'MultiConnector', 'MooncakeStoreConnector')
    lw = lc.tw(nm, 8.8, True) + 16
    lc.rect(rx0, RG_Y + 32, lw, 21, lc.C_KV_F if loadable else '#f1f5f9',
            lc.C_KV_S if loadable else lc.C_FAINT, rx=9, sw=1.0, dash=not loadable)
    lc.text(rx0 + lw / 2, RG_Y + 46, nm, 8.8, lc.C_KV_S if loadable else lc.C_MUTE,
            'middle', True, maxw=lw - 4, tag='rg:' + nm[:6])
    rx0 += lw + 8
lc.text(BXR - 16, RG_Y + 46, '虚线=实现体在外部包', 8, lc.C_MUTE, 'end', maxw=170,
        tag='rg:note')

# ---------------- 结论 + 页脚 ----------------
CONC_Y = RG_Y + 64 + 24
lc.text(MX, CONC_Y,
        '图注结论：契约不动、世界观随便换——分布式仓、外挂引擎、配电排，装的都是同一道 KV 边界。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 mooncake/store/connector.py:L88-L96 · store/scheduler.py:L81-L134（异步查询与差值语义）· store/worker.py:L1538-L1562（双 no-op）· lmcache_connector.py:L83-L113 · multi_connector.py:L399-L404',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · 栏色按「池在哪」分类：紫=池出引擎进程（跨进程 RPC）· 蓝=引擎在包外 · 绿=组合器；Mooncake 查询走真 ZMQ REQ 对脚本化 REP 服务实测',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-eco-three-fillings.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
