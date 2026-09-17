#!/usr/bin/env python3
"""ch38 机制图 m1 · contract-nesting（figure_spec ch38-fig-contract-nesting，模板 layout）

放大自 L2 站 1（① 池的构成·契约再嵌套）· L0：KV 边界+外部池。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签；最外虚线框即 L0 的
「KV 边界+外部池」，不另立第二种架构画法。

claim：池化 facade 的三重身份——best-effort（requires_kv_delivery=False：丢一次 save
只是未来一次 miss）、契约再嵌套（KVConnectorBase_V1 之内 OffloadingSpec 又是一份
双面契约：get_manager 住调度器进程、get_worker 住 worker 进程）、工厂选引擎
（spec_name 默认 CPUOffloadingSpec，可选 TieringOffloadingSpec）。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · requires_kv_delivery=False（注释原话 a dropped save is just a future cache miss）
  · role 分裂：2 个 connector 对象、非空半边恰 2（每对象只活一半）
  · spec_name 三选路：缺省=CPUOffloadingSpec、显式 CPU=同、Tiering=TieringOffloadingSpec、未知名 ValueError
  · 内层契约 6+4 原语：manager{lookup/prepare_load/touch/complete_load/prepare_store/complete_store}
    × worker{submit_store/submit_load/get_finished/wait}
  · 三钩子全空：wait_for_layer_load/save_kv_layer/wait_for_save
  · 生态注册表 6 条；已删条目（DecodeBenchConnector）不再注册
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
lc.text(MX, 36, '池化 facade 的三重身份：best-effort 门面 · 契约再嵌套 · 工厂选引擎', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 60, '决策在调度器进程、搬运在 worker 进程——ch16 的双面契约在池化里又递归了一层', 10.5,
        lc.C_MUTE, 'start', maxw=1050, tag='subtitle')
_ch = '放大自 L2 站 1（① 池的构成·契约再嵌套）· L0：KV 边界+外部池'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- L0 外框（= L0 的 KV 边界+外部池，不另立架构画法） ----------------
# FRM_BOT 在结论行算出后回填（先占位，末尾重画会覆盖不了——这里直接先算结论位置）
CONC_Y_PRE = 150 + 300 + 26 + 74 + 24
FRM_Y, FRM_BOT = 84, CONC_Y_PRE + 14
lc.rect(MX, FRM_Y, BXR - MX, FRM_BOT - FRM_Y, '#ffffff', lc.C_KV_S, rx=10, sw=1.6, dash=True)
lc.text(MX + 16, FRM_Y + 20, 'L0 放大区：KV 边界（ch16 立）+ 外部池——本章打开的那一块', 11,
        lc.C_KV_S, 'start', True, maxw=520, tag='l0frame')

# ---------------- 三栏面板 ----------------
PANEL_Y = 150
PW_L, PW_M, PW_R = 420, 480, 430
GAP = 40
LX = MX + 24
MX_ = LX + PW_L + GAP
RX = MX_ + PW_M + GAP
PH = 300

# 左：OffloadingConnector facade（EngineCore 橙）
lc.rect(LX, PANEL_Y, PW_L, PH, lc.C_ENG_F, lc.C_ENG_S, rx=8, sw=1.6)
lc.text(LX + 16, PANEL_Y + 24, 'OffloadingConnector（facade 门面）', 11.5, lc.C_ENG_S,
        'start', True, maxw=PW_L - 32, tag='facade:t')
lc.text(LX + 16, PANEL_Y + 44, 'best-effort 三件贴在门面上：', 9.5, lc.C_TXT, 'start', True,
        maxw=PW_L - 32, tag='facade:lead')
facade_lines = [
    '① requires_kv_delivery = False',
    '    注释原话：a dropped save is just a',
    '    future cache miss',
    '② request_finished → (False, None)',
    '    不接管块释放（对照 ch37 P/D 的 True）',
    '③ 不逐层 · 不同步：',
    '    wait_for_layer_load / save_kv_layer /',
    '    wait_for_save 三钩子全空',
]
for i, ln in enumerate(facade_lines):
    lc.text(LX + 16, PANEL_Y + 64 + i * 16.5, ln, 9.5, '#334155', 'start',
            maxw=PW_L - 30, tag='facade:l%d' % i)
lc.text(LX + 16, PANEL_Y + PH - 10, 'offloading_connector.py:L49-L80 · L103-L134', 8.5,
        lc.C_FAINT, 'start', maxw=PW_L - 30, tag='facade:file')

# 中：KVConnectorBase_V1 外框 内嵌 OffloadingSpec（双面再劈两半）
lc.rect(MX_, PANEL_Y, PW_M, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.5, dash=True)
lc.text(MX_ + 16, PANEL_Y + 22, 'KVConnectorBase_V1 双面契约（ch16 已立）', 11, lc.C_MUTE,
        'start', True, maxw=PW_M - 32, tag='outer:t')
lc.text(MX_ + PW_M - 16, PANEL_Y + 22, '内层再嵌一份同构契约 ↓', 9, lc.C_MUTE, 'end',
        maxw=200, tag='outer:note')
SPEC_Y = PANEL_Y + 38
SPEC_H = 176
lc.rect(MX_ + 18, SPEC_Y, PW_M - 36, SPEC_H, lc.C_ENG_F, lc.C_ENG_S, rx=7, sw=1.4)
lc.text(MX_ + PW_M / 2, SPEC_Y + 18, 'OffloadingSpec（内层契约）', 10.5, lc.C_TXT, 'middle',
        True, maxw=PW_M - 60, tag='spec:t')
HALF_H = (SPEC_H - 34) / 2
# 上半：get_manager（调度器进程，橙）
lc.rect(MX_ + 26, SPEC_Y + 28, PW_M - 52, HALF_H, '#ffffff', lc.C_ENG_S, rx=6, sw=1.3)
lc.text(MX_ + 34, SPEC_Y + 46, 'get_manager → 调度器进程（账本原语 6）', 9.5, lc.C_ENG_S,
        'start', True, maxw=PW_M - 76, tag='mgr:t')
for i, ln in enumerate(['lookup · prepare_load · touch',
                        'complete_load · prepare_store · complete_store']):
    lc.text(MX_ + 34, SPEC_Y + 64 + i * 15, ln, 9, '#334155', 'start',
            maxw=PW_M - 76, tag='mgr:l%d' % i)
# 下半：get_worker（worker 进程，绿）
WY = SPEC_Y + 28 + HALF_H + 8
lc.rect(MX_ + 26, WY, PW_M - 52, HALF_H, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
lc.text(MX_ + 34, WY + 18, 'get_worker → worker 进程（搬运原语 4）', 9.5, lc.C_GPU_S,
        'start', True, maxw=PW_M - 76, tag='wkr:t')
lc.text(MX_ + 34, WY + 36, 'submit_store · submit_load · get_finished · wait', 9,
        '#334155', 'start', maxw=PW_M - 76, tag='wkr:l0')
lc.text(MX_ + 34, WY + 52, 'role 分裂：2 个 connector 对象、非空半边', 8.5, lc.C_MUTE,
        'start', maxw=PW_M - 76, tag='wkr:l1')
lc.text(MX_ + 34, WY + 65, '恰 2——每对象只活一半', 8.5, lc.C_MUTE, 'start',
        maxw=PW_M - 76, tag='wkr:l2')
lc.text(MX_ + 16, PANEL_Y + PH - 10, 'vllm/v1/kv_offload/base.py:L162-L186 · L545-L566', 8.5,
        lc.C_FAINT, 'start', maxw=PW_M - 32, tag='outer:file')

# 右：OffloadingSpecFactory（工厂选引擎，青）
lc.rect(RX, PANEL_Y, PW_R, PH, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.6)
lc.text(RX + 16, PANEL_Y + 24, 'OffloadingSpecFactory（spec_name 选引擎）', 11, lc.C_KV_S,
        'start', True, maxw=PW_R - 32, tag='fact:t')
fact_lines = [
    '· spec_name 缺省 → CPUOffloadingSpec',
    '· 显式 CPU → 同上',
    '· Tiering → TieringOffloadingSpec',
    '· 未知名 → ValueError 拒启',
]
for i, ln in enumerate(fact_lines):
    lc.text(RX + 16, PANEL_Y + 46 + i * 16, ln, 9.5, '#334155', 'start',
            maxw=PW_R - 30, tag='fact:l%d' % i)
ENG_Y = PANEL_Y + 118
for j, (name, sub) in enumerate([('CPUOffloadingSpec', 'CPU 单层'),
                                 ('TieringOffloadingSpec', 'CPU+fs/obj/p2p 分层')]):
    ex = RX + 16 + j * ((PW_R - 32 - 12) / 2 + 12)
    ew = (PW_R - 32 - 12) / 2
    lc.rect(ex, ENG_Y, ew, 64, '#ffffff', lc.C_KV_S, rx=6, sw=1.2)
    lc.text(ex + ew / 2, ENG_Y + 20, name, 8.8, lc.C_TXT, 'middle', True, maxw=ew - 8,
            tag='eng:t%d' % j)
    lc.text(ex + ew / 2, ENG_Y + 38, sub, 8.2, lc.C_MUTE, 'middle', maxw=ew - 8,
            tag='eng:s%d' % j)
lc.text(RX + 16, PANEL_Y + PH - 10, 'vllm/v1/kv_offload/factory.py:L38-L52', 8.5,
        lc.C_FAINT, 'start', maxw=PW_R - 30, tag='fact:file')

# ---------------- 箭头（端点全部贴框边） ----------------
# facade →（实现）外层契约：左框右边 → 中框左边
AY1 = PANEL_Y + PH * 0.42
lc.seg(LX + PW_L, AY1, MX_, AY1, lc.C_ENG_S, 1.8, 'std')
lc.text((LX + PW_L + MX_) / 2, AY1 - 8, '实现', 9, lc.C_ENG_S, 'middle', True, tag='a1')
# facade → 工厂（spec_name）：左框顶 → 上方绕行 → 右框顶
TOPR_Y = FRM_Y + 34
lc.parrow([(LX + PW_L / 2, PANEL_Y), (LX + PW_L / 2, TOPR_Y), (RX + PW_R / 2, TOPR_Y),
           (RX + PW_R / 2, PANEL_Y)], lc.C_KV_S, 1.5, 'std')
lc.text((LX + PW_L / 2 + RX + PW_R / 2) / 2, TOPR_Y - 6, '构造期按 spec_name 建引擎', 9,
        lc.C_KV_S, 'middle', maxw=400, tag='a2')
# 工厂 → 内层 spec（产出两半）：右框左边 → 中框右边
AY2 = PANEL_Y + PH * 0.62
lc.seg(RX, AY2, MX_ + PW_M, AY2, lc.C_KV_S, 1.8, 'std')
lc.text((RX + MX_ + PW_M) / 2, AY2 - 8, '产出 spec 两半', 9, lc.C_KV_S, 'middle',
        maxw=180, tag='a3')

# ---------------- 底部：生态注册表 6 条 ----------------
REG_Y = PANEL_Y + PH + 26
lc.rect(MX + 24, REG_Y, BXR - MX - 48, 74, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 40, REG_Y + 20, '生态注册表 6 条（kv_connector/factory.py:L152-L243）——同一份双面契约的全部注册面孔', 10,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 80, tag='reg:t')
reg_names = ['OffloadingConnector', 'MultiConnector', 'MooncakeConnector',
             'MooncakeStoreConnector', 'LMCacheConnectorV1', 'NixlConnector']
rx0 = MX + 40
for i, nm in enumerate(reg_names):
    loadable = nm in ('OffloadingConnector', 'MultiConnector', 'MooncakeStoreConnector')
    lw = lc.tw(nm, 9, True) + 18
    lc.rect(rx0, REG_Y + 32, lw, 22, lc.C_KV_F if loadable else '#f1f5f9',
            lc.C_KV_S if loadable else lc.C_FAINT, rx=9, sw=1.1,
            dash=not loadable)
    lc.text(rx0 + lw / 2, REG_Y + 47, nm, 9, lc.C_KV_S if loadable else lc.C_MUTE,
            'middle', True, maxw=lw - 4, tag='reg:%d' % i)
    rx0 += lw + 10
lc.text(rx0 + 6, REG_Y + 47, '实线=本章树内可直接加载 3 个 · 虚线=实现体在外部包；已删条目（DecodeBenchConnector）不再注册', 9,
        lc.C_MUTE, 'start', maxw=BXR - MX - 60 - (rx0 - MX), tag='reg:note')

# 结论行（图注结论给读者）
CONC_Y = REG_Y + 74 + 24
lc.text(MX + 24, CONC_Y,
        '图注结论：池化 = 同一份契约的第三种世界观（ch16 立契约 · ch37 填 P/D 可靠交接 · 本章填 best-effort 池）——『决策与搬运分离』在契约内层又递归了一次。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX - 48, tag='conc')

# ---------------- 页脚 ----------------
FY = CONC_Y + 22
lc.text(MX, FY, '逐字锚 offloading_connector.py:L49-L80（best-effort 三件）· L103-L134（三钩子全空）· '
                'kv_offload/factory.py:L38-L52（spec_name 三选路）· kv_offload/base.py:L162-L186 · L545-L566（内层 6+4 原语）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 15, '行号基线 vLLM v0.27.1 · 配色：调度器侧 EngineCore 橙 / worker 侧 GPU 绿 / 池引擎青（全书角色色）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-contract-nesting.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
