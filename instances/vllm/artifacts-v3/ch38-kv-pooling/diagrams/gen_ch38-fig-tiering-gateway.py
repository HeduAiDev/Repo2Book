#!/usr/bin/env python3
"""ch38 机制图 m11 · tiering-gateway（figure_spec ch38-fig-tiering-gateway，模板 layout）

放大自 L2 站 11（⑨ 分层池·cascade/promotion）· L0：外部池的分层纵深（HBM/DRAM/SSD 的代码形态）。

claim：分层池五原则的几何：CPU primary 是唯一能 DMA GPU 的层（网关），secondary
（fs/obj/p2p）只能经它中转——store 完成即 cascade 全层下推、secondary 命中先 promotion
升回主层（期间 RETRY）、primary 满则 promotion 失败=MISS、ref_cnt 作驱逐保护。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · 五原则 docstring 原文锚（tiering/manager.py:L4-L21）
  · promotion 节奏：RETRY→RETRY→（2 轮 on_schedule_end）→HIT；负载 16 字节回主层逐字节齐
  · cascade 实测：complete_store 后 fs 层落 1 个 .bin（四级路径 model_digest_r0/000/00_g0/hash.bin）
  · 调度器进程也 mmap 同一池（rank=None）
  · primary 满 → promotion 失败 → MISS；store_threshold≥2 分层禁用（raise）
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1540
MX = 60
BXR = W - MX
C_SEC = '#64748b'   # secondary 层灰蓝

# ---------------- 标题区 ----------------
lc.text(MX, 36, '分层池是一座中转枢纽：只有 CPU 月台能通 GPU 的轨道', 16.5,
        lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 60, '货要进 SSD 仓 / 远端库都从月台再搬一程（cascade 下推），要从仓里取回也先搬回月台（promotion 升回）——期间查询一律答「稍后再问」（RETRY）',
        10.5, lc.C_MUTE, 'start', maxw=1130, tag='subtitle')
_ch = '放大自 L2 站 11（⑨ 分层池·cascade/promotion）· L0：外部池分层纵深'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 请求入口（左上） ----------------
ENT_X, ENT_Y, ENT_W, ENT_H = MX, 128, 220, 64
lc.rect(ENT_X, ENT_Y, ENT_W, ENT_H, '#ffffff', lc.C_ENG_S, rx=8, sw=1.4)
lc.text(ENT_X + ENT_W / 2, ENT_Y + 20, '新请求查池', 10, lc.C_ENG_S, 'middle', True,
        maxw=ENT_W - 12, tag='ent:t')
lc.text(ENT_X + ENT_W / 2, ENT_Y + 38, 'get_num_new_matched_tokens', 7.6, '#334155',
        'middle', maxw=ENT_W - 10, tag='ent:s')
lc.text(ENT_X + ENT_W / 2, ENT_Y + 54, '（先查 primary，miss 才下探）', 7.8, lc.C_MUTE,
        'middle', maxw=ENT_W - 10, tag='ent:n')

# ---------------- 三层横带（中区） ----------------
CX0, CX1 = 360, 1120
# GPU 带
GY, GH_ = 128, 84
lc.rect(CX0, GY, CX1 - CX0, GH_, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(CX0 + 18, GY + 24, 'GPU · HBM', 12, lc.C_GPU_S, 'start', True, maxw=200, tag='gpu:t')
lc.text(CX0 + 18, GY + 44, '快而小——真正的算力驻地', 9, '#334155', 'start', maxw=280,
        tag='gpu:s')
lc.text(CX0 + 18, GY + 66, 'KV 块由 KV cache manager 按块表分配', 8, lc.C_MUTE, 'start',
        maxw=320, tag='gpu:n')
# CPU primary 带
PY0, PH0 = 262, 128
lc.rect(CX0, PY0, CX1 - CX0, PH0, lc.C_KV_F, lc.C_KV_S, rx=10, sw=2.2)
lc.text(CX0 + 18, PY0 + 24, 'CPU primary · /dev/shm 池', 12, lc.C_KV_S, 'start', True,
        maxw=300, tag='pri:t')
gw = lc.tw('唯一能 DMA GPU 的层 · 网关', 9.5, True) + 20
lc.rect(CX1 - gw - 14, PY0 + 10, gw, 22, lc.C_KV_S, lc.C_KV_S, rx=9, sw=1.2)
lc.text(CX1 - gw / 2 - 14, PY0 + 25, '唯一能 DMA GPU 的层 · 网关', 9.5, '#ffffff', 'middle',
        True, maxw=gw - 6, tag='pri:badge')
for j, ln in enumerate(['· worker 半边只持有 primary 的 CPUOffloadingWorker——代码路径上不存在 secondary→GPU 的通道',
                        '· ref_cnt 作驱逐保护：cascade 传输期 prepare_read 钉住 · promotion 占位 prepare_write（ref_cnt=-1）']):
    lc.text(CX0 + 18, PY0 + 48 + j * 17, ln, 8.6, '#334155', 'start', maxw=CX1 - CX0 - 40,
            tag='pri:l%d' % j)
lc.text(CX0 + 18, PY0 + PH0 - 10, 'shared_offload_region.py · 调度器进程（rank=None）也 mmap 同一物理池', 8,
        lc.C_FAINT, 'start', maxw=CX1 - CX0 - 40, tag='pri:f')
# secondary 带（一行三框）
SY, SH_ = 452, 96
sec = [('fs', '文件系统层（线程池 I/O）', False),
       ('obj', '对象存储层（源码树已删）', True),
       ('p2p', '对端引擎的 CPU 池（ch37 同源）', False)]
SW = (CX1 - CX0 - 2 * 20) / 3
for i, (name, sub, deleted) in enumerate(sec):
    x = CX0 + i * (SW + 20)
    lc.rect(x, SY, SW, SH_, '#f8fafc', C_SEC, rx=8, sw=1.4, dash=deleted)
    nm = name + ('（已删）' if deleted else '')
    lc.text(x + SW / 2, SY + 26, nm, 11, C_SEC, 'middle', True, maxw=SW - 10,
            tag='sec%d' % i)
    lc.text(x + SW / 2, SY + 48, sub, 8.2, lc.C_MUTE, 'middle', maxw=SW - 12,
            tag='secs%d' % i)
    if deleted:
        ly = SY + 26
        lc.seg(x + SW / 2 - 34, ly - 4, x + SW / 2 + 34, ly - 4, lc.C_ABORT, 1.3)
lc.text(CX0 + 18, SY - 10, 'secondary 副层：只能经 primary 中转', 9, C_SEC, 'start', True,
        maxw=400, tag='sec:cap')

# ---------------- 通道箭头 ----------------
# GPU ↔ primary：DMA 双向（左）
dma_x = CX0 + 120
lc.seg(dma_x, GY + GH_, dma_x, PY0, lc.C_GPU_S, 2.0, 'std')
lc.text(dma_x + 8, GY + GH_ + 20, 'store 下卸（DMA·worker 进程）', 8, lc.C_GPU_S, 'start',
        maxw=200, tag='dma:down')
dma2_x = CX0 + 320
lc.seg(dma2_x, PY0, dma2_x, GY + GH_, lc.C_KV_S, 2.0, 'std')
lc.text(dma2_x + 8, GY + GH_ + 20, 'load 回流（同一条专用拷贝流）', 8, lc.C_KV_S, 'start',
        maxw=200, tag='dma:up')
# cascade ↓ 与 promotion ↑（右侧双通道）
ca_x, pr_x = CX1 - 260, CX1 - 80
lc.seg(ca_x, PY0 + PH0, ca_x, SY, C_SEC, 2.2, 'std')
lc.text(ca_x - 8, 424, 'cascade 全层下推', 9, C_SEC, 'end', True, maxw=130, tag='ca:t')
lc.text(ca_x - 8, 440, 'store 完成即下推（prepare_read 钉 ref_cnt）', 7.8, C_SEC, 'end',
        maxw=190, tag='ca:s')
lc.seg(pr_x, SY, pr_x, PY0 + PH0, lc.C_ENG_S, 2.2, 'std')
lc.text(pr_x + 8, 396, 'promotion 升回', 9, lc.C_ENG_S, 'start', True, maxw=120, tag='pr:t')
lc.text(pr_x + 8, 412, '命中先升回主层，期间 RETRY', 7.8, lc.C_ENG_S, 'start', maxw=170,
        tag='pr:s')
# 请求入口 → primary 查询箭头（终点落在 primary 带左边线上）
lc.seg(ENT_X + ENT_W, ENT_Y + ENT_H / 2, CX0, PY0 + 40, lc.C_ENG_S, 1.8, 'std')
lc.text((ENT_X + ENT_W + CX0) / 2, ENT_Y + ENT_H / 2 - 24, 'lookup', 8.5, lc.C_ENG_S,
        'middle', maxw=80, tag='q:t')
# RETRY 回环虚线（secondary 左缘 → 绕回请求入口底边）
lc.parrow([(CX0, SY + SH_ / 2), (ENT_X + 40, SY + SH_ / 2), (ENT_X + 40, ENT_Y + ENT_H)],
          lc.C_ABORT, 1.6, 'std', dash=True)
lc.text(ENT_X + 52, (SY + SH_ / 2 + ENT_Y + ENT_H) / 2 + 4, '查询答 RETRY：稍后再问', 8.5,
        lc.C_ABORT, 'start', maxw=150, tag='retry:t')

# ---------------- 右列：实测节奏与代价 ----------------
RX = 1160
RW = BXR - RX
boxes = [
    ('promotion 节奏实测', lc.C_ENG_S, [
        '· lookup#1 → RETRY（fs 异步查询入队）',
        '· lookup#2 → RETRY（prepare_write 占位）',
        '· 2 轮 on_schedule_end 批量提交后 → HIT',
        '· 最坏 3-4 个引擎步：『SSD 里的块',
        '   命中了却到不了 GPU』的具体节奏',
        '· 负载 16 字节回主层逐字节齐',
    ], 'tiering/manager.py:L282-L451'),
    ('cascade 实测', C_SEC, [
        '· complete_store 后 fs 层落 1 个 .bin',
        '· 四级路径：model_digest_r0/000/',
        '   00_g0/<hash>.bin（1 块 × 每副层 1 份）',
        '· store_threshold ≥ 2 分层禁用（显式 raise',
        '   ——cascade 要求全块上下文）',
    ], 'tiering/manager.py:L586-L630'),
    ('代价面', lc.C_ABORT, [
        '· primary 满 → promotion 失败 → lookup MISS',
        '   （『有数据也到不了 GPU』）',
        '· secondary 命中必两跳：fs→CPU→GPU',
        '· CPU↔secondary I/O 住调度器进程',
        '   （GPU↔CPU DMA 住 worker 进程）',
    ], 'tiering/spec.py:L170-L215'),
]
by = GY
for t, color, lines, foot in boxes:
    h = 30 + len(lines) * 15 + 20
    lc.rect(RX, by, RW, h, '#ffffff', color, rx=8, sw=1.4)
    lc.text(RX + 12, by + 19, t, 9.8, color, 'start', True, maxw=RW - 24, tag='rb:' + t)
    for j, ln in enumerate(lines):
        lc.text(RX + 12, by + 38 + j * 15, ln, 8.2, '#334155', 'start', maxw=RW - 22,
                tag='rb:%s:l%d' % (t, j))
    lc.text(RX + 12, by + h - 8, foot, 7.4, lc.C_FAINT, 'start', maxw=RW - 22,
            tag='rb:%s:f' % t)
    by += h + 16

# ---------------- 底条：五原则 ----------------
BP_Y = SY + SH_ + 24
lc.rect(MX, BP_Y, BXR - MX, 62, '#ffffff', lc.C_KV_S, rx=8, sw=1.2)
lc.text(MX + 16, BP_Y + 20, '五原则（tiering/manager.py:L4-L21 docstring 原文锚）', 9.8,
        lc.C_KV_S, 'start', True, maxw=BXR - MX - 32, tag='bp:t')
prin = ['Always offload to all tiers', 'Primary tier is the gateway', 'Staged promotion',
        'Transparent retry', 'ref_cnt as eviction protection']
px = MX + 16
for p in prin:
    pw = lc.tw(p, 8.5) + 16
    lc.rect(px, BP_Y + 30, pw, 20, lc.C_KV_F, lc.C_KV_S, rx=9, sw=1.0)
    lc.text(px + pw / 2, BP_Y + 44, p, 8.5, lc.C_KV_S, 'middle', maxw=pw - 4,
            tag='bp:' + p[:8])
    px += pw + 8

# ---------------- 结论 + 页脚 ----------------
CONC_Y = BP_Y + 62 + 24
lc.text(MX, CONC_Y,
        '图注结论：分层不是多修几条路，是把所有路都并进一座枢纽——代价是 secondary 数据必两跳（实测 RETRY×2 后 HIT）。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 vllm/v1/kv_offload/tiering/manager.py:L4-L21（五原则）· L282-L451（promotion 两跳与 RETRY）· L586-L630（cascade）· tiering/spec.py:L170-L215（rank=None mmap）· tiering/factory.py:L57-L77（副层工厂 fs/p2p）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · host 实测走真 TieringOffloadingManager + 真 fs 层（线程池文件 I/O）；store_threshold≥2 在分层模式显式 raise（实测 ValueError）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-tiering-gateway.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
