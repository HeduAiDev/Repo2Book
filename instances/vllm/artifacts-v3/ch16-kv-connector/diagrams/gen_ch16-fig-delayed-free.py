#!/usr/bin/env python3
"""ch16 机制图 8 · 延迟释放·已交接未送达（figure_spec ch16-fig-delayed-free，模板 before-after）

放大自 L0「KV 账本列 × GPU 列交界的回收格」（本章 l0_zoom）、L2 站 11（producer 终局·块的交接）。

claim：request_finished→True 后块进入『已交接未送达』挂起态——不释放、请求留账
（self.requests 不删），直到 worker 的 get_finished 报 finished_sending 才 _free_blocks；
has_finished_requests 让引擎在全部活请求结束后继续步进等这笔账。

数字全部取自 figure_spec.numbers（精简版 companion host 实测 trace：接管瞬间块仍挂 2、
free 61 不变、请求留册、has_finished_requests=true；finished_sending 到达出册、free 回 63；
交接内容=整块表，SupportsHMA 走 request_finished_all_groups 逐组块表）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX = 54
BXR = 1446

# ---------------- 标题区 ----------------
lc.text(MX, 36, '延迟释放：request_finished→True 之后，块要等签收才还池', 16.5, lc.C_TXT, 'start', True,
        maxw=980, tag='title')
lc.text(MX, 60, 'producer 的请求完成不是块生涯的终点而是交接点——所有权归 connector，free 停着不动，'
                '直到 worker 报 finished_sending（scheduler.py:L2577-L2612 / L2738-L2742）', 10.5, lc.C_MUTE,
        'start', maxw=1080, tag='subtitle')
_ch = '放大自 L2 站 11 producer 终局·块的交接 · L0：KV 账本列 × GPU 列交界·回收格'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 交接调用条 ----------------
HS_Y = 96
CHIPS = ['_free_request（请求完成）', '_connector_finished：窗外回收 + 按 num_computed_tokens 裁出整块表',
         'request_finished() 答 True = connector 接管（块不释放）']
hx = MX
for i, s in enumerate(CHIPS):
    fs = 9.5
    cw_ = lc.tw(s, fs, True) + 22
    lc.rect(hx, HS_Y, cw_, 34, '#ffffff', lc.C_ENG_S if i == 2 else lc.C_MUTE, rx=7, sw=1.4)
    lc.text(hx + cw_ / 2, HS_Y + 21, s, fs, lc.C_TXT, 'middle', True, maxw=cw_ - 10, tag=f'hc{i}')
    hx += cw_
    if i < len(CHIPS) - 1:
        lc.seg(hx + 2, HS_Y + 17, hx + 14, HS_Y + 17, lc.C_MUTE, 1.6, 'std')
        hx += 18

# ---------------- 三快照面板 ----------------
PN_Y = 164
PN_H = 344
PANELS = [
    ('步 1 · 异步准入（对照）', 'connector 答 (32, True)：占 2 块', False, '在册（等待传输）', 'r1 名下',
     61, 'r1', None),
    ('步 2 · 接管瞬间（挂起态 ★）', '请求完成，request_finished 答 True', True, '在册（已完成但留账）', '归 connector',
     61, 'r1', True),
    ('步 3 · finished_sending 到达', 'worker get_finished 报送达签收', False, '出册', '已还池',
     63, None, None),
]
PGAP = 24
PW = (BXR - MX - 2 * PGAP) / 3
for i, (t, ev, hot, ledger, holder, free, rchip, hfr) in enumerate(PANELS):
    x = MX + i * (PW + PGAP)
    stroke = lc.C_BEAT_S if hot else lc.C_MUTE
    fill = lc.C_BEAT_F if hot else '#ffffff'
    lc.rect(x, PN_Y, PW, PN_H, fill, stroke, rx=10, sw=2.0 if hot else 1.5)
    lc.text(x + PW / 2, PN_Y + 26, t, 11.5, lc.C_BEAT_T if hot else lc.C_TXT, 'middle', True,
            maxw=PW - 20, tag=f'pn{i}t')
    lc.text(x + PW / 2, PN_Y + 46, ev, 9, lc.C_MUTE, 'middle', maxw=PW - 20, tag=f'pn{i}e')
    # 请求册行
    ry = PN_Y + 66
    lc.text(x + 16, ry + 14, '请求册 self.requests：', 8.5, lc.C_MUTE, 'start', maxw=180, tag=f'pn{i}r')
    bw = 96
    if rchip:
        lc.rect(x + 170, ry, bw, 22, '#ffffff', stroke, rx=9, sw=1.2)
        lc.text(x + 170 + bw / 2, ry + 15, f'{rchip} 在队列里', 8.5, lc.C_TXT, 'middle', True,
                maxw=bw - 6, tag=f'pn{i}rc')
    lc.text(x + 170 + bw + 10, ry + 15, ledger, 9.5, lc.C_BEAT_T if hot else '#334155', 'start', True,
            maxw=PW - 200 - bw, tag=f'pn{i}rl')
    # 块格行
    byy = ry + 34
    lc.text(x + 16, byy + 34, '块表：', 8.5, lc.C_MUTE, 'start', maxw=70, tag=f'pn{i}bt')
    bs = 60
    for j, bid in enumerate([1, 2]):
        bx = x + 62 + j * (bs + 10)
        held = i < 2
        lc.rect(bx, byy, bs, 46, lc.C_BEAT_F if held else '#f1f5f9',
                lc.C_BEAT_S if held else lc.C_MUTE, rx=6, sw=1.4)
        lc.text(bx + bs / 2, byy + 19, f'块 {bid}', 10, lc.C_BEAT_T if held else lc.C_MUTE, 'middle', True,
                maxw=bs - 8, tag=f'pn{i}b{j}')
        lc.text(bx + bs / 2, byy + 37, holder if held else '归还', 7.5, lc.C_BEAT_T if held else lc.C_MUTE,
                'middle', maxw=bs - 6, tag=f'pn{i}bs{j}')
    # free 大数字
    lc.text(x + 16, PN_Y + 186, f'free {free}', 20, lc.C_BEAT_T if hot else lc.C_MUTE, 'start', True,
            maxw=140, tag=f'pn{i}f')
    if hfr:
        lc.text(x + 16, PN_Y + 210, 'has_finished_requests = true', 9.5, lc.C_BEAT_T, 'start', True,
                maxw=230, tag=f'pn{i}hf')
        lc.text(x + 16, PN_Y + 226, '引擎不会提前收工', 8.5, lc.C_MUTE, 'start', maxw=230, tag=f'pn{i}hf2')
    if i == 0:
        lc.text(x + 16, PN_Y + 210, '（池 64 · 基线 free 63）', 8.5, lc.C_MUTE, 'start', maxw=230, tag='pn0b')
    if i == 2:
        lc.text(x + 16, PN_Y + 210, '全部归还', 9.5, lc.C_GPU_S, 'start', True, maxw=230, tag='pn2ok')
        lc.text(x + 16, PN_Y + 226, '这笔账结清', 8.5, lc.C_MUTE, 'start', maxw=230, tag='pn2ok2')
    # 池条
    py = PN_Y + 250
    CELL, CGAP = (PW - 32 - 63 * 1.2) / 64 + 1.2 - 1.2, 1.2   # 让 64 格恰好铺满内宽
    CELL = (PW - 32 - 63 * CGAP) / 64
    for b in range(64):
        bx = x + 16 + b * (CELL + CGAP)
        if b == 0:
            f_, s_ = '#cbd5e1', lc.C_MUTE
        elif b in (1, 2) and i < 2:
            f_, s_ = lc.C_BEAT_S, lc.C_BEAT_S
        else:
            f_, s_ = '#ffffff', '#cbd5e1'
        lc.rect(bx, py, CELL, 18, f_, s_, rx=1.2, sw=0.7)
    lc.text(x + 16, py + 32, 'free ' + str(free), 8, lc.C_BEAT_T if hot else lc.C_MUTE, 'start', maxw=100,
            tag=f'pn{i}pf')
    lc.text(x + PW - 16, py + 32, '琥珀=仍被持有', 8, lc.C_FAINT, 'end', maxw=150, tag=f'pn{i}pl')

# 面板间箭头
for i in range(2):
    x0 = MX + (i + 1) * PW + i * PGAP + 6
    x1 = x0 + PGAP - 9
    lc.seg(x0, PN_Y + PN_H / 2, x1, PN_Y + PN_H / 2, lc.C_MUTE, 2.0, 'std')

# ---------------- 底部注记 ----------------
NT_Y = PN_Y + PN_H + 26
lc.rect(MX, NT_Y, BXR - MX, 96, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 18, NT_Y + 22, '交接内容 = 整块表：实测 per_group_tables [[1, 2]]（num_groups=1）；混合架构走 SupportsHMA.request_finished_all_groups 逐组交接（base.py:L93-L114）',
        9, '#334155', 'start', maxw=1340, tag='nt1')
lc.text(MX + 18, NT_Y + 42, '留账的意义：has_finished_requests（scheduler.py:L2394-L2404 注释原话——已完成的请求『remain in self.requests』等延迟清理）'
        '让引擎在全部活请求结束后继续步进，等这笔账结清', 9, '#334155', 'start', maxw=1340, tag='nt2')
lc.text(MX + 18, NT_Y + 62, '对照组：best-effort 缓存（offload）答 False → 块立即释放、请求出册——丢一次 save 只是未来一次 miss（requires_kv_delivery 语义的另一面）',
        9, '#334155', 'start', maxw=1340, tag='nt3')
lc.text(MX + 18, NT_Y + 82, '步 2→步 3 之间：producer 的 save 仍在途——先腾块就等于把还没寄出的货扔了',
        9, lc.C_MUTE, 'start', maxw=1340, tag='nt4')

# ---------------- 页脚 ----------------
FY = NT_Y + 96 + 28
lc.text(MX, FY, '逐字锚 vllm/v1/core/sched/scheduler.py:L2577-L2612（_connector_finished：窗外回收 + get_block_ids_for_computed_tokens 裁整块表）· '
                'L2738-L2742（finished_sending → _free_blocks）· L2394-L2404（has_finished_requests）· base.py:L93-L114（request_finished_all_groups）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '三快照读数（挂 2 块 / free 61 不变 / 留册 / has_finished_requests=true → 出册 / free 63）取自精简版 companion host 实测 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 36

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch16-fig-delayed-free.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
