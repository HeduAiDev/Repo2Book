#!/usr/bin/env python3
"""ch17 机制图 3 · FutureWrapper 的 FIFO 配对（figure_spec ch17-fig-futurewrapper-fifo，模板 state-table）

放大自 L0 GPU 执行臂行（gpu_column）Executor 块的收割半边——即 L2 章图 south
『output_rank 收割 + FutureWrapper』组件（站 13）的机制展开。架构归属回指 L2/L0
（FIGURE-SYSTEM §3.3）：图右上角指北小签。

claim：两条 FIFO（每 worker 应答 MQ、futures_queue）天然同序，result() 先排干先于自己的
future——第 k 次 dequeue 必是第 k 个 RPC 的回复：实测 4 个 RPC 背靠背连发 0.05ms 后只收
最新者，一次 result() 排空全部 4 个（None/{5}/None/{9} 各归各、队列清空），全程没有一个
RPC id。

数字全部取自 figure_spec.numbers（m12_futurewrapper_fifo host 实测 trace + pin 锚点，逐字
对齐）；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 1016
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '配对不靠 id 靠顺序：result() 先排干先于自己——第 k 次 dequeue = 第 k 个 RPC 的回复',
        16.5, lc.C_TXT, 'start', True, maxw=1070, tag='title')
lc.text(MX, 58, '广播 MQ FIFO（worker 按发出序处理）· 应答 MQ FIFO（回复序=处理序）· futures_queue '
        'appendleft/pop 即 FIFO——三条队列天然同序，RPC 协议里一个 id 都没有（multiproc_executor.py:L70-L100）',
        10.5, lc.C_MUTE, 'start', maxw=1070, tag='subtitle')
_ch = '放大自 L2 south output_rank 收割 + FutureWrapper（站 13）· L0：GPU 执行臂上层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶部：deque 机制（队内序演进） ----------------
lc.text(MX, 100, '队列机制——futures_queue 本体：appendleft 左进 / pop 右出 ⇒ deque 是 FIFO', 10.5,
        lc.C_TXT, 'start', True, maxw=960, tag='anat:t')
DY, DH = 112, 48
SQ_W, SQ_H, SQ_GAP = 44, 28, 6


def mini_deque(x, w, names):
    """小 deque 条：方块右对齐（appendleft 只往左加，老方块原地不动）。返回方块 x 坐标表。"""
    lc.rect(x, DY, w, DH, '#ffffff', lc.C_ZMQ_S, rx=6, sw=1.4)
    xs = {}
    n = len(names)
    x0 = x + w - 12 - n * SQ_W - (n - 1) * SQ_GAP
    for i, nm in enumerate(names):
        sx = x0 + i * (SQ_W + SQ_GAP)
        xs[nm] = sx
        lc.rect(sx, DY + 10, SQ_W, SQ_H, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.2)
        lc.text(sx + SQ_W / 2, DY + 28, nm, 9, lc.C_TXT, 'middle', True, maxw=SQ_W - 4,
                tag='md:' + nm)
    return xs


lc.rect(MX, DY, 140, DH, '#ffffff', lc.C_MUTE, rx=6, sw=1.2)
lc.text(MX + 70, DY + 21, 'FutureWrapper 构造', 8.5, lc.C_TXT, 'middle', True, maxw=128,
        tag='anat:lead')
lc.text(MX + 70, DY + 38, '（构造即入队）', 8, lc.C_MUTE, 'middle', maxw=128, tag='anat:lead2')
DQ = [(240, 170, ['f1']), (450, 190, ['f2', 'f1']), (680, 250, ['f3', 'f2', 'f1'])]
prev_r = MX + 140
for x, w, names in DQ:
    lc.seg(prev_r + 2, DY + DH / 2, x - 3, DY + DH / 2, lc.C_ZMQ_S, 1.6, 'std')
    lc.text((prev_r + x) / 2, DY + DH / 2 - 8, 'appendleft ' + names[0], 8, lc.C_ZMQ_S,
            'middle', maxw=110, tag='anat:a' + names[0])
    mini_deque(x, w, names)
    prev_r = x + w
lc.seg(prev_r + 2, DY + DH / 2, prev_r + 72, DY + DH / 2, lc.C_ENG_S, 1.8, 'up')
lc.text(prev_r + 37, DY + DH / 2 - 8, 'pop 右出 = 最旧先出', 8, lc.C_ENG_S, 'middle',
        maxw=130, tag='anat:pop')
NOTE_X = prev_r + 84
lc.text(NOTE_X, DY + 20, 'FutureWrapper.__init__ 末尾 futures_queue.appendleft(self)——发出序 = 入队序', 9.5,
        lc.C_TXT, 'start', maxw=BXR - NOTE_X, tag='anat:n1')
lc.text(NOTE_X, DY + 38, 'result(): while not self.done(): pop → _wait_for_response（替先发者收尸）', 9.5,
        lc.C_MUTE, 'start', maxw=BXR - NOTE_X, tag='anat:n2')

# ---------------- 主表：五轮状态 ----------------
COLS = [
    (MX, 150, '轮', 'trace 五轮'),
    (214, 290, '发起（在飞 RPC）', 'non_block=True'),
    (508, 440, 'futures_queue 快照', '左=最新 · 右=最老'),
    (952, 488, '收割动作 → 排空 → 配对结果', 'FIFO 配对不变式'),
]
HDR_Y, HDR_H = 204, 36
for x, w, t, s in COLS:
    lc.rect(x, HDR_Y, w - 4, HDR_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=4, sw=1.2)
    lc.text(x + (w - 4) / 2, HDR_Y + 16, t, 9.5, lc.C_BEAT_T, 'middle', True, maxw=w - 12,
            tag='th:' + t[:8])
    lc.text(x + (w - 4) / 2, HDR_Y + 30, s, 7.5, lc.C_MUTE, 'middle', maxw=w - 12,
            tag='ts:' + s[:8])

BAR_X, BAR_W = 530, 340
BS_W, BS_H, BS_GAP = 70, 24, 4


def row_deque(y, names, highlight, bubbles):
    """行内 deque 快照：右对齐方块 + 回复气泡（绿）；highlight 名字者加粗描边=正在收割。"""
    lc.rect(BAR_X, y + 14, BAR_W, 36, '#ffffff', lc.C_ZMQ_S, rx=6, sw=1.4)
    n = len(names)
    x0 = BAR_X + BAR_W - 10 - n * BS_W - (n - 1) * BS_GAP
    for i, nm in enumerate(names):
        sx = x0 + i * (BS_W + BS_GAP)
        hot = (nm == highlight)
        lc.rect(sx, y + 20, BS_W, BS_H, lc.C_ENG_F if bubbles else '#ffffff', lc.C_ENG_S,
                rx=4, sw=2.4 if hot else 1.2)
        lc.text(sx + BS_W / 2, y + 36, nm, 8.5, lc.C_TXT, 'middle', True, maxw=BS_W - 6,
                tag='bd:' + nm + str(y))
        if bubbles:
            lc.text(sx + BS_W / 2, y + 64, bubbles[i], 7.5, lc.C_GPU_S, 'middle',
                    maxw=BS_W + 14, tag='bb:' + nm + str(y))


ROWS = [
    dict(h=104, name='A · 纯队列', sub='真实 FutureWrapper+deque',
         issue=['f1 → f2 → f3 依次构造', '（脚本化应答记录排空序）'],
         deque_names=['f3', 'f2', 'f1'], highlight='f3',
         bubbles=['resp-3', 'resp-2', 'resp-1'],
         act=['收最新的 f3.result()（最新者先收）',
              '排干 drain_order=[1,2,3]——f3 替 f1、f2 收尸',
              'resp-1 / resp-2 / resp-3 各归各 · 三个 future 全 done']),
    dict(h=76, name='A · 两条边界', sub='异常 / 超时',
         issue=['异常应答 f_exc', '带 timeout 调 result()'],
         deque_names=[], highlight=None, bubbles=[],
         act=['异常：set_exception 转出——RuntimeError: mq says no',
              '超时：result(timeout) 直接 raise "timeout not implemented"',
              '（两代实现都没实现——正文照实说）']),
    dict(h=104, name='B · round1', sub='背靠背两跳',
         issue=['② execute_model(total=3)', '④ sample_tokens(None)——non_block 连发'],
         deque_names=['samp', 'exec'], highlight='samp',
         bubbles=['{scheduled=3}', 'None'],
         act=['收 fut_samp.result()——exec 先被排空（得 None、done）',
              'sample 得 scheduled=3 · grammar=False',
              'q_len 1→2 → 收割后归 0']),
    dict(h=104, name='B · round2', sub='带 grammar',
         issue=['② execute_model(total=7)', '④ sample_tokens(grammar 允许集)'],
         deque_names=['samp', 'exec'], highlight='samp',
         bubbles=['{scheduled=7}', 'None'],
         act=['engine 序：先收最旧——exec 得 None 才触发 ④',
              'sample 得 scheduled=7 · grammar=True',
              '配对证据：7 不串到 round1 的 3——第 k 次 dequeue=第 k 个 RPC']),
    dict(h=118, name='C · 两对在飞', sub='异步稳态',
         issue=['exec(5) → sample → exec(9) → sample', '连发 4 个 non_block · 0.05ms · 全 pending'],
         deque_names=['s4', 'e4', 's3', 'e3'], highlight='s4',
         bubbles=['{9}', 'None', '{5}', 'None'],
         act=['只收最新的 fut_s4.result()——一次排空 4 个',
              'None → {5} → None → {9} 各归各 · 4 个全 done',
              '队列清空 q_len 4→0 · 排空共 0.18ms']),
]
ry = 244
for r in ROWS:
    h = r['h']
    lc.rect(MX, ry, BXR - MX, h, '#ffffff', lc.C_MUTE, rx=6, sw=1.1)
    lc.text(MX + 14, ry + 24, r['name'], 10, lc.C_TXT, 'start', True, maxw=130,
            tag='rl:' + r['name'])
    lc.text(MX + 14, ry + 42, r['sub'], 8, lc.C_MUTE, 'start', maxw=130, tag='rls:' + r['name'])
    for i, ln in enumerate(r['issue']):
        lc.text(228, ry + 26 + i * 17, ln, 8.5, '#334155', 'start', maxw=266,
                tag='ri:' + ln[:8])
    if r['deque_names']:
        row_deque(ry, r['deque_names'], r['highlight'], r['bubbles'])
    else:  # 边界行：两条出路口 chip
        lc.rect(530, ry + 14, 180, 34, '#ffffff', lc.C_ABORT, rx=5, sw=1.2)
        lc.text(620, ry + 35, 'f_exc → set_exception', 8.5, lc.C_ABORT, 'middle', True,
                maxw=168, tag='bd:exc')
        lc.rect(730, ry + 14, 190, 34, '#ffffff', lc.C_ABORT, rx=5, sw=1.2)
        lc.text(825, ry + 35, 'result(timeout=…) → raise', 8.5, lc.C_ABORT, 'middle', True,
                maxw=178, tag='bd:to')
    for i, ln in enumerate(r['act']):
        lc.text(966, ry + 26 + i * 17, ln, 8.5, '#334155', 'start', maxw=464,
                tag='ra:' + ln[:8])
    ry += h + 4

# ---------------- 底部：round3 耗时对比 + 配对不变式 ----------------
TY = ry + 18
lc.text(MX, TY + 12, 'round3 的耗时对比——连发近乎免费，收割才付等待：', 9.5, lc.C_TXT, 'start',
        True, maxw=520, tag='clk:t')
lc.rect(MX, TY + 20, 80, 22, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=4, sw=1.3)
lc.text(MX + 40, TY + 35, '0.05ms', 8.5, lc.C_TXT, 'middle', True, maxw=72, tag='clk:b1')
lc.seg(MX + 80 + 2, TY + 31, MX + 176 - 2, TY + 31, lc.C_MUTE, 1.5, 'std')
lc.rect(MX + 176, TY + 20, 288, 22, lc.C_GPU_F, lc.C_GPU_S, rx=4, sw=1.3)
lc.text(MX + 176 + 144, TY + 35, '0.18ms', 8.5, lc.C_TXT, 'middle', True, maxw=72, tag='clk:b2')
lc.text(MX + 490, TY + 35, 'q_len 4 → 0（全程只调了一次 result()）', 9.5, lc.C_TXT, 'start',
        True, maxw=420, tag='clk:q')
lc.text(MX, TY + 62, '左：4 个 non_block 连发（enqueue + 包 Future，不等任何应答）→ 右：一次 result() 排空 4 个——'
        '后发者替先发者收尸，延迟被摊平', 9, lc.C_MUTE, 'start', maxw=1380, tag='clk:n')

IV_Y = TY + 84
lc.rect(MX, IV_Y, BXR - MX, 70, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(MX + 14, IV_Y + 18, '三条 FIFO 同序 ⇒ 配对不变式：第 k 次 dequeue = 第 k 个 RPC 的回复——全程无需 id / 独立通道 / 应答序号',
        10, lc.C_TXT, 'start', True, maxw=1380, tag='iv:t')
lc.text(MX + 14, IV_Y + 36, '广播 MQ 单写多读（worker 按发出序处理）· 每 worker 应答 MQ（回复序=处理序）· futures_queue（构造序=发出序）',
        8.5, '#334155', 'start', maxw=1380, tag='iv:l1')
lc.text(MX + 14, IV_Y + 52, '在飞上界 = 2 对（max_concurrent_batches=2 · config/vllm.py:L539-L548 注释 "Async scheduling requires '
        '2 concurrent batches to overlap"）——round3 即此稳态；背靠背连发形态 = step_with_batch_queue（core.py:L655-L673）',
        8.5, '#334155', 'start', maxw=1380, tag='iv:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = IV_Y + 94
lx = MX
lc.rect(lx, LEG_Y - 9, 20, 13, '#ffffff', lc.C_ZMQ_S, rx=4, sw=1.4)
lc.text(lx + 26, LEG_Y + 2, 'futures_queue（MQ 家族）', 9, lc.C_TXT, 'start', maxw=200,
        tag='leg1')
lx += 26 + lc.tw('futures_queue（MQ 家族）', 9) + 22
lc.rect(lx, LEG_Y - 9, 20, 13, lc.C_ENG_F, lc.C_ENG_S, rx=4, sw=1.4)
lc.text(lx + 26, LEG_Y + 2, 'future 方块（排空者染色）', 9, lc.C_TXT, 'start', maxw=200,
        tag='leg2')
lx += 26 + lc.tw('future 方块（排空者染色）', 9) + 22
lc.text(lx, LEG_Y + 2, '绿色小字 = 回复值', 9, lc.C_GPU_S, 'start', True, maxw=140, tag='leg3')
lx += lc.tw('绿色小字 = 回复值', 9, True) + 22
lc.rect(lx, LEG_Y - 9, 20, 13, '#ffffff', lc.C_ENG_S, rx=4, sw=2.4)
lc.text(lx + 26, LEG_Y + 2, '粗描边 = 正在收割的 future', 9, lc.C_TXT, 'start', maxw=200,
        tag='leg4')
lc.text(MX, LEG_Y + 26, 'verbatim vllm/v1/executor/multiproc_executor.py:L70-L100（FutureWrapper）· '
        'vllm/config/vllm.py:L539-L548 · vllm/v1/engine/core.py:L655-L673',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '实测取自精简版 companion host 实测（A=真实 FutureWrapper+deque · B/C=真执行器 e2e · '
        'ZMQ loopback seam——毫秒只取量级感）· 行号基线 vLLM v0.27.1',
        9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch17-fig-futurewrapper-fifo.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
