#!/usr/bin/env python3
"""ch38 机制图 m10 · flush-fence（figure_spec ch38-fig-flush-fence，模板 before-after）

放大自 L2 站 10（⑧ 完成回收·flush 围栏）· L0：外部池与 GPU 池之间「块复用与异步搬运竞态」那道闸。
before-after 左右对照共用块 1 的位置锚（视觉上是同一块显存）。

claim：flush 围栏三触发（GPU 块将被复用/请求被抢占/终局前 store 未完）→ jobs_to_flush →
worker handle_preemptions 先抢先提交延迟队列里的 store、再 wait() 同步——GPU 改写块之前
搬运必已落地。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · 终局例：request_finished 返回 (False, None)（不接管）；jobs_to_flush={0}、盯防账本 {1,2}
  · 复用例：新请求分到块 1 → jobs_to_flush={0}；围栏序列 submitted [] → [(0,store)] → wait({0})、延迟队列清空
  · 抢占例：preempted_req_ids → jobs_to_flush={0}
  · 围栏代码：先提交后 wait（wait=事件 synchronize）
  · 对照：ch16 wait_for_save 每拍强制等（P/D）vs 本章被复用才等（best-effort）
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1520
MX = 60
BXR = W - MX

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'flush 围栏：异步搬运与块复用是两列相向的火车——只在真要相撞时拉闸', 16.5,
        lc.C_TXT, 'start', True, maxw=1080, tag='title')
lc.text(MX, 60, '调度器不停车：三种信号出现时把延迟队列里的 store 抢先发车，然后站在站台等它到站（wait 同步），才放 GPU 改写这块显存',
        10.5, lc.C_MUTE, 'start', maxw=1100, tag='subtitle')
_ch = '放大自 L2 站 10（⑧ 完成回收·flush 围栏）· L0：块复用×异步搬运竞态闸'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')


def gpu_block(cx, y, w, h, label, sub):
    lc.rect(cx - w / 2, y, w, h, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.8)
    lc.text(cx, y + 22, label, 11, lc.C_TXT, 'middle', True, maxw=w - 8, tag='blk:' + label)
    lc.text(cx, y + 40, sub, 8.5, lc.C_MUTE, 'middle', maxw=w - 6, tag='blk:s' + label)


# ---------------- 左面板：前 · 两列相向 ----------------
PW = (BXR - MX - 28) / 2
PX = MX
PY, PH = 104, 384
lc.rect(PX, PY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(PX + 14, PY + 20, '前 · 两列相向（本步还没人等谁）', 10.5, lc.C_TXT, 'start', True,
        maxw=PW - 28, tag='bf:t')
# 三枚触发信号旗
trig = [
    ('块复用', '新分配块 ∩ 盯防账本'),
    ('抢占', 'preempted_req_ids 的在飞 job'),
    ('终局', '请求结束而 store 未完'),
]
FW, FH, FGAP = (PW - 28 - 2 * 12) / 3, 52, 12
for i, (name, sub) in enumerate(trig):
    fx = PX + 14 + i * (FW + FGAP)
    lc.rect(fx, PY + 34, FW, FH, lc.C_ENG_F, lc.C_ENG_S, rx=6, sw=1.3)
    lc.text(fx + FW / 2, PY + 34 + 18, '触发 · ' + name, 9.5, lc.C_ENG_S, 'middle', True,
            maxw=FW - 6, tag='trig%d' % i)
    lc.text(fx + FW / 2, PY + 34 + 36, sub, 7.6, lc.C_MUTE, 'middle', maxw=FW - 4,
            tag='trigs%d' % i)
# 闸门
GATE_Y = PY + 108
GW, GH = 210, 40
gcx = PX + PW / 2
lc.rect(gcx - GW / 2, GATE_Y, GW, GH, '#ffffff', lc.C_ABORT, rx=8, sw=1.6)
lc.text(gcx, GATE_Y + 16, '闸门 jobs_to_flush', 9.5, lc.C_ABORT, 'middle', True,
        maxw=GW - 8, tag='gate:t')
lc.text(gcx, GATE_Y + 32, '实测三种触发都得 {0}', 8, lc.C_ABORT, 'middle', maxw=GW - 8,
        tag='gate:s')
# 三旗 → 闸门 的虚线（端点贴旗底边/闸门顶边）
for i in range(3):
    fx = PX + 14 + i * (FW + FGAP) + FW / 2
    lc.seg(fx, PY + 34 + FH, gcx - GW / 2 + 18 + i * ((GW - 36) / 2), GATE_Y, lc.C_ENG_S,
           1.2, None, dash=True)
# 块 1（位置锚）与两侧相向箭头
BLK_Y = PY + 186
bcx = gcx
gpu_block(bcx, BLK_Y, 110, 52, 'GPU 块 1', '盯防账本 {1, 2}')
# 左：延迟队列 store job（虚框）→ 块1
SJ_X, SJ_Y, SJ_W, SJ_H = PX + 16, BLK_Y - 6, 150, 64
lc.rect(SJ_X, SJ_Y, SJ_W, SJ_H, '#ffffff', lc.C_ENG_S, rx=7, sw=1.3, dash=True)
lc.text(SJ_X + SJ_W / 2, SJ_Y + 18, '延迟队列', 9, lc.C_ENG_S, 'middle', True,
        maxw=SJ_W - 8, tag='sj:t')
lc.text(SJ_X + SJ_W / 2, SJ_Y + 34, 'store job 0', 8.5, '#334155', 'middle',
        maxw=SJ_W - 8, tag='sj:s')
lc.text(SJ_X + SJ_W / 2, SJ_Y + 50, '引用块 [1, 2]', 8, lc.C_MUTE, 'middle',
        maxw=SJ_W - 8, tag='sj:b')
lc.seg(SJ_X + SJ_W, SJ_Y + SJ_H / 2, bcx - 55, BLK_Y + 26, lc.C_ENG_S, 1.8, 'std', dash=True)
lc.text(470, BLK_Y - 8, '想搬出去（尚未提交）', 8, lc.C_ENG_S, 'middle', maxw=140,
        tag='bf:a1')
# 右：新请求分配 → 块1
NR_X = PX + PW - 16
lc.seg(NR_X, BLK_Y + 26, bcx + 55, BLK_Y + 26, lc.C_GPU_S, 1.8, 'std')
lc.text(600, BLK_Y + 4, '新请求分到块 1（将要改写）', 8, lc.C_GPU_S, 'middle', maxw=150,
        tag='bf:a2')
# 面板底注
lc.text(PX + 14, PY + PH - 42, '此刻：store 还在延迟队列（submitted=[]）、谁也不等谁——', 8.5,
        lc.C_MUTE, 'start', maxw=PW - 28, tag='bf:n1')
lc.text(PX + 14, PY + PH - 26, '块 1 同时被「未提交的搬运」与「新请求的改写」引用', 8.5,
        lc.C_MUTE, 'start', maxw=PW - 28, tag='bf:n2')

# ---------------- 右面板：后 · 拉闸 ----------------
QX = MX + PW + 28
lc.rect(QX, PY, PW, PH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(QX + 14, PY + 20, '后 · 拉闸：先发车、再站台上等它到站', 10.5, lc.C_TXT, 'start', True,
        maxw=PW - 28, tag='af:t')
# 步①
S1_Y = PY + 36
lc.rect(QX + 14, S1_Y, PW - 28, 64, lc.C_ENG_F, lc.C_ENG_S, rx=7, sw=1.3)
lc.text(QX + 26, S1_Y + 18, '① 抢先提交（handle_preemptions）', 9.5, lc.C_ENG_S, 'start',
        True, maxw=PW - 52, tag='af:s1t')
lc.text(QX + 26, S1_Y + 36, 'submitted: [] → [(0, store)]，延迟队列清空', 8.6, '#334155',
        'start', maxw=PW - 52, tag='af:s1l')
lc.text(QX + 26, S1_Y + 52, '（flush 的 store 弹出后才轮到本步任何会写这些块的前向）', 8,
        lc.C_MUTE, 'start', maxw=PW - 52, tag='af:s1m')
# 步②（时钟标记）
S2_Y = S1_Y + 78
lc.rect(QX + 14, S2_Y, PW - 28, 64, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.3)
ccx, ccy = QX + 40, S2_Y + 32
lc.ELEMS.append(((ccx - 11, ccy - 11, ccx + 11, ccy + 11),
                 f'<circle cx="{ccx}" cy="{ccy}" r="10" fill="#ffffff" stroke="{lc.C_KV_S}" '
                 f'stroke-width="1.6"/><line x1="{ccx}" y1="{ccy}" x2="{ccx}" y2="{ccy - 6}" '
                 f'stroke="{lc.C_KV_S}" stroke-width="1.6"/><line x1="{ccx}" y1="{ccy}" '
                 f'x2="{ccx + 5}" y2="{ccy + 3}" stroke="{lc.C_KV_S}" stroke-width="1.6"/>'))
lc.text(QX + 60, S2_Y + 18, '② wait({0})：事件 synchronize', 9.5, lc.C_KV_S, 'start', True,
        maxw=PW - 86, tag='af:s2t')
lc.text(QX + 60, S2_Y + 36, '返回即 DMA 已落地（GPU→CPU 字节到位）', 8.6, '#334155',
        'start', maxw=PW - 86, tag='af:s2l')
lc.text(QX + 60, S2_Y + 52, '等待上界 = 在飞 transfer 的剩余 DMA 时间', 8, lc.C_MUTE,
        'start', maxw=PW - 86, tag='af:s2m')
# 步①→② 箭头
lc.seg(QX + PW / 2, S1_Y + 64, QX + PW / 2, S2_Y, lc.C_MUTE, 1.6, 'std')
# 块 1（同一位置锚，与左面板同 y）+ 落下的写箭头
BLK2_Y = PY + 186 + 96
bcx2 = QX + PW / 2 - 40
gpu_block(bcx2, BLK2_Y, 110, 52, 'GPU 块 1', '同一块显存（位置锚）')
lc.seg(QX + PW / 2 + 110, BLK2_Y - 18, bcx2 + 55, BLK2_Y + 10, lc.C_GPU_S, 1.8, 'std')
lc.text(QX + PW / 2 + 96, BLK2_Y - 30, 'wait 返回后：新请求的写箭头才落到块 1', 8.2,
        lc.C_GPU_S, 'middle', maxw=PW - 130, tag='af:a2')
lc.text(QX + 14, PY + PH - 26, '围栏顺序 = 先 submit 再 wait——ch16 m7 handle_preemptions 挂点在池化世界的填实', 8.5,
        lc.C_MUTE, 'start', maxw=PW - 28, tag='af:n1')

# 面板间箭头（前 → 后）
lc.seg(PX + PW, GATE_Y + GH / 2, QX, GATE_Y + GH / 2, lc.C_ABORT, 2.2, 'std')
lc.text((PX + PW + QX) / 2, GATE_Y + GH / 2 - 8, '三触发', 8.5, lc.C_ABORT, 'middle',
        maxw=28, tag='mid')

# ---------------- 底条：不接管对照 + 与 ch16 的对照 ----------------
BT_Y = PY + PH + 22
lc.rect(MX, BT_Y, BXR - MX, 74, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 16, BT_Y + 20, '不接管对照：request_finished 返回 (False, None)——块释放不延迟，只把引用块记入盯防账本等 flush 检查', 9.5,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='bt:t')
lc.text(MX + 16, BT_Y + 40, '· 对照 ch37 P/D 分离章：那里 request_finished 返 True 接管块释放（可靠交接）；本章 best-effort——丢一次 save 只是未来一次 miss',
        8.6, '#334155', 'start', maxw=BXR - MX - 32, tag='bt:l1')
lc.text(MX + 16, BT_Y + 58, '· 对照 ch16 逐层契约：wait_for_save 每拍强制等（P/D 可靠语义）；本章「被复用才等」——best-effort 换吞吐（三钩子全空）',
        8.6, '#334155', 'start', maxw=BXR - MX - 32, tag='bt:l2')

# ---------------- 结论 + 页脚 ----------------
CONC_Y = BT_Y + 74 + 24
lc.text(MX, CONC_Y,
        '图注结论：围栏是 best-effort 世界里唯一的同步点——只在两列火车真的要相撞时拉闸，平时谁也不等谁；GPU 块被改写之前，引用它的 store 必已提交且搬运已完成。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 offloading/scheduler.py:L1280-L1358（三触发检查）· L1228-L1255（盯防账本）· offloading/worker.py:L292-L317（先提交后 wait）· offloading_connector.py:L103-L134（三钩子空实现）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · 实测三触发各得 jobs_to_flush={0}；围栏动作序列 submitted [] → [(0,store)] → wait({0})',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-flush-fence.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
