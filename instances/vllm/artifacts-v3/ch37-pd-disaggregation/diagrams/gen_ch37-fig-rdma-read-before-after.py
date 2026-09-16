#!/usr/bin/env python3
"""ch37 机制图 m6 · rdma-read-before-after（figure_spec ch37-fig-rdma-read-before-after，模板 before-after）

放大自 L0「双实例+KV 边界」的 KV 边界本体（中排⑥ READ 直读 P 显存）、L2 章图站 8。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签；块格=小方格、KV 青系，
与 ch13/ch16 的 KV 块视觉语言同源。

claim：单边 READ 的前后对照：发起前 D 的落地块全零（校验和 0.0）、P 的块装着 KV
（7.0 填充，块校验和 896.0）；一条 make_prepped_xfer('READ') 把 P 显存里 4 块 × 512 B
原样搬进 D——P 的 CPU 零参与，完成后 D 靠一条 'req-1:1' notif 让 P 放块。

数字全部取自 figure_spec.numbers（traces 实测 + pin 源码锚点，逐字核对）：
  · 每块 512 B（128 个 float32，HND 张量 (8,2,4,16)）× 4 块 = 2048 B
  · 前：P 块校验和 896.0（7.0 填充）、D 同号块 0.0
  · 后：D 四块 torch.equal 全 True、校验和 896.0；handle 1 个（非阻塞）
  · P 侧完成信号 = 对端 notif 'req-1:1'（格式 req:tp_size，携消费者数）；P 本地 handle 0 个
  · 全命中对照：handle=0、只发 notif；部分命中对照：远端裁尾 [[2,3]]（拉 2 跳 2）
  · READ 本体：make_prepped_xfer('READ')+transfer 非阻塞、handle 入 _recving_transfers
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1600
MX = 54
BXR = 1546

N_BLOCKS = 8
FULL = [0, 1, 2, 3]
C_EMPTY, C_EMPTY_T = '#e2e8f0', '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 36, '一条单边 READ：D 按描述符裸地址从 P 显存整块抱走，店员不用动', 16.5,
        lc.C_TXT, 'start', True, maxw=1250, tag='title')
lc.text(MX, 60, '每块 512 B（128 个 float32，HND 张量 (8,2,4,16)）× 4 块 = 2048 B——READ 非阻塞、handle 1 个入 _recving_transfers 轮询；'
                'P 的引擎线程零参与（本地 handle 0 个）',
        10.5, lc.C_MUTE, 'start', maxw=1430, tag='subtitle')
_ch = '放大自 L2 站 8（⑥ READ 直读 P 显存）· L0：KV 边界本体'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 双面板 ----------------
PNL_Y, PNL_H = 96, 360
BF_X, AF_X, PNL_W = MX, 900, 646
MID_X0, MID_X1 = BF_X + PNL_W + 16, AF_X - 16


def block_strip(px, py, title, sub, filled, cell_vals, chip):
    """一组块格：title 行 + 8 个方格（filled=着色块号列表）+ 块号轴 + 校验和 chip。"""
    lc.text(px, py, title, 10.5, lc.C_TXT, 'start', True, maxw=380, tag=title[:8])
    lc.text(px + PNL_W, py, sub, 8.5, lc.C_MUTE, 'end', maxw=250, tag=sub[:8])
    cw_, ch_, gap = 64, 44, 6
    x0 = px
    for i in range(N_BLOCKS):
        cx = x0 + i * (cw_ + gap)
        if i in filled:
            lc.rect(cx, py + 12, cw_, ch_, lc.C_KV_S, 'none', rx=4, sw=0)
            lc.text(cx + cw_ / 2, py + 12 + 18, cell_vals[i], 9.5, '#ffffff', 'middle', True,
                    maxw=cw_ - 6, tag='cell%d' % i)
            lc.text(cx + cw_ / 2, py + 12 + 34, '块 %d' % i, 7.5, '#cffafe', 'middle',
                    maxw=cw_ - 6, tag='celln%d' % i)
        else:
            lc.rect(cx, py + 12, cw_, ch_, C_EMPTY, 'none', rx=4, sw=0)
            lc.text(cx + cw_ / 2, py + 12 + 18, cell_vals[i], 9.5, C_EMPTY_T, 'middle',
                    maxw=cw_ - 6, tag='cellg%d' % i)
            lc.text(cx + cw_ / 2, py + 12 + 34, '块 %d' % i, 7.5, C_EMPTY_T, 'middle',
                    maxw=cw_ - 6, tag='cellgn%d' % i)
    # 校验和 chip（条右下）
    lc.text(px + PNL_W, py + 12 + ch_ + 16, chip, 9, lc.C_KV_S, 'end', True, maxw=300,
            tag='chip:' + chip[:8])


# ---- 面板：发起前 ----
lc.rect(BF_X, PNL_Y, PNL_W, PNL_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(BF_X + 16, PNL_Y + 24, '发起前', 12, lc.C_TXT, 'start', True, maxw=120, tag='bf:t')
lc.text(BF_X + PNL_W - 16, PNL_Y + 24, '首遇远端：握手在飞、READ 未发（handle=0）', 8.5,
        lc.C_MUTE, 'end', maxw=420, tag='bf:s')
block_strip(BF_X + 16, PNL_Y + 56, 'P 的 KV 张量（8 块）', '块 [0,1,2,3] 填 7.0',
            FULL, {i: ('7.0' if i in FULL else '空') for i in range(N_BLOCKS)},
            'P 块校验和 896.0')
block_strip(BF_X + 16, PNL_Y + 170, 'D 的落地缓冲（未入哈希表的新块）', '同号块清零',
            [], {i: '0.0' for i in range(N_BLOCKS)},
            'D 校验和 0.0')
lc.text(BF_X + PNL_W / 2, PNL_Y + PNL_H - 14, 'D 的落地缓冲是未入哈希表的新块——全零等货', 8.5,
        lc.C_MUTE, 'middle', maxw=PNL_W - 32, tag='bf:note')

# ---- 面板：READ 完成后 ----
lc.rect(AF_X, PNL_Y, PNL_W, PNL_H, '#ffffff', lc.C_KV_S, rx=9, sw=1.6)
lc.text(AF_X + 16, PNL_Y + 24, 'READ 完成后', 12, lc.C_KV_S, 'start', True, maxw=140, tag='af:t')
lc.text(AF_X + PNL_W - 16, PNL_Y + 24, 'handle=1 · D 四块 torch.equal 全 True', 8.5,
        lc.C_KV_S, 'end', maxw=420, tag='af:s')
block_strip(AF_X + 16, PNL_Y + 56, 'P 的 KV 张量（8 块）', '块还在货架上（等回条才腾）',
            FULL, {i: ('7.0' if i in FULL else '空') for i in range(N_BLOCKS)},
            'P 块校验和 896.0')
block_strip(AF_X + 16, PNL_Y + 170, 'D 的落地缓冲', '字节逐块相等',
            FULL, {i: ('7.0' if i in FULL else '空') for i in range(N_BLOCKS)},
            'D 校验和 896.0')
# 回条：D → P 的 notif（沿反方向走的小信封）
nx = AF_X + PNL_W - 30
lc.seg(nx, PNL_Y + 228, nx, PNL_Y + 116, lc.C_MUTE, 1.5, 'std', dash=True)
lc.text(nx - 10, PNL_Y + 252, "notif 'req-1:1'", 8.5, lc.C_MUTE, 'end', tag='notif:1')
lc.text(nx - 10, PNL_Y + 266, '（格式 req:tp_size · 携消费者数）', 8, lc.C_MUTE, 'end',
        tag='notif:2')
lc.text(nx - 10, PNL_Y + 280, '→ P done_sending={req-1} · 放块', 8, lc.C_MUTE, 'end',
        tag='notif:3')
lc.text(AF_X + PNL_W / 2, PNL_Y + PNL_H - 14, 'D 轮询本地 handle 即知收完；P 没参与传输、本地 handle 0 个——只能等回条（不对称）',
        8.5, lc.C_MUTE, 'middle', maxw=PNL_W - 32, tag='af:note')

# ---- 中缝：READ 粗箭头 ----
mid_y = PNL_Y + 150
lc.seg(MID_X0 + 4, mid_y, MID_X1 - 6, mid_y, lc.C_KV_S, 7.0, 'kv')
lc.text((MID_X0 + MID_X1) / 2, mid_y - 58, 'READ', 15, lc.C_KV_S, 'middle', True, tag='read:t')
read_lines = [
    "make_prepped_xfer('READ')",
    '+ transfer',
    '· 非阻塞',
    '· 按描述符裸地址取数',
    '  （握手期换的货位表）',
    '· 4 块 × 512 B = 2048 B',
]
for j, ln in enumerate(read_lines):
    lc.text((MID_X0 + MID_X1) / 2, mid_y - 36 + j * 15, ln, 8.5, lc.C_TXT, 'middle',
            maxw=MID_X1 - MID_X0 - 8, tag='read:l%d' % j)
# P 的 CPU 零参与：店员打盹（一处比喻即止）
zz_x, zz_y = (MID_X0 + MID_X1) / 2, mid_y + 52
lc.circle(zz_x, zz_y, 16, lc.C_MUTE, 1.4, dash=True)
lc.text(zz_x, zz_y + 3, 'zzz', 9, lc.C_MUTE, 'middle', True, tag='zz:t')
lc.text(zz_x, zz_y + 34, 'P 的引擎线程', 8.5, lc.C_MUTE, 'middle', tag='zz:l1')
lc.text(zz_x, zz_y + 48, '零参与', 8.5, lc.C_MUTE, 'middle', True, tag='zz:l2')

# ---------------- 底部：三个对照小结 ----------------
SM_Y = PNL_Y + PNL_H + 20
SM_W = (BXR - MX - 2 * 16) / 3
summaries = [
    ('对照 · 本地全命中', lc.C_KV_S, [
        '· 落地块表为空 → 不发 READ',
        '· handle=0 · 0 次传输',
        '· 只发一条 notif——P 收到即放块',
        '· 全命中免拉：notif 本身就是收据',
    ], 'pull_worker.py:L266-L285'),
    ('对照 · 部分本地前缀命中', lc.C_KV_S, [
        '· 本地命中 [[0,1]]（如共享 system prompt）',
        '· 远端裁到尾段 [[2,3]]',
        '· 实拉 2 块、跳过 2 块（传输量对半）',
        '· 只拉未命中的尾段',
    ], '_apply_prefix_caching（实测裁尾）'),
    ('完成信号 · 两源不对称', lc.C_ENG_S, [
        "· D：轮询本地 handle 全 DONE",
        '  → done_recving={req-1}',
        "· P：翻对端 notif 收件箱（'req-1:1'）",
        '  → done_sending={req-1} · 放块+删请求',
    ], 'pull_worker.py:L346-L399'),
]
for i, (t, color, lines, foot) in enumerate(summaries):
    x = MX + i * (SM_W + 16)
    lc.rect(x, SM_Y, SM_W, 122, '#ffffff', color, rx=8, sw=1.4)
    lc.text(x + 14, SM_Y + 20, t, 10, color, 'start', True, maxw=SM_W - 28, tag='sm:t%d' % i)
    for j, ln in enumerate(lines):
        lc.text(x + 14, SM_Y + 40 + j * 16, ln, 8.5, '#334155', 'start', maxw=SM_W - 26,
                tag='sm:l%d_%d' % (i, j))
    lc.text(x + 14, SM_Y + 122 - 10, foot, 8, lc.C_FAINT, 'start', maxw=SM_W - 26,
            tag='sm:f%d' % i)

# ---------------- 页脚 ----------------
FY = SM_Y + 142
lc.text(MX, FY, '逐字锚 nixl/pull_worker.py:L225-L344（READ 本体：make_prepped_xfer+transfer · handle 入 _recving_transfers）· '
                'L266-L285（全命中 notif-only）· L346-L399（notif 收件箱→放块）· 几何与块值同源 traces（8 块池、块 [0,1,2,3]、7.0/896.0）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 15, '行号基线 vLLM v0.27.1 · 本机取证：数据面为进程内替身（单边语义保留——只按描述符裸地址取数；真 RDMA 异步，顺序约束与完成判定路径不变）· 块格=KV 青，与 ch13/ch16 同源',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 34

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS,
       '<defs><marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" markerHeight="4.6" orient="auto">'
       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker></defs>']
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch37-fig-rdma-read-before-after.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
