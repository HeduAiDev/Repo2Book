#!/usr/bin/env python3
"""ch21 机制图 ② · 平台优先级表(figure_spec ch21-fig-m04-priorities,模板 state-table)

放大自 L0 GPU 执行臂(绿)·装配期裁决(站 3):『装配期决定每层跑哪个 kernel』的那张裁决表。
架构归属回指 L0(FIGURE-SYSTEM §3.3),不另立第二种架构画法。

claim:同一层到底跑哪个 kernel 由平台优先级表决定:use_mla × 算力代(major 10=Blackwell/12/
其他)× fp8 KV × num_heads 分叉出候选名单,表序即默认最优——名单是经验参数(注释挂 benchmark),
换硬件代要重测。

数字全部取自 figure_spec.numbers:非 MLA+SM100 五候选、非 MLA+SM90 FLASH_ATTN 提到表头、
MLA+SM100 七候选(含 TOKENSPEED_MLA 注释 wins past bs≈8/regresses at bs≤2)、MLA+其他代
(Hopper 等)六家(major=12 两家不上图,记档于 figure_spec.numbers[4])——pin 源码
vllm/platforms/cuda.py:L83-L163 逐字对拍。坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 792
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '裁决表:同一层跑哪个 kernel,先查平台优先级表——表头是「一切合法时」的默认最优',
        16.5, lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, 'use_mla × 算力代 × fp8 KV × num_heads 分叉出候选名单;名单序 = 「能订到时优先吃哪家」,但名单 ≠ 胜者——逐个 validate 之后才定',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂(绿)·装配期裁决(站 3)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 函数签名条 ----------------
FS_Y, FS_H = 88, 40
lc.rect(MX, FS_Y, BXR - MX, FS_H, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.4)
lc.text(MX + 16, FS_Y + 25, '_get_backend_priorities(use_mla, device_capability, num_heads, kv_cache_dtype, use_non_causal)  →  候选名单(按优先级排序)',
        10, lc.C_GPU_S, 'start', True, maxw=1120, tag='fs:call')
lc.text(BXR - 14, FS_Y + 25, 'vllm/platforms/cuda.py:L83-L163', 9, lc.C_FAINT, 'end',
        maxw=220, tag='fs:file')

# ---------------- 表头 ----------------
TB_Y = FS_Y + FS_H + 18
COND_X, COND_W = MX, 268
PILL_X0 = COND_X + COND_W + 20
lc.text(COND_X + 4, TB_Y, '分叉条件(use_mla × major)', 10, lc.C_TXT, 'start', True,
        maxw=COND_W, tag='th:c')
lc.text(PILL_X0, TB_Y, '候选名单(① = 表头 = 默认最优;序号 = 名单位次)', 10, lc.C_TXT,
        'start', True, maxw=800, tag='th:p')

RANKS = '①②③④⑤⑥⑦'

rows = [
    dict(cond=('非 MLA · major=10', '(Blackwell)· causal'),
         cands=['FLASHINFER', 'FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION', 'TURBOQUANT'],
         swap=0, extra=0, note=None),
    dict(cond=('非 MLA · 其他代', '(如 SM90)· 或 non-causal'),
         cands=['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION', 'TURBOQUANT'],
         swap=1, extra=0, note=None),
    dict(cond=('MLA · major=10', '(Blackwell)'),
         cands=['FLASHINFER_MLA', 'TOKENSPEED_MLA', 'CUTLASS_MLA', 'FLASH_ATTN_MLA',
                'FLASHMLA', 'TRITON_MLA', '稀疏尾(*sparse)'],
         swap=-1, extra=30, note='TOKENSPEED_MLA'),
    dict(cond=('MLA · 其他代', '(Hopper 等)'),
         cands=['FLASH_ATTN_MLA', 'FLASHMLA', 'FLASHINFER_MLA', 'TRITON_MLA',
                'FLASH_ATTN_MLA_SPARSE', 'FLASHMLA_SPARSE'],
         swap=-1, extra=0, note=None),
]

ROW_Y = TB_Y + 16
PILL_H = 34
PILL_FS = 9.5
for r in rows:
    rh = 64 + r['extra']
    lc.rect(COND_X, ROW_Y, COND_W, rh, '#ffffff', lc.C_MUTE, rx=7, sw=1.2)
    lc.text(COND_X + 12, ROW_Y + 26, r['cond'][0], 10, lc.C_TXT, 'start', True,
            maxw=COND_W - 24, tag='c0')
    if r['cond'][1]:
        lc.text(COND_X + 12, ROW_Y + 43, r['cond'][1], 8.5, lc.C_MUTE, 'start',
                maxw=COND_W - 24, tag='c1')
    px = PILL_X0
    py = ROW_Y + 8
    for k, name in enumerate(r['cands']):
        head = (k == 0)
        tail = (name.startswith('稀疏尾'))
        nw = lc.tw(name, PILL_FS, True)
        bw = 26                                     # ① 徽标宽
        pw = bw + nw + 18
        if tail:
            lc.rect(px, py, pw, PILL_H, '#ffffff', lc.C_FAINT, rx=16, sw=1.2, dash=True)
            bc, nc = lc.C_FAINT, lc.C_MUTE
        elif head:
            lc.rect(px, py, pw, PILL_H, lc.C_GPU_F, lc.C_GPU_S, rx=16, sw=1.8)
            bc, nc = lc.C_GPU_S, lc.C_GPU_S
        else:
            lc.rect(px, py, pw, PILL_H, '#ffffff', lc.C_MUTE, rx=16, sw=1.2)
            bc, nc = lc.C_MUTE, '#334155'
        swapped = (r['swap'] == 0 and name == 'FLASHINFER') or \
                  (r['swap'] == 1 and name == 'FLASH_ATTN')
        if swapped:
            lc.rect(px - 3, py - 3, pw + 6, PILL_H + 6, 'none', lc.C_BEAT_S, rx=18,
                    sw=1.6, dash=True)
        lc.text(px + 13, py + 22, RANKS[k], 11, bc, 'middle', True, tag='rank')
        lc.text(px + bw + 8, py + 22, name, PILL_FS, nc, 'start', True, maxw=nw + 8,
                tag='pill')
        px += pw + 10
    if r['note']:
        # TOKENSPEED_MLA 经验注释:虚线小注挂在该行下方(左缘对齐候选列,避开条件框)
        nk = r['cands'].index(r['note'])
        nx = PILL_X0 + sum(lc.tw(n, PILL_FS, True) + 26 + 18 + 10 for n in r['cands'][:nk])
        nw_ = lc.tw(r['note'], PILL_FS, True)
        ncx = nx + (26 + nw_ + 18) / 2
        ny = ROW_Y + 64
        nb_w, nb_h = 620, 22
        nb_x = PILL_X0                     # 锚在候选列左缘:绝不与条件框(x≤328)重叠
        lc.seg(ncx, ROW_Y + 8 + PILL_H, ncx, ny + 6, lc.C_BEAT_S, 1.2, None, dash=True)
        lc.rect(nb_x, ny + 6, nb_w, nb_h, lc.C_BEAT_F, lc.C_BEAT_S, rx=6,
                sw=1.1, dash=True)
        lc.text(nb_x + nb_w / 2, ny + 21, '注释自述:R1 dims + FP8 KV only;wins past bs≈8, regresses at bs≤2 —— batch size 域优劣相反,表序只能取折中',
                8, lc.C_BEAT_T, 'middle', maxw=nb_w - 10, tag='ts:note')
    ROW_Y += rh + 10

# ---------------- 底部两块 ----------------
BB_Y = ROW_Y + 6
BB_W = (BXR - MX - 20) / 2
BB_H = 108
# 左:名单只是名单
lc.rect(MX, BB_Y, BB_W, BB_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 14, BB_Y + 20, '名单只是名单——胜者另定(站 4-5,正文下一节六轮实测)', 10, lc.C_TXT,
        'start', True, maxw=BB_W - 28, tag='bb1:t')
for k, ln in enumerate([
        '· 逐个 validate:没装的(ImportError)当一条落选原因,不挡后面的候选',
        '· 幸存者取 min(priority) 定胜者(位次互异,唯一)',
        '· 显式点名不合适 → 直接 ValueError:不回退、不静默换后端']):
    lc.text(MX + 14, BB_Y + 40 + k * 16, ln, 8.6, '#334155', 'start', maxw=BB_W - 26,
            tag=f'bb1:l{k}')
# 右:表序是经验的
bx = MX + BB_W + 20
lc.rect(bx, BB_Y, BB_W, BB_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.3)
lc.text(bx + 14, BB_Y + 20, '表序是经验的,换硬件代要重测', 10, lc.C_BEAT_T, 'start', True,
        maxw=BB_W - 200, tag='bb2:t')
for k, ln in enumerate([
        '· 表序由 benchmark 定:源码注释挂 issue #35807 的测速结果',
        '· non-causal 提前:SM100 FlashInfer 的 non-causal cutlass 路径已知有问题,故 FLASH_ATTN 提到表头(cuda.py:L144-L163)',
        '· 稀疏尾两家(*sparse_backends)永远补在主链尾——不占表头;',
        '   对内序:fp8 与 BF16·heads≤16 → FLASHINFER_MLA_SPARSE 先;BF16·heads>16 对调']):
    lc.text(bx + 14, BB_Y + 40 + k * 16, ln, 8.6, lc.C_BEAT_T, 'start', maxw=BB_W - 26,
            tag=f'bb2:l{k}')

# ---------------- 页脚 ----------------
FY = BB_Y + BB_H + 22
lc.text(MX, FY, '图例:绿 = 表头(一切合法时的默认最优) · 灰 = 其余候选(按位次) · 灰虚线 = 收拢的稀疏尾 · 橙虚线框 = 相互交换头两位的一对(FLASH_ATTN ⇄ FLASHINFER) · 橙块 = 经验性警告',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, FY + 18, 'vllm/platforms/cuda.py:L83-L163(四条分叉与名单)· 表序经验性:L96-L97 注释挂 benchmark issue · 四行名单与本章精简版 host 实测逐字对拍 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch21-fig-m04-priorities.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
