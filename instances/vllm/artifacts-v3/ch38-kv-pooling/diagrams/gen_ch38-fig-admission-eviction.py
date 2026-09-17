#!/usr/bin/env python3
"""ch38 机制图 m4 · admission-eviction（figure_spec ch38-fig-admission-eviction，模板 flow）

放大自 L2 站 5（④ 池准入与驱逐·prepare_store）· L0：外部池内部的「门房」。

claim：prepare_store 的五道闸门按序全过才落位：频次过滤（counts ≥ threshold）→ 已存去重 →
需逐数 vs 可逐数（不足即 None 拒收）→ evict(n, protected) 只逐空闲且保护本次输入 →
落位 ref_cnt=-1 写入中，complete_store 才翻 0 可读可逐。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · ref_cnt 状态机：-1 写入中（HIT_PENDING）→ complete_store 翻 0（HIT）；prepare_load +1 钉住、complete_load 归 0
  · LRU+touch：池 3 存 k0/k1/k2、touch(k0) 后存 k9 → 逐 k1 留 k0
  · 驱逐失败：需逐 1 > 可逐 0 → None（无副作用拒收）
  · store_threshold=2：查 1 次不收、查 2 次才收（counts=2）
  · 策略可插拔：lru/arc 内置 + out-of-tree cache_policy_module_path
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1460
MX = 60
BXR = W - MX

# ---------------- 标题区 ----------------
lc.text(MX, 36, '池的准入像一间满员旅馆：五道闸门全过才落位，任一关失败零副作用', 16.5,
        lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 60, '新客进门先问『有没有空房』，没有就按 LRU 赶最久没人住的——正在被读取的房间（ref_cnt>0）与本次名单上的客人赶不得；赶不够就直接拒单（None，不硬塞）',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L2 站 5（④ 池准入与驱逐）· L0：外部池的门房'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左列：五道闸门主链（竖排） ----------------
FL_X, FL_W = MX, 640
GATE_W = 470
GATE_X = FL_X + 40
gates = [
    ('①', '频次过滤', True, [
        'counts[key] ≥ store_threshold（默认 1）',
        'threshold=2 时：查 1 次不收、查 2 次才收',
        '频次不够 → 本批不收该键（零副作用）',
    ]),
    ('②', '已存去重（输入保护）', False, [
        '输入里已存的键直接留下、不算新客',
        '源码注释原话：已存输入块必须留下',
    ]),
    ('③', '需逐数 vs 可逐数', True, [
        '需逐 = 新键数 − 空闲数 · 可逐 = 空闲可逐块',
        '（扣除保护）——实测：池 2 全被钉住时',
        '需逐 1 > 可逐 0',
    ]),
    ('④', 'evict(n, protected)', False, [
        'LRU 按鲜度逐最旧空闲块（k1 最旧→先走）',
        '保护牌：ref_cnt>0 钉住 · 本次输入集',
        'policy.evict 原子：足额列表 或 None 不动状态',
    ]),
    ('⑤', '落位 _allocate_blocks', False, [
        '驱逐完成后一次性执行（每新键恰占 1 块）',
        '新块 ref_cnt = -1『写入中』：查得到、读不得',
    ]),
]
GY0, GH, GGAP = 122, 74, 40
for i, (num, name, is_dec, lines) in enumerate(gates):
    y = GY0 + i * (GH + GGAP)
    color = lc.C_KV_S if is_dec else lc.C_ENG_S
    fill = lc.C_KV_F if is_dec else lc.C_ENG_F
    lc.rect(GATE_X, y, GATE_W, GH, fill, color, rx=8, sw=1.5)
    # 判定闸门加菱形小标（矩形动作无）
    if is_dec:
        d = 11
        s = (f'<path d="M{GATE_X + 20:.1f},{y + GH / 2:.1f} L{GATE_X + 20 + d:.1f},{y + GH / 2 - d:.1f} '
             f'L{GATE_X + 20 + 2 * d:.1f},{y + GH / 2:.1f} L{GATE_X + 20 + d:.1f},{y + GH / 2 + d:.1f} Z" '
             f'fill="#ffffff" stroke="{lc.C_KV_S}" stroke-width="1.6"/>')
        lc.ELEMS.append(((GATE_X + 18, y + GH / 2 - d, GATE_X + 42, y + GH / 2 + d), s))
        tx = GATE_X + 50
    else:
        tx = GATE_X + 16
    lc.text(tx, y + 20, num + ' ' + name, 10, color, 'start', True,
            maxw=GATE_W - (tx - GATE_X) - 12, tag='g%d:t' % i)
    for j, ln in enumerate(lines):
        lc.text(tx, y + 38 + j * 15, ln, 8.8, '#334155', 'start',
                maxw=GATE_W - (tx - GATE_X) - 10, tag='g%d:l%d' % (i, j))
    # 主链下行箭头（闸底边 → 下一闸顶边）
    if i < len(gates) - 1:
        ay = y + GH
        lc.seg(GATE_X + GATE_W / 2, ay, GATE_X + GATE_W / 2, ay + GGAP, lc.C_MUTE, 1.8, 'std')
        lc.text(GATE_X + GATE_W / 2 + 8, ay + GGAP / 2 + 3, '过', 8.5, lc.C_MUTE, 'start',
                tag='pass%d' % i)
    globals()['gate_y%d' % i] = y

# None 拒收旁路箱（③④ 右侧汇入；与右列各箱错开 x 不重叠）
NB_X, NB_W = GATE_X + GATE_W + 90, 170
NB_Y = gate_y2 + 6
NB_H = (gate_y3 + GH) - NB_Y + 4
lc.rect(NB_X, NB_Y, NB_W, NB_H, '#fef2f2', lc.C_ABORT, rx=8, sw=1.5)
lc.text(NB_X + 12, NB_Y + 20, '返回 None——拒收', 10, lc.C_ABORT, 'start', True,
        maxw=NB_W - 24, tag='none:t')
lc.text(NB_X + 12, NB_Y + 38, '· 无任何副作用：不动任何块', 8.8, '#334155', 'start',
        maxw=NB_W - 22, tag='none:l1')
lc.text(NB_X + 12, NB_Y + 53, '· 实测 evict(10) 超容返', 8.8, '#334155', 'start',
        maxw=NB_W - 22, tag='none:l2')
lc.text(NB_X + 12, NB_Y + 68, '  None 后 get 仍在', 8.8, '#334155', 'start',
        maxw=NB_W - 22, tag='none:l3')
lc.text(NB_X + 12, NB_Y + 83, '· best-effort：放弃本批，', 8.8, '#334155', 'start',
        maxw=NB_W - 22, tag='none:l4')
lc.text(NB_X + 12, NB_Y + 98, '  不重试、只记指标', 8.8, '#334155', 'start',
        maxw=NB_W - 22, tag='none:l5')
# ③ → None / ④ → None 旁路箭头（闸右边 → 箱左边）
lc.seg(GATE_X + GATE_W, gate_y2 + GH / 2, NB_X, gate_y2 + GH / 2, lc.C_ABORT, 1.5, 'std')
lc.text((GATE_X + GATE_W + NB_X) / 2, gate_y2 + GH / 2 - 7, '需逐 > 可逐', 8.5, lc.C_ABORT,
        'middle', maxw=NB_X - GATE_X - GATE_W - 4, tag='byp3')
lc.seg(GATE_X + GATE_W, gate_y3 + GH / 2, NB_X, gate_y3 + GH / 2, lc.C_ABORT, 1.5, 'std')
lc.text((GATE_X + GATE_W + NB_X) / 2, gate_y3 + GH / 2 - 7, 'evict 失败', 8.5, lc.C_ABORT,
        'middle', maxw=NB_X - GATE_X - GATE_W - 4, tag='byp4')

# ---------------- 右列上：ref_cnt 状态机 ----------------
SM_X, SM_W = 860, BXR - 860
SM_Y, SM_H = 122, 108
lc.rect(SM_X, SM_Y, SM_W, SM_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(SM_X + 12, SM_Y + 20, 'ref_cnt 三态：一块的池内生命周期', 10, lc.C_TXT, 'start', True,
        maxw=SM_W - 24, tag='sm:t')
st_w, st_h = (SM_W - 24 - 2 * 78) / 3, 34
st_y = SM_Y + 40
states = [('-1 写入中', 'HIT_PENDING', lc.C_ENG_S), ('0 可读可逐', 'HIT', lc.C_KV_S),
          ('1 被读·钉住', 'prepare_load 期间', lc.C_GPU_S)]
st_x = []
for i, (name, sub, color) in enumerate(states):
    x = SM_X + 12 + i * (st_w + 78)
    st_x.append(x)
    lc.rect(x, st_y, st_w, st_h, '#ffffff', color, rx=7, sw=1.5)
    lc.text(x + st_w / 2, st_y + 14, name, 9, color, 'middle', True, maxw=st_w - 6,
            tag='st%d' % i)
    lc.text(x + st_w / 2, st_y + 27, sub, 7.5, lc.C_MUTE, 'middle', maxw=st_w - 6,
            tag='sts%d' % i)
# 状态间转移箭头（端点贴状态框边）
lc.seg(st_x[0] + st_w, st_y + st_h / 2, st_x[1], st_y + st_h / 2, lc.C_MUTE, 1.6, 'std')
lc.text((st_x[0] + st_w + st_x[1]) / 2, st_y + st_h / 2 - 6, 'complete_store', 7.8, lc.C_MUTE,
        'middle', maxw=76, tag='tr1')
lc.seg(st_x[1] + st_w, st_y + st_h / 2, st_x[2], st_y + st_h / 2, lc.C_GPU_S, 1.6, 'std')
lc.text((st_x[1] + st_w + st_x[2]) / 2, st_y + st_h / 2 - 6, 'prepare_load +1', 7.5,
        lc.C_GPU_S, 'middle', maxw=76, tag='tr2')
lc.seg(st_x[2], st_y + st_h / 2 + 8, st_x[1] + st_w, st_y + st_h / 2 + 8, lc.C_GPU_S, 1.4,
       'std')
lc.text((st_x[1] + st_w + st_x[2]) / 2, st_y + st_h / 2 + 20, 'complete_load→0', 7.5,
        lc.C_GPU_S, 'middle', maxw=76, tag='tr3')
lc.text(SM_X + SM_W / 2, SM_Y + SM_H - 8, '『写入中→可读可逐』的翻转点 = complete_store 结算', 8,
        lc.C_MUTE, 'middle', maxw=SM_W - 20, tag='sm:note')

# ---------------- 右列中：LRU+touch 实测 ----------------
LR_Y = SM_Y + SM_H + 18
LR_H = 118
lc.rect(SM_X, LR_Y, SM_W, LR_H, '#ffffff', lc.C_KV_S, rx=8, sw=1.3)
lc.text(SM_X + 12, LR_Y + 20, 'LRU + touch 实测（池 3 块）', 10, lc.C_KV_S, 'start', True,
        maxw=SM_W - 24, tag='lr:t')
# 三个已存键 + 新键
ky = LR_Y + 34
for i, (k, fate, color, dash) in enumerate([('k0', 'touch 刷新→存活 HIT', lc.C_GPU_S, False),
                                            ('k1', 'LRU 最旧→被逐 MISS', lc.C_ABORT, True),
                                            ('k2', '', lc.C_MUTE, False),
                                            ('k9', '新客进场', lc.C_ENG_S, False)]):
    x = SM_X + 12 + i * ((SM_W - 24 - 3 * 10) / 4 + 10)
    kw = (SM_W - 24 - 3 * 10) / 4
    lc.rect(x, ky, kw, 26, '#ffffff' if not dash else '#fef2f2', color, rx=6,
            sw=1.3, dash=dash)
    lc.text(x + kw / 2, ky + 17, k, 9.5, color, 'middle', True, maxw=kw - 4, tag='k%d' % i)
    if fate:
        lc.text(x + kw / 2, ky + 42, fate, 7.8, color if color != lc.C_MUTE else lc.C_MUTE,
                'middle', maxw=kw + 6, tag='kf%d' % i)
lc.text(SM_X + 12, LR_Y + LR_H - 22, '存 k0/k1/k2 后 touch(k0) 再存 k9 → 需逐 1：', 8.5,
        '#334155', 'start', maxw=SM_W - 24, tag='lr:l1')
lc.text(SM_X + 12, LR_Y + LR_H - 8, '逐 k1、净占用不变；事件 BlockRemoved(keys=[1])', 8.5,
        '#334155', 'start', maxw=SM_W - 24, tag='lr:l2')

# ---------------- 右列下：钉/解钉 + 策略 ----------------
PP_Y = LR_Y + LR_H + 18
PP_H = 122
lc.rect(SM_X, PP_Y, SM_W, PP_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(SM_X + 12, PP_Y + 20, '钉 / 解钉 的两侧 + 策略可插拔', 10, lc.C_TXT, 'start', True,
        maxw=SM_W - 24, tag='pp:t')
pp_lines = [
    '· load k0 期间存 k9 → 逐 k1（k0 被钉免逐）',
    '· complete_load 解钉后存 k10 → 逐 k0',
    '  （while_pinned_evicted=[1]、after=[0]）',
    '· 策略可插拔：lru / arc 内置，',
    '  out-of-tree 走 cache_policy_module_path',
]
for j, ln in enumerate(pp_lines):
    lc.text(SM_X + 12, PP_Y + 40 + j * 15.5, ln, 8.6, '#334155', 'start',
            maxw=SM_W - 22, tag='pp:l%d' % j)

# ⑤ 落位 → 状态机 的虚线（落位动作标 -1）
lc.parrow([(GATE_X + GATE_W, gate_y4 + GH / 2), (SM_X - 8, gate_y4 + GH / 2),
           (SM_X - 8, SM_Y + SM_H / 2), (SM_X, SM_Y + SM_H / 2)],
          lc.C_ENG_S, 1.3, 'std', dash=True)
lc.text((GATE_X + GATE_W + SM_X) / 2, gate_y4 + GH / 2 - 8, '落位即 -1（见状态机）', 8.5,
        lc.C_ENG_S, 'middle', maxw=SM_X - GATE_X - GATE_W, tag='link45')

# ---------------- 结论 + 页脚 ----------------
BOT = max(gate_y4 + GH, PP_Y + PP_H) + 24
lc.text(MX, BOT,
        '图注结论：全过五关才落位、任一关失败零副作用——要么整体成功（每新键恰占 1 块、驱逐数恰等于需逐数），要么返回 None 且池状态原样；计数守恒 free + allocated ≤ num_blocks。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = BOT + 22
lc.text(MX, FT_Y, '逐字锚 vllm/v1/kv_offload/cpu/manager.py:L166-L236（prepare_store 五闸 + evict 原子性）· cpu/spec.py:L94-L97（store_threshold）· cpu/policies/factory.py:L12-L40（策略可插拔）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · 菱形小标 = 判定闸（可走旁路）；矩形 = 动作闸；状态机三态取自 LookupResult 与 ref_cnt 实测',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-admission-eviction.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
