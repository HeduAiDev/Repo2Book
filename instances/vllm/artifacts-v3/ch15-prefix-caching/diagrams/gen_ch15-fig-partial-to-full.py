#!/usr/bin/env python3
"""ch15 机制图 · 写回时序：partial 条目同拍就贴上，满块晋升等跨块边界那一拍（figure_spec
ch15-fig-partial-to-full，模板 flow·一次调用的两条登记路径 + 四拍时间轴带）

放大自 L0 KV 账本列缓存区·命中主循环——「写回」一拍内部的时序放大（与
ch15-fig-writeback-mask 相邻成对：那张讲登记什么、这张讲何时登记）。

claim：partial 条目在写回的同一拍贴上（短路挡不住它），满块晋升发生在 token 跨过块边界
的那一拍（闸自动翻转放行）——两条登记路径共享一次 cache_blocks 调用、先后依序。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑：full(64)+mamba(64,align)、
hash 粒度 16、prompt 48 + 16 个生成 token）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印符号免疫

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440

BLUE, BLUE_F = '#2563eb', '#eff6ff'      # 满块登记路 / 主哈希
ORNG, ORNG_F = '#ea580c', '#fff7ed'      # partial 尾登记路 / 别名
VIOL, VIOL_F = '#7c3aed', '#f5f3ff'      # mamba 组条目
GREEN, GREEN_F = '#16a34a', '#f0fdf4'   # 闸翻转放行（第 16 拍）
GRAY = '#94a3b8'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '写回时序：partial 条目同拍就贴上（短路挡不住），满块晋升等跨块边界的那一拍', 16.5,
        lc.C_TXT, 'start', True, maxw=1150, tag='title')
lc.text(MX, 58, '48-token prompt 落在 64-token 块内部（hash 粒度 16）：两条登记路径共享一次 cache_blocks 调用、'
                '先后依序；账本只被满块登记推进——跨边界那拍 num_full_blocks 涨而账本没涨，下一拍闸必放行',
        10.5, lc.C_MUTE, 'start', maxw=1340, tag='subtitle')
_ch = 'L0 放大 · KV 账本列缓存区 · 「写回」一拍内部'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶部流程带：一次 cache_blocks 调用、两条登记路径 ----------------
FY0, FBH = 88, 70


def flow_box(x, w, title, lines, stroke, fill, anchor=''):
    lc.rect(x, FY0, w, FBH, fill, stroke, rx=7, sw=1.4)
    lc.text(x + 12, FY0 + 18, title, 9.8, lc.C_TXT, 'start', True, maxw=w - 24, tag='fb:' + title[:8])
    for i, ln in enumerate(lines):
        lc.text(x + 12, FY0 + 36 + i * 14, ln, 8.2, '#334155', 'start', maxw=w - 22,
                tag='fb:%d' % i)
    if anchor:
        lc.text(x + 12, FY0 + FBH - 5, anchor, 7.2, lc.C_FAINT, 'start', maxw=w - 22,
                tag='fb:an')


FW_A, FW_B, FW_C1, FW_C2, FGAP = 150, 270, 300, 336, 37
XA_ = MX
XB_ = XA_ + FW_A + FGAP
XC1 = XB_ + FW_B + FGAP
XC2 = XC1 + FW_C1 + FGAP
lc.rect(XA_, FY0 + 14, FW_A, 42, '#ffffff', GRAY, rx=7, sw=1.2, dash=True)
lc.text(XA_ + FW_A / 2, FY0 + 31, 'allocate_slots 尾', 9.2, lc.C_MUTE, 'middle', True,
        maxw=FW_A - 12, tag='fa')
lc.text(XA_ + FW_A / 2, FY0 + 47, '每拍写回入口', 8, lc.C_MUTE, 'middle', maxw=FW_A - 12,
        tag='fa:sub')
flow_box(XB_, FW_B, 'FullAttentionManager.cache_blocks（覆写）',
         ['两条登记路径都从这里走'], BLUE, BLUE_F, 'single_type:L779-L789')
flow_box(XC1, FW_C1, '① super().cache_blocks',
         ['幂等闸 num_cached >= num_full → 短路', '放行：满块区间登记 · 推进账本'],
         BLUE, BLUE_F, '幂等闸 L448')
flow_box(XC2, FW_C2, '② _cache_partial_tail_block',
         ['boundary = prompt 尾 //16*16 → 注册块内条目', '不推进账本——短路管不着它'],
         ORNG, ORNG_F, 'single_type:L791-L819')
for x0, x1, lab in [(XA_ + FW_A, XB_, ''), (XB_ + FW_B, XC1, ''), (XC1 + FW_C1, XC2, '随后依序')]:
    lc.seg(x0 + 2, FY0 + FBH / 2, x1 - 2, FY0 + FBH / 2, lc.C_MUTE, 1.8, 'std')
    if lab:
        lc.text((x0 + x1) / 2, FY0 + FBH / 2 - 8, lab, 8.4, lc.C_MUTE, 'middle', tag='fl:' + lab)
lc.text(MX, FY0 + FBH + 18, '一次调用两条路：短路只挡 ① 的满块区间，② 在 super() 之后照跑——partial 条目'
        '同拍就上墙（prefill 拍 map 即 +1）', 9, lc.C_MUTE, 'start', maxw=900, tag='flow:cap')

# ---------------- 四拍时间轴带 ----------------
TY = 208
lc.text(MX, TY, '四拍时间轴 · block_size=64 · hash 粒度 16 · prompt 48 token', 9.5, lc.C_TXT,
        'start', True, maxw=420, tag='tl:t')

CARD_Y, CARD_W, CARD_H, CARD_G = TY + 14, 325, 226, 18
COLS = [
    dict(hdr='prefill', sub='num_tokens=48', hl=False,
         rows=[('请求侧哈希', '3 枚（@16/@32/@48，构造即算）', lc.C_TXT),
               ('num_full_blocks', '0', lc.C_TXT),
               ('幂等闸', '0 >= 0 → 短路（满块零登记）', GRAY),
               ('满块登记路', '—（无满块可登记）', GRAY),
               ('partial 尾路', '@48 注册（boundary=48//16*16）', ORNG),
               ('map 条目', '1（块 1 @48）', lc.C_TXT)]),
    dict(hdr='decode 拍 2', sub='num_tokens=50', hl=False,
         rows=[('请求侧哈希', '3（未跨 16 边界，零成本）', lc.C_TXT),
               ('num_full_blocks', '0', lc.C_TXT),
               ('幂等闸', '0 >= 0 → 短路（幂等）', GRAY),
               ('满块登记路', '幂等 · 零登记', GRAY),
               ('partial 尾路', 'already_cached → 幂等', GRAY),
               ('map 条目', '1', lc.C_TXT)]),
    dict(hdr='decode 拍 15', sub='num_tokens=63', hl=False,
         rows=[('请求侧哈希', '3', lc.C_TXT),
               ('num_full_blocks', '0', lc.C_TXT),
               ('幂等闸', '0 >= 0 → 短路（幂等）', GRAY),
               ('满块登记路', '幂等 · 零登记', GRAY),
               ('partial 尾路', 'already_cached → 幂等', GRAY),
               ('map 条目', '1', lc.C_TXT)]),
    dict(hdr='decode 拍 16', sub='num_tokens=64 · 第 16 个生成 token', hl=True,
         rows=[('请求侧哈希', '4——append 补第 4 枚', BLUE),
               ('num_full_blocks', '1（台阶跳变）', BLUE),
               ('幂等闸', '0 >= 1？否——翻转放行', GREEN),
               ('满块登记路', '晋升：摘 @48 插 @64 · 主哈希 48→64', BLUE),
               ('partial 尾路', '别名补登 @48（prompt 边界恒 48）', ORNG),
               ('map 条目', '3（主 @64 + 别名 @48 + mamba @64）', lc.C_TXT)]),
]
for i, c in enumerate(COLS):
    cx = MX + i * (CARD_W + CARD_G)
    stroke, fill = (GREEN, GREEN_F) if c['hl'] else (lc.C_MUTE, '#ffffff')
    lc.rect(cx, CARD_Y, CARD_W, CARD_H, fill, stroke, rx=8, sw=1.6 if c['hl'] else 1.2)
    lc.rect(cx, CARD_Y, CARD_W, 30, GREEN_F if c['hl'] else '#f1f5f9', stroke, rx=8, sw=0)
    lc.text(cx + 12, CARD_Y + 20, c['hdr'], 10, lc.C_TXT, 'start', True, maxw=120, tag='cd:h')
    lc.text(cx + CARD_W - 12, CARD_Y + 20, c['sub'], 8.6, lc.C_MUTE, 'end',
            maxw=CARD_W - 140, tag='cd:s')
    for j, (lab, val, col) in enumerate(c['rows']):
        ry = CARD_Y + 44 + j * 30
        lc.text(cx + 12, ry, lab, 7.6, lc.C_MUTE, 'start', maxw=78, tag='cd:l%d' % j)
        lc.text(cx + 96, ry, val, 8.4, col, 'start', bold=(c['hl'] and j >= 2),
                maxw=CARD_W - 108, tag='cd:v%d' % j)
    if i < 3:
        ax0, ax1 = cx + CARD_W, cx + CARD_W + CARD_G
        dash = (i == 1)
        lc.seg(ax0 + 2, CARD_Y + 15, ax1 - 2, CARD_Y + 15, lc.C_MUTE, 1.6, 'std', dash=dash)
# 拍 2-15 省略括注（骑在卡 2-3 头顶）
BX0 = MX + (CARD_W + CARD_G)
BX1 = MX + 2 * (CARD_W + CARD_G) + CARD_W
lc.seg(BX0, CARD_Y - 8, BX1, CARD_Y - 8, GRAY, 1.1, dash=True)
lc.text((BX0 + BX1) / 2, CARD_Y - 13, '拍 2-15 之间共 14 拍省略——双路幂等、零新增', 8, GRAY,
        'middle', tag='tl:brace')

# ---------------- 底部左：晋升解剖（map 1→3，一块挂两条目） ----------------
BY = CARD_Y + CARD_H + 18
LP_W = 660
lc.rect(MX, BY, LP_W, 158, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(MX + 16, BY + 22, '第 16 拍的账：map 1→3——一块挂主哈希 + 别名两条目', 11.5, lc.C_TXT,
        'start', True, maxw=LP_W - 32, tag='lp:t')
_blk_x, _blk_y = MX + 430, BY + 40
lc.rect(_blk_x, _blk_y, 190, 56, BLUE_F, BLUE, rx=7, sw=1.5)
lc.text(_blk_x + 95, _blk_y + 22, '块 1', 11, lc.C_TXT, 'middle', True, maxw=180, tag='blk')
lc.text(_blk_x + 95, _blk_y + 40, 'num_tokens=64 · alias_keys=1', 8, '#334155', 'middle',
        maxw=180, tag='blk:s')
for k, (chip, note, col, cfill) in enumerate([
        ('主 @64', '覆盖边界 48→64（晋升换主）', BLUE, BLUE_F),
        ('别名 @48', 'prompt 边界恒 48 · 反向索引记账', ORNG, ORNG_F)]):
    ky = BY + 40 + k * 52
    lc.rect(MX + 24, ky, 108, 28, cfill, col, rx=6, sw=1.3)
    lc.text(MX + 78, ky + 18, chip, 9.5, col, 'middle', True, maxw=100, tag='kchip:%d' % k)
    lc.text(MX + 24, ky + 40, note, 7.6, '#334155', 'start', maxw=240, tag='knote:%d' % k)
    lc.seg(MX + 132, ky + 14, _blk_x - 2, BY + (58 if k == 0 else 78), col, 1.4, 'std')
lc.rect(MX + 24, BY + 132, 150, 20, VIOL_F, VIOL, rx=6, sw=1.3)
lc.text(MX + 99, BY + 146, 'mamba 组 @64', 8.6, VIOL, 'middle', True, maxw=140, tag='mchip')
lc.text(MX + 184, BY + 146, '（mamba 管家自己的边界条目，非别名）', 7.6, '#334155', 'start',
        maxw=280, tag='mnote')
lc.text(MX + 420, BY + 116, '摘 1 插 1（晋升）+ 补登 1 + mamba 1', 8.4, lc.C_TXT,
        'start', True, maxw=230, tag='lp:sum1')
lc.text(MX + 420, BY + 132, '→ map 1→3', 8.4, lc.C_TXT, 'start', True, maxw=120, tag='lp:sum2')

# ---------------- 底部右：两问直答 ----------------
RX = MX + LP_W + 28
RW = BXR - RX
lc.rect(RX, BY, RW, 158, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(RX + 16, BY + 22, '两个常见疑问，这张图直答', 11.5, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='rp:t')
lc.text(RX + 16, BY + 44, '问：哈希什么时候算？', 9.4, BLUE, 'start', True, maxw=RW - 32,
        tag='rp:q1')
for i, ln in enumerate(['· 构造时 3 枚（@16/@32/@48）+ append 第 64 token 时补第 4 枚——长在请求身上',
                        '· cache_full_blocks 从不算哈希，只消费 request.block_hashes']):
    lc.text(RX + 28, BY + 61 + i * 15, ln, 8.4, '#334155', 'start', maxw=RW - 44, tag='rp:a1%d' % i)
lc.text(RX + 16, BY + 96, '问：partial 什么时候写？', 9.4, ORNG, 'start', True, maxw=RW - 32,
        tag='rp:q2')
for i, ln in enumerate(['· 与满块写回同一次 cache_blocks 调用、在 super() 之后——短路挡不住它',
                        '· 「短路」只挡满块区间；跨块边界那拍闸自动翻转、晋升是常态不是特例']):
    lc.text(RX + 28, BY + 113 + i * 15, ln, 8.4, '#334155', 'start', maxw=RW - 44, tag='rp:a2%d' % i)
lc.text(RX + 16, BY + 148, '（15 个 decode 拍双路幂等零新增——账不动、map 不动）', 8, lc.C_MUTE,
        'start', maxw=RW - 32, tag='rp:note')

# ---------------- 不变量条 ----------------
IY = BY + 158 + 16
lc.rect(MX, IY, BXR - MX, 56, lc.C_KV_F, lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, IY + 22, '不变量：幂等闸只管满块区间；num_cached_blocks 账本只被满块登记推进，'
        'num_full_blocks 是 tokens//block_size 的台阶函数', 9.6, lc.C_KV_S, 'start', True,
        maxw=BXR - MX - 32, tag='inv:1')
lc.text(MX + 16, IY + 42, '每个块边界处差值转正、下一拍闸必放行——「短路」与「partial→full 晋升」不矛盾：'
        '前者是满块区间的幂等，后者是每次写满块的前奏', 9, '#334155', 'start',
        maxw=BXR - MX - 32, tag='inv:2')

# ---------------- 图例 ----------------
LY = IY + 76
lx = MX
for stroke, fill, name, dash in [(BLUE, BLUE_F, '满块登记路 / 主哈希', False),
                                 (ORNG, ORNG_F, 'partial 尾登记路 / 别名', False),
                                 (VIOL, VIOL_F, 'mamba 组条目', False),
                                 (GREEN, GREEN_F, '闸翻转放行（第 16 拍）', False)]:
    lc.rect(lx, LY - 9, 20, 13, fill, stroke, rx=3, sw=1.2)
    lc.text(lx + 26, LY + 1, name, 8.8, lc.C_TXT, 'start', maxw=190, tag='lg:' + name[:6])
    lx += 26 + lc.tw(name, 8.8) + 18
lc.rect(lx, LY - 9, 20, 13, '#ffffff', GRAY, rx=3, sw=1.2, dash=True)
lc.text(lx + 26, LY + 1, '短路 / 幂等（无登记）', 8.8, lc.C_TXT, 'start', maxw=170, tag='lg:sc')
lx += 26 + lc.tw('短路 / 幂等（无登记）', 8.8) + 18
lc.text(lx, LY + 1, '时间轴 = 拍序（prefill → 16 个 decode 拍）', 8.8, lc.C_MUTE, 'start',
        maxw=BXR - lx, tag='lg:axis')

# ---------------- 页脚 ----------------
FY = LY + 26
lc.text(MX, FY, '逐字锚 vllm/v1/core/single_type_kv_cache_manager.py:L779-L789（覆写：super() 之后 _cache_partial_tail_block）'
        ' · L448（幂等闸） · L791-L819（partial 尾）', 8.2, lc.C_FAINT, 'start', maxw=BXR - MX,
        tag='foot1')
lc.text(MX, FY + 16, 'vllm/v1/core/block_pool.py:L284（晋升：唯一合法的「新满块已有哈希」场景） · L495-L499（already_cached 判等）'
        ' · L240-L241（docstring：cache_full_blocks 只消费 request.block_hashes）', 8.2, lc.C_FAINT,
        'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, FY + 32, '数字取自配套精简版 host 实跑（full(64)+mamba(64, align) · hash 粒度 16 · prompt 48 token'
        ' + 16 个生成 token）· 行号基线 vLLM v0.27.1', 8.2, lc.C_FAINT, 'start', maxw=BXR - MX,
        tag='foot3')

# ---------------- 装配输出 ----------------
H = FY + 52
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch15-fig-partial-to-full.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
