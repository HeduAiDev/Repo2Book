#!/usr/bin/env python3
"""ch14 机制图 6 · 一份账喂两侧的调用全景（figure-request ch14-fig-init-call-panorama，add）

放大自 L0 启动装配带 → 显存账本列与 GPU 列的双喂线——本章 L2 章图中排拍片④「一份账喂两侧」
的代码级放大：既有 ch14-fig-one-ledger-two-sides 讲「同一份账的两次投影」（数据投影视角），
本张讲「调用链上谁调谁、锚在哪几行」（调用视角），互补不重复。

claim：站 7 全局调用图——一份账喂两侧的调用关系总览：EngineCore._initialize_kv_caches 总编排，
上游吃站 3 的 available_gpu_memory（一行减法的产物）与站 4 的 kv_cache_specs（每层自报形状），
中游 get_kv_cache_configs 单点定账（护栏四道 / override 折算 / PP 取最小＝上一节，站 6），
下游分两支：generate_scheduler_kv_cache_config 拍平版喂调度器（写回 cache_config 四件套）、
model_executor.initialize_from_config 布局版喂 worker（CuMem 池真分配）。两侧 num_blocks 同 320。

numbers（逐字取自 figure-request，provenance 见 request 条目）：
  41943040（40MiB）→ 320（num_blocks）· block_size 16 · 容量 5120（=320×16）·
  并发 1.25（=320/256）· 两侧 num_blocks 同 320 ·
  core.py:L301-L330 · kv_cache_utils.py:L1855-L1874 · gpu_worker.py:L649-L676

坐标全部由常量/游标计算，零手写魔数；文本全走 lc.text（内部 esc()）。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # GBK 控制台打印符号免疫

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与骨架常量 ----------------
W, MX, BXR = 1500, 60, 1440
CW = BXR - MX                       # 1380
COL_GAP = 60                        # 左右双栏中缝
HALF_W = (CW - COL_GAP) / 2         # 660
LX, RX = MX, MX + HALF_W + COL_GAP  # 60 / 780
LCX, RCX = LX + HALF_W / 2, RX + HALF_W / 2   # 390 / 1110
MIDX = MX + CW / 2                  # 750

# 语义色（l0_common 常量，与 L0/L2 同源：引擎=橙 / 显存账本与调度列=青 / GPU 执行臂=绿）
C_EN, C_EN_F = lc.C_ENG_S, lc.C_ENG_F
C_KV, C_KV_F = lc.C_KV_S, lc.C_KV_F
C_GPU, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_GREY = lc.C_MUTE

EXTRA_DEFS = ('<defs>'
              f'<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_KV}"/></marker>'
              f'<marker id="gpu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
              f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_GPU}"/></marker>'
              '</defs>')

MK = {C_EN: 'up', C_KV: 'kv', C_GPU: 'gpu', C_GREY: 'std'}
BADGE_FS = 8.5                      # 站号徽标字号（与 L2 站号 tab 同款）


# ---------------- 文本换行（CJK 逐字、ASCII 按词；禁手写断行） ----------------
def _sep(a, b):
    if not a:
        return ''
    return '' if (ord(a[-1]) > 0x2E80 or ord(b[0]) > 0x2E80) else ' '


def wrap(s, fs, maxw, bold=False):
    if lc.tw(s, fs, bold) <= maxw:
        return [s]
    lines, cur = [], ''
    for word in s.split(' '):
        cand = cur + _sep(cur, word) + word
        if lc.tw(cand, fs, bold) <= maxw:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        while lc.tw(word, fs, bold) > maxw:            # 单词本身超宽 → 逐字切
            cut = len(word)
            while cut > 1 and lc.tw(word[:cut], fs, bold) > maxw:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        cur = word
    if cur:
        lines.append(cur)
    return lines


def rrow(t=None, fs=9.2, c=lc.C_TXT, bold=False, ind=0, lead=17, chips=None):
    """节点内一行（可自动换行）；chips = 该行改画一排数字小胶囊。"""
    return dict(t=t, fs=fs, c=c, bold=bold, ind=ind, lead=lead, chips=chips)


def chiprow(x, w, ymid, items, stroke, fill):
    """框内一排数字胶囊（宽度自算、整体居中）。"""
    fs = 8.6
    ws = [lc.tw(it, fs, True) + 16 for it in items]
    cx = x + (w - (sum(ws) + 8 * (len(items) - 1))) / 2
    for it, cw in zip(items, ws):
        lc.rect(cx, ymid - 11, cw, 22, fill, stroke, rx=10, sw=1.0)
        lc.text(cx + cw / 2, ymid + 4.6, it, fs, stroke, 'middle', True, maxw=cw - 6,
                tag='chip:' + it)
        cx += cw + 8


def nbox(x, y, w, title, tfile, rows, stroke, fill='#ffffff', badge=None,
         dash=False, tfs=11.5, pad=10):
    """调用链节点框：标题（粗）+ 源码路径（灰）+ 内容行（自动换行）。
    badge = 站号 tab（骑框顶，与 L2 站号徽标同款）。返回 dict(y=框底, chips=[胶囊行中线 y…])。"""
    inner = w - 28
    laid = []
    for r in rows:
        if r['chips']:
            laid.append((None, 0.0, None, False, 0, r['lead'], r['chips']))
            continue
        for ln in wrap(r['t'], r['fs'], inner - r['ind'], r['bold']):
            laid.append((ln, r['fs'], r['c'], r['bold'], r['ind'], r['lead'], None))
    h = (30 if tfile else 22) + sum(r[5] for r in laid) + pad + 4
    lc.rect(x, y, w, h, fill, stroke, rx=8, sw=1.5, dash=dash)
    tx = x + 14
    lc.text(tx, y + 21, title, tfs, lc.C_TXT, 'start', True, maxw=inner, tag='t:' + title[:14])
    if tfile:
        lc.text(tx, y + 37, tfile, 8.6, lc.C_FAINT, 'start', maxw=inner, tag='f:' + tfile[:16])
    cy = y + (30 if tfile else 22)
    chips_y = []
    for ln, fs, c, bold, ind, lead, chips in laid:
        cy += lead
        if chips:
            chips_y.append(cy - 5)
            continue
        lc.text(tx + ind, cy, ln, fs, c, 'start', bold, maxw=inner - ind, tag='b:' + ln[:18])
    if badge:
        bw = 14 + 9.0 * len(badge)
        lc.rect(x + w - bw - 6, y - 8, bw, 17, lc.C_BADGE_F, C_EN, rx=8, sw=1.0)
        lc.text(x + w - bw / 2 - 6, y + 3, badge, BADGE_FS, C_EN, 'middle', True,
                tag='bg:' + badge)
    return dict(y=y + h, chips=chips_y)


def vlabel(x, y, s, fs=8.8, c=None, maxw=None, tag='vl'):
    """竖线旁的调用表达式标注（箭头中段）。"""
    lc.text(x + 8, y, s, fs, c or lc.C_TXT, 'start', maxw=maxw or (BXR - x - 8),
            tag=tag + s[:14])


def merge(xs, y0s, x_in, y_in, color, sw=2.0, bus_gap=26):
    """汇聚：各源竖落 → 横向母线 → 中轴落到下一框顶。返回母线 y。"""
    bus = max(y0s) + bus_gap
    for x, y in zip(xs, y0s):
        lc.seg(x, y, x, bus, color, sw)
    lc.seg(min(xs + [x_in]), bus, max(xs + [x_in]), bus, color, sw)
    lc.seg(x_in, bus, x_in, y_in, color, sw, MK[color])
    return bus


def fork(x0, y0, targets, color, sw=2.0, bus_gap=26, drop_gap=26):
    """分岔：中轴下潜 → 横向母线 → 各目标竖落（落点带箭头）。返回 (母线 y, 落点终点 y)。"""
    bus = y0 + bus_gap
    lc.seg(x0, y0, x0, bus, color, sw)
    lc.seg(min(targets + [x0]), bus, max(targets + [x0]), bus, color, sw)
    for t in targets:
        lc.seg(t, bus, t, bus + drop_gap, color, sw, MK[color])
    return bus, bus + drop_gap


# ================= 标题区 =================
lc.text(MX, 34, '调用全景：一份账喂两侧——谁调谁，账从哪来、单点在哪、喂到哪',
        17, lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 58, '站 7 一图读完：站 3 的一行减法（41943040 B）与站 4 的每层自报（spec）汇进总编排 → '
                'get_kv_cache_configs 单点定账 320 块 → 拍平版喂调度器、布局版喂 worker，'
                '两侧 num_blocks 同 320',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='sub1')
lc.text(MX, 74, '读图自上而下；橙 = 引擎总编排（core.py）｜青 = 账本定账与调度器侧（kv_cache_utils.py）｜'
                '绿 = worker 侧（gpu_worker.py / gpu_model_runner.py）｜箭头旁是调用表达式',
        9.2, lc.C_MUTE, 'start', maxw=1330, tag='sub2')
_ch = '放大自 L0 · 启动装配带 · L2 拍片④'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

GAP = 46                            # 节点之间的箭头段高（放得下两行标注）
y = 100

# ================= 上游两支（站 3 / 站 4 · 绿 = worker 侧） =================
up_l = nbox(LX, y, HALF_W,
            '一行减法：连图池一起扣干净',
            'vllm/v1/worker/gpu_worker.py · Worker.determine_available_memory',
            [rrow('available_kv = 预算 − 非 KV 占用 − CUDA 图估计（一行减法）', 9.2, '#334155'),
             rrow('available_gpu_memory = 41943040 B（40 MiB）', 10.5, C_GPU, True, lead=20),
             rrow('多卡时是一张逐 worker 的清单；这就是 KV 池的全部本金', 8.8, C_GREY)],
            C_GPU, C_GPU_F, badge='站 3')
up_r = nbox(RX, y, HALF_W,
            '每层自报形状：spec 是全部原料',
            'vllm/v1/worker/gpu_model_runner.py · get_kv_cache_spec',
            [rrow('每层交上来的自我介绍：FullAttentionSpec（全历史）/ SlidingWindowSpec（滑窗）/ '
                  'MambaSpec（状态型）', 9.2, '#334155'),
             rrow('kv_cache_specs（每层自报形状）', 10.5, C_GPU, True, lead=20),
             rrow('spec 是分组与字节换块数的全部原料——账本精确到层的前提', 8.8, C_GREY)],
            C_GPU, C_GPU_F, badge='站 4')

# 上游 → 总编排（汇聚）
y_eng = max(up_l['y'], up_r['y']) + GAP
bus_up = merge([LCX, RCX], [up_l['y'], up_r['y']], MIDX, y_eng, C_GPU, sw=2.0, bus_gap=30)
vlabel(LCX, up_l['y'] + 20, 'available_gpu_memory', 9.0, C_GPU, maxw=HALF_W - 40)
vlabel(RCX, up_r['y'] + 20, 'kv_cache_specs', 9.0, C_GPU, maxw=HALF_W - 40)

# ================= 总编排（EngineCore · 橙） =================
eng = nbox(MX, y_eng, CW,
           'EngineCore._initialize_kv_caches —— 总编排：一份账的去向全在这段里',
           'vllm/v1/engine/core.py:L301-L330',
           [rrow('① 定账：调 get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)',
                 9.2, '#334155'),
            rrow('② 拍平与写回：generate_scheduler_kv_cache_config(kv_cache_configs) → '
                 '写回 cache_config 四件套', 9.2, '#334155'),
            rrow('③ 落地：self.model_executor.initialize_from_config(kv_cache_configs)',
                 9.2, '#334155'),
            rrow('块内还夹一步 auto-fit 善后：max_model_len 被缩就 collective_rpc 同步 worker'
                 '（边界步，不占本图主线）', 8.8, C_GREY)],
           C_EN, C_EN_F)

# 总编排 → 定账单点（引擎发起的那次调用）
y_calc = eng['y'] + GAP + 8
lc.seg(MIDX, eng['y'], MIDX, y_calc, C_EN, 2.2, 'up')
vlabel(MIDX, eng['y'] + 22,
       'kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)',
       9.0, C_EN, maxw=CW / 2 + 300)
vlabel(MIDX, eng['y'] + 39, '引擎里只此一次调用（单点）——两侧拿到的都是它的返回值',
       8.8, C_GREY, maxw=CW / 2 + 300)

# ================= 单点定账（站 6 · 青 = 账本） =================
calc = nbox(MX, y_calc, CW,
            'get_kv_cache_configs —— 单点定账：护栏四道 / override 折算 / PP 取最小',
            'vllm/v1/core/kv_cache_utils.py · get_kv_cache_configs',
            [rrow('字节换块数的总算术：41943040 // 65536 // 2 = 320', 11, C_KV, True, lead=20),
             rrow('先除一页的字节数得页数、再除每块的层数得块数——两次整除，零头丢掉',
                  9.0, '#334155'),
             rrow('护栏四道、override 折算、PP 各 rank 取最小都是它的内部步骤（上一节已逐条拆开）；'
                  '这里只认一件事：它是 KVCacheConfig 的唯一产出点', 8.8, C_GREY)],
            C_KV, C_KV_F, badge='站 6')

# 定账 → 一份账（唯一产出）
y_cfg = calc['y'] + GAP - 6
lc.seg(MIDX, calc['y'], MIDX, y_cfg, C_KV, 2.0, 'kv')
vlabel(MIDX, calc['y'] + 24, '产出：KVCacheConfig（num_blocks + 分组 + 张量布局）',
       8.8, C_KV, maxw=CW / 2 + 300)

CFG_W, CFG_H = 520, 46
lc.rect(MIDX - CFG_W / 2, y_cfg, CFG_W, CFG_H, '#ffffff', C_KV, rx=CFG_H / 2, sw=2.2)
lc.text(MIDX, y_cfg + 19, '一份 KVCacheConfig（两侧唯一的账）', 11.5, C_KV, 'middle', True,
        maxw=CFG_W - 24, tag='cfg:t')
lc.text(MIDX, y_cfg + 36, 'num_blocks = 320', 10.5, C_KV, 'middle', True, maxw=CFG_W - 24,
        tag='cfg:n')

# ================= 喂两侧（站 7）：分岔题头 + 两支 =================
fd_y = y_cfg + CFG_H + 30           # 题头行基线（分岔竖线两侧留空）
head = '喂两侧（站 7）：一份账的两次投影'
lc.text(MX, fd_y, head, 11.5, lc.C_TXT, 'start', True, maxw=520, tag='band:t')
bw = 14 + 9.0 * len('站 7')
bx = MX + lc.tw(head, 11.5, True) + 12
lc.rect(bx, fd_y - 11, bw, 17, lc.C_BADGE_F, C_EN, rx=8, sw=1.0)
lc.text(bx + bw / 2, fd_y, '站 7', BADGE_FS, C_EN, 'middle', True, tag='band:bg')
lc.text(BXR, fd_y, '拍平版喂调度器 / 布局版喂 worker——数字必然相等，靠结构不靠对账',
        9.0, C_GREY, 'end', maxw=560, tag='band:s')

bus_dn, y_br = fork(MIDX, y_cfg + CFG_H, [LCX, RCX], C_KV, sw=2.2, bus_gap=fd_y - (y_cfg + CFG_H) + 16,
                    drop_gap=30)
vlabel(LCX, y_br - 12, '拍平（无损投影）', 8.8, C_KV, maxw=HALF_W - 40)
vlabel(RCX, y_br - 12, '张量布局（真分配）', 8.8, C_GPU, maxw=HALF_W - 40)

br_l = nbox(LX, y_br, HALF_W,
            '拍平版喂调度器：generate_scheduler_kv_cache_config —— 无损拍平',
            'vllm/v1/core/kv_cache_utils.py:L1855-L1874',
            [rrow('assert 全部 worker 的 num_blocks 相等（PP 各 rank 取最小已保证）'
                  '→ 代表 spec 任取一层', 9.2, '#334155'),
             rrow('写回 cache_config 四件套（前端日志与 API 看到的就是这些值）：', 9.2, '#334155'),
             rrow(chips=['num_gpu_blocks 320', 'block_size 16', '容量 5120 = 320×16',
                         '并发 1.25 = 320 / 256'], lead=32),
             rrow('调度器侧：KVCacheManager + BlockPool 就位，此后按 320 块做准入与抢占的账',
                  8.8, C_GREY)],
            C_KV, C_KV_F)
for cy in br_l['chips']:
    chiprow(LX, HALF_W, cy, ['num_gpu_blocks 320', 'block_size 16', '容量 5120 = 320×16',
                             '并发 1.25 = 320 / 256'], C_KV, C_KV_F)

br_r = nbox(RX, y_br, HALF_W,
            '布局版喂 worker：initialize_from_config —— 在池里真分配',
            'vllm/v1/worker/gpu_worker.py:L649-L676',
            [rrow('self.model_executor.initialize_from_config(kv_cache_configs)：'
                  'worker 按 config 的张量布局分配', 9.2, '#334155'),
             rrow('落点在 tag="kv_cache" 的 CuMem 池：按显式 tag 分池记账，kv_cache 一池、'
                  'weights 一池', 9.2, '#334155'),
             rrow('池里每一页都是真显存——块号是调度器侧与 worker 共用的唯一键', 8.8, C_GREY),
             rrow('worker 侧 num_blocks = 320（executor_got_same_config = true）',
                  10.5, C_GPU, True, lead=20)],
            C_GPU, C_GPU_F)

# ================= 两侧同数（会合括号 + 结论） =================
jy = max(br_l['y'], br_r['y']) + 26
for x in (LCX, RCX):
    lc.seg(x, max(br_l['y'], br_r['y']), x, jy, C_GREY, 1.8)
lc.seg(LCX, jy, RCX, jy, C_GREY, 1.8)
lc.text(MIDX, jy + 24, '两侧 num_blocks 同 320（executor_got_same_config = true）——'
                       '同一份账的两次投影', 10, lc.C_TXT, 'middle', True, maxw=CW, tag='join:t')
lc.text(MIDX, jy + 42, '相等靠单源结构：任何一侧想看到不同的块数都必须绕过 get_kv_cache_configs，'
                       '而装配序里没有第二条路', 8.8, C_GREY, 'middle', maxw=CW, tag='join:s')

# ================= 图例 =================
LY = jy + 68
lx = MX
for c, f, name in [(C_EN, C_EN_F, '引擎总编排（core.py）'),
                   (C_KV, C_KV_F, '账本定账与调度器侧（kv_cache_utils.py）'),
                   (C_GPU, C_GPU_F, 'worker 侧（测量与真分配）')]:
    lc.rect(lx, LY - 10, 20, 13, f, c, rx=3, sw=1.4)
    lc.text(lx + 26, LY, name, 8.8, lc.C_TXT, 'start', maxw=280, tag='lg' + name[:6])
    lx += 26 + lc.tw(name, 8.8) + 22
lc.rect(lx, LY - 11, 14 + 9.0 * 3, 17, lc.C_BADGE_F, C_EN, rx=8, sw=1.0)
lc.text(lx + (14 + 9.0 * 3) / 2, LY, '站 3', BADGE_FS, C_EN, 'middle', True, tag='lg:bg')
lx += 14 + 9.0 * 3 + 8
lc.text(lx, LY, '= 本章站号（站号 = 账本从诞生到把门的阅读顺序）', 8.8, lc.C_TXT, 'start',
        maxw=340, tag='lg:bgt')
lx += lc.tw('= 本章站号（站号 = 账本从诞生到把门的阅读顺序）', 8.8) + 22
lc.seg(lx, LY - 4, lx + 30, LY - 4, C_EN, 2.2, 'up')
lc.text(lx + 36, LY, '调用/喂入方向（自上而下，旁注即调用表达式）', 8.8, lc.C_TXT, 'start',
        maxw=BXR - lx - 36, tag='lg:ar')

# ================= 读图 + 图注（给结论） + 页脚锚 =================
CY0 = LY + 26
lc.text(MX, CY0, '读图：自上而下四段——上游两站（绿，worker 侧测量与自报）→ 总编排（橙，core.py 一段）'
                 '→ 单点定账（青）→ 一份账分两支（左青=调度器侧、右绿=worker 侧）',
        8.6, C_GREY, 'start', maxw=CW, tag='cp0')
lc.text(MX, CY0 + 20, '账从哪来（站 3 一行减法、站 4 每层自报）、单点在哪（get_kv_cache_configs）、'
                      '喂到哪两侧（拍平版喂调度器、布局版喂 worker）——两侧 num_blocks 同 320，'
                      '因为它们是同一份 KVCacheConfig 的两次投影，不是两笔账',
        9.5, lc.C_TXT, 'start', True, maxw=CW, tag='cap')
lc.text(MX, CY0 + 38, '逐字锚 vllm/v1/engine/core.py:L301-L330（总编排）· '
                      'vllm/v1/core/kv_cache_utils.py:L1855-L1874（拍平）· '
                      'vllm/v1/worker/gpu_worker.py:L649-L676（CuMem 池分配）',
        8.2, lc.C_FAINT, 'start', maxw=CW, tag='ft1')
lc.text(MX, CY0 + 52, '数字取自配套精简版实跑（2 层 full 玩具 · page 65536 B · available 40 MiB）· '
                      '行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=CW, tag='ft2')

# ================= 装配输出 =================
H = CY0 + 52 + 20
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>',
       lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-init-call-panorama.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
