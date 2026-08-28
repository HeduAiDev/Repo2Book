#!/usr/bin/env python3
"""ch14 机制图 5 · 一份账喂两侧（figure_spec ch14-fig-one-ledger-two-sides，模板 tensor-flow）

放大自 L0 启动段 → KV 账本列与 GPU 列的双喂线——本章 L2 章图拍片④「一份账喂两侧」
的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：get_kv_cache_configs 是 KVCacheConfig 的唯一产出点——拍平版喂调度器、
张量布局版喂 worker，两侧 num_blocks 由结构保证相等（PP 再取各 rank 最小并缩张量）。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑）。坐标由常量/循环计算；
文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440
MIDX = 750

# ---------------- 标题区 ----------------
lc.text(MX, 34, '一份账喂两侧：get_kv_cache_configs 是 KVCacheConfig 的唯一产出点——单源即防漂',
        16.5, lc.C_TXT, 'start', True, maxw=990, tag='title')
lc.text(MX, 58, '拍平版喂调度器（建 KVCacheManager）、张量布局版喂 worker（真分配）——两侧 num_blocks 相等靠结构，不靠运行时对账',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = 'L0 放大 · 启动段双喂线 · L2 拍片④'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶部：装配序 ----------------
AW = 900
lc.rect(MIDX - AW / 2, 90, AW, 56, lc.C_ENG_F, lc.C_ENG_S, rx=7, sw=1.5)
lc.text(MIDX, 109, 'EngineCore._initialize_kv_caches · 装配序', 11.5, lc.C_ENG_S, 'middle',
        True, maxw=AW - 24, tag='top:t')
lc.text(MIDX, 130, '收每层 KVCacheSpec → profile 可用显存 → 定账 → 写回 cache_config → worker initialize → 调度器 resolve',
        9, '#334155', 'middle', maxw=AW - 24, tag='top:s')

# ---------------- 定账（唯一产出点） ----------------
DW = 760
DY = 170
lc.rect(MIDX - DW / 2, DY, DW, 84, lc.C_KV_F, lc.C_KV_S, rx=7, sw=2.0)
lc.text(MIDX, DY + 21, 'get_kv_cache_configs · 定账（唯一产出点）', 11.5, lc.C_KV_S, 'middle',
        True, maxw=DW - 24, tag='calc:t')
lc.text(MIDX, DY + 42, 'available 41943040 B（40 MiB）// 65536（page）// 2（组 = 2 层）= 320 块',
        10, '#334155', 'middle', maxw=DW - 24, tag='calc:l1')
lc.text(MIDX, DY + 62, 'page × 2 层 = 131072 B / 块 · max_model_len 4096 下每请求 256 块',
        8.8, lc.C_MUTE, 'middle', maxw=DW - 24, tag='calc:l2')
lc.seg(MIDX, 146, MIDX, DY - 3, lc.C_ENG_S, 2.0, 'std')

# ---------------- KVCacheConfig 单点 ----------------
KY = 282
KW = 420
lc.rect(MIDX - KW / 2, KY, KW, 44, '#ffffff', lc.C_KV_S, rx=22, sw=2.2)
lc.text(MIDX, KY + 19, '一份 KVCacheConfig', 12, lc.C_KV_S, 'middle', True, maxw=KW - 24,
        tag='cfg:t')
lc.text(MIDX, KY + 36, 'num_blocks = 320', 10.5, lc.C_KV_S, 'middle', True, maxw=KW - 24,
        tag='cfg:n')
lc.seg(MIDX, DY + 84, MIDX, KY - 3, lc.C_KV_S, 2.0, 'std')

# ---------------- 两侧面板 ----------------
PY, PH = 366, 150
LX, LW2 = MX, 620
RX, RW2 = BXR - 620, 620
# 左：调度器侧
lc.rect(LX, PY, LW2, PH, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.6)
lc.text(LX + 16, PY + 22, '调度器侧 · 拍平版', 11.5, lc.C_KV_S, 'start', True, maxw=200,
        tag='sch:t')
lc.text(LX + LW2 - 12, PY + 22, 'kv_cache_utils.py:L1855-L1874', 8.2, lc.C_FAINT, 'end',
        maxw=250, tag='sch:f')
for i, ln in enumerate([
        'generate_scheduler_kv_cache_config：无损拍平',
        '→ FullAttentionSpec 代表组 · num_blocks 原样断言相等',
        '→ KVCacheManager + BlockPool 就位（watermark 在此注入）',
        '调度器从此按 320 块做准入与抢占的账']):
    lc.text(LX + 16, PY + 44 + i * 19, ln, 9.2, '#334155', 'start', maxw=LW2 - 32,
            tag='sch:l%d' % i)
lc.text(LX + 16, PY + 44 + 4 * 19 + 6, 'num_blocks = 320', 11, lc.C_KV_S, 'start', True,
        maxw=200, tag='sch:n')
# 右：worker 侧
lc.rect(RX, PY, RW2, PH, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.6)
lc.text(RX + 16, PY + 22, 'worker 侧 · 张量布局版', 11.5, lc.C_GPU_S, 'start', True, maxw=220,
        tag='wk:t')
lc.text(RX + RW2 - 12, PY + 22, 'gpu_worker.py:L650-L676', 8.2, lc.C_FAINT, 'end', maxw=250,
        tag='wk:f')
for i, ln in enumerate([
        'initialize_from_config：按 config 的张量布局',
        '→ CuMemAllocator tag="kv_cache" 池内真分配',
        '→ worker_initialized_num_blocks = 320',
        '池里每一页都是真显存，块号与调度器同源']):
    lc.text(RX + 16, PY + 44 + i * 19, ln, 9.2, '#334155', 'start', maxw=RW2 - 32,
            tag='wk:l%d' % i)
lc.text(RX + 16, PY + 44 + 4 * 19 + 6, 'num_blocks = 320（executor_got_same_config = true）',
        11, lc.C_GPU_S, 'start', True, maxw=RW2 - 32, tag='wk:n')
# 喂线箭头（config → 两侧）
lc.parrow([(MIDX - KW / 2, KY + 22), (LX + LW2 / 2, KY + 22), (LX + LW2 / 2, PY - 3)],
          lc.C_KV_S, 2.0, 'std')
lc.parrow([(MIDX + KW / 2, KY + 22), (RX + RW2 / 2, KY + 22), (RX + RW2 / 2, PY - 3)],
          lc.C_GPU_S, 2.0, 'std')
lc.text(MIDX - KW / 2 - 10, KY + 14, '拍平（无损投影）', 8.8, lc.C_KV_S, 'end', maxw=110,
        tag='al:l')
lc.text(MIDX + KW / 2 + 10, KY + 14, '张量布局（真分配）', 8.8, lc.C_GPU_S, 'start', maxw=120,
        tag='al:r')

# ---------------- 中缝：写回 cache_config 四件套 ----------------
WBY = PY + PH + 26
WBW = 660
lc.rect(MIDX - WBW / 2, WBY, WBW, 74, '#ffffff', lc.C_MUTE, rx=7, sw=1.4)
lc.text(MIDX, WBY + 19, '写回 cache_config 四件套（前端日志与 API 看到的就是这些值）', 10,
        lc.C_TXT, 'middle', True, maxw=WBW - 20, tag='wb:t')
CHIPS = ['num_gpu_blocks=320', 'block_size=16', 'kv_cache_size_tokens=5120',
         'kv_cache_max_concurrency=1.25']
chw = [lc.tw(c, 8.8, True) + 16 for c in CHIPS]
tot = sum(chw) + 3 * 8
cxx = MIDX - tot / 2
for c, cw2 in zip(CHIPS, chw):
    lc.rect(cxx, WBY + 32, cw2, 22, lc.C_KV_F, lc.C_KV_S, rx=10, sw=1.1)
    lc.text(cxx + cw2 / 2, WBY + 47, c, 8.8, lc.C_KV_S, 'middle', True, maxw=cw2 - 6,
            tag='wb:' + c)
    cxx += cw2 + 8
lc.text(MIDX, WBY + 70, '容量 5120 token = 320 × 16 · 并发 1.25 = 320 / 256（4096 长度每请求 256 块）',
        8.4, '#475569', 'middle', maxw=WBW - 20, tag='wb:sub')
lc.seg(MIDX, KY + 44, MIDX, WBY - 3, lc.C_MUTE, 1.6, 'std', dash=True)

# ---------------- 底部：两条边界注记 ----------------
NY = WBY + 96
NW = (BXR - MX - 20) / 2
lc.rect(MX, NY, NW, 72, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
lc.text(MX + 14, NY + 19, 'PP 场景：各 rank 取全场最小', 9.8, lc.C_TXT, 'start', True,
        maxw=NW - 28, tag='pp:t')
lc.text(MX + 14, NY + 37, 'get_kv_cache_configs 末段显式把各 rank 的 num_blocks 改写为', 8.8,
        '#334155', 'start', maxw=NW - 28, tag='pp:l1')
lc.text(MX + 14, NY + 53, '全场最小、张量按比例缩——任何一侧想看到不同的块数都必须绕过这个函数',
        8.8, '#334155', 'start', maxw=NW - 28, tag='pp:l2')
lc.text(MX + 14, NY + 69, '而装配序里没有第二条路', 8.8, '#334155', 'start', maxw=NW - 28,
        tag='pp:l3')
lc.rect(MX + NW + 20, NY, NW, 72, '#ffffff', lc.C_MUTE, rx=7, sw=1.2, dash=True)
nx = MX + NW + 34
lc.text(nx, NY + 19, 'auto-fit 的同步尾巴', 9.8, lc.C_TXT, 'start', True, maxw=NW - 28,
        tag='af:t')
lc.text(nx, NY + 37, 'auto-fit 缩了 max_model_len 时，还要 collective_rpc 调', 8.8,
        '#334155', 'start', maxw=NW - 28, tag='af:l1')
lc.text(nx, NY + 53, 'update_max_model_len 同步 worker——worker 先于 profile 启动、', 8.8,
        '#334155', 'start', maxw=NW - 28, tag='af:l2')
lc.text(nx, NY + 69, '缓存的是旧值，不补一发就账实不符', 8.8, '#334155', 'start',
        maxw=NW - 28, tag='af:l3')

# ---------------- 页脚 ----------------
FY = NY + 96
lc.text(MX, FY, '逐字锚 vllm/v1/core/kv_cache_utils.py:L2094-L2242（get_kv_cache_configs 本体）· L1855-L1874（拍平）· '
        'vllm/v1/worker/gpu_worker.py:L650-L676（initialize_from_config）· vllm/v1/engine/core.py:L250-L359（装配序）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 16, '数字取自配套精简版 host 实跑（2 层 full 玩具 · page 65536 B）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = FY + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-one-ledger-two-sides.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
