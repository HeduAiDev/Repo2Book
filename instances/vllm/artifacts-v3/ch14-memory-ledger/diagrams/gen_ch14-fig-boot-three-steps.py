#!/usr/bin/env python3
"""ch14 机制图 1 · 启动三步定账（figure_spec ch14-fig-boot-three-steps，模板 flow）

放大自 L0 启动带（EngineCore 装配）→ KV 账本列的起点——即本章 L2 章图北行
第 1-3 站（快照定预算 / dummy 前向测峰值 / cudagraph 估计入账）+ 中排拍片③
定块数的机制展开。架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：右上角指北小签。

claim：一行减法定出 KV 池本金：available_kv = requested − non_kv − cudagraph 估计，
再 available // page_size // group_size 换成块数——启动一次算清、全程不再重算。

数字全部取自 figure_spec.numbers（配套精简版 host 实跑，设备读数按 HOST SEAM
注入，算术与 pin 源码逐字一致）。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1500
MX, BXR = 60, 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '启动三步定账：预算 − 峰值账 − 图池估计 = KV 池本金——一次算清、全程不再重算',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '12.5 GiB 假想卡（账目刻意选整）× util 0.8 = 10 GiB 预算；dummy 前向记下 non_kv 峰值账；一行减法剩下 8 GiB 全给 KV，再 //页//组 换成 1024 块',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = 'L0 放大 · 启动带 → KV 账本列起点 · L2 北行站1-3 → 拍片③'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：四步管道 ----------------
PX, PW_ = MX, 640
STAGES = [
    ('s1', '步① request_memory · 定预算', 88,
     ['ceil(13421772800 × 0.8) = 10737418240 B（10.0 GiB）',
      'free 5 GiB < 10 GiB → 直接 raise：预算先于一切'],
     'vllm/v1/worker/utils.py:L409-L429', lc.C_KV_S, lc.C_KV_F, True),
    ('s2', '步② memory_profiling · dummy 前向记峰值账', 100,
     ['total_consumed 1342177280（权重 0.75 + 非 torch 0.5）',
      '+ transient 268435456（激活峰 0.25）',
      '→ non_kv = 1610612736 B（1.5 GiB）'],
     'vllm/utils/mem_utils.py:L233-L326', lc.C_ENG_S, lc.C_ENG_F, True),
    ('s3', '步③ 一行减法 · 定 KV 本金', 88,
     ['10737418240 − 1610612736 − 536870912（cudagraph 估计）',
      '= available_kv 8589934592 B（8.0 GiB）'],
     'vllm/v1/worker/gpu_worker.py:L544-L548', lc.C_KV_S, lc.C_KV_F, True),
    ('s4', '步④ 字节换块数 · get_num_blocks', 100,
     ['8589934592 // 262144（page）// 32（组=32 层）= 1024 块',
      '护栏 needed(4096) = 2147483648 B（2.0 GiB）≤ 8 GiB → 过',
      '（同一份 available 喂护栏与定块数——账不漂移）'],
     'vllm/v1/core/kv_cache_utils.py:L993-L1010', lc.C_KV_S, lc.C_KV_F, True),
]
SY, SGAP = 92, 22
ypos = {}
yy = SY
for key, title, h, notes, file, stroke, fill, hot in STAGES:
    lc.rect(PX, yy, PW_, h, fill, stroke, rx=7, sw=1.8 if hot else 1.3)
    lc.text(PX + 16, yy + 21, title, 11, stroke, 'start', True, maxw=PW_ - 32, tag='st:' + key)
    for i, ln in enumerate(notes):
        lc.text(PX + 16, yy + 40 + i * 17, ln, 9.2, '#334155', 'start', maxw=PW_ - 32,
                tag='sn:%s%d' % (key, i))
    lc.text(PX + PW_ - 12, yy + h - 9, file, 8.2, lc.C_FAINT, 'end', maxw=PW_ - 30,
            tag='sf:' + key)
    ypos[key] = (yy, h)
    if key != 's4':
        lc.seg(PX + PW_ / 2, yy + h + 2, PX + PW_ / 2, yy + h + SGAP - 3, lc.C_KV_S, 2.0, 'std')
    yy += h + SGAP
PIPE_BOT = yy - SGAP

# ---------------- 右：钱包瀑布条（10 GiB 预算怎么被切） ----------------
WX, WBW = 830, 168
BAR_Y0, GIB = 118, 46.0          # 46px = 1 GiB → 10 GiB = 460px
SEGS = [
    ('non_kv 峰值账 1.5 GiB', 1.5, lc.C_ENG_S, lc.C_ENG_F,
     '权重 0.75 + 激活峰 0.25 + 非 torch 0.5（= 1610612736 B）'),
    ('cudagraph 估计 0.5 GiB', 0.5, lc.C_GPU_S, lc.C_GPU_F,
     '启用 CUDA graph 才入账（= 536870912 B）'),
    ('KV 池本金 8.0 GiB', 8.0, lc.C_KV_S, lc.C_KV_F,
     '= 8589934592 B——此后运行期每道门照这份账放行'),
]
lc.text(WX + WBW / 2, BAR_Y0 - 26, '预算 requested = 10737418240 B（10.0 GiB）', 10.5,
        lc.C_TXT, 'middle', True, maxw=320, tag='bar:total')
lc.text(WX + WBW / 2, BAR_Y0 - 10, '12.5 GiB 卡 × gpu_memory_utilization 0.8', 8.8,
        lc.C_MUTE, 'middle', maxw=320, tag='bar:src')
sy = BAR_Y0
seg_pos = {}
for i, (name, gib, stroke, fill, note) in enumerate(SEGS):
    sh = gib * GIB
    lc.rect(WX, sy, WBW, sh, fill, stroke, rx=4, sw=1.5)
    if i > 0:
        lc.text(WX - 14, sy + 4 + 6, '−', 15, lc.C_ABORT, 'end', True, maxw=20, tag='minus%d' % i)
    lc.text(WX + WBW + 14, sy + sh / 2 - 2, name, 10.5, stroke, 'start', True, maxw=240,
            tag='seg%d:t' % i)
    lc.text(WX + WBW + 14, sy + sh / 2 + 14, note, 8.6, '#475569', 'start',
            maxw=BXR - (WX + WBW + 14) - 6, tag='seg%d:n' % i)
    seg_pos[i] = (sy, sh)
    sy += sh
BAR_BOT = sy
lc.text(WX - 14, BAR_BOT - 6, '=', 15, lc.C_KV_S, 'end', True, maxw=20, tag='eq')
# 条内竖排份额标签（大段才有空间）
kv_y, kv_h = seg_pos[2]
lc.text(WX + WBW / 2, kv_y + kv_h / 2 - 6, 'KV 池', 12, lc.C_KV_S, 'middle', True,
        maxw=WBW - 10, tag='bar:kv1')
lc.text(WX + WBW / 2, kv_y + kv_h / 2 + 12, '1024 块', 15, lc.C_KV_S, 'middle', True,
        maxw=WBW - 10, tag='bar:kv2')
# 瀑布条与步③ 的呼应线（available_kv 同一个数）
s3y, s3h = ypos['s3']
lc.parrow([(PX + PW_, s3y + s3h / 2), (WX - 44, s3y + s3h / 2), (WX - 44, kv_y + kv_h / 2),
           (WX - 4, kv_y + kv_h / 2)], lc.C_KV_S, 1.6, 'std', dash=True)
lc.text((PX + PW_ + WX) / 2 - 10, s3y + s3h / 2 - 8, 'available_kv 8589934592 B', 8.6,
        lc.C_KV_S, 'middle', maxw=200, tag='link:kv')

# ---------------- 底部：换块收据（全宽） ----------------
RC_Y = max(PIPE_BOT, BAR_BOT) + 26
lc.rect(MX, RC_Y, BXR - MX, 78, '#ffffff', lc.C_KV_S, rx=7, sw=1.4)
lc.text(MX + 16, RC_Y + 21, '换块收据 · Llama-2-7B FP16（32 层 · kv_heads 32 · head_dim 128 · block_size 16）',
        10.5, lc.C_KV_S, 'start', True, maxw=700, tag='rc:t')
lc.text(MX + 16, RC_Y + 42, 'page = 2 × 16 × 32 × 128 × 2 = 262144 B（每层每块）　→　8589934592 // 262144 // 32 = 1024 块',
        9.6, '#334155', 'start', maxw=BXR - MX - 32, tag='rc:l1')
lc.text(MX + 16, RC_Y + 60, '容量 16384 token（= 1024 × 16）· max_model_len 4096 下每请求 256 块 → 并发 4.0×（= 1024 / 256）· 每 token KV 524288 B',
        9.6, '#334155', 'start', maxw=BXR - MX - 32, tag='rc:l2')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = RC_Y + 100
lx = MX
for stroke, fill, name in [(lc.C_ENG_S, lc.C_ENG_F, 'non_kv 峰值账（KV 之前的债主）'),
                           (lc.C_GPU_S, lc.C_GPU_F, 'cudagraph 图池估计'),
                           (lc.C_KV_S, lc.C_KV_F, 'KV 池本金')]:
    lc.rect(lx, LEG_Y - 9, 20, 13, fill, stroke, rx=3, sw=1.4)
    lc.text(lx + 26, LEG_Y + 1, name, 8.8, lc.C_TXT, 'start', maxw=220, tag='lg')
    lx += 26 + lc.tw(name, 8.8) + 26
lc.text(lx, LEG_Y + 1, '整条 profile 只跑 O(1) 次 dummy 前向 + O(1) 次减法——启动后不再重算，运行期每道门的预算都来自这份账',
        8.8, lc.C_MUTE, 'start', maxw=BXR - lx, tag='lg:note')

lc.text(MX, LEG_Y + 26, '逐字锚 vllm/v1/worker/utils.py:L409-L429（request_memory）· vllm/v1/worker/gpu_worker.py:L459-L548（profile 序与减法）· '
        'vllm/utils/mem_utils.py:L233-L326（峰值账两项）· vllm/v1/core/kv_cache_utils.py:L993-L1010（get_num_blocks）',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 42, '数字取自配套精简版 host 实跑（设备读数按 HOST SEAM 注入，算术与 pin 源码逐字一致）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
H = LEG_Y + 58
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-boot-three-steps.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
