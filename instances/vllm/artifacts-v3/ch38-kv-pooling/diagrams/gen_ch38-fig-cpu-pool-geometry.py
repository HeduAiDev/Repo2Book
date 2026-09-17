#!/usr/bin/env python3
"""ch38 机制图 m2 · cpu-pool-geometry（figure_spec ch38-fig-cpu-pool-geometry，模板 layout）

放大自 L2 站 2（② CPU 池开张·钉页 mmap）· L0：外部池的物理本体（/dev/shm 一块）。

claim：CPU 池一块池位的物理布局是 |--- W0-B0 ---|--- W1-B0 ---| ... | maybe-pad |：
全体 worker 的页拼进行内、末尾页对齐补零，num_blocks = cpu_bytes_to_use ÷
round_up(worker 每块字节 × world_size × blocks_per_chunk, PAGESIZE)。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · 主例：512B × 2 × 2 = 2048B → round_up(2048, 4096) = 4096B，垫 2048B
  · num_blocks = 100MiB ÷ 4096B = 25600；每 worker 页 = 2048 ÷ 2 = 1024B
  · 对照：1024B/块 → 4096B 恰对齐、垫 0、仍 25600 块
  · 布局注释原文 |--- W0-B0---|---- W1-B0---| ... | maybe-pad |（cpu/spec.py:L77-L110）
  · /dev/shm/vllm_offload_{engine_id}.mmap；槽偏移 = rank × cpu_page_size；rank=None 也 mmap
  · canonical 规范化：2 层 → 1 条 (8,512) 张量、组引用 2 条（同 data_ptr）
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1560
MX = 60
BXR = W - MX

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'CPU 池开张：一整块预算按「全 worker 拼起来的一块」计价，物理上只有一份池', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 60, '先定价再铺位：池位里先排 W0 的页、再排 W1 的页、末尾补零凑整页——预算按全部 worker 计，而非 per-worker', 10.5,
        lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L2 站 2（② CPU 池开张·钉页 mmap）· L0：外部池'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 定价公式条 ----------------
FY0 = 86
lc.rect(MX, FY0, BXR - MX, 44, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.2)
lc.text((MX + BXR) / 2, FY0 + 18, '定价公式：num_blocks = cpu_bytes_to_use ÷ round_up(每 worker 块字节 × world_size × blocks_per_chunk, PAGESIZE=4096)',
        10.5, lc.C_TXT, 'middle', True, maxw=BXR - MX - 30, tag='formula')
lc.text((MX + BXR) / 2, FY0 + 34, '「预算是必填项——不定价不开张：缺 cpu_bytes_to_use 直接 Exception」· 小池例：8192B ÷ 4096B = 2 块',
        8.5, lc.C_MUTE, 'middle', maxw=BXR - MX - 30, tag='formula:sub')

# ---------------- 主例 / 对照 两箱 ----------------
CMP_Y = FY0 + 58
CW2 = (BXR - MX - 20) / 2
cases = [
    ('主例 · 每 worker 块 512B', lc.C_KV_S, [
        '· 512B × 2 worker × 2 块 = 2048B',
        '· round_up(2048, 4096) = 4096B → 垫 2048B',
        '   （padding 率 50%）',
        '· num_blocks = 100MiB ÷ 4096B = 25600',
        '· 每 worker 页 = 2048 ÷ 2 = 1024B',
    ]),
    ('对照 · 每 worker 块 1024B', lc.C_GPU_S, [
        '· 1024B × 2 × 2 = 4096B 恰对齐',
        '· 垫 0（无 padding 布局公式仍成立）',
        '· num_blocks 仍 = 25600',
        '· 每 worker 页 = 2048B',
        '· world=1 另例：512B 垫 3584B · 1MiB=256 块',
    ]),
]
for i, (t, color, lines) in enumerate(cases):
    x = MX + i * (CW2 + 20)
    h = 30 + len(lines) * 15.5 + 8
    lc.rect(x, CMP_Y, CW2, h, '#ffffff', color, rx=8, sw=1.4)
    lc.text(x + 14, CMP_Y + 20, t, 10, color, 'start', True, maxw=CW2 - 28, tag='cmp:t%d' % i)
    for j, ln in enumerate(lines):
        lc.text(x + 14, CMP_Y + 40 + j * 15.5, ln, 9, '#334155', 'start',
                maxw=CW2 - 26, tag='cmp:l%d_%d' % (i, j))
CMP_H = 30 + 5 * 15.5 + 8

# ---------------- 一块池位的横条布局 ----------------
BAR_SEC_Y = CMP_Y + CMP_H + 26
lc.text(MX, BAR_SEC_Y, '一块池位（chunk）的物理布局——源码注释原文：|--- W0-B0 ---|--- W1-B0 ---| ... | maybe-pad |', 11,
        lc.C_TXT, 'start', True, maxw=BXR - MX - 340, tag='bar:sec')
lc.text(BXR, BAR_SEC_Y, '主例代入（4096B/池位）', 9, lc.C_MUTE, 'end', tag='bar:scale')

FILE_LX, FILE_RX = MX, 218
lc.rect(FILE_LX, BAR_SEC_Y + 16, FILE_RX - FILE_LX, 68, '#ffffff', lc.C_KV_S, rx=8, sw=1.2,
        dash=True)
lc.text((FILE_LX + FILE_RX) / 2, BAR_SEC_Y + 34, '/dev/shm/', 9, lc.C_KV_S, 'middle', True,
        maxw=FILE_RX - FILE_LX - 10, tag='file:l1')
lc.text((FILE_LX + FILE_RX) / 2, BAR_SEC_Y + 50, 'vllm_offload_', 8.2, lc.C_KV_S, 'middle',
        maxw=FILE_RX - FILE_LX - 10, tag='file:l2')
lc.text((FILE_LX + FILE_RX) / 2, BAR_SEC_Y + 64, '{engine_id}.mmap', 8.2, lc.C_KV_S, 'middle',
        maxw=FILE_RX - FILE_LX - 10, tag='file:l3')
lc.text((FILE_LX + FILE_RX) / 2, BAR_SEC_Y + 80, '共享 mmap 文件', 8.2, lc.C_MUTE, 'middle',
        maxw=FILE_RX - FILE_LX - 10, tag='file:l4')

BAR_X0, BAR_X1 = FILE_RX + 24, BXR - 10
BAR_Y, BAR_H = BAR_SEC_Y + 24, 56
BAR_W = BAR_X1 - BAR_X0
BYTES = 4096.0


def bx(b):
    return BAR_X0 + b / BYTES * BAR_W


# 分区：W0 两个块页 | W1 两个块页 | pad
zones = [
    ('W0-B0', 0, 512, lc.C_GPU_F, lc.C_GPU_S),
    ('W0-B1', 512, 1024, lc.C_GPU_F, lc.C_GPU_S),
    ('W1-B0', 1024, 1536, '#dcfce7', lc.C_GPU_S),
    ('W1-B1', 1536, 2048, '#dcfce7', lc.C_GPU_S),
]
for name, b0, b1, fill, stroke in zones:
    lc.rect(bx(b0), BAR_Y, bx(b1) - bx(b0), BAR_H, fill, stroke, rx=3, sw=1.2)
    lc.text((bx(b0) + bx(b1)) / 2, BAR_Y + 24, name, 9, lc.C_TXT, 'middle', True,
            maxw=bx(b1) - bx(b0) - 4, tag='zone:' + name)
    lc.text((bx(b0) + bx(b1)) / 2, BAR_Y + 42, '512B', 8.5, lc.C_MUTE, 'middle',
            maxw=bx(b1) - bx(b0) - 4, tag='zone:b' + name)
# pad 斜纹
PAD_B0, PAD_B1 = 2048, 4096
svgp = (f'<rect x="{bx(PAD_B0):.1f}" y="{BAR_Y}" width="{bx(PAD_B1) - bx(PAD_B0):.1f}" '
        f'height="{BAR_H}" fill="url(#padhatch)" stroke="#94a3b8" stroke-width="1.2" rx="3"/>')
lc.ELEMS.append(((bx(PAD_B0) - 2, BAR_Y - 2, bx(PAD_B1) + 2, BAR_Y + BAR_H + 2), svgp))
lc.text((bx(PAD_B0) + bx(PAD_B1)) / 2, BAR_Y + 24, 'maybe-pad（补零凑整页）', 9, lc.C_MUTE,
        'middle', True, maxw=bx(PAD_B1) - bx(PAD_B0) - 8, tag='pad:t')
lc.text((bx(PAD_B0) + bx(PAD_B1)) / 2, BAR_Y + 42, '2048B', 8.5, lc.C_MUTE, 'middle',
        maxw=bx(PAD_B1) - bx(PAD_B0) - 8, tag='pad:b')
# 上方 worker 页括标
lc.seg(bx(0), BAR_Y - 8, bx(1024), BAR_Y - 8, lc.C_GPU_S, 1.2)
lc.seg(bx(0), BAR_Y - 12, bx(0), BAR_Y - 4, lc.C_GPU_S, 1.0)
lc.seg(bx(1024), BAR_Y - 12, bx(1024), BAR_Y - 4, lc.C_GPU_S, 1.0)
lc.text(bx(512), BAR_Y - 18, 'W0 页 1024B（槽偏移 = 0 × cpu_page_size）', 8.5, lc.C_GPU_S,
        'middle', maxw=bx(1024) - bx(0), tag='w0:lab')
lc.seg(bx(1024), BAR_Y - 8, bx(2048), BAR_Y - 8, '#15803d', 1.2)
lc.seg(bx(2048), BAR_Y - 12, bx(2048), BAR_Y - 4, '#15803d', 1.0)
lc.text(bx(1536), BAR_Y - 18, 'W1 页 1024B（偏移 = 1 × cpu_page_size）', 8.5, '#15803d',
        'middle', maxw=bx(2048) - bx(1024), tag='w1:lab')
# 文件标签框 → 池位条 的箭头（两端贴框边）
lc.seg(FILE_RX, BAR_SEC_Y + 50, BAR_X0, BAR_SEC_Y + 50, lc.C_KV_S, 1.6, 'std')
# 下方字节标尺
for b in (0, 1024, 2048, 4096):
    lc.seg(bx(b), BAR_Y + BAR_H, bx(b), BAR_Y + BAR_H + 5, lc.C_MUTE, 1.0)
    lab = '%dB' % b if b else '0'
    lc.text(bx(b), BAR_Y + BAR_H + 18, lab, 8.5, lc.C_MUTE, 'middle', tag='tick%d' % b)

# ---------------- 三类执行者（同一物理池的视觉锚） ----------------
BUS_Y = BAR_Y + BAR_H + 40
lc.seg(BAR_X0, BUS_Y, BAR_X1, BUS_Y, lc.C_KV_S, 2.0)
lc.text(BAR_X1, BUS_Y - 8, '同一物理池（一份 mmap 文件）——「共享数据、零共享状态」', 9.5,
        lc.C_KV_S, 'end', True, maxw=520, tag='bus:t')
CHIP_Y = BUS_Y + 44
CHIP_W2 = (BAR_X1 - BAR_X0 - 2 * 24) / 3
executors = [
    ('① 全体 TP worker', lc.C_GPU_S, [
        '· 各占一 slot：worker k 区间 =',
        '   [k × cpu_page_size, (k+1) × …)',
        '   互不重叠（线性偏移保证）',
        '· cudaHostRegister 钉页',
    ]),
    ('② 调度器进程（rank=None）', lc.C_ENG_S, [
        '· 不属任何 worker，也 mmap',
        '   同一文件——看到同一物理池',
        '· tiering 下 CPU↔secondary I/O',
        '   线程就住这一侧',
    ]),
    ('③ secondary tier I/O 线程', lc.C_ZMQ_S, [
        '· 经 primary 出让的',
        '   memoryview 直读写池位',
        '· 不自建通道——物理池仍只',
        '   这一份',
    ]),
]
for i, (t, color, lines) in enumerate(executors):
    x = BAR_X0 + i * (CHIP_W2 + 24)
    h = 30 + len(lines) * 15 + 8
    lc.rect(x, CHIP_Y, CHIP_W2, h, '#ffffff', color, rx=8, sw=1.4)
    lc.text(x + 12, CHIP_Y + 20, t, 9.5, color, 'start', True, maxw=CHIP_W2 - 24,
            tag='exe:t%d' % i)
    for j, ln in enumerate(lines):
        lc.text(x + 12, CHIP_Y + 38 + j * 15, ln, 8.5, '#334155', 'start',
                maxw=CHIP_W2 - 22, tag='exe:l%d_%d' % (i, j))
    # 总线 → 芯片 的引线（竖直虚线，起点落在总线上、终点贴芯片顶边）
    cx = x + CHIP_W2 / 2
    lc.seg(cx, BUS_Y, cx, CHIP_Y, color, 1.4, 'std', dash=True)
CHIP_H = 30 + 4 * 15 + 8

# ---------------- canonical 规范化 + 不变量 ----------------
BOT_Y = CHIP_Y + CHIP_H + 24
BW = (BXR - MX - 20) / 2
# 左：canonical mini
b1x = MX
lc.rect(b1x, BOT_Y, BW, 108, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(b1x + 14, BOT_Y + 20, 'canonical 规范化：任意 attention 布局先拍平', 10, lc.C_TXT,
        'start', True, maxw=BW - 28, tag='can:t')
_LY = BOT_Y + 40
for j, lab in enumerate(['层 0 引用', '层 1 引用']):
    lx_ = b1x + 18 + j * 96
    lc.rect(lx_, _LY, 84, 24, lc.C_GPU_F, lc.C_GPU_S, rx=5, sw=1.1)
    lc.text(lx_ + 42, _LY + 16, lab, 8.5, lc.C_GPU_S, 'middle', True, maxw=80, tag='can:ref%d' % j)
    lc.seg(lx_ + 84, _LY + 12, b1x + 262, _LY + 12, lc.C_GPU_S, 1.2, 'std')
lc.rect(b1x + 262, _LY - 6, 190, 36, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.3)
lc.text(b1x + 357, _LY + 8, 'canonical 张量 (8, 512)', 9, lc.C_KV_S, 'middle', True,
        maxw=186, tag='can:ten')
lc.text(b1x + 357, _LY + 23, '1 条物理张量（data_ptr 相同）', 8, lc.C_MUTE, 'middle',
        maxw=186, tag='can:ptr')
lc.text(b1x + 14, BOT_Y + 94, '组 1 × 引用 2——层共享同一物理张量，搬运描述符按「组 × 层引用」展开', 8.5,
        lc.C_MUTE, 'start', maxw=BW - 28, tag='can:note')
# 右：不变量
b2x = MX + BW + 20
lc.rect(b2x, BOT_Y, BW, 108, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(b2x + 14, BOT_Y + 20, '开张即守恒（构造期同一条公式算出，无并发）', 10, lc.C_TXT,
        'start', True, maxw=BW - 28, tag='inv:t')
inv_lines = [
    '· 槽位互不重叠：worker k 区间端点相接不交（基例 rank 0 从 0 起，归纳步相邻区间相接）',
    '· 预算上界：num_blocks × 4096B ≤ 100MiB（整除定义即上界）',
    '· 单一事实源：num_blocks / cpu_page_size / kv_bytes_per_chunk 都在 spec 构造期同源算出',
]
for j, ln in enumerate(inv_lines):
    lc.text(b2x + 14, BOT_Y + 42 + j * 17, ln, 8.8, '#334155', 'start', maxw=BW - 26,
            tag='inv:l%d' % j)

# ---------------- 结论 + 页脚 ----------------
CONC_Y = BOT_Y + 108 + 24
lc.text(MX, CONC_Y,
        '图注结论：预算按全 worker 计价、池位按 worker 拼页、末尾补零凑整页——三类执行者看到的是同一份物理池，「共享数据、零共享状态」。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 vllm/v1/kv_offload/cpu/spec.py:L77-L110（布局注释原文与定价公式）· '
                  'shared_offload_region.py:L28-L116（/dev/shm mmap + rank=None）· tiering/spec.py:L170-L215 · offloading/worker.py:L69-L243（canonical 规范化）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · 图中字节数为 world_size=2 · blocks_per_chunk=2 的主例；host 无 /dev/shm 时走源码自带的 no-mmap 张量回退分支',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS,
       '<defs><pattern id="padhatch" width="8" height="8" patternUnits="userSpaceOnUse" '
       'patternTransform="rotate(45)"><rect width="8" height="8" fill="#f1f5f9"/>'
       '<line x1="0" y1="0" x2="0" y2="8" stroke="#cbd5e1" stroke-width="3"/></pattern></defs>']
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-cpu-pool-geometry.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
