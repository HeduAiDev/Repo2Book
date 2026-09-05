#!/usr/bin/env python3
"""ch20 机制图 ⑧ · 分页 KV 读取(figure_spec ch20-fig-paged-kv-read,模板 layout)

放大自 L0 中列『GPU 执行臂』(绿色列)『模型层 forward + 编译』块与 A 列『BlockPool + 前缀缓存』
(青,C_KV_S)的交界——kernel 读侧穿过页表取分页 KV;写侧(slot_mapping scatter)归 ch19/ch22。
primer 推导链支线:论文的连续 K,V ∈ R^(N×d) 在 vLLM 的真实形态。架构归属回指 L0(FIGURE-SYSTEM §3.3)。

claim:论文假设连续的 K/V,在 vLLM 是『块池 + 每请求页表』——kernel 沿 block_table 逐块取片
(槽位 = 块号×block_size + 页内偏移);接口契约『带 block_table 必须 seqused_k』;tiling 的
SRAM 块(B_c,容量切)与页池块(16 token,寻址粒度切)是两个互不相干的『块』,名字撞车须澄清。

数字全部取自 figure_spec.numbers(block_size=16 config/cache.py:L47;槽位换算 block_table.py:
L430-L440;三断言 flash_attn_interface.py:L270-L278;页池 block_pool.py:L175-L181 与 PagedAttention
arXiv:2309.06180 §2.2——pin 源码逐字)。页表块号为明标『示意』的结构示例(乱序/可共享)。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 706
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'kernel 的 K/V 不在连续显存里:拿着 block_table 逐页取片',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '论文假设连续的 K,V ∈ R^(N×d);vLLM 里它是 16 token 一页、散在块池货架上——kernel 沿每请求页表逐块取片,槽位 = 块号 × block_size + 页内偏移',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '推导链支线 · 放大自 L0 GPU 执行臂(绿)× A 列『BlockPool+前缀缓存』(青)交界'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 上:块池货架 ----------------
lc.text(MX, 96, 'GPU 显存:BlockPool 块池(整块 KV 显存切成 num_gpu_blocks 个等大页块;示意只画 12 块)', 10.5,
        lc.C_KV_S, 'start', True, maxw=1030, tag='pool:t')
BLK_W, BLK_H, BLK_GAP, NBLK = 90, 52, 8, 12
POOL_Y = 112
for i in range(NBLK):
    bx = MX + i * (BLK_W + BLK_GAP)
    if i == 10:                                     # NULL 块:padding 专用
        lc.rect(bx, POOL_Y, BLK_W, BLK_H, '#f1f5f9', '#94a3b8', rx=5, sw=1.2, dash=True)
        lc.text(bx + BLK_W / 2, POOL_Y + 24, 'NULL', 9, '#94a3b8', 'middle', True, maxw=BLK_W - 8,
                tag=f'pool:null')
        lc.text(bx + BLK_W / 2, POOL_Y + 40, 'padding 专用', 7.5, '#94a3b8', 'middle',
                maxw=BLK_W - 6, tag='pool:null2')
    else:
        lc.rect(bx, POOL_Y, BLK_W, BLK_H, lc.C_KV_F, lc.C_KV_S, rx=5, sw=1.5)
        lc.text(bx + BLK_W / 2, POOL_Y + 24, f'块 {i}', 10, lc.C_KV_S, 'middle', True,
                maxw=BLK_W - 8, tag=f'pool:b{i}')
        lc.text(bx + BLK_W / 2, POOL_Y + 40, '16 token', 7.5, '#334155', 'middle',
                maxw=BLK_W - 6, tag=f'pool:s{i}')
lc.text(MX, POOL_Y + BLK_H + 18, '每块 16 token(block_size=16)· 尾部最多浪费 block_size−1 = 15 个 token 位 · 块号 = 调度器与 kernel 的唯一共享键',
        8.5, lc.C_MUTE, 'start', maxw=1120, tag='pool:sub')

BLK_CX = [MX + i * (BLK_W + BLK_GAP) + BLK_W / 2 for i in range(NBLK)]

# ---------------- 中下:每请求页表 ----------------
lc.text(MX, 208, '每请求一张 block_table(请求级页表;示意块号:乱序、可共享——真实页表由分配器决定)', 10.5,
        lc.C_TXT, 'start', True, maxw=1030, tag='pt:t')
CELL_W, CELL_H = 54, 34


def page_row(y, name, ids, arrow_y, color, dash, xoff, toff):
    """y=行顶;ids=页表块号;arrow_y=水平肘线高度;xoff=行内格盘右移;toff=入块目标 x 偏移(共享块分流)。"""
    lc.text(MX, y + CELL_H / 2 + 3, f'{name} →', 10, lc.C_TXT, 'start', True, maxw=60,
            tag='pt:' + name)
    cx0 = 100 + xoff
    centers = []
    for k, bid in enumerate(ids):
        x = cx0 + k * (CELL_W + 4)
        lc.rect(x, y, CELL_W, CELL_H, '#ffffff', color, rx=5, sw=1.5)
        lc.text(x + CELL_W / 2, y + CELL_H / 2 + 3.5, str(bid), 11, color, 'middle', True,
                maxw=CELL_W - 8, tag=f'pt:{name}{k}')
        centers.append(x + CELL_W / 2)
    for cx, bid in zip(centers, ids):
        bx = BLK_CX[bid] + toff
        lc.parrow([(cx, y), (cx, arrow_y), (bx, arrow_y), (bx, POOL_Y + BLK_H + 2)],
                  color, 1.6, 'std', dash=dash)
    return centers


page_row(222, 'A', [5, 2, 9], 192, lc.C_KV_S, False, xoff=0, toff=-8)
lc.text(100, 272, '(示意)3 页 × 16 = 48 token 的 KV', 8.5, lc.C_MUTE, 'start', maxw=180,
        tag='pt:anote')
page_row(310, 'B', [2, 7, 4], 284, '#0e7490', True, xoff=300, toff=8)
lc.text(400, 366, 'B 的首页 = 块 2:与 A 共享', 8.5, '#0e7490', 'start', True, maxw=215,
        tag='pt:bnote')
lc.text(400, 381, '同一页(引用计数,免复制)', 8.5, '#0e7490', 'start', maxw=215,
        tag='pt:bnote2')
lc.text(400, 396, '——账本在调度列(回指 ch13)', 8.5, '#0e7490', 'start', maxw=215,
        tag='pt:bnote3')

# ---------------- 右中:kernel 读侧 ----------------
KB_X, KB_W, KB_Y, KB_H = 620, 380, 300, 172
lc.rect(KB_X, KB_Y, KB_W, KB_H, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.8)
lc.text(KB_X + 14, KB_Y + 20, 'kernel 读侧:flash_attn_varlen_func 内部', 10.5, lc.C_GPU_S,
        'start', True, maxw=KB_W - 28, tag='kb:t')
lc.text(KB_X + 14, KB_Y + 40, '沿 block_table 逐块取 K/V 片,拼成 KV 列块参与 tiling', 9,
        '#334155', 'start', maxw=KB_W - 28, tag='kb:l1')
# SRAM tile 小格
lc.rect(KB_X + 14, KB_Y + 56, 150, 42, '#bbf7d0', lc.C_GPU_S, rx=5, sw=1.4)
lc.text(KB_X + 89, KB_Y + 72, 'SRAM KV tile', 9, '#166534', 'middle', True, maxw=140,
        tag='kb:tile')
lc.text(KB_X + 89, KB_Y + 88, 'B_c = 64/128 token', 8, '#166534', 'middle', maxw=140,
        tag='kb:tile2')
lc.text(KB_X + 178, KB_Y + 70, 'tile 尺寸由片上容量定:', 8.5, '#334155', 'start', maxw=190,
        tag='kb:l2')
lc.text(KB_X + 178, KB_Y + 86, 'Bc = ⌈M/4d⌉(Alg.1 line 1,', 8.5, '#334155', 'start', maxw=190,
        tag='kb:l3')
lc.text(KB_X + 178, KB_Y + 102, '工程上取 {64,128})', 8.5, '#334155', 'start', maxw=190,
        tag='kb:l4')
lc.text(KB_X + 14, KB_Y + 118, '接口断言(三断言之一,逐字):', 8.7, lc.C_TXT, 'start', True,
        maxw=KB_W - 28, tag='kb:as1')
lc.text(KB_X + 14, KB_Y + 135, 'assert block_table is None or seqused_k is not None', 8,
        lc.C_GPU_S, 'start', True, maxw=KB_W - 28, tag='kb:as2')
lc.text(KB_X + 14, KB_Y + 152, '——片在池里,长度必须按请求给(seqused_k)', 8.5, '#334155',
        'start', maxw=KB_W - 28, tag='kb:as3')
# 池 → kernel 箭头
lc.seg(900, POOL_Y + BLK_H + 2, 900, KB_Y - 2, lc.C_GPU_S, 2.2, 'std')
lc.text(908, 240, '逐块取片', 8.5, lc.C_GPU_S, 'start', True, maxw=90, tag='kb:fetch')
lc.text(908, 254, '(读侧只读)', 8, lc.C_MUTE, 'start', maxw=90, tag='kb:fetch2')

# ---------------- 右:名字撞车警示 ----------------
NC_X, NC_W = 1040, 400
lc.rect(NC_X, 300, NC_W, 172, '#ffffff', lc.C_ENG_S, rx=8, sw=1.5)
lc.text(NC_X + 14, 320, '别被『块』字骗了:两种『块』互不相干', 10, lc.C_ENG_S, 'start', True,
        maxw=NC_W - 28, tag='nc:t')
lc.rect(NC_X + 14, 334, 372, 44, '#f0fdf4', lc.C_GPU_S, rx=6, sw=1.3)
lc.text(NC_X + 24, 351, 'SRAM tile 的块:B_c = 64/128 token', 8.7, '#166534', 'start', True,
        maxw=354, tag='nc:c1')
lc.text(NC_X + 24, 368, '按片上容量/算力切——HBM→SRAM 按它搬', 8.2, '#334155', 'start',
        maxw=354, tag='nc:c1b')
lc.rect(NC_X + 14, 386, 372, 44, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.3)
lc.text(NC_X + 24, 403, '页池的块:16 token', 8.7, lc.C_KV_S, 'start', True, maxw=354,
        tag='nc:c2')
lc.text(NC_X + 24, 420, '按池寻址粒度切——池内寻址按它走', 8.2, '#334155', 'start', maxw=354,
        tag='nc:c2b')
lc.text(NC_X + 14, 450, '粒度独立:一个 tile 可跨多页取片,一页也可被多个 tile 读', 8.2,
        lc.C_BEAT_T, 'start', True, maxw=NC_W - 28, tag='nc:l')
lc.text(NC_X + 14, 466, 'arXiv:2205.14135 Alg.1 line 1 vs vllm/config/cache.py:L47', 7.5,
        lc.C_FAINT, 'start', maxw=NC_W - 28, tag='nc:f')

# ---------------- 槽位换算条 + 写侧 ----------------
SY2 = 494
lc.rect(MX, SY2, 900, 56, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(MX + 14, SY2 + 19, '槽位换算(GPU 上由 Triton kernel _compute_slot_mapping_kernel 算):slot = block_table[req][pos // block_size] × block_size + pos % block_size',
        8.7, lc.C_TXT, 'start', True, maxw=872, tag='sf:t')
lc.text(MX + 14, SY2 + 40, '逐字:slot_ids = block_numbers * block_size + slot_offsets(vllm/v1/worker/block_table.py:L440)——读侧寻址与写侧 scatter 用同一张页表',
        8, lc.C_MUTE, 'start', maxw=872, tag='sf:v')
lc.rect(980, SY2, 460, 56, '#ffffff', '#94a3b8', rx=8, sw=1.2, dash=True)
lc.text(994, SY2 + 19, '写侧(slot_mapping scatter)不在本章:回指 ch19(捕获时的固定地址)· 预告 ch22', 8.5,
        '#475569', 'start', True, maxw=432, tag='ws:t')
lc.text(994, SY2 + 40, '虚线 = 后续章节内容(预告),本章只画读侧', 8, '#94a3b8', 'start',
        maxw=432, tag='ws:l')

# ---------------- 页脚 ----------------
FY = SY2 + 78
lc.text(MX, FY, '图例:青 = KV 页池(调度显存账本列) · 绿 = kernel / GPU 执行臂 · 灰虚线 = 预告后续章节(ch22) · 橙框 = 概念澄清(两种『块』)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, FY + 18, 'vllm/config/cache.py:L47(DEFAULT_BLOCK_SIZE=16)· vllm/v1/worker/block_table.py:L430-L440(槽位换算)· vllm/vllm_flash_attn/flash_attn_interface.py:L270-L278(三断言)· flash_attn.py:L82-L84(kernel block_size 须为 16 的倍数)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, FY + 36, '页池与 PagedAttention:arXiv:2309.06180 §2.2(归 ch13 已讲,本章只引用不重讲)· 论文类比原话 blocks as pages, tokens as bytes, requests as processes · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch20-fig-paged-kv-read.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
