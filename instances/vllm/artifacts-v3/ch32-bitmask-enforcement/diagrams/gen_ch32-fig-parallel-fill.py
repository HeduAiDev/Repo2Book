#!/usr/bin/env python3
"""ch32 机制图 · 并行填行的任务几何（figure_spec ch32-fig-parallel-fill，模板 tiling）

放大自 L0 采样列·结构化输出组『批装配』段的并行分支展开（L2 站 3→4 之间的算法放大；
串行分支归 spec 窗口图），架构归属回指 L2 章图，不另立第二种架构画法（FIGURE-SYSTEM §3）。

claim：129 行按 16 行一任务切成 8 摞满批+1 摞尾批、提交最多 8 线程的专用池——行间各写
各行是可并行的全部结构性理由，阈值 128 与建池前提 max_num_seqs>128 是两个不同的 128。

数字全部取自 figure_spec.numbers（129 请求 → 9 任务 8×16+1 / 恰 128 不触池 /
max_num_seqs=128 不建池·129 建池 / 串行 0.824ms vs 并行 2.138ms（GIL 持有，tvm-ffi DLL
零释放符号）/ workers=max(1, min(cpu//2, 8)) 本机 8 / 构造锚 __init__.py:L60-L68）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 900
MX, BXR = 60, 1440

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>')


def fp_color(i):
    """行 i 允许 token i%64：色阶随指纹 0..63 走、64 行一循环。"""
    v = (i % 64) / 63.0
    return f'#{int(0x18 + v * 0x28):02x}{int(0x50 + v * 0x48):02x}{int(0x34 + v * 0x38):02x}'


# ---------------- 标题区 ----------------
lc.text(MX, 34, '并行填行的任务几何：129 行 → 9 个任务（8 摞满批 ×16 行 + 1 摞尾批）→ 8 线程专用池', 16.5,
        lc.C_TXT, 'start', True, maxw=1180, tag='title')
lc.text(MX, 58, '每请求恰一行（无 spec 时）；行区间互不相交 = 无锁可并行的全部结构性理由——但「允许并行」≠「实测加速」：'
               '本机 xgrammar 0.2.6（tvm-ffi 绑定）持有 GIL，并行实测未回本',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='subtitle')
_ch = '放大自 L0 采样列·结构化输出组批装配 · L2 站 3→4 之间的算法放大'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 任务列栅格：9 列 × 16 行（末列 1 格） =================
GX, GY = 96, 108
CELL_W, CELL_H, COL_GAP = 56, 17, 12
N_TASKS, ROWS_FULL = 9, 16
GRID_W = N_TASKS * CELL_W + (N_TASKS - 1) * COL_GAP   # 600
GRID_BOT = GY + 24 + ROWS_FULL * CELL_H + 6           # 单元格区底

lc.text(GX, GY - 12, '129 行 × 任务的切分（每列 = 一个任务，恰 16 行）', 10.5, lc.C_TXT, 'start', True,
        maxw=640, tag='g:t')

for t in range(N_TASKS):
    x = GX + t * (CELL_W + COL_GAP)
    n_rows = ROWS_FULL if t < 8 else 1
    r0 = t * ROWS_FULL
    lc.rect(x - 3, GY + 24, CELL_W + 6, n_rows * CELL_H + 6, '#ffffff',
            lc.C_SAM_S if t < 8 else lc.C_MUTE, rx=5, sw=1.4, dash=(t == 8))
    for r in range(n_rows):
        i = r0 + r
        cy = GY + 27 + r * CELL_H
        lc.ELEMS.append(((x, cy, x + CELL_W, cy + CELL_H - 1),
                         f'<rect x="{x}" y="{cy}" width="{CELL_W}" height="{CELL_H - 1}" '
                         f'fill="{fp_color(i)}" stroke="none"/>'))
    head = f'任务 {t + 1} · 16 行' if t < 8 else '任务 9（尾批）'
    lc.text(x + CELL_W / 2, GY + 14, head, 8.2 if t < 8 else 7.6,
            lc.C_SAM_S if t < 8 else lc.C_MUTE, 'middle', True, maxw=CELL_W + 22,
            tag='task' + str(t))
    lc.text(x + CELL_W / 2, GY + 24 + n_rows * CELL_H + 16,
            f'{r0}..{r0 + n_rows - 1}', 7.4, lc.C_FAINT, 'middle', maxw=CELL_W + 14,
            tag='rng' + str(t))

# 指纹色带（64 格一循环，恰覆盖 4 个满批列）
FS_Y = GRID_BOT + 40
fw = GRID_W / 64.0
for j in range(64):
    lc.ELEMS.append(((GX + j * fw, FS_Y, GX + (j + 1) * fw, FS_Y + 14),
                     f'<rect x="{GX + j * fw:.1f}" y="{FS_Y}" width="{fw + 0.5:.1f}" height="14" '
                     f'fill="{fp_color(j)}" stroke="none"/>'))
for j, lab in [(0, 'token 0'), (16, '16'), (32, '32'), (63, 'token 63')]:
    lc.text(GX + j * fw + (fw if j == 63 else 0), FS_Y + 28, lab, 7.4, lc.C_FAINT,
            'middle' if j not in (0, 63) else ('start' if j == 0 else 'end'),
            maxw=60, tag='fs' + str(j))
lc.text(GX + GRID_W + 14, FS_Y + 8, '格子色阶 = 行 i 允许 token i%64', 8, lc.C_MUTE, 'start',
        maxw=300, tag='fs:cap')
lc.text(GX + GRID_W + 14, FS_Y + 22, '（64 行一循环；129 行逐行核验全对）', 7.6, lc.C_MUTE, 'start',
        maxw=300, tag='fs:cap2')
lc.text(GX, FS_Y + 46, '129 = 8×16 + 1：8 摞满批加 1 摞尾批——切分本身给出两两不相交的行区间',
        8.4, lc.C_MUTE, 'start', maxw=700, tag='g:foot')

# ================= 线程池框（右上） =================
PX, PY, PW, PH = 1000, 108, 400, 262
lc.rect(PX, PY, PW, PH, lc.C_BEAT_F, lc.C_BEAT_S, rx=10, sw=2.0)
lc.text(PX + 16, PY + 22, 'executor_for_fillmask —— 专用线程池', 10.5, lc.C_BEAT_T, 'start', True,
        maxw=PW - 32, tag='p:t')
lc.text(PX + 16, PY + 40, '与编译线程池是两个池（各自独立）', 8, lc.C_MUTE, 'start',
        maxw=PW - 32, tag='p:s')
SLOT_W, SLOT_H, SLOT_GAP = (PW - 32 - 7 * 6) / 8, 62, 6
for k in range(8):
    x = PX + 16 + k * (SLOT_W + SLOT_GAP)
    lc.rect(x, PY + 54, SLOT_W, SLOT_H, '#ffffff', lc.C_BEAT_S, rx=5, sw=1.2)
    lc.text(x + SLOT_W / 2, PY + 54 + 25, f'W{k + 1}', 8.2, lc.C_BEAT_T, 'middle', True,
            maxw=SLOT_W, tag='w' + str(k))
    lc.text(x + SLOT_W / 2, PY + 54 + 44, '线程', 7.2, lc.C_MUTE, 'middle', maxw=SLOT_W,
            tag='wq' + str(k))
lc.text(PX + 16, PY + 138, 'workers = max(1, min(cpu//2, 8))，本机 = 8', 8.4, lc.C_BEAT_T,
        'start', True, maxw=PW - 32, tag='p:f1')
lc.text(PX + 16, PY + 156, 'promise.result() 收口：批次边界上等齐全部 9 个任务', 8,
        '#334155', 'start', maxw=PW - 30, tag='p:f2')
lc.text(PX + 16, PY + 174, '每任务只写自己的行区间 [16k, 16k+16)：互不相交', 8,
        '#334155', 'start', maxw=PW - 30, tag='p:f3')
lc.text(PX + 16, PY + 192, '→ 无共享可变状态、无锁（正确性边界 = 区间划分）', 8,
        '#334155', 'start', maxw=PW - 30, tag='p:f4')
lc.text(PX + 16, PY + PH - 12, 'vllm/v1/structured_output/__init__.py:L242-L271（填行）· L60-L68（建池）',
        7.6, lc.C_FAINT, 'start', maxw=PW - 30, tag='p:file')

# 汇交提交箭头：栅格整体 → 池
ARW_Y = PY + PH / 2
lc.parrow([(GX + GRID_W + 10, ARW_Y), (PX - 8, ARW_Y)], lc.C_BEAT_S, 2.2, 'std')
lc.text((GX + GRID_W + PX) / 2, ARW_Y - 26, 'executor.submit ×9', 8.8, lc.C_BEAT_T, 'middle', True,
        maxw=200, tag='arw:t')
lc.text((GX + GRID_W + PX) / 2, ARW_Y - 12, '（每任务一个 Future）', 7.6, lc.C_MUTE, 'middle',
        maxw=200, tag='arw:s')

# ================= 右中：128 的两义 =================
NY, NH = PY + PH + 16, 150
lc.rect(PX, NY, PW, NH, '#ffffff', lc.C_MUTE, rx=8, sw=1.2, dash=True)
lc.text(PX + 14, NY + 20, '「128」的两义（两个不同的 128）', 9.4, lc.C_ENG_S, 'start', True,
        maxw=PW - 28, tag='b:t')
for j, ln in enumerate(['· 分支阈值：本步请求数 > 128 且 num_spec==0 才进并行',
                        '  （恰 128 请求 = 串行、129 才并行）——一步的批',
                        '· 建池前提：max_num_seqs > 128（部署上限）——',
                        '  max_num_seqs=128 不建池、129 建池：小部署该分支是死代码',
                        '· 串行分支独占 spec 场景：同 grammar 推进/回滚须按草稿序']):
    lc.text(PX + 14, NY + 38 + j * 16.5, ln, 8.2, '#334155', 'start', maxw=PW - 26,
            tag='b:l' + str(j))

# ================= 右下：实测未回本（诚实条） =================
NY2 = NY + NH + 14
NH2 = 174
lc.rect(PX, NY2, PW, NH2, '#fef2f2', lc.C_ABORT, rx=8, sw=1.3)
lc.text(PX + 14, NY2 + 20, '实测未回本（取证环境，正文须挑明）', 9.4, lc.C_ABORT, 'start', True,
        maxw=PW - 28, tag='t:t')
for j, ln in enumerate(['· 计时（256 行 × 50257 词表，本机实测中位）：',
                        '  串行 0.824ms vs 并行 2.138ms（speedup 0.39）',
                        '· 放大 40 倍工作量仍 0.51x；DLL 逐字节扫描',
                        '  gil_scoped_release 符号 0 命中（tvm-ffi 绑定）',
                        '  ⇒ fill_next_token_bitmask 持有 GIL',
                        '· 结论：并行分支的存在理由是结构许可（行间独写），',
                        '  实际收益依赖后端释放 GIL 与单行成本——不得写成',
                        '  已验证加速']):
    lc.text(PX + 14, NY2 + 38 + j * 16.5, ln, 8.2, '#334155', 'start', maxw=PW - 26,
            tag='t:l' + str(j))

# ================= 左下：为什么允许并行（阅卷分工） =================
QY, QH = FS_Y + 66, 240
lc.rect(GX, QY, GRID_W, QH, '#ffffff', lc.C_SAM_S, rx=9, sw=1.4)
lc.text(GX + 16, QY + 22, '为什么允许并行：阅卷分工（无共享可变状态）', 10, lc.C_SAM_S, 'start', True,
        maxw=GRID_W - 32, tag='q:t')
QLINES = [
    '· 129 份卷子互不批注：每个任务只写自己的 16 行区间；',
    '  唯一共享对象 _grammar_bitmask 是行主序连续张量——',
    '  不同行在内存上不相交，torch 行写无跨行伪共享语义',
    '· 正确性不需要锁；但「线程安全」≠「有真并行」：',
    '  GIL 是否释放由后端实现决定（右下红框实测）',
    '· 卷子太少或批改太快时，发卷收卷（提交/收割）开本',
    '  反超一个人全改完——轻语法单行仅 3.22µs',
]
for j, ln in enumerate(QLINES):
    lc.text(GX + 16, QY + 44 + j * 18, ln, 8.4, '#334155', 'start', maxw=GRID_W - 30,
            tag='q:l' + str(j))
lc.seg(GX + 16, QY + 44 + len(QLINES) * 18 + 6, GX + GRID_W - 16, QY + 44 + len(QLINES) * 18 + 6,
       '#e2e8f0', 1.0)
lc.text(GX + 16, QY + 44 + len(QLINES) * 18 + 26,
        '区间两两不相交是「能并行」的全部理由；', 8.4, lc.C_MUTE, 'start', maxw=GRID_W - 30,
        tag='q:f1')
lc.text(GX + 16, QY + 44 + len(QLINES) * 18 + 44,
        '是否值得并行是另一个问题（右下红框）', 8.4, lc.C_MUTE, 'start', maxw=GRID_W - 30,
        tag='q:f2')

# ================= 图例 + 页脚 =================
LY = QY + QH + 30
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, '#ffffff', lc.C_SAM_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '满批任务（16 行）', 8.8, lc.C_TXT, 'start', maxw=160, tag='lg1')
lx0 += 21 + lc.tw('满批任务（16 行）', 8.8) + 16
lc.rect(lx0, LY - 9, 16, 11, '#ffffff', lc.C_MUTE, rx=3, sw=1.2)
lc.text(lx0 + 21, LY + 1, '尾批任务（虚线，1 行）', 8.8, lc.C_TXT, 'start', maxw=190, tag='lg2')
lx0 += 21 + lc.tw('尾批任务（虚线，1 行）', 8.8) + 16
lc.rect(lx0, LY - 9, 16, 11, lc.C_BEAT_F, lc.C_BEAT_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '线程池（半 CPU、上限 8）', 8.8, lc.C_TXT, 'start', maxw=200, tag='lg3')
lx0 += 21 + lc.tw('线程池（半 CPU、上限 8）', 8.8) + 16
lc.ELEMS.append(((lx0, LY - 11, lx0 + 22, LY + 4),
                 f'<rect x="{lx0}" y="{LY - 11}" width="22" height="15" fill="{fp_color(32)}" '
                 f'stroke="{lc.C_MUTE}" stroke-width="0.8" rx="2"/>'))
lc.text(lx0 + 27, LY + 1, '格子色阶 = 允许 token i%64', 8.8, lc.C_TXT, 'start', maxw=220, tag='lg4')

lc.text(MX, LY + 28, 'vllm/v1/structured_output/__init__.py:L242-L271（并行填行：16 个一任务提交 executor_for_fillmask）'
                     '· L60-L68（workers=max(1, min(cpu//2, 8))；max_num_seqs>128 才建池）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LY + 46, '任务切分 / 行内容 129 行逐行核验 / 边界（恰 128 串行、max_num_seqs 128/129 建池） / 计时 0.824 vs 2.138ms（GIL 持有）'
                     ' ＝ 本章驱动脚本实测（xgrammar 0.2.6，host 20 核）· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch32-fig-parallel-fill.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
