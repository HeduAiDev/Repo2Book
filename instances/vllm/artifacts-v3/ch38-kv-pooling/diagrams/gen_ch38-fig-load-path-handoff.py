#!/usr/bin/env python3
"""ch38 机制图 m8 · load-path-handoff（figure_spec ch38-fig-load-path-handoff，模板 flow）

放大自 L2 站 9（⑦ 加载·接 ch16 异步路径）· L0：外部池→GPU 池回流通道（接 ch16 站 9-10
的到货结算提升；请求侧等待停在站 5-6 的 WAITING_FOR_REMOTE_KVS）。ch16 < ch38 → 全部用「回指」措辞（exp-2026-07-18-04）。

claim：加载路径五步：命中 keys → manager.prepare_load（ref_cnt+1 防逐）→ dst GPU 块打包
GPULoadStoreSpec（group_sizes/block_indices 支持半块对齐）→ load job 随 meta 过线
submit_load（CPU→GPU DMA）→ finished_recving 接 ch16 提升路径（补缓存+全命中退一 token）。

数字全部取自 spec.numbers（实测 + pin 源码锚点，逐字核对）：
  · 16 token 命中 → load job 1：src CPU 槽 [0,1,2,3] → dst GPU 块 [4,5,6,7]
  · 提交序 [(0,store),(1,load)]（store 先于 load）、finished_recving={r2}、dst 与 CPU 槽逐字节相等
  · GPULoadStoreSpec 契约：group_sizes 分组、block_indices 记逻辑起点（skip=1 偏移 [300,400,500] 实测）
  · prepare_load 钉 ref_cnt；完成信号 load 发 finished_recving（store 不发 finished_sending）
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
lc.text(MX, 36, '命中之后不是「复制粘贴」而是「挂号领床位」：加载路径五步把货送到 ch16 的站台', 16.5,
        lc.C_TXT, 'start', True, maxw=1120, tag='title')
lc.text(MX, 60, '先把 CPU 侧的块钉住（防驱逐），再在 GPU 池按块表发床位号（offload 块更大时还能半块对齐地从床中间躺下），搬完才递条子',
        10.5, lc.C_MUTE, 'start', maxw=1120, tag='subtitle')
_ch = '放大自 L2 站 9（⑦ 加载·接 ch16 异步路径）· L0：外部池→GPU 池回流'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 五站横排（右缘留 150px 通道给 ⑤ 的虚线出口） ----------------
ST_Y, ST_H = 110, 260
ST_W = (BXR - MX - 4 * 22 - 150) / 5
stations = [
    ('①', '查命中', lc.C_KV_S, lc.C_KV_F, [
        '· 同前缀新请求查池',
        '· num_hit = 16 token',
        '· load_async = True',
        '· CPU 池成了「第二个',
        '   前缀缓存」',
    ], 'offloading/scheduler.py:L816-L871'),
    ('②', '钉块（防逐）', lc.C_ENG_S, lc.C_ENG_F, [
        '· manager.prepare_load',
        '· ref_cnt +1：加载保护期',
        '   内不可逐',
        '· 对照：未钉时可逐 0 →',
        '   驱逐失败 None',
    ], 'cpu/manager.py:L129-L146'),
    ('③', '发床位', lc.C_ENG_S, lc.C_ENG_F, [
        '· dst GPU 块打包成',
        '   GPULoadStoreSpec',
        '· dst = [4, 5, 6, 7]',
        '· group_sizes 分组 ·',
        '   block_indices 记逻辑起点',
    ], 'kv_offload/base.py:L409-L440'),
    ('④', 'DMA 回流', lc.C_GPU_S, lc.C_GPU_F, [
        '· load job 1 随 meta 过线',
        '   → worker submit_load',
        '· src CPU 槽 [0,1,2,3]',
        '   → dst GPU 块 [4,5,6,7]',
        '· 提交序 [(0,store),(1,load)]',
        '   ——store 恒先于 load',
    ], 'offloading/worker.py:L326-L381'),
    ('⑤', '递条子', lc.C_ZMQ_S, '#f5f3ff', [
        '· finished_recving = {r2}',
        '· 请求停在 ch16 站 5-6 的',
        '   WAITING_FOR_REMOTE_KVS',
        '· 提升轨道 = ch16 站 9-10：',
        '   补缓存 + 全命中退一',
        '   token 重算',
    ], 'scheduler.py:L2635-L2676（ch16 已立）'),
]
for i, (num, name, color, fill, lines, foot) in enumerate(stations):
    x = MX + i * (ST_W + 22)
    dash = (i == 4)
    lc.rect(x, ST_Y, ST_W, ST_H, fill, color, rx=8, sw=1.5, dash=dash)
    lc.text(x + 12, ST_Y + 22, num + ' ' + name, 10.5, color, 'start', True,
            maxw=ST_W - 24, tag='st%d:t' % i)
    for j, ln in enumerate(lines):
        lc.text(x + 12, ST_Y + 44 + j * 16.5, ln, 8.6, '#334155', 'start',
                maxw=ST_W - 22, tag='st%d:l%d' % (i, j))
    lc.text(x + 12, ST_Y + ST_H - 10, foot, 7.6, lc.C_FAINT, 'start',
            maxw=ST_W - 20, tag='st%d:f' % i)
    if i < 4:
        ax0 = x + ST_W
        ax1 = x + ST_W + 22
        ay = ST_Y + 26
        lc.seg(ax0, ay, ax1, ay, stations[i + 1][2], 2.0, 'std')
# 站间流转一行（在行下方，替代逐箭头小标——22px 缝里放不下文字）
lc.text(MX, ST_Y + ST_H + 16,
        '站间流转：①→② 命中 keys · ②→③ 块已钉住 · ③→④ load job 随 meta 过线 · ④→⑤ DMA 完成',
        8.8, lc.C_MUTE, 'start', maxw=BXR - MX, tag='seq:flow')

# 半块对齐小图标（挂在③框内下方）
x3 = MX + 2 * (ST_W + 22)
icy = ST_Y + 150
lc.rect(x3 + 12, icy, ST_W - 24, 36, '#ffffff', lc.C_GPU_S, rx=5, sw=1.0)
for k in range(4):
    lc.rect(x3 + 18 + k * ((ST_W - 36) / 4), icy + 7, (ST_W - 36) / 4 - 4, 22,
            '#dcfce7' if k else '#bbf7d0', lc.C_GPU_S, rx=3, sw=1.0)
lc.text(x3 + ST_W / 2, icy + 58, '半块对齐：offload 块 > GPU 块时', 7.8,
        lc.C_GPU_S, 'middle', maxw=ST_W - 10, tag='halfalign1')
lc.text(x3 + ST_W / 2, icy + 71, '首块可从块中部起读', 7.8, lc.C_GPU_S, 'middle',
        maxw=ST_W - 10, tag='halfalign2')

# ⑤ 之后的虚线接出画面右缘（回指 ch16，不重画）
EX_X = MX + 4 * (ST_W + 22) + ST_W
lc.seg(EX_X, ST_Y + 26, BXR - 4, ST_Y + 26, lc.C_ZMQ_S, 1.8, 'std', dash=True)
lc.text(BXR - 8, ST_Y + 12, '→ ch16 站 9-10 提升', 8.5, lc.C_ZMQ_S, 'end', maxw=145, tag='exit1')
lc.text(BXR - 8, ST_Y + 46, '（已立·本章消费）', 8, lc.C_ZMQ_S, 'end', maxw=145, tag='exit2')

# ---------------- 字节证据条 ----------------
EV_Y = ST_Y + ST_H + 24
lc.rect(MX, EV_Y, BXR - MX, 56, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.2)
lc.text(MX + 16, EV_Y + 22, '字节级回程验证：dst 4 块与 CPU 槽逐字节相等（校验和 24576 / 25088 / 25600 / 26112 = 512B × 48..51）——先存出去、再原样搬回来',
        9.5, lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='ev:t')
lc.text(MX + 16, EV_Y + 42, '完成信号：load 发 finished_recving；store 永不发 finished_sending（只在账本上销号——完成计数通道）',
        8.8, lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='ev:s')

# ---------------- 结论 + 页脚 ----------------
CONC_Y = EV_Y + 56 + 24
lc.text(MX, CONC_Y,
        '图注结论：命中只走半程——「换回来」的另一半是 ch16 立好的异步等待与提升机器，本章只负责把货送到站台。',
        10.5, lc.C_TXT, 'start', True, maxw=BXR - MX, tag='conc')
FT_Y = CONC_Y + 22
lc.text(MX, FT_Y, '逐字锚 offloading/scheduler.py:L873-L970（update_state_after_alloc 五步）· offloading/worker.py:L326-L381（submit_load 与完成信号）· '
                  'kv_offload/base.py:L409-L440（GPULoadStoreSpec 契约）· v1/core/sched/scheduler.py:L2635-L2676（ch16 提升链，回指）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FT_Y + 15, '行号基线 vLLM v0.27.1 · ①②③ 调度器进程（橙）· ④ worker 进程 DMA（绿）· ⑤ 接 ch16 轨道（虚线紫）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FT_Y + 34
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch38-fig-load-path-handoff.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
