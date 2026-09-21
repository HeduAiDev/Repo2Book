#!/usr/bin/env python3
"""ch14 机制图 · KV cache 初始化自顶向下全链调用树（figure_spec ch14-fig-kvcache-init-chain，模板 flow/layout）

用户点名的「完整调用关系架构图」：EngineCore._initialize_kv_caches 八步编排
（注册→广播收 spec→non_causal 旁路→profile→四路分发定账→拍平喂调度器→
worker 双路径分配→切片绑定），竖向主干 + 关键分支侧挂，每站一行职责、每站
file:line 锚。与既有 ch14-fig-init-call-panorama（站 7 一份账喂两侧的调用全景）
互补：那张聚焦「账从哪来、喂到哪两侧」，本张是覆盖收集→定账→落地的全链主干。

claim：KV cache 初始化是一条自顶向下的直线调用链——模型差异被 spec 申报面
整段挡在收集站，下游零模型分支。

数字/锚点全部取自 figure_spec.numbers（provenance = 初始化调用树补充档案对
pin 源码逐行核验；61 层实算 = pin 算法实跑复算）。坐标由常量/游标计算；文本
全 esc()。
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

# ---------------- 画布与骨架常量 ----------------
W, MX, BXR = 1500, 60, 1440
TX, TW = MX, 780                      # 左侧主干列
PX, PW = 872, BXR - 872               # 右侧放大列（568）
AMBER = '#b45309'                     # 申报面闸门（与既有图的设计证据标注同色）
C_EN, C_EN_F = lc.C_ENG_S, lc.C_ENG_F
C_KV, C_KV_F = lc.C_KV_S, lc.C_KV_F
C_GPU, C_GPU_F = lc.C_GPU_S, lc.C_GPU_F
C_RPC = lc.C_ZMQ_S                    # executor 广播层 = 紫（IPC 色同 L0）
BADGE_FS = 8.5
GAP = 16                              # 主干站间箭头段高

EXTRA_DEFS = ('<defs>'
              f'<marker id="en" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              f'markerHeight="4.6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{C_EN}"/></marker>'
              '</defs>')


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
        while lc.tw(word, fs, bold) > maxw:
            cut = len(word)
            while cut > 1 and lc.tw(word[:cut], fs, bold) > maxw:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        cur = word
    if cur:
        lines.append(cur)
    return lines


def station(y, num, name, duty, anchor, stroke, fill='#ffffff', badge=None,
            anchor2=None, anchor2c=None):
    """主干站框：编号圆徽 + 站名（粗）+ 职责行（自动换行）+ 锚点行（灰）。
    anchor2 = 第二行锚点（可换色，如 executor 包装行用紫）。返回框底 y。"""
    inner = TW - 86                          # 让开左侧编号徽的缩进
    duty_lines = wrap(duty, 9.2, inner)
    h = 26 + len(duty_lines) * 15 + 15 * (1 + (1 if anchor2 else 0)) + 8
    lc.rect(TX, y, TW, h, fill, stroke, rx=8, sw=1.5)
    # 编号圆徽（骑框左缘）
    lc.circle(TX + 20, y + h / 2, 13, stroke, sw=1.6, dash=False)
    lc.text(TX + 20, y + h / 2 + 4, num, 11.5, stroke, 'middle', True, tag='num:' + num)
    tx = TX + 46
    lc.text(tx, y + 21, name, 11.2, lc.C_TXT, 'start', True, maxw=TW - 100, tag='st:' + name[:14])
    cy = y + 38
    for ln in duty_lines:
        lc.text(tx, cy, ln, 9.2, '#334155', 'start', maxw=inner, tag='d:' + ln[:16])
        cy += 15
    lc.text(tx, cy + 2, anchor, 8.4, lc.C_FAINT, 'start', maxw=TW - 60, tag='a:' + anchor[:18])
    if anchor2:
        lc.text(tx, cy + 17, anchor2, 8.4, anchor2c or lc.C_FAINT, 'start',
                maxw=TW - 60, tag='a2:' + anchor2[:18])
    if badge:
        bw = 14 + 9.0 * len(badge)
        lc.rect(TX + TW - bw - 6, y - 8, bw, 17, lc.C_BADGE_F, C_EN, rx=8, sw=1.0)
        lc.text(TX + TW - bw / 2 - 6, y + 3, badge, BADGE_FS, C_EN, 'middle', True, tag='bg:' + badge)
    return y + h


def panel(y, title, anchor, rows, stroke, fill='#ffffff', badge=None):
    """右侧放大面板：题头 + 锚点 + 内容行（自动换行）。返回框底 y。"""
    inner = PW - 28
    laid = []
    for t, c, bold, ind in rows:
        for ln in wrap(t, 8.8, inner - ind, bold):
            laid.append((ln, c, bold, ind))
    h = 30 + len(laid) * 14.5 + 12
    lc.rect(PX, y, PW, h, fill, stroke, rx=8, sw=1.4, dash=True)
    lc.text(PX + 14, y + 19, title, 10.4, stroke, 'start', True, maxw=inner - 50, tag='pt:' + title[:12])
    if badge:
        bw = 14 + 9.0 * len(badge)
        lc.rect(PX + PW - bw - 6, y - 8, bw, 17, lc.C_BADGE_F, C_EN, rx=8, sw=1.0)
        lc.text(PX + PW - bw / 2 - 6, y + 3, badge, BADGE_FS, C_EN, 'middle', True, tag='pbg:' + badge)
    lc.text(PX + 14, y + 35, anchor, 8.0, lc.C_FAINT, 'start', maxw=inner, tag='pa:' + anchor[:16])
    cy = y + 51
    for ln, c, bold, ind in laid:
        lc.text(PX + 14 + ind, cy, ln, 8.8, c, 'start', bold, maxw=inner - ind, tag='pr:' + ln[:14])
        cy += 14.5
    return y + h


def vseg(y_from, y_to, color=C_EN, sw=2.0, marker='en'):
    """主干站间竖直箭头（框底 → 下一框顶）。"""
    lc.seg(TX + 46, y_from, TX + 46, y_to, color, sw, marker)


def link_to_panel(y_station, y_panel_mid):
    """主干站右缘 → 放大面板左缘 的虚线肘形连接。"""
    mid_x = (TX + TW + PX) / 2
    lc.parrow([(TX + TW, y_station), (mid_x, y_station), (mid_x, y_panel_mid), (PX, y_panel_mid)],
              lc.C_MUTE, 1.3, 'std', dash=True)


# ================= 标题区 =================
lc.text(MX, 34, 'KV cache 初始化全链：一条自顶向下的直线调用链', 17, lc.C_TXT, 'start', True,
        maxw=1050, tag='title')
lc.text(MX, 58, 'EngineCore._initialize_kv_caches 八步编排——注册→广播收 spec→non_causal 旁路→profile→四路分发定账→'
                '拍平喂调度器→worker 双路径分配→切片绑定；每站一行职责、每站 file:line 锚',
        10.5, lc.C_MUTE, 'start', maxw=1330, tag='sub1')
lc.text(MX, 75, '读图：左列主干自上而下即调用序（橙=编排与检查 core.py·紫=executor 广播·青=定账 kv_cache_utils·绿=worker 侧）；'
                '右列虚线相连的是该站的放大；琥珀虚线=spec 申报面闸门',
        9.0, lc.C_MUTE, 'start', maxw=1330, tag='sub2')
_ch = '放大自 L0 · 启动装配带 × KV 账本列'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 主干：入口 + 八站 =================
y = 100
y_root_bot = station(y, '⌂', 'EngineCore._initialize_kv_caches —— 总编排',
                     '八步在 L250-L332 里直线排开：无回跳、无部分初始化状态，任何一站抛错即中止启动',
                     'vllm/v1/engine/core.py:L250-L332', C_EN, C_EN_F)
y = y_root_bot + GAP
vseg(y_root_bot, y)

# ---- 收集段题头 ----
lc.text(TX + 46, y + 13, '收集段（①—④）：把「每种缓存归谁管、每层要多少」收上来', 9.6, C_EN, 'start', True,
        maxw=TW - 60, tag='ph1')
y += 22

y1 = station(y, '①', '注册菜单 register_all_kvcache_specs',
             'engine-core 进程注册 11 项内置 spec→manager 映射，再加平台钩子——先公示「每种缓存归谁管」的菜单',
             'vllm/v1/engine/core.py:L254 · vllm/v1/core/single_type_kv_cache_manager.py:L1881-L1942', C_EN)
y = y1 + GAP
vseg(y1, y)

y2 = station(y, '②', '广播收 spec model_executor.get_kv_cache_specs()',
             '广播到每个 worker：每个注意力模块经收集面三分支自报形状（放大见右上）——模型差异的唯一入口',
             'vllm/v1/engine/core.py:L257',
             C_EN, badge='站 4',
             anchor2='executor 两层包装：vllm/v1/executor/abstract.py:L149-L150 = collective_rpc("get_kv_cache_spec")',
             anchor2c=C_RPC)
y = y2 + GAP

# ---- 放大面板 A（收集面三分支，右侧，与 ② 同段） ----
pa_y = y1 - 6
pa_bot = panel(pa_y, '② 的放大 · 收集面三分支',
               'GPUModelRunner.get_kv_cache_spec · gpu_model_runner.py:L7800-L7837',
               [('分支一 EC 专职 producer：不分配本地 KV，直接返回空表', '#334155', False, 0),
                ('分支二 KV 共享层：登记进共享表、不为本层建 spec', '#334155', False, 0),
                ('分支三 逐层自报：每个注意力模块自己申报缓存格式', '#334155', False, 0),
                ('枚举对象是 AttentionLayerBase 全体子类——新缓存组件进 forward_context 就会被收集', lc.C_MUTE, False, 0)],
               C_GPU, badge='站 4')
link_to_panel(y2 - 24, pa_y + (pa_bot - pa_y) / 2)

# ---- spec 申报面闸门（横贯全宽的琥珀虚线） ----
gate_y = y + 6
lc.seg(MX - 6, gate_y, BXR + 6, gate_y, AMBER, 1.8, dash=True)
lc.text((MX + BXR) / 2, gate_y + 16, 'spec 申报面闸门 —— 模型差异到此为止：③以下各站只认 spec 自报字段，'
                                     '分组 / 布局 / 分配 / 切片零模型分支', 9.6, AMBER, 'middle', True,
        maxw=BXR - MX, tag='gate')
y = gate_y + 30

y3 = station(y, '③', 'non_causal 旁路',
             '任一层 spec 带 non_causal=True（Prefix LM 双向注意力）→ 强制关 chunked prefill 与 prefix caching'
             '——非因果 prefill 会腐蚀缓存',
             'vllm/v1/engine/core.py:L265-L279', C_EN)
y = y3 + GAP
vseg(y3, y, C_EN)

y4 = station(y, '④', 'profile 定可用 determine_available_memory',
             '广播到 worker 跑 dummy 前向，量出峰值后的剩余显存——这就是 KV 池的全部本金',
             'vllm/v1/engine/core.py:L293', C_GPU, badge='站 3')
y = y4 + GAP

# ---- 定账段题头 ----
lc.text(TX + 46, y + 13, '定账段（⑤—⑥）：单点出账，拍平与带布局两版分发', 9.6, C_KV, 'start', True,
        maxw=TW - 60, tag='ph2')
y += 22

y5 = station(y, '⑤', '定账 get_kv_cache_configs',
             '合并全 worker 的 spec → 四路分发（V4 走第三路重组）→ packed 布局 → 护栏四道 → 出 KVCacheConfig'
             '（四路放大见右）',
             'vllm/v1/engine/core.py:L304-L306 · vllm/v1/core/kv_cache_utils.py:L2094-L2242', C_KV, badge='站 6')
y = y5 + GAP
vseg(y5, y, C_KV, marker='std')

y6 = station(y, '⑥', '拍平喂调度器 generate_scheduler_kv_cache_config',
             '解包出代表 spec、写回 cache_config；block_size = min(各组 spec.block_size)——调度器只拿一份拍平的账',
             'vllm/v1/engine/core.py:L315-L321', C_KV, badge='站 7')
y = y6 + GAP

# ---- 放大面板 B（四路分发 → packed 布局，与 ⑤ 对齐） ----
pb_y = y5 - 6
pb_bot = panel(pb_y, '⑤ 的放大 · 四路分发 → packed 布局',
               'get_kv_cache_groups · kv_cache_utils.py:L1781-L1852 · 布局 L1283-L1358',
               [('路 1 全同 spec → 单组直装（绝大多数单型模型）', '#334155', False, 0),
                ('路 2 同型异宽 → 等量装包（同 token 槽位、页宽可异）', '#334155', False, 0),
                ('路 3 混合多页宽 → 组化重组 + 近似 GCD（V4 走这条）', C_KV, True, 0),
                ('路 4 一般混合 → 页宽统一（不齐则兜底组）', '#334155', False, 0),
                ('↓ packed 布局：block_stride = 最宽组组页和，每层在块内固定 offset（跨组同 offset 物理重叠）', lc.C_MUTE, False, 0)],
               C_KV, badge='站 5')
link_to_panel(y5 + 20, pb_y + (pb_bot - pb_y) / 2)

# ---- 落地段题头 ----
lc.text(TX + 46, y + 13, '落地段（⑦—⑧）：worker 进程按布局真分配、逐层切视图', 9.6, C_GPU, 'start', True,
        maxw=TW - 60, tag='ph3')
y += 22

y7 = station(y, '⑦', 'worker 双路径分配 initialize_from_config',
             '经 executor 广播落 worker：先 ensure_kv_transfer_initialized，再 initialize_kv_cache——内部 10 步，'
             '含 prepare_kernel_block_sizes（管理块 256 拆 4×64）（判据放大见右）',
             'vllm/v1/worker/gpu_worker.py:L650-L665 · vllm/v1/worker/gpu_model_runner.py:L7624-L7681',
             C_GPU, badge='站 12')
y = y7 + GAP
vseg(y7, y, C_GPU, marker='std')

y8 = station(y, '⑧', '切片绑定',
             '四步切片 view(-1, block_stride)[:, offset:offset+page].view(dtype).view(shape) 逐层切出条带视图 → '
             'bind_kv_cache 双写：runner 平铺列表 + forward_context 每层实例',
             'vllm/v1/worker/gpu/attn_utils.py:L225-L233 · vllm/v1/worker/utils.py:L466-L525', C_GPU)
y8_bot = y8
y = y8 + 18

# ---- 放大面板 C（分配双路径，与 ⑦ 对齐） ----
pc_y = y7 - 6
pc_bot = panel(pc_y, '⑦ 的放大 · 分配双路径',
               'gpu_model_runner.py:L7541-L7594 · kv_connector_model_runner_mixin.py:L116-L162',
               [('判据 use_uniform_kv_cache：优化路（整块连续，供 KV connector 跨层搬运）', '#334155', False, 0),
                ('通用路 _allocate + _reshape——V4 packed 走这条', C_GPU, True, 0),
                ('packed_backing 一次分配，组内全部张量别名同一块 slab', '#334155', False, 0),
                ('四步链：view(-1,stride) → [:,off:off+page] → .view(dtype) → .view(shape)', lc.C_MUTE, False, 0)],
               C_GPU)
link_to_panel(y7 + 20, pc_y + (pc_bot - pc_y) / 2)

# ================= 底部：实算流量 + 结论 + 图例 + 页脚 =================
by = max(y, pa_bot + 14, pb_bot + 14, pc_bot + 14) + 6
lc.rect(MX, by, BXR - MX, 74, '#ffffff', lc.C_MUTE, rx=9, sw=1.2)
lc.text(MX + 16, by + 20, '61 层 V4 一次启动的实算：61 层自报 243 条 spec → 四桶装包 → 终五组、202 个 offset 的张量声明，'
                          '全部别名同一块 slab', 10.0, lc.C_TXT, 'start', True, maxw=BXR - MX - 32, tag='b1')
lc.text(MX + 16, by + 40, '整条链零模型分支：链上每站的输入都是上一站的输出或 spec 字段——模型差异被 spec 申报面'
                          '整段挡在收集站', 9.4, '#334155', 'start', maxw=BXR - MX - 32, tag='b2')
lc.text(MX + 16, by + 58, '61 层数字 = pin 算法对 61 层配置实跑复算（层分布为社区实读口径）· 行号基线 vLLM v0.27.1',
        8.2, lc.C_FAINT, 'start', maxw=BXR - MX - 32, tag='b3')

# 图例行
LY = by + 88
lx = MX
for c, f, name in [(C_EN, C_EN_F, '编排与检查（core.py）'), (C_RPC, '#f5f3ff', 'executor 广播层'),
                   (C_KV, C_KV_F, '定账与分发（kv_cache_utils）'), (C_GPU, C_GPU_F, 'worker 侧执行')]:
    lc.rect(lx, LY - 10, 20, 13, f, c, rx=3, sw=1.4)
    lc.text(lx + 26, LY, name, 8.8, lc.C_TXT, 'start', maxw=200, tag='lg' + name[:6])
    lx += 26 + lc.tw(name, 8.8) + 20
lc.seg(lx, LY - 4, lx + 30, LY - 4, AMBER, 1.8, dash=True)
lc.text(lx + 36, LY, '申报面闸门', 8.8, lc.C_TXT, 'start', maxw=120, tag='lg:gate')
lx += 36 + lc.tw('申报面闸门', 8.8) + 20
lc.rect(lx, LY - 11, 14 + 9.0 * 3, 17, lc.C_BADGE_F, C_EN, rx=8, sw=1.0)
lc.text(lx + (14 + 9.0 * 3) / 2, LY, '站 3', BADGE_FS, C_EN, 'middle', True, tag='lg:bg')
lx += 14 + 9.0 * 3 + 8
lc.text(lx, LY, '= 本章站号（账本从诞生到把门的阅读顺序）', 8.8, lc.C_TXT, 'start', maxw=340, tag='lg:bgt')
lx += lc.tw('= 本章站号（账本从诞生到把门的阅读顺序）', 8.8) + 20
lc.seg(lx, LY - 4, lx + 30, LY - 4, C_EN, 2.0, 'en')
lc.text(lx + 36, LY, '调用方向（自上而下）', 8.8, lc.C_TXT, 'start', maxw=BXR - lx - 36, tag='lg:ar')

# ================= 装配输出 =================
H = int(LY + 24)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch14-fig-kvcache-init-chain.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
