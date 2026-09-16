#!/usr/bin/env python3
"""ch37 机制图 m9 · push-pairing-machine（figure_spec ch37-fig-push-pairing-machine，模板 flow）

放大自 L0「双实例+KV 边界」下排推模式对照（同一 KV 边界的第二种填法）、L2 章图站 12。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签；配对机器框与 L2 站 12
组件框同源视觉；两腿输入 D 侧 API 蓝 / P 侧 GPU 绿、汇合菱形与 WRITE 箭头 KV 青。

claim：推模式配对机器：D 分配即发 PUSH_REG 注册（11 键载荷，携 D 自己的坐标+块号），
P 的 nixl-push-writer 后台线程把「D 注册」与「P 完成块」两腿配对（谁先到先存表，
剥 8 位随机后缀归一匹配）后 WRITE 直写 D 预分配块——传输与 D 的排队等待重叠。

数字全部取自 figure_spec.numbers（traces 实测 + pin 源码锚点，逐字核对）：
  · 注册载荷 11 键：decode_engine_id/host/port/tp_size + remote_engine_id/host/port/tp_size
    + request_id + local_block_ids + remote_pp_size；watchdog 480s
  · writer 四收件箱（注册待发/延迟推送/完成块/淘汰）+ notif 路由；自轮询 1ms 仅在有未配对完成块时
  · 后缀剥离匹配：'req-a-1a2b3c4d'→'req-a'；精确键优先；大小写不匹配不配
  · WRITE 直写：块 [0,1] 填 3.0 → D 块校验和 384.0=P 侧；D 本地 handle=0（完成靠 P 的 notif 物化空条目上报）
  · P 侧完成=WRITE handle 轮询 DONE（done_sending）；与拉模式两源换边
  · 保活：无活请求+has_pending_push_work=True → 引擎继续步进
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1600
MX = 54
BXR = 1546
EXTRA_DEFS = ('<defs><marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" '
              'markerHeight="4.6" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker></defs>')


def diamond(cx, cy, w, h, stroke, fill='#ffffff', sw=1.6):
    d = (f'M{cx:.1f},{cy - h / 2:.1f} L{cx + w / 2:.1f},{cy:.1f} '
         f'L{cx:.1f},{cy + h / 2:.1f} L{cx - w / 2:.1f},{cy:.1f} Z')
    s = f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    lc.ELEMS.append(((cx - w / 2 - 2, cy - h / 2 - 2, cx + w / 2 + 2, cy + h / 2 + 2), s))


# ---------------- 标题区 ----------------
lc.text(MX, 36, '推模式配对机器：送货上门让传输躲开排队，代价是一整套配对机器', 16.5,
        lc.C_TXT, 'start', True, maxw=1250, tag='title')
lc.text(MX, 60, 'D 刚分到货架就寄地址（注册）、P 配好货由专职快递员线程直接写进 D 的货架——两腿谁先到先记账，凑齐一对就发货（传输不再排在 D 的排队后面）',
        10.5, lc.C_MUTE, 'start', maxw=1430, tag='subtitle')
_ch = '放大自 L2 站 12（推模式对照）· L0：双实例+KV 边界下排'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 左：两条输入腿 ----------------
IN_X, IN_W = MX, 300
# 腿 1：D 分配即注册（API 蓝）
R1_Y, R1_H = 116, 218
lc.rect(IN_X, R1_Y, IN_W, R1_H, '#ffffff', lc.C_API_S, rx=9, sw=1.8)
lc.text(IN_X + 14, R1_Y + 22, 'D 分配即注册（不等被调度）', 11, lc.C_API_S, 'start', True,
        maxw=IN_W - 28, tag='r1:t')
r1_lines = [
    '· update_state_after_alloc 暂存',
    '· build_meta 打包 PUSH_REG:',
    '· 注册载荷 11 键——',
    '  decode_engine_id/host/port/tp_size',
    '  + remote_engine_id/host/port/tp_size',
    '  + request_id + local_block_ids',
    '  + remote_pp_size（D 自己的坐标+块号）',
    '· watchdog 480s（注册后没等到 WRITE',
    '  即弃）',
]
for j, ln in enumerate(r1_lines):
    lc.text(IN_X + 14, R1_Y + 44 + j * 15.5, ln, 8.5, '#334155', 'start', maxw=IN_W - 26,
            tag='r1:l%d' % j)
lc.text(IN_X + 14, R1_Y + R1_H - 9, 'nixl/push_scheduler.py:L156-L206', 8, lc.C_FAINT, 'start',
        maxw=IN_W - 26, tag='r1:f')

# 腿 2：P 终局完成块（GPU 绿）
R2_Y, R2_H = 366, 190
lc.rect(IN_X, R2_Y, IN_W, R2_H, '#ffffff', lc.C_GPU_S, rx=9, sw=1.8)
lc.text(IN_X + 14, R2_Y + 22, 'P 终局 · 完成块入库', 11, lc.C_GPU_S, 'start', True,
        maxw=IN_W - 28, tag='r2:t')
r2_lines = [
    '· request_finished 把完成块交 writer',
    '  （delay_free=True）',
    '· 完成块 {req-1: [[0,1]]}',
    '· has_pending_push_work=True',
    '· 租约戳 30s（P 钉住块等配对）',
]
for j, ln in enumerate(r2_lines):
    lc.text(IN_X + 14, R2_Y + 44 + j * 15.5, ln, 8.5, '#334155', 'start', maxw=IN_W - 26,
            tag='r2:l%d' % j)
lc.text(IN_X + 14, R2_Y + R2_H - 9, 'nixl/push_worker.py:L210-L272', 8, lc.C_FAINT, 'start',
        maxw=IN_W - 26, tag='r2:f')

# ---------------- 中：配对机器 ----------------
MC_X, MC_W = 440, 558
MC_Y, MC_H = 116, 444
mc_mid = MC_X + MC_W / 2
lc.rect(MC_X, MC_Y, MC_W, MC_H, lc.C_KV_F, lc.C_KV_S, rx=12, sw=2.2)
lc.text(MC_X + 16, MC_Y + 24, '配对机器 · nixl-push-writer 线程（P 侧常驻）', 12.5,
        lc.C_KV_S, 'start', True, maxw=MC_W - 32, tag='mc:t')

# 四收件箱
ib_y, ib_h = MC_Y + 40, 46
ib_names = ['注册待发', '延迟推送', '完成块', '淘汰']
ib_w = (MC_W - 32 - 3 * 10) / 4
lc.text(MC_X + 16, ib_y - 6, '四收件箱 + notif 路由', 9, lc.C_TXT, 'start', True,
        maxw=240, tag='ib:t')
for i, nm in enumerate(ib_names):
    x = MC_X + 16 + i * (ib_w + 10)
    lc.rect(x, ib_y, ib_w, ib_h, '#ffffff', lc.C_KV_S, rx=6, sw=1.2)
    lc.text(x + ib_w / 2, ib_y + 20, nm, 9, lc.C_TXT, 'middle', True, maxw=ib_w - 8,
            tag='ib:%d' % i)
    lc.text(x + ib_w / 2, ib_y + 36, '收件箱', 7.5, lc.C_MUTE, 'middle', maxw=ib_w - 8,
            tag='ib:s%d' % i)
lc.text(MC_X + MC_W - 16, ib_y - 6, '自轮询 1ms 仅在有未配对完成块时（其余事件驱动）', 8,
        lc.C_MUTE, 'end', maxw=330, tag='ib:poll')

# 两张匹配表
tb_y, tb_h = MC_Y + 116, 96
tb_w = (MC_W - 32 - 12) / 2
tables = [
    ('D 注册表（先到存表）', lc.C_API_S, [
        'req-a-1a2b3c4d → 11 键载荷',
        '（地址+块号）',
        '未配对 → 留表等另一腿',
    ]),
    ('P 完成块表（先到存表）', lc.C_GPU_S, [
        'req-a-1a2b3c4d → [[0,1]]',
        '（配好货的块）',
        '未配对 → 留表等另一腿',
    ]),
]
for i, (t, color, lines) in enumerate(tables):
    x = MC_X + 16 + i * (tb_w + 12)
    lc.rect(x, tb_y, tb_w, tb_h, '#ffffff', color, rx=7, sw=1.4)
    lc.text(x + 12, tb_y + 18, t, 9, color, 'start', True, maxw=tb_w - 24, tag='tb:t%d' % i)
    for j, ln in enumerate(lines):
        lc.text(x + 12, tb_y + 38 + j * 15, ln, 8.5, '#334155', 'start', maxw=tb_w - 22,
                tag='tb:l%d_%d' % (i, j))
    # 存表箭头（收件箱 → 匹配表）
    lc.seg(x + tb_w / 2, ib_y + ib_h, x + tb_w / 2, tb_y, color, 1.4, 'std')

# 配对菱形
dia_cx, dia_cy, dia_w, dia_h = mc_mid, MC_Y + 288, 230, 68
for i in (0, 1):
    x = MC_X + 16 + i * (tb_w + 12) + tb_w / 2
    lc.parrow([(x, tb_y + tb_h), (x, dia_cy - 40), (dia_cx - (dia_w / 2 - 12) if i == 0 else dia_cx + (dia_w / 2 - 12), dia_cy - 40),
               (dia_cx - (dia_w / 2 - 12) if i == 0 else dia_cx + (dia_w / 2 - 12), dia_cy - dia_h / 2)],
              tables[i][1], 1.6, 'std')
diamond(dia_cx, dia_cy, dia_w, dia_h, lc.C_KV_S)
lc.text(dia_cx, dia_cy - 6, '两腿齐备？', 9.5, lc.C_TXT, 'middle', True, maxw=dia_w - 40,
        tag='dia:1')
lc.text(dia_cx, dia_cy + 10, '（剥后缀归一 · 精确键优先）', 8, lc.C_MUTE, 'middle',
        maxw=dia_w - 40, tag='dia:2')
# 未齐 → 回存表（回环虚线）
lc.parrow([(dia_cx + dia_w / 2, dia_cy + 14), (dia_cx + dia_w / 2 + 34, dia_cy + 14),
           (dia_cx + dia_w / 2 + 34, tb_y + tb_h / 2), (MC_X + 16 + tb_w + 12 + tb_w, tb_y + tb_h / 2)],
          lc.C_MUTE, 1.2, 'std', dash=True)
lc.text(dia_cx + dia_w / 2 + 40, dia_cy + 30, '未齐 → 留表等', 8, lc.C_MUTE, 'start',
        maxw=120, tag='loop:t')
# 齐 → _do_start_push_kv → WRITE（出机器右缘）
lc.seg(dia_cx, dia_cy + dia_h / 2, dia_cx, dia_cy + dia_h / 2 + 26, lc.C_KV_S, 1.8, 'kv')
lc.text(dia_cx, dia_cy + dia_h / 2 + 44, '_do_start_push_kv：恰一次（pop 删除性读出——同一请求不可能二次配对）',
        8.5, lc.C_KV_S, 'middle', maxw=MC_W - 32, tag='fire:t')
lc.text(mc_mid, MC_Y + MC_H - 14, '配对键 = get_base_request_id 剥 8 位十六进制随机后缀（两腿 request_id 各带一个，剥掉才配得上）',
        8, lc.C_MUTE, 'middle', maxw=MC_W - 32, tag='mc:note')

# 输入腿 → 配对机器
lc.seg(IN_X + IN_W, R1_Y + R1_H / 2, MC_X - 4, R1_Y + R1_H / 2, lc.C_API_S, 2.0, 'dn')
lc.text((IN_X + IN_W + MC_X) / 2, R1_Y + R1_H / 2 - 8, 'PUSH_REG 注册', 8.5, lc.C_API_S,
        'middle', maxw=MC_X - IN_X - IN_W - 8, tag='a1')
lc.seg(IN_X + IN_W, R2_Y + R2_H / 2, MC_X - 4, R2_Y + R2_H / 2, lc.C_GPU_S, 2.0, 'dn')
lc.text((IN_X + IN_W + MC_X) / 2, R2_Y + R2_H / 2 - 8, '完成块', 8.5, lc.C_GPU_S,
        'middle', maxw=MC_X - IN_X - IN_W - 8, tag='a2')

# ---------------- 右：WRITE 直写 D 预分配块 ----------------
OP_X, OP_W = 1064, BXR - 1064
OP_Y, OP_H = 116, 250
lc.rect(OP_X, OP_Y, OP_W, OP_H, '#ffffff', lc.C_KV_S, rx=9, sw=1.6)
lc.text(OP_X + 14, OP_Y + 22, 'WRITE 直写 D 的预分配块', 11, lc.C_KV_S, 'start', True,
        maxw=OP_W - 28, tag='op:t')
# 块格（8 格，0-1 着色 3.0）
cw_, ch_, gap = 52, 40, 5
gx0, gy = OP_X + 16, OP_Y + 44
for i in range(8):
    x = gx0 + i * (cw_ + gap)
    if i < 2:
        lc.rect(x, gy, cw_, ch_, lc.C_KV_S, 'none', rx=4, sw=0)
        lc.text(x + cw_ / 2, gy + 17, '3.0', 9, '#ffffff', 'middle', True, maxw=cw_ - 6,
                tag='op:c%d' % i)
        lc.text(x + cw_ / 2, gy + 32, '块 %d' % i, 7.5, '#cffafe', 'middle', maxw=cw_ - 6,
                tag='op:cn%d' % i)
    else:
        lc.rect(x, gy, cw_, ch_, '#e2e8f0', 'none', rx=4, sw=0)
        lc.text(x + cw_ / 2, gy + 17, '空', 9, '#94a3b8', 'middle', maxw=cw_ - 6,
                tag='op:g%d' % i)
        lc.text(x + cw_ / 2, gy + 32, '块 %d' % i, 7.5, '#94a3b8', 'middle', maxw=cw_ - 6,
                tag='op:gn%d' % i)
lc.text(OP_X + OP_W - 14, gy + ch_ + 14, 'D 块校验和 384.0 = P 侧（[0,1] 填 3.0）', 9,
        lc.C_KV_S, 'end', True, maxw=OP_W - 28, tag='op:chip')
op_lines = [
    '· D 本地 handle = 0（收端零轮询）',
    '· 完成判定与拉模式互为镜像（两源换边）：',
    '  P 侧 = WRITE handle 轮询 DONE → done_sending={req-1}',
    '  D 侧 = 收 P 的 WRITE 完成 notif（物化空条目上报）',
    '  → done_recving={req-1}',
]
for j, ln in enumerate(op_lines):
    lc.text(OP_X + 14, OP_Y + 128 + j * 15.5, ln, 8.5, '#334155', 'start', maxw=OP_W - 26,
            tag='op:l%d' % j)
lc.text(OP_X + 14, OP_Y + OP_H - 9, 'nixl/push_worker.py:L656-L686 · L748-L774', 8,
        lc.C_FAINT, 'start', maxw=OP_W - 26, tag='op:f')

# WRITE 粗箭头（机器右缘 → 右板左缘）
wr_y = OP_Y + 78
lc.seg(MC_X + MC_W + 4, wr_y, OP_X - 6, wr_y, lc.C_KV_S, 6.0, 'kv')
lc.text((MC_X + MC_W + OP_X) / 2, wr_y - 26, 'WRITE', 13, lc.C_KV_S, 'middle', True,
        maxw=OP_X - MC_X - MC_W, tag='wr:t')
lc.text((MC_X + MC_W + OP_X) / 2, wr_y - 10, '直写预分配块', 8.5, lc.C_KV_S, 'middle',
        maxw=OP_X - MC_X - MC_W, tag='wr:s')
lc.text((MC_X + MC_W + OP_X) / 2, wr_y + 16, '2 块 × 512 B', 8, lc.C_MUTE, 'middle',
        maxw=OP_X - MC_X - MC_W, tag='wr:b')

# 右板下方：配对规则小注（后缀剥离示例）
SR_Y = OP_Y + OP_H + 18
lc.rect(OP_X, SR_Y, OP_W, 118, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(OP_X + 14, SR_Y + 20, '后缀剥离匹配', 9.5, lc.C_TXT, 'start', True, maxw=OP_W - 28,
        tag='sr:t')
sr_lines = [
    "· 'req-a-1a2b3c4d' → 'req-a'",
    '· 精确键优先；大小写不匹配不配',
    "  （'REQ-A' 不配 'req-a'，实测）",
    '· 后缀 = 两腿 request_id 各带的 8 位',
    '  十六进制随机尾巴（input_processor 追加）',
]
for j, ln in enumerate(sr_lines):
    lc.text(OP_X + 14, SR_Y + 38 + j * 14.5, ln, 8, '#334155', 'start', maxw=OP_W - 26,
            tag='sr:l%d' % j)
lc.text(OP_X + 14, SR_Y + 118 - 9, 'nixl/push_worker.py:L369-L399 · utils.py:get_base_request_id', 8,
        lc.C_FAINT, 'start', maxw=OP_W - 26, tag='sr:f')

# ---------------- 底部：watchdog + 保活 ----------------
BT_Y = SR_Y + 132
bw = (BXR - MX - 16) / 2
lc.rect(MX, BT_Y, bw, 84, 'none', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(MX + 14, BT_Y + 18, '注册 watchdog 480s', 9.5, lc.C_TXT, 'start', True, maxw=bw - 28,
        tag='bt1:t')
for j, ln in enumerate([
        '· 注册后 480s 内没等到 WRITE → 过期注册丢弃、不再打包',
        '· P 侧 WRITE 失败不能标 invalid 块（无 recv 元数据）——只能弃单等租约/watchdog']):
    lc.text(MX + 14, BT_Y + 38 + j * 15, ln, 8.5, '#334155', 'start', maxw=bw - 26,
            tag='bt1:l%d' % j)
lc.rect(MX + bw + 16, BT_Y, bw, 84, 'none', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(MX + bw + 30, BT_Y + 18, '保活：has_pending_push_work', 9.5, lc.C_TXT, 'start', True,
        maxw=bw - 28, tag='bt2:t')
for j, ln in enumerate([
        '· 无活请求 + 有待推送块 → 引擎继续步进（has_requests 纳入 pending）',
        '· WRITE 可能在全部活请求结束后还在飞——引擎不许提前歇（scheduler.py:L2406-L2420）']):
    lc.text(MX + bw + 30, BT_Y + 38 + j * 15, ln, 8.5, '#334155', 'start', maxw=bw - 26,
            tag='bt2:l%d' % j)

# ---------------- 页脚 ----------------
FY = BT_Y + 104
lc.text(MX, FY, '逐字锚 nixl/push_scheduler.py:L156-L206（PUSH_REG 11 键 + watchdog）· push_worker.py:L210-L272（writer 四收件箱）· '
                'L369-L399（后缀剥离）· L656-L686（WRITE 直写）· L748-L774（handle 轮询）· vllm/v1/core/sched/scheduler.py:L2406-L2420（保活）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 15, '行号基线 vLLM v0.27.1 · 角色色即身份：蓝=D 侧注册腿 / 绿=P 侧完成块腿 / 青=配对与 WRITE（KV 过户），与全书 L0/L2 同源 · '
                     '配对机器框 = L2 站 12 组件框的放大（同源视觉）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 34

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch37-fig-push-pairing-machine.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
