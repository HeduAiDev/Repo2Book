#!/usr/bin/env python3
"""ch37 机制图 m1 · dual-engine-halves（figure_spec ch37-fig-dual-engine-halves，模板 layout）

放大自 L0「双实例+KV 边界」（本章 l0_zoom）、L2 章图站 1（① 双实例与角色）。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签；KV 边界本图只画占位
（数据面的放大归本章握手/READ 图，不另立画法）。

claim：P/D 是同一份 NixlBaseConnector 代码的两种角色上岗：每台引擎内部 scheduler 侧
与 worker 侧各建一份 connector、facade 按 role 只建半边（另一半恒 None）——全局
4 个 connector 对象、每个恰一半非空（非空半边合计 4：scheduler 半 2 + worker 半 2）。

数字全部取自 figure_spec.numbers（traces 实测 + pin 源码锚点，逐字核对）：
  · 注册名 3 个、别名 1 对（'NixlConnector' = NixlPullConnector，connector.py:L387；
    factory.py:L176-L192 三条 register_connector）
  · kv_role 部署门：缺省/非法即 ValueError（kv_transfer.py:L92-L105，
    报错原文逐字取自源码 L98/L104）
  · engine_id 默认 uuid4（36 位）两两不同、side channel 端口每引擎独占
  · 每台 2 份 connector（scheduler 侧 + worker 侧）→ 4 个对象、非空半边每台 2 /
    合计 4（scheduler 半 2 + worker 半 2）
  · 非 MLA 强制 HND：张量 (8,2,4,16)、每块 512 B
  · kv_both 仍可构建但打弃用告警（connector.py:L124）
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W = 1560
MX = 54
BXR = 1506

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'P/D 不是两种程序：同一份代码，两种工牌，每台引擎里 connector 只活一半', 16.5,
        lc.C_TXT, 'start', True, maxw=1160, tag='title')
lc.text(MX, 60, '两台完整引擎（P=kv_role:kv_producer、D=kv_role:kv_consumer）· 布局门：非 MLA 强制 HND——'
                '本例张量 (8,2,4,16)、每块 512 B（块粒度传输的物理基础）',
        10.5, lc.C_MUTE, 'start', maxw=1250, tag='subtitle')
_ch = '放大自 L2 站 1（① 双实例与角色）· L0：双实例+KV 边界'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 部署带：三列 ----------------
DB_Y, DB_H = 84, 142
COL_W = (BXR - MX - 2 * 16) / 3
db_cols = [
    ('部署门：kv_role 二选一', lc.C_API_S, [
        '· 不填 → ValueError：',
        '  「Please specify kv_role when kv_connector is set,」',
        '· 非法（kv_watcher）→ ValueError：',
        '  「Unsupported kv_role: kv_watcher. …」',
        '· 角色是部署门——不选角色不许启动（构造期即拦）',
    ], 'vllm/config/kv_transfer.py:L92-L105'),
    ('身份：engine_id 与端口', lc.C_ZMQ_S, [
        '· engine_id 默认 uuid4（36 位）· 两实例两两不同',
        '· notif 寻址与握手都用它',
        '· side channel 端口每引擎独占',
        '  （P/D 各一个 ZMQ ROUTER 待命）',
        '· 部署 2 台引擎 = 2 条 vllm serve',
        '  （各带一份 --kv-transfer-config）',
    ], 'kv_transfer.py 默认 engine_id=uuid4'),
    ('注册名 3 个（别名 1 对）', lc.C_ENG_S, [
        '· NixlPullConnector（拉模式 READ）',
        '· NixlPushConnector（推模式 WRITE）',
        '· NixlConnector = NixlPullConnector（别名）',
        '· 部署命令行里写 NixlConnector 就是拉模式',
        '· 本图两台都用拉模式（同一份代码）',
    ], 'nixl/connector.py:L387 · factory.py:L176-L192'),
]
for i, (t, color, lines, foot) in enumerate(db_cols):
    x = MX + i * (COL_W + 16)
    lc.rect(x, DB_Y, COL_W, DB_H, '#ffffff', color, rx=8, sw=1.6)
    lc.text(x + 14, DB_Y + 22, t, 11.5, color, 'start', True, maxw=COL_W - 28, tag='db:t' + str(i))
    for j, ln in enumerate(lines):
        lc.text(x + 14, DB_Y + 44 + j * 15.5, ln, 8.5, '#334155', 'start',
                maxw=COL_W - 26, tag='db:l%d_%d' % (i, j))
    lc.text(x + 14, DB_Y + DB_H - 9, foot, 8, lc.C_FAINT, 'start', maxw=COL_W - 26,
            tag='db:f' + str(i))

# ---------------- 双引擎框 + 中缝 KV 边界 ----------------
EG_Y, EG_H = 252, 430
EG_W = 620
P_X, D_X = MX, BXR - EG_W
MIDX = (P_X + EG_W + D_X) / 2


def engine_frame(x, name, role, badge, badge_color, plaque):
    lc.rect(x, EG_Y, EG_W, EG_H, '#ffffff', lc.C_TXT, rx=10, sw=2.0)
    lc.text(x + 16, EG_Y + 26, name, 13, lc.C_TXT, 'start', True, maxw=330, tag=name + ':t')
    bw = lc.tw(badge, 9.5, True) + 16
    lc.rect(x + EG_W - bw - 12, EG_Y + 9, bw, 21, '#ffffff', badge_color, rx=9, sw=1.2)
    lc.text(x + EG_W - bw / 2 - 12, EG_Y + 24, badge, 9.5, badge_color, 'middle', True,
            maxw=bw - 6, tag=name + ':b')
    lc.text(x + 16, EG_Y + 46, plaque, 9, lc.C_MUTE, 'start', maxw=EG_W - 32, tag=name + ':p')
    # 两个进程行：调度器半边 / worker 半边
    proc_y0 = EG_Y + 62
    proc_h = (EG_H - 62 - 44 - 14) / 2
    rows = [
        ('调度器进程 · role=SCHEDULER', lc.C_ENG_S, lc.C_ENG_F,
         'facade：NixlPullConnector',
         ('connector_scheduler', '= NixlPullConnectorScheduler'),
         ('connector_worker', '= None')),
        ('worker 进程 · role=WORKER', lc.C_GPU_S, lc.C_GPU_F,
         'facade：NixlPullConnector',
         ('connector_worker', '= NixlPullConnectorWorker'),
         ('connector_scheduler', '= None')),
    ]
    for k, (pt, sc, sf, ft, live, dead) in enumerate(rows):
        py = proc_y0 + k * (proc_h + 14)
        lc.rect(x + 16, py, EG_W - 32, proc_h, sf, sc, rx=7, sw=1.4)
        lc.text(x + 28, py + 19, pt, 10.5, sc, 'start', True, maxw=EG_W - 56, tag=name + ':pt' + str(k))
        # facade 框（进程内的那份 connector 拷贝）
        fy, fh = py + 28, proc_h - 40
        lc.rect(x + 28, fy, EG_W - 56, fh, '#ffffff', sc, rx=6, sw=1.2)
        lc.text(x + 40, fy + 17, ft + '（本进程那份拷贝）', 9, lc.C_TXT, 'start', True,
                maxw=EG_W - 80, tag=name + ':ft' + str(k))
        # 两个半边槽位：活的着色、死的灰虚
        sy, sh = fy + 25, fh - 34
        sw_ = (EG_W - 56 - 24 - 10) / 2
        for m, (field, value, alive) in enumerate([
                (live[0], live[1], True), (dead[0], dead[1], False)]):
            sx = x + 28 + 12 + m * (sw_ + 10)
            if alive:
                lc.rect(sx, sy, sw_, sh, '#ffffff', sc, rx=5, sw=1.4)
                lc.text(sx + sw_ / 2, sy + 16, field, 8.5, lc.C_TXT, 'middle', True,
                        maxw=sw_ - 12, tag=name + ':sv%d_%d' % (k, m))
                lc.text(sx + sw_ / 2, sy + 32, value, 8.5, lc.C_TXT, 'middle',
                        maxw=sw_ - 12, tag=name + ':sv2_%d_%d' % (k, m))
                lc.text(sx + sw_ / 2, sy + sh - 8, '● 本半边活', 8, sc, 'middle',
                        maxw=sw_ - 12, tag=name + ':sa%d_%d' % (k, m))
            else:
                lc.rect(sx, sy, sw_, sh, '#f1f5f9', lc.C_FAINT, rx=5, sw=1.1, dash=True)
                lc.text(sx + sw_ / 2, sy + 16, field, 8.5, lc.C_MUTE, 'middle',
                        maxw=sw_ - 12, tag=name + ':dv%d_%d' % (k, m))
                lc.text(sx + sw_ / 2, sy + 32, value, 8.5, lc.C_MUTE, 'middle',
                        maxw=sw_ - 12, tag=name + ':dv2_%d_%d' % (k, m))
                lc.text(sx + sw_ / 2, sy + sh - 8, '○ 死半边（恒 None）', 8, lc.C_FAINT, 'middle',
                        maxw=sw_ - 12, tag=name + ':da%d_%d' % (k, m))


engine_frame(P_X, 'Prefill 引擎（P）', 'kv_role=kv_producer', '工牌：算完 prefill 把块钉住等人来取', lc.C_ENG_S,
             'vllm serve …--kv-transfer-config（P 侧一份）· engine_id 默认 uuid4（36 位）')
engine_frame(D_X, 'Decode 引擎（D）', 'kv_role=kv_consumer', '工牌：prompt 的 KV 在别人那里，我去取', lc.C_GPU_S,
             'vllm serve …--kv-transfer-config（D 侧一份）· 同一份 connector 代码，只换工牌')

# 计数条（贴双引擎底、引到两框）：spec 数字逐字
CNT_Y = EG_Y + EG_H + 8
lc.rect(MX, CNT_Y, BXR - MX, 40, lc.C_ENG_F, lc.C_ENG_S, rx=7, sw=1.2)
lc.text((MX + BXR) / 2, CNT_Y + 17, '每台 2 份 connector（scheduler 侧 + worker 侧）→ 4 个对象、非空半边每台 2 / 合计 4'
        '（scheduler 半 2 + worker 半 2）（trace 实测：SCHEDULER 拷贝只建 connector_scheduler、'
        'WORKER 拷贝只建 connector_worker，另一半恒 None）',
        10, lc.C_TXT, 'middle', True, maxw=BXR - MX - 30, tag='cnt:1')
lc.text((MX + BXR) / 2, CNT_Y + 32, '同一份代码按角色只活一半——分开构建、不共享状态（ch16 role-split 在真实后端的形态）',
        8.5, lc.C_MUTE, 'middle', maxw=BXR - MX - 30, tag='cnt:2')

# ---------------- 中缝：KV 边界占位 ----------------
BND_X0, BND_X1 = P_X + EG_W + 18, D_X - 18
bmid = (BND_X0 + BND_X1) / 2
lc.rect(bmid - 14, EG_Y + 40, 28, EG_H - 80, 'none', lc.C_KV_S, rx=8, sw=1.6, dash=True)
lc.text(bmid, EG_Y + 24, 'KV 边界', 10.5, lc.C_KV_S, 'middle', True, maxw=BND_X1 - BND_X0, tag='bnd:t')
lc.text(bmid, EG_Y + EG_H - 44, '（本图只画占位', 8, lc.C_MUTE, 'middle', maxw=BND_X1 - BND_X0, tag='bnd:p1')
lc.text(bmid, EG_Y + EG_H - 30, '数据面放大→后图）', 8, lc.C_MUTE, 'middle', maxw=BND_X1 - BND_X0, tag='bnd:p2')

# ---------------- 兼容位注记 + 页脚 ----------------
KB_Y = CNT_Y + 52
lc.rect(MX, KB_Y, BXR - MX, 34, 'none', lc.C_FAINT, rx=7, sw=1.1, dash=True)
lc.text((MX + BXR) / 2, KB_Y + 15, '旧角色 kv_both 仍可构建但打弃用告警：'
        '「Using kv_role=‘kv_both’ with NixlConnector is deprecated …」（P/D 各设一职才是正统）',
        8.5, lc.C_MUTE, 'middle', maxw=BXR - MX - 30, tag='kb:1')
lc.text((MX + BXR) / 2, KB_Y + 28, '配置指纹（compatibility hash）与「先握手再开门」→ 见本章握手图 · '
        'READ 直读/WRITE 直写的两种填法 → 见本章数据面两图',
        8.5, lc.C_MUTE, 'middle', maxw=BXR - MX - 30, tag='kb:2')

FY = KB_Y + 52
lc.text(MX, FY, '逐字锚 vllm/config/kv_transfer.py:L92-L105（kv_role 门·报错原文 L98/L104）· '
                'nixl/connector.py:L322-L341（按 role 建半边）· L387（NixlConnector=NixlPullConnector 别名）· '
                'L124（kv_both 弃用告警）· kv_connector/factory.py:L176-L192（注册名 3 个）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 15, '行号基线 vLLM v0.27.1 · 角色色即身份：橙=EngineCore（调度器半边）/ 绿=GPU（worker 半边）/ 青=KV 边界，与全书 L0/L2 同源',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 34

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch37-fig-dual-engine-halves.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
