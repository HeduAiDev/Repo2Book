#!/usr/bin/env python3
"""ch37 机制图 m5 · oob-handshake-gate（figure_spec ch37-fig-oob-handshake-gate，模板 flow）

放大自 L0「双实例+KV 边界」下排启动待命 → 中排⑤后台握手的 side channel、L2 章图站 2/7。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签；不另立第二种架构画法。

claim：带外握手三件套：side channel（每引擎调度器侧一个 ZMQ ROUTER 常驻握手元数据）+
两段解码（先比 compatibility hash、过门才解 agent 元数据）+ RTT 中点时钟偏移——
先证明两台引擎配置一致，再让它们互读显存。

数字全部取自 figure_spec.numbers（traces 实测 + pin 源码锚点，逐字核对）：
  · P 待命：ROUTER 监听线程 nixl_handshake_listener 启动期就位；agent 元数据 8 块 × 512 B、布局 HND、物理块比 1
  · 兼容 hash 64 位十六进制、10 因子；model 名或注意力后端任一变 → hash 变
  · 不匹配在解 agent 元数据之前报 RuntimeError（两段解码）
  · REQ 超时 5000ms 防死等；本机整趟 4.998ms、时钟偏移 0.00028s（RTT 中点估计，取最低 RTT 样本）
  · 单飞：在飞两次返回 future、完成后返回 None；agent 表由完成回调落账
  · 协议版本 NIXL_CONNECTOR_VERSION=5（版本史：2=remote_request_id、4=心跳续租、5=到期时刻+时钟同步）
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
EXTRA_DEFS = ('<defs>'
              '<marker id="zmq" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" markerHeight="4.6" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ZMQ_S}"/></marker>'
              '<marker id="kv" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6.5" markerHeight="4.6" orient="auto">'
              f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_KV_S}"/></marker>'
              '</defs>')


def diamond(cx, cy, w, h, stroke, fill='#ffffff', sw=1.6):
    d = (f'M{cx:.1f},{cy - h / 2:.1f} L{cx + w / 2:.1f},{cy:.1f} '
         f'L{cx:.1f},{cy + h / 2:.1f} L{cx - w / 2:.1f},{cy:.1f} Z')
    s = f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    lc.ELEMS.append(((cx - w / 2 - 2, cy - h / 2 - 2, cx + w / 2 + 2, cy + h / 2 + 2), s))


# ---------------- 标题区 ----------------
lc.text(MX, 36, '带外握手：对暗号在侧门完成，数据面大门只对指纹一致的两台引擎打开', 16.5,
        lc.C_TXT, 'start', True, maxw=1250, tag='title')
lc.text(MX, 60, '握手全程走 ZMQ side channel（跟数据面完全分开——这就是「带外」）：先比配置指纹（compatibility hash），过门才换钥匙（agent 描述符），顺带量出两机时钟偏移（租约要用）',
        10.5, lc.C_MUTE, 'start', maxw=1430, tag='subtitle')
_ch = '放大自 L2 站 2/7（启动待命 · 后台握手）· L0：双实例+KV 边界'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 主区左：D worker ↔ P scheduler（上：侧门 / 下：锁着的数据面） ----------------
D_X, D_W = MX, 280
P_X, P_W = 646, 304
CH_X0, CH_X1 = D_X + D_W + 14, P_X - 14          # 通道带 348..632
cm = (CH_X0 + CH_X1) / 2

lc.rect(D_X, 118, D_W, 208, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=2.0)
lc.text(D_X + 14, 118 + 24, 'D worker · 发起方', 12.5, lc.C_GPU_S, 'start', True,
        maxw=D_W - 28, tag='d:t')
d_lines = [
    '· 首遇远端 engine_id → 后台 future 握手',
    '  （不堵调度热路径）',
    '· 握手产物：agent 表 {(0, 0): <uuid>}',
    '  + 两机时钟偏移',
    '· 每个 engine_id 至多一个在飞',
    '  （重复调用拿同一个 future）',
]
for j, ln in enumerate(d_lines):
    lc.text(D_X + 14, 118 + 48 + j * 16, ln, 8.5, '#334155', 'start', maxw=D_W - 26,
            tag='d:l%d' % j)
lc.text(D_X + 14, 118 + 208 - 10, 'nixl/base_worker.py:L853-L941', 8, lc.C_FAINT, 'start',
        maxw=D_W - 26, tag='d:f')

lc.rect(P_X, 118, P_W, 208, lc.C_ENG_F, lc.C_ENG_S, rx=9, sw=2.0)
lc.text(P_X + 14, 118 + 24, 'P scheduler · 握手服务端', 12.5, lc.C_ENG_S, 'start', True,
        maxw=P_W - 28, tag='p:t')
p_lines = [
    '· 启动即待命（EngineCore 装配期注入）',
    '· 聚合全 worker 握手元数据 {(pp, tp): payload}',
    '· ROUTER 监听线程 nixl_handshake_listener',
    '· agent 元数据：8 块 × 512 B · 布局 HND',
    '  · 物理块比 1',
]
for j, ln in enumerate(p_lines):
    lc.text(P_X + 14, 118 + 48 + j * 16, ln, 8.5, '#334155', 'start', maxw=P_W - 26,
            tag='p:l%d' % j)
lc.text(P_X + 14, 118 + 208 - 10, 'base_scheduler.py:L281-L322 · core.py:L181-L200', 8,
        lc.C_FAINT, 'start', maxw=P_W - 26, tag='p:f')

# 上通道：侧门（ZMQ 紫）
SB_Y, SB_H = 138, 96
lc.rect(CH_X0, SB_Y, CH_X1 - CH_X0, SB_H, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=8, sw=1.8, dash=True)
lc.text(cm, SB_Y - 8, '侧门 · side channel（ZMQ）', 9.5, lc.C_ZMQ_S, 'middle', True,
        maxw=CH_X1 - CH_X0, tag='sb:t')
lc.seg(D_X + D_W, SB_Y + 30, P_X - 4, SB_Y + 30, lc.C_ZMQ_S, 1.8, 'zmq')
lc.text(cm, SB_Y + 22, 'REQ：(GET_META, 0, 0)', 8.5, lc.C_ZMQ_S, 'middle',
        maxw=CH_X1 - CH_X0 - 8, tag='sb:req')
lc.seg(P_X, SB_Y + 66, D_X + D_W + 4, SB_Y + 66, lc.C_ZMQ_S, 1.8, 'zmq')
lc.text(cm, SB_Y + 58, '回显：握手字节 + 时戳', 8.5, lc.C_ZMQ_S, 'middle',
        maxw=CH_X1 - CH_X0 - 8, tag='sb:rep')

# RTT 注记（上通道下方）
lc.text(cm, SB_Y + SB_H + 18, '整趟 4.998ms（本机实测）· REQ 超时 5000ms 防死等', 8.5,
        lc.C_MUTE, 'middle', maxw=CH_X1 - CH_X0 + 120, tag='rtt:1')
lc.text(cm, SB_Y + SB_H + 34, '时钟偏移 = remote_perf − (start+recv)/2 ≈ 0.00028s', 8.5,
        lc.C_MUTE, 'middle', maxw=CH_X1 - CH_X0 + 120, tag='rtt:2')
lc.text(cm, SB_Y + SB_H + 50, '（RTT 中点估计 · 取最低 RTT 样本）', 8, lc.C_FAINT, 'middle',
        maxw=CH_X1 - CH_X0 + 120, tag='rtt:3')

# 下通道：数据面（KV 青，锁着的门）
DB_Y, DB_H = 300, 64
lc.rect(CH_X0, DB_Y, CH_X1 - CH_X0, DB_H, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.8, dash=True)
lc.text(cm, DB_Y - 8, '数据面 · KV 边界', 9.5, lc.C_KV_S, 'middle', True,
        maxw=CH_X1 - CH_X0, tag='db:t')
lc.text(cm, DB_Y + 20, '锁着：hash 对上才开', 9, lc.C_KV_S, 'middle', True,
        maxw=CH_X1 - CH_X0 - 8, tag='db:l1')
lc.text(cm, DB_Y + 40, '开了才许 READ/WRITE 直读显存', 8, lc.C_MUTE, 'middle',
        maxw=CH_X1 - CH_X0 - 8, tag='db:l2')
lc.text(cm, DB_Y + 58, '（→ 见本章数据面图）', 8, lc.C_FAINT, 'middle',
        maxw=CH_X1 - CH_X0 - 8, tag='db:l3')

# ---------------- 主区右：两段解码 ----------------
RX = 990
RW = BXR - RX
lc.rect(RX, 118, RW, 236, '#ffffff', lc.C_TXT, rx=9, sw=1.8)
lc.text(RX + 16, 118 + 24, '两段解码：回包先过 ① 再到 ②', 12.5, lc.C_TXT, 'start', True,
        maxw=RW - 32, tag='2p:t')

# ① hash 菱形门
dia_cx, dia_cy, dia_w, dia_h = RX + 150, 118 + 92, 190, 64
diamond(dia_cx, dia_cy, dia_w, dia_h, lc.C_ZMQ_S)
lc.text(dia_cx, dia_cy - 4, '① compatibility hash', 9, lc.C_TXT, 'middle', True,
        maxw=dia_w - 30, tag='dia:1')
lc.text(dia_cx, dia_cy + 12, '相等？', 9, lc.C_TXT, 'middle', True, maxw=dia_w - 30, tag='dia:2')
# 不匹配出口（右）→ RuntimeError
lc.seg(dia_cx + dia_w / 2, dia_cy, RX + RW - 158, dia_cy, lc.C_ABORT, 1.6, 'ab')
lc.text(dia_cx + dia_w / 2 + 12, dia_cy - 8, '不匹配', 8, lc.C_ABORT, 'start', tag='ex:no')
lc.rect(RX + RW - 154, dia_cy - 40, 154, 80, '#fef2f2', lc.C_ABORT, rx=7, sw=1.3)
lc.text(RX + RW - 77, dia_cy - 24, 'RuntimeError:', 8.5, lc.C_ABORT, 'middle', True,
        maxw=142, tag='err:1')
lc.text(RX + RW - 77, dia_cy - 10, 'NIXL compatibility', 8.5, lc.C_ABORT, 'middle',
        maxw=142, tag='err:2')
lc.text(RX + RW - 77, dia_cy + 4, 'hash mismatch', 8.5, lc.C_ABORT, 'middle',
        maxw=142, tag='err:3')
lc.text(RX + RW - 77, dia_cy + 22, '在解 agent 元数据之前', 8, lc.C_MUTE, 'middle',
        maxw=142, tag='err:4')
lc.text(RX + RW - 77, dia_cy + 34, '炸——不等数据传错才炸', 8, lc.C_MUTE, 'middle',
        maxw=142, tag='err:5')
# 匹配出口（下）→ ②
lc.seg(dia_cx, dia_cy + dia_h / 2, dia_cx, dia_cy + dia_h / 2 + 24, lc.C_ZMQ_S, 1.6, 'zmq')
lc.text(dia_cx + 8, dia_cy + dia_h / 2 + 14, '匹配', 8, lc.C_ZMQ_S, 'start', tag='ex:yes')
b2_y = dia_cy + dia_h / 2 + 26
lc.rect(dia_cx - 95, b2_y, 190, 58, lc.C_ZMQ_F, lc.C_ZMQ_S, rx=7, sw=1.4)
lc.text(dia_cx, b2_y + 17, '② 解 agent metadata', 9, lc.C_TXT, 'middle', True, maxw=178,
        tag='b2:1')
lc.text(dia_cx, b2_y + 32, '→ 注册 agent 描述符（钥匙）', 8.5, lc.C_TXT, 'middle', maxw=178,
        tag='b2:2')
lc.text(dia_cx, b2_y + 47, '货位表：(基址, 块长) 按块号索引', 8, lc.C_MUTE, 'middle',
        maxw=178, tag='b2:3')
lc.text(RX + 16, 118 + 236 - 12, 'nixl/base_worker.py:L661-L683（hash 门先于 agent 解码）', 8,
        lc.C_FAINT, 'start', maxw=RW - 32, tag='2p:f')

# hash 因子盒（右栏下方）
HF_Y = 368
lc.rect(RX, HF_Y, RW, 130, '#ffffff', lc.C_ZMQ_S, rx=8, sw=1.4)
lc.text(RX + 14, HF_Y + 20, 'compatibility hash：64 位十六进制 · 10 因子', 10,
        lc.C_ZMQ_S, 'start', True, maxw=RW - 28, tag='hf:t')
hf_lines = [
    'vLLM 版本 · 连接器版本(=5) · 模型 · dtype · KV 头数',
    '· 头维 · 层数 · 注意力后端 · cache dtype · HMA/跨层',
    '· model 名或注意力后端任一变 → hash 变（实测）',
    '· TP / block_size / 布局不在 hash 里',
    '  （运行期另校验——支持异构）',
]
for j, ln in enumerate(hf_lines):
    lc.text(RX + 14, HF_Y + 42 + j * 15.5, ln, 8.5, '#334155', 'start', maxw=RW - 26,
            tag='hf:l%d' % j)
lc.text(RX + 14, HF_Y + 130 - 10, 'nixl/metadata.py:L81-L141', 8, lc.C_FAINT, 'start',
        maxw=RW - 26, tag='hf:f')

# ---------------- 底部：单飞 + 版本史 ----------------
BT_Y = 368
lc.rect(MX, BT_Y, CH_X1 - MX + 20, 130, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 14, BT_Y + 20, '单飞：每个远端至多一个在飞握手', 10, lc.C_TXT, 'start', True,
        maxw=CH_X1 - MX, tag='sf:t')
sf_lines = [
    '· 在飞期两次 _ensure_handshake 都拿到同一个 future',
    '· 完成后第三次返回 None',
    '· agent 表与时钟偏移由完成回调落账（锁内先删表键再落表，',
    '  不可能并发两个握手）',
    '· 握手在后台 future 里做——不堵调度热路径',
]
for j, ln in enumerate(sf_lines):
    lc.text(MX + 14, BT_Y + 42 + j * 15.5, ln, 8.5, '#334155', 'start',
            maxw=CH_X1 - MX - 6, tag='sf:l%d' % j)
lc.text(MX + 14, BT_Y + 130 - 10, 'nixl/base_worker.py:L853-L941', 8, lc.C_FAINT, 'start',
        maxw=CH_X1 - MX - 6, tag='sf:f')

VR_Y = HF_Y + 142
lc.rect(MX, VR_Y, BXR - MX, 44, lc.C_BEAT_F, lc.C_BEAT_S, rx=8, sw=1.2)
lc.text((MX + BXR) / 2, VR_Y + 18, '协议版本 NIXL_CONNECTOR_VERSION = 5（版本史：2 = remote_request_id · '
        '4 = 心跳续租 · 5 = 到期时刻 + 时钟同步）', 9.5, lc.C_BEAT_T, 'middle', True,
        maxw=BXR - MX - 30, tag='vr:1')
lc.text((MX + BXR) / 2, VR_Y + 34, '握手失败/超时不会写坏任何一端：D 拿不到描述符就不发 READ，P 的门不开（nixl/metadata.py:L26-L45）',
        8.5, lc.C_MUTE, 'middle', maxw=BXR - MX - 30, tag='vr:2')

FY = VR_Y + 66
lc.text(MX, FY, '逐字锚 nixl/base_worker.py:L608-L643（ZMQ REQ · RCVTIMEO 5000ms 在 L631）· L661-L683（hash 门）· '
                'L853-L941（单飞 future）· base_scheduler.py:L281-L322（ROUTER 待命）· vllm/v1/engine/core.py:L181-L200（元数据聚合）· '
                'nixl/metadata.py:L26-L45（版本史）· L81-L141（hash 因子）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 15, '行号基线 vLLM v0.27.1 · 角色色即身份：紫=ZMQ side channel / 青=KV 数据面 / 绿=D worker / 橙=P 调度器侧，与全书 L0/L2 同源',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

H = FY + 34

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS, EXTRA_DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch37-fig-oob-handshake-gate.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
