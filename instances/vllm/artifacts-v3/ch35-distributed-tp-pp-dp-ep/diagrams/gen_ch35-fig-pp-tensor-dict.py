#!/usr/bin/env python3
"""ch35 机制图 · m8 PP 段间接力：一张张量字典、两条道（figure_spec ch35-fig-pp-tensor-dict）

放大自 L0 多实例视角 ④ PP·段间接力（L2 站 12）的载荷通道展开。

claim：PP 段间传张量字典按数据类型分道：pickle 序列化的 metadata 先 send_object
走 cpu_group(gloo) 对象通道、每个张量各自 isend/irecv 走 device_group——
双群组分工在 P2P 上的体现，接收侧包成 AsyncIntermediateTensors 懒同步。

数字全部取自 explainer figure_spec.numbers（traces/m01 pp_dict 实测 + pin 锚点）；
坐标由常量/循环计算；文本全 esc()；配色走 l0_common（cpu 通道 ZMQ 紫细虚、
device 通道 GPU 绿粗实）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1340, 780
MX = 64
BXR = 1276

DEFS = lc.DEFS.replace('</defs>',
                       '<marker id="pu" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="6.5" markerHeight="4.6" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_ZMQ_S}"/></marker>'
                       '<marker id="gn" viewBox="0 0 10 6" refX="9" refY="3" '
                       'markerWidth="7" markerHeight="5" orient="auto">'
                       f'<path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_GPU_S}"/></marker>'
                       '</defs>')


def chip(x_right, y, label, color):
    w = lc.tw(label, 9.5, True) + 14
    x = x_right - w
    lc.rect(x, y, w, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
    lc.text(x + w / 2, y + 14.5, label, 9.5, color, 'middle', True,
            maxw=w - 4, tag='chip:' + label[:10])
    return x


# ---------------- 标题区 ----------------
lc.text(MX, 36, 'PP 段间接力：一张张量字典、两条道——装箱单走 gloo，大箱走设备通道', 16.5,
        lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 62, '先走邮政寄装箱单（键名/形状/dtype——CPU 小对象），再让卡车逐件发大箱（张量）；'
        '上一节的双群组分工在点对点上同样成立', 10.5, lc.C_MUTE, 'start', maxw=1150,
        tag='subtitle')
chip(BXR, 12, '放大自 L2 站 12 · L0：多实例视角', lc.C_GPU_S)

# ---------------- 左：被拆分的字典 ----------------
DX, DY, DW, DH = MX, 116, 250, 300
lc.rect(DX, DY, DW, DH, '#ffffff', lc.C_ENG_S, rx=9, sw=1.8)
lc.text(DX + 14, DY + 24, 'output.tensors', 12, lc.C_TXT, 'start', True,
        maxw=DW - 28, tag='dict:t')
lc.text(DX + 14, DY + 42, '首段输出的张量字典 · 3 键（实测）', 9.5, lc.C_MUTE, 'start',
        maxw=DW - 28, tag='dict:s')
keys = [
    ('hidden', '张量 [3,4]', True, 12, '12 元素（0..11）'),
    ('residual', '张量 [3,4]', True, 12, '全 7.0'),
    ('scalar_meta', '非张量对象', False, 0, '{"num": 42}'),
]
for i, (k, kind, is_tensor, nel, val) in enumerate(keys):
    ky = DY + 64 + i * 74
    stroke = lc.C_GPU_S if is_tensor else lc.C_ZMQ_S
    lc.rect(DX + 14, ky, DW - 28, 62, '#ffffff' if is_tensor else lc.C_ZMQ_F, stroke,
            rx=6, sw=1.4)
    lc.text(DX + 26, ky + 21, k, 10.5, lc.C_TXT, 'start', True, maxw=DW - 52, tag='k%d' % i)
    lc.text(DX + 26, ky + 38, '%s · %s' % (kind, val), 9, '#334155', 'start',
            maxw=DW - 52, tag='k%d:v' % i)
    badge = 'isend ×1' if is_tensor else 'send_object ×1'
    lc.text(DX + 26, ky + 54, badge, 8.5, stroke, 'start', tag='k%d:b' % i)

# ---------------- 上/下：两段 Worker 泳道 ----------------
WX, WW = 380, 660
UY, UH = 116, 148
lc.rect(WX, UY, WW, UH, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(WX + 16, UY + 24, '首段 Worker（PP rank0 · 非末段）', 12.5, lc.C_GPU_S, 'start',
        True, maxw=WW - 260, tag='up:t')
lc.text(WX + WW - 14, UY + 24, 'vllm/v1/worker/gpu_worker.py', 8.5, lc.C_FAINT, 'end',
        tag='up:f')
up_lines = [
    '① send_tensor_dict(output.tensors, dst=1)：先 send_object(metadata_list)（L1055-L1058）',
    '② 逐张量 isend——hidden / residual 各拿 1 个句柄（TensorMetadata 进装箱单）',
    '③ 句柄存 _pp_send_work——发完即走，下一轮 execute_model 开头统一收割',
]
for i, ln in enumerate(up_lines):
    lc.text(WX + 16, UY + 50 + i * 19, ln, 9.5, '#334155', 'start', maxw=WW - 30,
            tag='up:l%d' % i)
lc.text(WX + 16, UY + 118, 'CUDA 张量 isend 后 record_stream——防缓冲被提前回收（L1068-L1071）',
        9, lc.C_MUTE, 'start', maxw=WW - 30, tag='up:note')

LY_, LH = 420, 148
lc.rect(WX, LY_, WW, LH, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.0)
lc.text(WX + 16, LY_ + 24, '次段 Worker（PP rank1 · 非首段）', 12.5, lc.C_GPU_S, 'start',
        True, maxw=WW - 260, tag='dn:t')
lc.text(WX + WW - 14, LY_ + 24, 'vllm/v1/worker/gpu_worker.py:L1064-L1105', 8.5,
        lc.C_FAINT, 'end', tag='dn:f')
dn_lines = [
    '① irecv_tensor_dict：先同步收 metadata（cpu_group 对象通道=同步门）',
    '② 按装箱单逐张量 irecv——句柄先揣兜里、不等待就返回',
    '③ 包成 AsyncIntermediateTensors：碰 .tensors 那一刻才逐句柄 wait（懒同步）',
]
for i, ln in enumerate(dn_lines):
    lc.text(WX + 16, LY_ + 50 + i * 19, ln, 9.5, '#334155', 'start', maxw=WW - 30,
            tag='dn:l%d' % i)
lc.text(WX + 16, LY_ + 118, '实测：receiver 逐值复原——hidden 0..11 · residual 全 7.0 · '
        'scalar_meta={"num": 42}', 9, lc.C_MUTE, 'start', maxw=WW - 30, tag='dn:note')

# ---------------- 中：两条通道 ----------------
CH1, CH2 = 308, 368            # 上=cpu 通道线 y；下=device 通道线 y
chan_x0, chan_x1 = 460, 960    # cpu 通道主线；device 通道起点前移到 440（馈线不穿文字）
# 通道底带
lc.rect(420, 278, 580, 126, '#f8fafc', lc.C_FAINT, rx=10, sw=1.0)
# cpu 通道（细虚紫）
lc.seg(chan_x0, CH1, chan_x1, CH1, lc.C_ZMQ_S, 2.0, 'pu', dash=True)
lc.text(470, 294, 'cpu_group（gloo）· 对象通道——细虚线', 10, lc.C_ZMQ_S, 'start',
        True, maxw=380, tag='cpu:t')
lc.text(470, 326, '装箱单 metadata_list（键名/形状/dtype × 3 + {"num": 42}）· send_object 1 次',
        9, '#334155', 'start', maxw=440, tag='cpu:l')
# device 通道（粗实绿）
lc.seg(440, CH2, chan_x1, CH2, lc.C_GPU_S, 4.0, 'gn')
lc.text(470, 352, 'device_group（真 pin=NCCL · host 实跑 gloo）· 张量通道——粗实线',
        10, lc.C_GPU_S, 'start', True, maxw=440, tag='dev:t')
lc.text(470, 386, '逐张量 isend / irecv · 2 个张量句柄（hidden 12 元素 · residual 全 7.0）',
        9, '#334155', 'start', maxw=440, tag='dev:l')

# 字典 → 首段（左向喂入；箭头下移避开泳道步骤行文字）
lc.seg(DX + DW, 204, WX, 204, lc.C_ENG_S, 2.2, 'std')
lc.alabel((DX + DW + WX) / 2, 197, 'output.tensors', 9, lc.C_ENG_S)
# 首段 → 两通道（发送侧）
lc.seg(chan_x0, UY + UH, chan_x0, CH1 - 4, lc.C_ZMQ_S, 1.8, 'pu', dash=True)
lc.seg(440, UY + UH, 440, CH2 - 4, lc.C_GPU_S, 2.2, 'gn')
# 两通道 → 次段（接收侧，落在泳道顶边）
lc.seg(860, CH1, 860, LY_, lc.C_ZMQ_S, 1.8, 'pu', dash=True)
lc.seg(900, CH2, 900, LY_, lc.C_GPU_S, 2.4, 'gn')

# ---------------- 右侧 why 注 ----------------
NX = WX + WW + 24
NW = BXR - NX
lc.rect(NX, 116, NW, 240, 'none', lc.C_FAINT, rx=8, sw=1.1, dash=True)
lc.text(NX + 14, 138, 'why · 为什么分两条道', 10, lc.C_TXT, 'start', True, maxw=NW - 28,
        tag='why:t')
why = [
    '· 装箱单是 CPU 上的小对象——',
    '  走 NCCL 要无谓的 GPU↔CPU 来回',
    '· 大箱是张量，必须走设备通道',
    '  ——一张字典两种货、两条道',
    '· metadata 先行是硬顺序：接收侧',
    '  要先知道形状才能 irecv',
    '· 发送句柄不等待：首段拍内发完',
    '  即走（_pp_send_work 下拍收割），',
    '  双向都不阻塞 busy loop',
]
for i, ln in enumerate(why):
    lc.text(NX + 14, 160 + i * 18, ln, 9, '#334155', 'start', maxw=NW - 28,
            tag='why:l%d' % i)
lc.rect(NX, 372, NW, 196, '#f8fafc', lc.C_MUTE, rx=8, sw=1.1)
lc.text(NX + 14, 394, '实测账（traces 单机 gloo）', 10, lc.C_TXT, 'start', True,
        maxw=NW - 28, tag='acct:t')
acct = [
    '· keys_sent=3：hidden / residual /',
    '  scalar_meta',
    '· n_tensor_handles=2（2 键张量）',
    '· receiver 逐值复原：hidden=',
    '  [3,4] 12 元素 0..11、residual',
    '  全 7.0、scalar_meta={"num":42}',
    '· 异步 stream 语义在 host 退化为',
    '  同步完成（gloo 无 stream；调用',
    '  方随后必然同步，结果等价）',
]
for i, ln in enumerate(acct):
    lc.text(NX + 14, 416 + i * 17, ln, 9, '#334155', 'start', maxw=NW - 28,
            tag='acct:l%d' % i)

# ---------------- 底部结论条 ----------------
BY = 600
lc.rect(MX, BY, BXR - MX, 84, lc.C_GPU_F, lc.C_GPU_S, rx=8, sw=1.1)
lc.text(MX + 16, BY + 24, '双群组分工（GroupCoordinator 的 cpu 线路与 device 线路）在点对点上同样成立：'
        '装箱单 1 次 send_object + 大箱 2 个 isend 句柄', 10.5, lc.C_TXT, 'start', True,
        maxw=BXR - MX - 32, tag='bot:l1')
lc.text(MX + 16, BY + 46, '接收侧 AsyncIntermediateTensors 懒同步：单子（句柄）揣兜里先干本地的活'
        '（KV 取址、注意力元数据），碰 .tensors 才等快递员——谁先碰箱子谁负责等', 9.5,
        '#334155', 'start', maxw=BXR - MX - 32, tag='bot:l2')
lc.text(MX + 16, BY + 68, '发送侧非阻塞 + 下拍收割：接力不占 busy loop——PP 气泡只剩传输本身',
        9.5, lc.C_MUTE, 'start', maxw=BXR - MX - 32, tag='bot:l3')

# ---------------- 图例 + 页脚 ----------------
LY2 = BY + 84 + 26
lc.seg(MX + 2, LY2 - 3, MX + 32, LY2 - 3, lc.C_ZMQ_S, 2.0, 'pu', dash=True)
lc.text(MX + 40, LY2 + 1, 'cpu 通道（gloo·装箱单/小对象）', 9.5, lc.C_TXT, 'start',
        tag='leg:cpu')
lx2 = MX + 40 + lc.tw('cpu 通道（gloo·装箱单/小对象）', 9.5) + 22
lc.seg(lx2 + 2, LY2 - 3, lx2 + 32, LY2 - 3, lc.C_GPU_S, 4.0, 'gn')
lc.text(lx2 + 40, LY2 + 1, 'device 通道（真 pin=NCCL·张量大件）', 9.5, lc.C_TXT, 'start',
        tag='leg:dev')
lc.text(MX, LY2 + 24,
        '行号基线 vLLM v0.27.1（6e448d0ea）· 实测值标 host（gloo 单机退化——真 pin 张量走 NCCL '
        'P2P）· 框内灰字 = 规范源码路径', 9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='footer')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch35-fig-pp-tensor-dict.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
