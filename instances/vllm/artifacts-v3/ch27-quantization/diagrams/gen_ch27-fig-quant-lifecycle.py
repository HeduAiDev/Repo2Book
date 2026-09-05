#!/usr/bin/env python3
"""ch27 机制图 9 · 量化权重的一生:离线数学 + 运行期四重门(figure_spec ch27-fig-quant-lifecycle,模板 flow)

放大自 L0 中列『GPU 执行臂』(恒绿 C_GPU_S)整列——GPUModelRunner(构造/装载)与
『模型层 forward + 编译』(每拍消费)串起运行期全生命周期;配置期算力门在 L0 外围
(VllmConfig 装配,启动视角),图上以入口关卡回指。不另立第二种架构画法(FIGURE-SYSTEM §3)。

claim:离线数学产网格 + 运行期四重门:①配置期算力硬门(门槛来自 kernel);②构造期按
优先级表 + can_implement 挑 kernel(同一检查点 H100 走 Machete、A100 落 Marlin);
③装载期把检查点格式重排成 kernel 格式;④编译期算子选择(标准=整条编译管线更快)——
聪明的部分全在离线,运行期只是按硬件消费网格。

数字全部取自本章参考实现实跑 + pin 源码 file:L 锚点;坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 892
MX = 60
BXR = 1440
GRID = '#e2e8f0'

lc.text(MX, 34, '聪明在工厂、快在店里:一副量化权重的生命周期 = 离线数学 + 运行期四重门',
        16.5, lc.C_TXT, 'start', True, maxw=1060, tag='title')
lc.text(MX, 58, 'GPTQ/AWQ/SmoothQuant 的数学全发生在离线加工厂;到了卡上,vLLM 只做四件事:进门查算力、柜台挑 kernel、后厨换包装、编译间选路——速度的来源写在论文里:提速几乎全部来自少搬显存',
        10.5, lc.C_MUTE, 'start', maxw=1080, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂(整列)· 配置期门回指 L0 启动视角(VllmConfig 装配)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')


def gate(x, y, n):
    """关卡圆徽标。"""
    lc.ELEMS.append(((x - 11, y - 11, x + 11, y + 11),
                     f'<circle cx="{x}" cy="{y}" r="10" fill="#0f172a"/>'))
    lc.text(x, y + 3.5, n, 9.5, '#ffffff', 'middle', True, tag='gate' + n)


# ================= 离线工厂(plain) =================
FX, FY, FW, FH = MX, 200, 300, 262
lc.rect(FX, FY, FW, FH, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
lc.text(FX + 14, FY + 22, '离线加工厂', 12, lc.C_TXT, 'start', True, maxw=FW - 28, tag='f:h')
lc.text(FX + 14, FY + 40, '聪明的部分全在这里——一次成型,产检查点', 8.5, lc.C_MUTE, 'start',
        maxw=FW - 28, tag='f:sub')
FACT = [
    'GPTQ:逐列取整 + 逆 Hessian 找补',
    'AWQ:显著通道 ×s(看激活不看权重)',
    'SmoothQuant:难度搬家(α=0.5 配平)',
]
for i, ln in enumerate(FACT):
    yy = FY + 62 + i * 30
    lc.rect(FX + 12, yy - 12, FW - 24, 24, '#f8fafc', GRID, rx=5, sw=1.0)
    lc.text(FX + 20, yy + 4, ln, 8.5, '#334155', 'start', maxw=FW - 40, tag='fl' + str(i))
lc.rect(FX + 12, FY + 168, FW - 24, 72, '#f1f5f9', lc.C_MUTE, rx=5, sw=1.2)
lc.text(FX + 20, FY + 186, '产出检查点:', 9, lc.C_TXT, 'start', True, maxw=FW - 40, tag='f:o1')
lc.text(FX + 20, FY + 204, 'qweight · scales · [g_idx(置换)]', 9, lc.C_TXT, 'start', True,
        maxw=FW - 40, tag='f:o2')
lc.text(FX + 20, FY + 226, '数学到此为止——运行期一行都看不见', 8, lc.C_MUTE, 'start',
        maxw=FW - 40, tag='f:o3')

# ================= 关卡 ①(配置期,回指 L0 外围) =================
GX, GY, GW, GH = MX, 486, 300, 154
lc.rect(GX, GY, GW, GH, '#ffffff', lc.C_MUTE, rx=8, sw=1.4)
gate(GX + 26, GY + 26, '1')
lc.text(GX + 46, GY + 30, '配置期:算力硬门', 11, lc.C_TXT, 'start', True, maxw=GW - 60,
        tag='g:h')
GLINES = [
    'get_min_capability 不满足 → 直接 ValueError',
    '门槛来自 kernel(docstring 原话:',
    'due to the custom CUDA kernels)',
    '',
    '回指 L0 外围:启动视角(VllmConfig 装配)',
]
for i, ln in enumerate(GLINES):
    if ln:
        col = lc.C_MUTE if i == 4 else '#334155'
        lc.text(GX + 16, GY + 54 + i * 17, ln, 8.3, col, 'start', maxw=GW - 28,
                tag='gl' + str(i))
lc.text(GX + 16, GY + GH - 12, 'base_config.py:L119-L126 · vllm.py:L706-L739', 7.5,
        lc.C_FAINT, 'start', maxw=GW - 28, tag='g:src')

# 工厂 → 关卡① → 绿带
lc.seg(FX + FW / 2, FY + FH, FX + FW / 2, GY, lc.C_MUTE, 2.0, 'std')
lc.seg(GX + GW, GY + GH / 2 - 30, 390, GY + GH / 2 - 30, lc.C_MUTE, 2.0, 'std')
lc.text(FX + FW / 2 + 8, FY + FH + 24, '检查点进卡', 8, lc.C_MUTE, 'start', tag='a:ck')

# ================= GPU 执行臂(恒绿) =================
BX_, BY_, BW_, BH_ = 390, 200, 1050, 470
lc.rect(BX_, BY_, BW_, BH_, lc.C_GPU_F, lc.C_GPU_S, rx=10, sw=2.2)
lc.text(BX_ + 16, BY_ + 24, 'GPU 执行臂(L0 中列,恒绿)', 12.5, lc.C_GPU_S, 'start', True,
        maxw=500, tag='b:h')
lc.text(BX_ + BW_ - 14, BY_ + 24, 'GPUModelRunner(构造/装载)→ 模型层 forward + 编译(每拍消费)', 8.5,
        lc.C_MUTE, 'end', maxw=520, tag='b:sub')

S_X, S_W = BX_ + 16, BW_ - 32


def station(y, h, n, title, lines, src, extra_chip=None):
    lc.rect(S_X, y, S_W, h, '#ffffff', lc.C_GPU_S, rx=7, sw=1.2)
    gate(S_X + 22, y + 20, n)
    lc.text(S_X + 42, y + 24, title, 10.5, lc.C_TXT, 'start', True, maxw=S_W - 60,
            tag='st' + n)
    yy = y + 42
    for ln in lines:
        lc.text(S_X + 16, yy, ln, 8.3, '#334155', 'start', maxw=S_W - 32, tag='stl' + n + ln[:8])
        yy += 15
    if src:
        lc.text(S_X + S_W - 12, y + h - 8, src, 7.5, lc.C_FAINT, 'end', maxw=S_W - 32,
                tag='sts' + n)
    if extra_chip:
        cx, cy, ct = extra_chip
        cw_ = lc.tw(ct, 8, True) + 12
        lc.rect(S_X + S_W - cw_ - 12, y + 10, cw_, 16, '#ffedd5', lc.C_ENG_S, rx=8, sw=1.0)
        lc.text(S_X + S_W - cw_ / 2 - 12, y + 21.5, ct, 8, lc.C_ENG_S, 'middle', True,
                maxw=cw_ - 4, tag='stc' + n)
    return y


# ---- ② 构造期(优先级表柜台) ----
S2_Y, S2_H = 236, 176
station(S2_Y, S2_H, '2', '构造期:柜台挑 kernel —— choose_mp_linear_kernel',
        ['逐个过三道闸:黑名单 / 算力 / can_implement,第一个全过的中选,构造期一次定死'],
        'kernels/linear/__init__.py:L411-L439(表)· L747-L789(循环)')
KERNELS = ['CutlassW4A8', 'Machete', 'AllSpark', 'Marlin',
           'Conch', 'Exllama', 'TritonW4A16', 'Humming']
HILITE = {'Machete': 'H100(cap 90)走它 · min 90', 'Marlin': 'A100(cap 80)落它 · min 75'}
kx0, ky0 = S_X + 16, S2_Y + 62
KW_, KH_, KG = 118, 30, 12
for i, k in enumerate(KERNELS):
    r, c = divmod(i, 4)
    x = kx0 + c * (KW_ + KG)
    y = ky0 + r * (KH_ + 10)
    hot = k in HILITE
    lc.rect(x, y, KW_, KH_, '#f0fdf4' if hot else '#f8fafc', lc.C_GPU_S if hot else GRID,
            rx=5, sw=1.4 if hot else 1.0)
    lc.text(x + KW_ / 2, y + 13, ('> ' if i in (1, 4) else '') + k, 8.2,
            lc.C_GPU_S if hot else '#334155', 'middle', True, maxw=KW_ - 8, tag='k' + k)
    lc.text(x + KW_ / 2, y + 25, HILITE.get(k, ''), 6.8, lc.C_GPU_S, 'middle',
            maxw=KW_ - 6, tag='kh' + k)
    if i < 7 and c < 3:
        lc.text(x + KW_ + KG / 2, y + 14, '>', 9, lc.C_MUTE, 'middle', tag='kg' + str(i))
lc.text(kx0, S2_Y + S2_H - 32, '同一 GPTQ 检查点,两张卡两个 kernel:Machete min_cap=90(machete.py:L26-L27)· Marlin min_cap=75(marlin.py:L37-L38)',
        8, '#334155', 'start', maxw=S_W - 32, tag='s2:v')

# ---- ③ 装载期 ----
S3_Y, S3_H = S2_Y + S2_H + 12, 76
station(S3_Y, S3_H, '3', '装载期:后厨换包装 —— process_weights_after_loading',
        ['全模型遍历:检查点格式 ≠ kernel 格式,重排成 kernel 自己的格式',
         'Marlin repack · FP8 转置合并 shard scale · NVFP4 alpha 预计算'],
        'model_loader/utils.py:L100-L122')

# ---- ④ 编译期 ----
S4_Y, S4_H = S3_Y + S3_H + 12, 92
station(S4_Y, S4_H, '4', '编译期:算子选择 —— 标准是「整条编译管线更快」',
        ['块状权重:强制 +quant_fp8 手工 kernel(注释自述 On H100 the CUDA kernel is faster)',
         'query 量化:刻意用普通 torch 算子,让 compile 融合 · fuse_* 三谓词开关'],
        'vllm.py:L1253-L1268 · L275-L290 · attention.py:L514-L524',
        extra_chip=(0, 0, '回收 ch19 编译章的伏笔'))

# ---- ⑤ 每拍 apply ----
S5_Y, S5_H = S4_Y + S4_H + 12, 42
lc.rect(S_X, S5_Y, S_W, S5_H, '#ffffff', lc.C_GPU_S, rx=7, sw=1.6)
lc.text(S_X + 16, S5_Y + 17, '⑤ 每拍 apply:W4A16 / W8A8 两条消费线——forward 每拍搬运的权重字节由格式决定(见下方带宽账)',
        8.8, lc.C_GPU_S, 'start', True, maxw=S_W - 32, tag='s5')
lc.text(S_X + 16, S5_Y + 33, 'GPTQ §5:almost all of the speedup is due to our kernels(动态反量化矩阵-向量积)',
        7.8, lc.C_MUTE, 'start', maxw=S_W - 32, tag='s5:q')

# 站间纵箭头
for y0, y1 in ((S2_Y + S2_H, S3_Y), (S3_Y + S3_H, S4_Y), (S4_Y + S4_H, S5_Y)):
    lc.seg(S_X + 22, y0, S_X + 22, y1, lc.C_GPU_S, 1.8, 'std')

# ================= 底部:带宽账 =================
BY = 690
lc.rect(MX, BY, 1380, 148, '#ffffff', GRID, rx=8, sw=1.2)
lc.text(MX + 16, BY + 20, '带宽账:为什么少搬字节 = 快(算术强度 = 16/bits,decode 矩阵-向量积)', 10.5,
        lc.C_TXT, 'start', True, maxw=1340, tag='bw:h')
# 左:强度小图(1/2/4 vs 165)
mini_x = MX + 30
base_y = BY + 118
for i, (bits, v, col) in enumerate([('FP16', 1, lc.C_API_S), ('INT8', 2, lc.C_ENG_S),
                                    ('INT4', 4, lc.C_GPU_S)]):
    x = mini_x + i * 90
    hgt = v * 4.5
    lc.rect(x, base_y - hgt, 46, max(hgt, 3), col, col, rx=2, sw=0)
    lc.text(x + 23, base_y - hgt - 8, str(v), 8, col, 'middle', True, tag='bwv' + bits)
    lc.text(x + 23, base_y + 12, bits, 8, lc.C_MUTE, 'middle', tag='bwn' + bits)
thr_y = base_y - 120
lc.seg(mini_x - 10, thr_y, mini_x + 300, thr_y, lc.C_ABORT, 1.4, dash=True)
lc.text(mini_x + 306, thr_y + 3, '4090 平衡点 165', 8, lc.C_ABORT, 'start', True,
        maxw=110, tag='bw:thr')
lc.rect(mini_x - 12, base_y, 314, 2, '#334155', '#334155', rx=1, sw=0)
lc.text(mini_x - 12, BY + 138, 'FLOPs/字节——三种格式全部深陷 memory-bound 区', 7.8,
        lc.C_MUTE, 'start', maxw=340, tag='bw:axis')
# 右:字节账 + 实测
TX0 = MX + 420
lc.text(TX0, BY + 44, '4096×4096 层权重字节:33554432(FP16) → 8388608(INT4) = 4 倍', 9,
        '#334155', 'start', True, maxw=640, tag='bw:bytes')
lc.text(TX0, BY + 66, '论文实测(3-bit OPT-175B,batch 1,len 128):A100 230ms → 71ms = 3.24×', 9,
        lc.C_GPU_S, 'start', True, maxw=640, tag='bw:t1')
lc.text(TX0, BY + 84, 'A6000 589ms → 130ms = 4.53×(GPTQ §5 Table 6,逐字)', 9, lc.C_GPU_S,
        'start', True, maxw=640, tag='bw:t2')
lc.text(TX0, BY + 108, '§6 定性:speedups from reduced memory movement,', 8.5, lc.C_MUTE,
        'start', maxw=640, tag='bw:q1')
lc.text(TX0, BY + 124, 'and does not lead to computational reductions', 8.5, lc.C_MUTE,
        'start', maxw=640, tag='bw:q2')
# 中:强度公式
lc.text(MX + 780, BY + 66, '强度 = FLOPs / 权重字节', 9.5, lc.C_TXT, 'start', True,
        maxw=280, tag='bw:f1')
lc.text(MX + 780, BY + 84, '= 2/(bits/8) = 16/bits', 9.5, lc.C_TXT, 'start', True,
        maxw=280, tag='bw:f2')
lc.text(MX + 780, BY + 108, 'AWQ §4.1:any workload with', 8, lc.C_MUTE, 'start', maxw=300,
        tag='bw:aw1')
lc.text(MX + 780, BY + 124, 'intensity < 165 is memory bounded', 8, lc.C_MUTE, 'start',
        maxw=300, tag='bw:aw2')

# ================= 页脚 =================
lc.text(MX, BY + 164, 'vllm/model_executor/kernels/linear/__init__.py:L411-L439 · L747-L789 · mixed_precision/machete.py:L26-L27 / marlin.py:L37-L38 · model_loader/utils.py:L100-L122 · vllm/config/vllm.py:L706-L739 · L1253-L1268 · L275-L290',
        8.5, lc.C_FAINT, 'start', maxw=1380, tag='ft1')
lc.text(MX, BY + 182, 'attention.py:L514-L524 · 论文口径 arXiv:2210.17323 §5-§6(Table 6 逐字)· arXiv:2306.00978 §4.1 · 数值:本章参考实现实跑 + pin 源码锚点 · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=1380, tag='ft2')

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch27-fig-quant-lifecycle.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
