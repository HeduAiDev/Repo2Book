#!/usr/bin/env python3
"""ch25 机制图 · MQA 吸收腿:同一分数、两条算路(figure ch25-fig-mqa-leg,模板 before-after)

放大自 L0『模型层 MLA 框』一拍前向的 MQA 腿展开——L2 站 13(bmm 吸收→forward_mqa→_v_up_proj)。
上半:结合律双路对照(路线 A 先上投影 K / 路线 B 先吸收 q,同值 1.569422、差 0.000000);
下半:B 路线的批量三站(bmm 吸收 → 单头 MQA kernel 只读 cache → v_up bmm)+ 字节账双箭头。

claim:decode 腿用乘法结合律把『每头上投影 K』换成『把 q 先吸收进潜空间』:两条路线算出
同一分数(本例 1.569422,差 0.000000),cache 里 576 维潜向量一行不动、不为 128 头物化任何 K。

数字全部取自 figure spec 的 numbers(结合律 1.569422/0.000000 · bmm 形状账
(4,3,16)×(4,16,512)→(4,3,512) 与 (4,3,512)×(4,512,16)→(4,3,16) · (3,4,576) ·
DSV3 fp16 25 行 28800 B vs 2048000 B · 两次 bmm 各 196608 MAC · 三锚点 L888/L919/L949)。
坐标常量/循环;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 860
MX = 52
BXR = 1448
C_LAT = lc.C_API_S
C_LAT_T = '#1e40af'

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'MQA 吸收腿:同一分数、两条算路——乘法结合律让 cache 一行不动',
        16, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, 'decode 每步只有零星新 token:不为 128 个头物化 K,而是把每头 q 先乘进潜空间,与同一份 576 维潜向量点积',
        10.5, lc.C_MUTE, 'start', maxw=1060, tag='subtitle')
_ch = '放大自 L0『模型层 MLA 框』· L2 站 13 的机制展开'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 上半:结合律双路对照 =================
lc.text(MX, 94, '上半 · 一个分数的两种算法(取 head 0、请求 0、其第 7 个历史 token 实测)', 10.5,
        lc.C_TXT, 'start', True, maxw=700, tag='up:t')

RAISE_Y = 128          # 两路线面板顶
PNL_H = 174            # 面板高

# ---- 路线 A(左):先上投影 K ----
AX0, AW = 60, 560
lc.rect(AX0, RAISE_Y, AW, PNL_H, '#ffffff', lc.C_MUTE, rx=9, sw=1.4)
lc.text(AX0 + AW / 2, RAISE_Y + 20, '路线 A · 先上投影 K(复印词典)', 10.5, lc.C_TXT, 'middle', True,
        tag='ra:t')
lc.rect(AX0 + 24, RAISE_Y + 36, 150, 28, '#eff6ff', C_LAT, rx=5, sw=1.4)
lc.text(AX0 + 99, RAISE_Y + 54, 'c(512 维潜向量)', 8.5, C_LAT_T, 'middle', True, maxw=142, tag='ra:c')
lc.seg(AX0 + 174, RAISE_Y + 50, AX0 + 234, RAISE_Y + 50, lc.C_GPU_S, 1.6, 'std')
lc.text(AX0 + 204, RAISE_Y + 44, '× W_UK', 7.5, lc.C_GPU_S, 'middle', tag='ra:w')
lc.rect(AX0 + 234, RAISE_Y + 36, 130, 28, '#dcfce7', lc.C_GPU_S, rx=5, sw=1.4)
lc.text(AX0 + 299, RAISE_Y + 54, 'k_nope(16 维)', 8.5, '#166534', 'middle', True, maxw=122, tag='ra:k')
lc.text(AX0 + 460, RAISE_Y + 54, '每头复印一份 K', 8, lc.C_MUTE, 'middle', tag='ra:n1')
lc.rect(AX0 + 24, RAISE_Y + 84, 130, 28, '#dcfce7', lc.C_GPU_S, rx=5, sw=1.4)
lc.text(AX0 + 89, RAISE_Y + 102, 'q_nope(16 维)', 8.5, '#166534', 'middle', True, maxw=122, tag='ra:q')
DOTA = (AX0 + 190, RAISE_Y + 84, 200, 28)
lc.rect(*DOTA, '#f8fafc', lc.C_MUTE, rx=5, sw=1.2)
lc.text(DOTA[0] + DOTA[2] / 2, RAISE_Y + 102, 'dot(q_nope, k_nope)', 8.5, lc.C_TXT, 'middle', True,
        tag='ra:d')
lc.seg(AX0 + 154, RAISE_Y + 98, DOTA[0], RAISE_Y + 98, lc.C_GPU_S, 1.6, 'std')
lc.parrow([(AX0 + 299, RAISE_Y + 64), (AX0 + 299, DOTA[1] - 22), (DOTA[0] + 150, DOTA[1] - 22),
           (DOTA[0] + 150, DOTA[1])], lc.C_GPU_S, 1.5, 'std')
lc.text(AX0 + AW / 2, RAISE_Y + 142, '先还原成 16 维 k 再点积', 8, lc.C_MUTE, 'middle', tag='ra:f')

# ---- 路线 B(右):先吸收 q ----
BX0, BW_ = 880, 560
lc.rect(BX0, RAISE_Y, BW_, PNL_H, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.8)
lc.text(BX0 + BW_ / 2, RAISE_Y + 20, '路线 B · 先吸收 q(把提问翻译成页码语言)', 10.5, '#166534',
        'middle', True, tag='rb:t')
lc.rect(BX0 + 24, RAISE_Y + 36, 130, 28, '#dcfce7', lc.C_GPU_S, rx=5, sw=1.4)
lc.text(BX0 + 89, RAISE_Y + 54, 'q_nope(16 维)', 8.5, '#166534', 'middle', True, maxw=122, tag='rb:q')
lc.seg(BX0 + 154, RAISE_Y + 50, BX0 + 214, RAISE_Y + 50, lc.C_GPU_S, 1.6, 'std')
lc.text(BX0 + 184, RAISE_Y + 44, '× W_UK^T', 7.5, lc.C_GPU_S, 'middle', tag='rb:w')
lc.rect(BX0 + 214, RAISE_Y + 36, 160, 28, '#eff6ff', C_LAT, rx=5, sw=1.4)
lc.text(BX0 + 294, RAISE_Y + 54, 'q_l(512 维潜语言)', 8.5, C_LAT_T, 'middle', True, maxw=152, tag='rb:ql')
lc.text(BX0 + 460, RAISE_Y + 54, '翻译,不复印', 8, lc.C_MUTE, 'middle', tag='rb:n1')
lc.rect(BX0 + 24, RAISE_Y + 84, 150, 28, '#eff6ff', C_LAT, rx=5, sw=1.4)
lc.text(BX0 + 99, RAISE_Y + 102, 'c(512 维潜向量)', 8.5, C_LAT_T, 'middle', True, maxw=142, tag='rb:c')
DOTB = (BX0 + 210, RAISE_Y + 84, 160, 28)
lc.rect(*DOTB, '#f8fafc', lc.C_MUTE, rx=5, sw=1.2)
lc.text(DOTB[0] + DOTB[2] / 2, RAISE_Y + 102, 'dot(q_l, c)', 8.5, lc.C_TXT, 'middle', True, tag='rb:d')
lc.seg(BX0 + 174, RAISE_Y + 98, DOTB[0], RAISE_Y + 98, lc.C_GPU_S, 1.6, 'std')
lc.parrow([(BX0 + 294, RAISE_Y + 64), (BX0 + 294, DOTB[1] - 22), (DOTB[0] + 120, DOTB[1] - 22),
           (DOTB[0] + 120, DOTB[1])], lc.C_GPU_S, 1.5, 'std')
lc.text(BX0 + BW_ / 2, RAISE_Y + 142, 'q 换说潜语言,与整份潜向量点积', 8, lc.C_MUTE, 'middle', tag='rb:f')

# ---- 中间等号 + 双分数 ----
lc.circle(750, RAISE_Y + 74, 28, lc.C_GPU_S, 2.0, dash=False)
lc.text(750, RAISE_Y + 82, '=', 22, lc.C_GPU_S, 'middle', True, tag='eq')
EQB = (560, RAISE_Y + PNL_H + 12, 380, 30)
lc.rect(*EQB, '#f0fdf4', lc.C_GPU_S, rx=6, sw=1.5)
lc.text(EQB[0] + EQB[2] / 2, EQB[1] + 20, '两路分数同为 1.569422,差 0.000000', 10, '#166534',
        'middle', True, maxw=370, tag='eq:v')
lc.text(750, EQB[1] + 48, 'dot(q, c·W_UK) = dot(q·W_UK_T, c)——矩阵乘结合律,先算哪个括号不改变结果',
        9, lc.C_MUTE, 'middle', maxw=640, tag='eq:f')

# ================= 下半:B 路线批量三站 =================
lc.text(MX, EQB[1] + 72, '下半 · 路线 B 的批量形态(B=3 请求 × N=4 头,全批 decode)', 10.5,
        lc.C_TXT, 'start', True, maxw=640, tag='lo:t')
lc.text(1120, EQB[1] + 72, '与上投影 MHA 参照 max diff 0.000000', 9, '#166534', 'start', True,
        maxw=300, tag='lo:v')

LB_Y = EQB[1] + 86
LB_H = 148
st_names = [
    ('① bmm 吸收', 'torch.bmm(q_nope, W_UK_T)', '(4,3,16)×(4,16,512) → (4,3,512)',
     '每头 q 换说 512 维潜语言', 'mla_attention.py:L888'),
    ('② 拼 rope 段 → 单头 MQA kernel', 'forward_mqa', 'q (3,4,576) 直接读分页 cache',
     'kernel 只见 1 个 KV『头』', 'mla_attention.py:L919'),
    ('③ 输出上投影', '_v_up_proj:bmm 乘 W_UV', '(4,3,512)×(4,512,16) → (4,3,16)',
     '输出 [3,64]=N×V 写回 decode 段', 'mla_attention.py:L949'),
]
bw_, bgap = 420, 40
bx0 = MX + 8
for i, (t, m, shp, note, anc) in enumerate(st_names):
    x = bx0 + i * (bw_ + bgap)
    lc.rect(x, LB_Y, bw_, LB_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.6)
    lc.text(x + bw_ / 2, LB_Y + 24, t, 10.5, '#166534', 'middle', True, maxw=bw_ - 16, tag='st%d:t' % i)
    lc.text(x + bw_ / 2, LB_Y + 47, m, 8.5, lc.C_TXT, 'middle', True, maxw=bw_ - 16, tag='st%d:m' % i)
    lc.text(x + bw_ / 2, LB_Y + 67, shp, 8.5, lc.C_GPU_S, 'middle', maxw=bw_ - 16, tag='st%d:s' % i)
    lc.text(x + bw_ / 2, LB_Y + 90, note, 8, '#334155', 'middle', maxw=bw_ - 16, tag='st%d:n' % i)
    lc.text(x + bw_ / 2, LB_Y + 130, anc, 7.5, lc.C_FAINT, 'middle', tag='st%d:a' % i)
    if i < 2:
        lc.seg(x + bw_, LB_Y + LB_H / 2, x + bw_ + bgap, LB_Y + LB_H / 2, lc.C_GPU_S, 2.0, 'std')

# cache 窄条(② 下方,只读)
CB = (bx0 + bw_ + bgap, LB_Y + LB_H + 16, bw_, 44)
lc.rect(*CB, '#ecfeff', lc.C_KV_S, rx=6, sw=1.6)
lc.text(CB[0] + CB[2] / 2, CB[1] + 18, '分页 cache(只读)——每行 576 维潜向量', 8.5, '#155e75',
        'middle', True, maxw=CB[2] - 14, tag='cache:t')
lc.text(CB[0] + CB[2] / 2, CB[1] + 34, 'decode_seq_lens 8/4/13,本步共读 25 行', 8, '#155e75', 'middle',
        tag='cache:s')
lc.seg(CB[0] + CB[2] / 2, LB_Y + LB_H, CB[0] + CB[2] / 2, CB[1], lc.C_KV_S, 1.8, 'std')
lc.text(CB[0] + CB[2] / 2, CB[1] + 60, '写腿早在 forward 入口完成(站 10)——这条腿只读', 7.5,
        lc.C_MUTE, 'middle', tag='cache:n')

# ---------------- 字节账(底部双箭头) ----------------
BY = CB[1] + 76
lc.rect(60, BY, 1384, 120, lc.C_KV_F, lc.C_KV_S, rx=8, sw=1.5)
lc.text(80, BY + 22, '读字节账:decode 每步把全部历史 KV 搬进计算单元(DSV3 fp16,25 行 cache)', 10,
        lc.C_KV_S, 'start', True, maxw=800, tag='ba:t')
ay = BY + 46
lc.seg(100, ay, 148, ay, C_LAT, 6.0, 'std')
lc.text(158, ay + 4, 'MQA 腿 28800 B——576 维单份潜向量 × 25 行 × 2B', 9, C_LAT_T, 'start', True,
        maxw=520, tag='ba:mq')
ay2 = BY + 80
lc.seg(100, ay2, 640, ay2, lc.C_ABORT, 14.0, 'std')
lc.text(652, ay2 + 4, 'MHA 等价 2048000 B——40960 维全展开 × 25 行 × 2B(71 倍)', 9, lc.C_ABORT,
        'start', True, maxw=560, tag='ba:mh')
lc.text(1050, BY + 44, '代价两笔明码:', 9, lc.C_TXT, 'start', True, maxw=200, tag='ba:c')
lc.text(1050, BY + 62, '· 每 decode token 多两次 bmm:吸收 196608 + 上投影 196608 MAC', 8,
        '#334155', 'start', maxw=390, tag='ba:c1')
lc.text(1050, BY + 80, '· W_UK_T/W_UV bmm 副本与原权重同驻(站 4 重排)', 8, '#334155', 'start',
        maxw=390, tag='ba:c2')
lc.text(1050, BY + 103, '算得多一点、搬得少一个数量级', 9, lc.C_KV_S, 'start', True, maxw=390, tag='ba:c3')

# ---------------- 页脚 ----------------
lc.text(MX, BY + 140, '图例:绿 = 执行流(GPU 执行臂角色) · 蓝 = 潜向量/潜空间(延续 ch24 蓝框语义) · 青 = cache 只读 · 箭头粗细 = 搬运字节量',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, BY + 160, '数值 = host 实测(vLLM v0.27.1 精简版实跑,float32;字节账按源码口径 DSV3 fp16 单列) · 行号基线 v0.27.1(6e448d0ea)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, BY + 180, '前提有二:nope 段无 RoPE(上一张图的解耦不变量)· W_UK_T/W_UV 与 kv_b_proj 拆自同一权重(站 4 重排只动排布不动数值)',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch25-fig-mqa-leg.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
