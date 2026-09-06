#!/usr/bin/env python3
"""ch21 机制图 ① · AttentionBackend 协议(figure_spec ch21-fig-m01-protocol,模板 flow)

放大自 L0 GPU 执行臂(绿)·模型层『模型层 forward + 编译』框:Attention() 插座背面(站 1-4)。
把插座背面那一圈『接口簧片』(协议)单独拉出来画:四张抽象 staticmethod 身份证 + 能力探针族 +
validate_configuration 聚合器。架构归属回指 L0(FIGURE-SYSTEM §3.3),不另立第二种架构画法。

claim:协议=四个 staticmethod 身份证(get_name/get_impl_cls/get_builder_cls/get_kv_cache_shape)
+ 能力探针族 + validate_configuration 聚合器(失败逐条追加原因、空列表=合法),全 staticmethod
因为选后端发生在『还没实例化任何东西』的装配期。

数字全部取自 figure_spec.numbers:四张抽象 staticmethod(backend.py:L60-L97)、
validate_configuration 语义(L319-L393)、forward_includes_kv_cache_update 默认 True/
FlashAttentionBackend False(L67 + flash_attn.py:L86)——pin 源码逐字。
坐标由常量/循环计算;文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 782
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, '插座背面:接上 Attention() 之前,后端先交四张「类级别」身份证',
        16.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '能跑哪个 kernel 的谈判发生在装配期——用的是类与静态声明,不是对象;身份证全是 staticmethod,正因为谈判时连一个实例都不存在',
        10.5, lc.C_MUTE, 'start', maxw=1030, tag='subtitle')
_ch = '放大自 L0 GPU 执行臂(绿)·模型层 Attention() 插座背面(站 1-4)'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 装配期时间线(为什么全 staticmethod) ----------------
lc.text(MX, 108, '装配期时间线——身份证为什么必须是 staticmethod', 10.5, lc.C_GPU_S,
        'start', True, maxw=560, tag='tl:t')
lc.text(BXR, 108, '站号 = 本章 L2 章图的请求走线站号', 9, lc.C_FAINT, 'end', maxw=300, tag='tl:r')

TL_Y, TL_H, TL_W, TL_GAP = 120, 92, 318, 36
tl_nodes = [
    ('站 1 · 建层', ['每层 Attention.__init__:构造方没显式', '传后端时,调 get_attn_backend(...)',
                     '此刻:只有类对象,零实例'], False),
    ('站 2-3 · 打包 + 查表', ['零散参数收进可哈希的 AttentionSelector-', 'Config;平台按 use_mla×算力代给出候选',
                              '名单(表序 = 经验参数,benchmark 出的)'], False),
    ('站 4 · 逐个 validate', ['对候选依序翻身份证 + 过探针;没装的', '(ImportError)也只当一条落选原因',
                              '本图:协议在这里被翻检'], True),
    ('站 6-7 · 建组定形', ['get_impl_cls() / get_builder_cls()', '这时才 new 出 impl 与 builder',
                           '实例出现在谈判之后(虚线 = 本章后文)'], False),
]
tl_x = [MX + i * (TL_W + TL_GAP) for i in range(4)]
for i, (t, lines, hot) in enumerate(tl_nodes):
    x = tl_x[i]
    if hot:
        lc.rect(x, TL_Y, TL_W, TL_H, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.8)
        ct, cl = lc.C_BEAT_T, lc.C_BEAT_T
    else:
        lc.rect(x, TL_Y, TL_W, TL_H, '#ffffff', lc.C_GPU_S, rx=7, sw=1.4,
                dash=(i == 3))
        ct, cl = lc.C_GPU_S, '#334155'
    lc.text(x + 12, TL_Y + 20, t, 10, ct, 'start', True, maxw=TL_W - 24, tag=f'tl{i}:t')
    for k, ln in enumerate(lines):
        c = cl if k < 2 else (lc.C_MUTE if i != 3 else '#94a3b8')
        lc.text(x + 12, TL_Y + 37 + k * 15, ln, 8.2, c, 'start',
                maxw=TL_W - 22, tag=f'tl{i}:l{k}')
    if i < 3:
        lc.seg(x + TL_W, TL_Y + TL_H / 2, x + TL_W + TL_GAP, TL_Y + TL_H / 2,
               lc.C_MUTE, 1.6, 'std')
lc.text(MX + (TL_W * 4 + TL_GAP * 3) / 2, TL_Y + TL_H + 18,
        '谈判窗口(站 1-4)没有实例 → 问题必须是「类能答的」→ 四张身份证全 staticmethod',
        9, lc.C_MUTE, 'middle', maxw=1200, tag='tl:note')

# ---------------- 左:AttentionBackend 协议卡 ----------------
P_X, P_Y, P_W, P_H = MX, 252, 700, 366
lc.rect(P_X, P_Y, P_W, P_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(P_X + 16, P_Y + 22, 'AttentionBackend(ABC)——插座背面的接口簧片', 11.5, lc.C_GPU_S,
        'start', True, maxw=520, tag='p:t')
lc.text(P_X + P_W - 14, P_Y + 22, 'vllm/v1/attention/backend.py', 9, lc.C_FAINT, 'end',
        maxw=200, tag='p:file')
lc.text(P_X + 16, P_Y + 38, '协议只规定「插孔长什么样」,不规定吹风机内部怎么造风——每个后端各造各的 kernel',
        8.5, lc.C_MUTE, 'start', maxw=P_W - 30, tag='p:sub')

ID_W, ID_H, ID_GAP = 326, 94, 12
id_cards = [
    ('get_name()', '后端名——日志与枚举反查登记的键', ''),
    ('get_impl_cls()', '计算体 impl:forward(读腿)', '+ do_kv_cache_update(写腿)'),
    ('get_builder_cls()', 'metadata builder:Common 通用单', '→ 后端专属单的翻译官'),
    ('get_kv_cache_shape(num_blocks,', 'block_size, num_kv_heads, head_size)', 'KV 张量逻辑形(K 与 V 打包进 content 维)'),
]
for k, (name, d1, d2) in enumerate(id_cards):
    cx = P_X + 16 + (k % 2) * (ID_W + ID_GAP)
    cy = P_Y + 48 + (k // 2) * (ID_H + ID_GAP)
    lc.rect(cx, cy, ID_W, ID_H, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.3)
    lc.text(cx + 12, cy + 19, name, 9.5, lc.C_TXT, 'start', True, maxw=ID_W - 100, tag=f'id{k}:n')
    lc.text(cx + 12, cy + 37, d1, 8.2, '#334155', 'start', maxw=ID_W - 22, tag=f'id{k}:d1')
    if d2:
        lc.text(cx + 12, cy + 52, d2, 8.2, '#334155', 'start', maxw=ID_W - 22, tag=f'id{k}:d2')
    _tag = 'staticmethod'
    _tw = lc.tw(_tag, 7.5)
    lc.rect(cx + ID_W - _tw - 20, cy + 6, _tw + 12, 15, '#ffffff', lc.C_GPU_S, rx=7, sw=1.0)
    lc.text(cx + ID_W - _tw / 2 - 14, cy + 17, _tag, 7.5, lc.C_GPU_S, 'middle', maxw=_tw + 8, tag=f'id{k}:tag')
lc.text(P_X + 16, P_Y + 48 + 2 * ID_H + ID_GAP + 20,
        '四个全是 @abstractmethod + @staticmethod:连实例都没有就要能答——这是「身份证」的含义',
        8.8, lc.C_MUTE, 'start', maxw=P_W - 30, tag='p:note')

PR_Y = P_Y + 48 + 2 * (ID_H + ID_GAP) + 30
lc.rect(P_X + 16, PR_Y, P_W - 32, P_Y + P_H - PR_Y - 12, '#ffffff', lc.C_MUTE, rx=6,
        sw=1.1, dash=True)
lc.text(P_X + 28, PR_Y + 17, '能力探针族 supports_*(共十几个,逐个返回 bool)', 9, lc.C_TXT,
        'start', True, maxw=400, tag='pr:t')
probes = ['supports_head_size', 'supports_dtype', 'supports_kv_cache_dtype',
          'supports_block_size', 'supports_compute_capability', 'supports_attn_type',
          'supports_sliding_window', 'supports_non_causal', 'supports_mm_prefix',
          '… 等十几个(见聚合器逐项)']
for k, p in enumerate(probes):
    col, row = k // 5, k % 5
    lc.text(P_X + 28 + col * 336, PR_Y + 34 + row * 14, p, 8, '#475569', 'start',
            maxw=330, tag=f'pr{k}')

# ---------------- 右:validate_configuration 聚合器 ----------------
V_X, V_W = 790, 650
lc.rect(V_X, P_Y, V_W, P_H, '#ffffff', lc.C_GPU_S, rx=8, sw=1.8)
lc.text(V_X + 16, P_Y + 22, 'validate_configuration——聚合器:「合法」= 原因清单为空', 11.5,
        lc.C_GPU_S, 'start', True, maxw=520, tag='v:t')
lc.text(V_X + V_W - 14, P_Y + 22, 'backend.py:L319-L393', 9, lc.C_FAINT, 'end',
        maxw=180, tag='v:file')
lc.text(V_X + 16, P_Y + 38, '入参 = 全部配置维度(head_size/dtype/kv_cache_dtype/block_size/use_mla/…)',
        8.5, lc.C_MUTE, 'start', maxw=V_W - 30, tag='v:sub')

S_X, S_W = V_X + 16, V_W - 32
s1_y, s1_h = P_Y + 48, 84
lc.rect(S_X, s1_y, S_W, s1_h, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.3)
lc.text(S_X + 12, s1_y + 17, '十几个探针逐个检查,失败就追加一条字符串原因(源码字面):', 8.8,
        lc.C_TXT, 'start', True, maxw=S_W - 24, tag='s1:t')
lc.text(S_X + 24, s1_y + 34, 'if not cls.supports_head_size(head_size):', 8.2, '#334155',
        'start', maxw=S_W - 40, tag='s1:c1')
lc.text(S_X + 24, s1_y + 48, 'invalid_reasons.append("head_size not supported")', 8.2, '#334155',
        'start', maxw=S_W - 40, tag='s1:c2')
lc.text(S_X + 24, s1_y + 64, '⋮ 逐项:dtype / kv_cache_dtype / block_size / use_mla / sink / sparse / compute_capability / attn_type / …',
        7.8, lc.C_MUTE, 'start', maxw=S_W - 36, tag='s1:c3')
lc.seg(S_X + S_W / 2, s1_y + s1_h, S_X + S_W / 2, s1_y + s1_h + 14, lc.C_GPU_S, 1.6, 'std')

s2_y = s1_y + s1_h + 14
s2_h = 44
lc.rect(S_X, s2_y, S_W, s2_h, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.3)
lc.text(S_X + 12, s2_y + 17, 'supports_combination(head_size, dtype, …) 组合探针收尾', 8.8,
        lc.C_TXT, 'start', True, maxw=S_W - 24, tag='s2:t')
lc.text(S_X + 12, s2_y + 33, '单查「组合」是否合法:返回 None 或一条原因,同样追加进清单', 8.2,
        '#334155', 'start', maxw=S_W - 24, tag='s2:l')
lc.seg(S_X + S_W / 2, s2_y + s2_h, S_X + S_W / 2, s2_y + s2_h + 14, lc.C_GPU_S, 1.6, 'std')

s3_y = s2_y + s2_h + 14
s3_h = 34
lc.rect(S_X, s3_y, S_W, s3_h, '#ffffff', lc.C_GPU_S, rx=6, sw=1.3)
lc.text(S_X + 12, s3_y + 21, 'return invalid_reasons  —— 字符串清单,不是布尔', 9, lc.C_TXT,
        'start', True, maxw=S_W - 24, tag='s3:t')

OC_Y = s3_y + s3_h + 12
OC_H = P_Y + P_H - OC_Y - 12
oc_w = (S_W - 12) / 2
lc.rect(S_X, OC_Y, oc_w, OC_H, lc.C_GPU_F, lc.C_GPU_S, rx=6, sw=1.5)
lc.text(S_X + oc_w / 2, OC_Y + 20, '返回 [ ] 空清单', 10, lc.C_GPU_S, 'middle', True,
        maxw=oc_w - 12, tag='ocA:t')
lc.text(S_X + oc_w / 2, OC_Y + 38, '= 该后端在此配置下合法 ✓', 8.8, lc.C_GPU_S, 'middle',
        maxw=oc_w - 12, tag='ocA:l')
lc.text(S_X + oc_w / 2, OC_Y + 54, '幸存候选再取 min(priority) 定胜者', 8, lc.C_MUTE, 'middle',
        maxw=oc_w - 12, tag='ocA:n')
ox = S_X + oc_w + 12
lc.rect(ox, OC_Y, oc_w, OC_H, '#ffffff', lc.C_BEAT_S, rx=6, sw=1.5)
lc.text(ox + oc_w / 2, OC_Y + 20, '非空 = 逐条落选原因', 10, lc.C_BEAT_T, 'middle', True,
        maxw=oc_w - 12, tag='ocB:t')
lc.text(ox + oc_w / 2, OC_Y + 38, '例:"head_size not supported" · "block_size not supported"', 8,
        lc.C_BEAT_T, 'middle', maxw=oc_w - 12, tag='ocB:l')
lc.text(ox + oc_w / 2, OC_Y + 54, '没装依赖(ImportError)也当一条原因,不挡后面的候选', 8, lc.C_MUTE,
        'middle', maxw=oc_w - 12, tag='ocB:n')

# ---------------- 底条:forward_includes_kv_cache_update ----------------
FB_Y = P_Y + P_H + 22
lc.text(MX, FB_Y + 14, '协议里唯一的「行为开关」:forward_includes_kv_cache_update——写 cache 归不归注意力算子管',
        10, lc.C_TXT, 'start', True, maxw=900, tag='fb:t')
fb_y = FB_Y + 24
lc.rect(MX, fb_y, 660, 48, '#ffffff', lc.C_MUTE, rx=7, sw=1.4)
lc.text(MX + 14, fb_y + 19, '默认 True:注意力前向里顺手把 K/V 写进 cache', 9, '#334155',
        'start', True, maxw=632, tag='fb:a1')
lc.text(MX + 14, fb_y + 36, '后端不拆腿,写与算在同一份 forward 里完成(backend.py:L67)', 8, lc.C_MUTE,
        'start', maxw=632, tag='fb:a2')
lc.rect(MX + 680, fb_y, 700, 48, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(MX + 694, fb_y + 19, 'FlashAttentionBackend = False:先散写(写腿)再算(读腿),两腿拆开', 9,
        lc.C_BEAT_T, 'start', True, maxw=672, tag='fb:b1')
lc.text(MX + 694, fb_y + 36, '(flash_attn.py:L86)写腿与读腿怎么各走各的,见本章后文两幅机制图', 8, lc.C_MUTE,
        'start', maxw=672, tag='fb:b2')

# ---------------- 页脚 ----------------
FY = fb_y + 74
lc.text(MX, FY, '图例:绿 = 装配期协议面(L0 GPU 执行臂·模型层) · 橙 = 本图主角站 / FA 的行为差异 · 灰虚线 = 本章后文展开的上下文',
        8.5, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:leg')
lc.text(MX, FY + 18, 'vllm/v1/attention/backend.py:L60-L97(四张抽象 staticmethod)· L319-L393(validate_configuration 逐探针追加)· vllm/v1/attention/backends/flash_attn.py:L86(forward_includes_kv_cache_update=False)· 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:src')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch21-fig-m01-protocol.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
