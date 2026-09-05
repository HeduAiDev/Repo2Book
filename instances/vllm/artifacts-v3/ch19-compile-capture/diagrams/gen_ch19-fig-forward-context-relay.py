#!/usr/bin/env python3
"""ch19 机制图 2 · forward context 三段接力（figure_spec ch19-fig-forward-context-relay，模板 flow）

放大自 L0『GPU 执行臂』中列『执行臂中层』——即本章 L2 章图 center ② 拍片
『构造期算子层』（站 4，Attention 自注册）与 center ⑧ 拍片『喂图·算子与回放』
（站 13，set_forward_context 注入）之间的接力展开。架构归属回指 L0/L2：右上角指北小签。

claim：attention 的执行环境三段接力——构造期自注册层表、每拍全局变量注入、
算子内按 layer_name 取回——模型 forward 签名从此不见 attn_metadata。

数字/引语全部取自 figure_spec.numbers（attention.py:L443-L446 自注册+重名 raise；
forward_context.py:L259-L344 set_forward_context + L229-L231 no_compile_layers 拷贝 +
L199-L205 不设 context assert 崩；attention.py:L732-L772 op 内取回）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX = 60
BXR = 1440

# ---------------- 标题区 ----------------
lc.text(MX, 34, 'attn_metadata 透传的终结：三段接力把执行环境搬进模块级全局变量',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '构造期每层自注册 → 每拍 runner 注入 → 算子内按 layer_name 现取——模型定义与执行环境解耦，模型文件签名不再被绑架',
        10.5, lc.C_MUTE, 'start', maxw=1010, tag='subtitle')
_ch = '放大自 L2 拍片 ②⇢⑧ · L0：GPU 执行臂中层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 三段泳道 ----------------
SW_, GAP = 400, 90
S1X = MX
S2X = S1X + SW_ + GAP
S3X = S2X + SW_ + GAP
SY, SH_ = 96, 470

def stage(x, no, title, sub, color, fill):
    lc.rect(x, SY, SW_, SH_, fill, color, rx=10, sw=2.0)
    lc.rect(x, SY, SW_, 40, '#ffffff', color, rx=10, sw=0)
    lc.rect(x, SY + 20, 44, 24, lc.C_BADGE_F, color, rx=8, sw=1.2)
    lc.text(x + 22, SY + 16.5, no, 11, lc.C_ENG_S, 'middle', True, maxw=40, tag='no' + no)
    lc.text(x + 54, SY + 16.5, title, 11, color, 'start', True, maxw=SW_ - 130, tag='st' + no)
    lc.text(x + SW_ - 12, SY + 16.5, sub, 8, lc.C_FAINT, 'end', maxw=170, tag='stsu' + no)

stage(S1X, '段①', '构造期 · 自注册层表', 'load_model 一次', lc.C_BEAT_S, '#ffffff')
stage(S2X, '段②', '每拍 · 全局变量注入', 'execute_model 每拍', lc.C_GPU_S, '#ffffff')
stage(S3X, '段③', 'op 内 · 按 layer_name 取回', '前向每层一次', lc.C_BEAT_S, '#ffffff')

# ===== 段① Attention.__init__ → static_forward_context =====
AY = SY + 64
lc.rect(S1X + 20, AY, SW_ - 40, 96, '#ffffff', lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(S1X + 34, AY + 22, 'Attention.__init__（每层一次）', 9.5, lc.C_TXT, 'start', True,
        maxw=SW_ - 68, tag='a:t')
lc.text(S1X + 34, AY + 40, 'vllm/model_executor/layers/attention/attention.py:L443-L446',
        7.6, lc.C_FAINT, 'start', maxw=SW_ - 68, tag='a:f')
lc.text(S1X + 34, AY + 58, 'if prefix in …static_forward_context:', 8.2, '#334155', 'start',
        maxw=SW_ - 68, tag='a:l1')
lc.text(S1X + 34, AY + 74, '    raise ValueError(f"Duplicate layer name: {prefix}")', 8.2,
        lc.C_ABORT, 'start', maxw=SW_ - 68, tag='a:l2')

DXY = SY + 186
lc.rect(S1X + 20, DXY, SW_ - 40, 120, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(S1X + 34, DXY + 22, 'compilation_config.static_forward_context', 9.2, lc.C_BEAT_T,
        'start', True, maxw=SW_ - 68, tag='d:t')
lc.text(S1X + 34, DXY + 38, '{ prefix: 层实例 }（模型定义域产出的层注册表）', 8.2, '#334155',
        'start', maxw=SW_ - 68, tag='d:l')
ENTRIES = ['model.layers.0.self_attn.attn', 'model.layers.1.self_attn.attn',
           'model.layers.2.self_attn.attn']
for i, e in enumerate(ENTRIES):
    ey = DXY + 56 + i * 20
    lc.rect(S1X + 34, ey - 12, 190, 17, '#ffffff', lc.C_MUTE, rx=3, sw=0.8)
    lc.text(S1X + 41, ey, e, 7.4, '#334155', 'start', maxw=176, tag='e' + str(i))
    lc.text(S1X + 232, ey, '→ Attention 实例', 7.4, lc.C_MUTE, 'start', maxw=140, tag='ev' + str(i))
lc.text(S1X + 20, DXY + 140, '重名即 raise：层名是全模型唯一的 key', 8.2, lc.C_BEAT_T,
        'start', maxw=SW_ - 40, tag='d:note')

# ①→② 注册表拷贝（构造产物 → 每拍可用）
lc.parrow([(S1X + SW_, DXY + 90), (S2X, DXY + 90)], lc.C_BEAT_S, 1.6, dash=True)
lc.text((S1X + SW_ + S2X) / 2, DXY + 78, 'no_compile_layers', 7.2, lc.C_BEAT_T, 'middle',
        True, maxw=84, tag='c1')
lc.text((S1X + SW_ + S2X) / 2, DXY + 94, '= 本表拷贝', 7.2, lc.C_BEAT_T, 'middle', True,
        maxw=84, tag='c2')
lc.text((S1X + SW_ + S2X) / 2, DXY + 112, '（构造期产物）', 7.0, lc.C_MUTE, 'middle',
        maxw=84, tag='c2b')

# ===== 段② set_forward_context 包住 _model_forward =====
BCY = SY + 64
lc.rect(S2X + 20, BCY, SW_ - 40, 96, '#ffffff', lc.C_GPU_S, rx=7, sw=1.4)
lc.text(S2X + 34, BCY + 22, 'set_forward_context(attn_metadata, …)', 9.3, lc.C_GPU_S,
        'start', True, maxw=SW_ - 68, tag='sfc:t')
lc.text(S2X + 34, BCY + 40, 'vllm/forward_context.py:L259-L344 · runner 站 13 注入', 7.6,
        lc.C_FAINT, 'start', maxw=SW_ - 68, tag='sfc:f')
lc.text(S2X + 34, BCY + 58, '参数袋：mode + batch_descriptor + slot_mapping +', 8.2,
        '#334155', 'start', maxw=SW_ - 68, tag='sfc:l1')
lc.text(S2X + 34, BCY + 74, 'dp_metadata / is_padding —— 包住 _model_forward 的 yield', 8.2,
        '#334155', 'start', maxw=SW_ - 68, tag='sfc:l2')

CHY = SY + 186
lc.rect(S2X + 20, CHY, SW_ - 40, 120, lc.C_GPU_F, lc.C_GPU_S, rx=7, sw=1.6)
lc.text(S2X + 34, CHY + 22, '全局变量通道：ForwardContext', 9.5, lc.C_GPU_S, 'start',
        True, maxw=SW_ - 68, tag='ch:t')
CHL = ['· attn_metadata（dict / list，speculative 双形态）',
       '· cudagraph_runtime_mode + batch_descriptor（站 15 回放的 key）',
       '· no_compile_layers = static_forward_context 的拷贝',
       '· slot_mapping: { layer_name: tensor }']
for i, ln in enumerate(CHL):
    lc.text(S2X + 34, CHY + 44 + i * 18, ln, 8.0, '#334155', 'start', maxw=SW_ - 68,
            tag='chl' + str(i))
lc.text(S2X + 20, CHY + 140, '一次注入、全层共享：模型 forward 无须携带任何执行环境', 8.2,
        lc.C_GPU_S, 'start', maxw=SW_ - 40, tag='ch:note')

# ===== 段③ op 内取回 =====
OPY = SY + 64
lc.rect(S3X + 20, OPY, SW_ - 40, 96, '#ffffff', lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(S3X + 34, OPY + 22, 'unified_attention_with_output(…, layer_name)', 8.8, lc.C_BEAT_T,
        'start', True, maxw=SW_ - 68, tag='op:t')
lc.text(S3X + 34, OPY + 40, 'attention.py:L817-L846 · 签名里只有一个层名', 7.6, lc.C_FAINT,
        'start', maxw=SW_ - 68, tag='op:f')
lc.text(S3X + 34, OPY + 58, 'attn_metadata, self, kv_cache, _ =', 8.2, '#334155', 'start',
        maxw=SW_ - 68, tag='op:l1')
lc.text(S3X + 34, OPY + 74, '    get_attention_context(layer_name)', 8.2, '#334155', 'start',
        maxw=SW_ - 68, tag='op:l2')

GEY = SY + 186
lc.rect(S3X + 20, GEY, SW_ - 40, 120, lc.C_BEAT_F, lc.C_BEAT_S, rx=7, sw=1.4)
lc.text(S3X + 34, GEY + 22, 'get_attention_context(layer_name) 三支取回', 9.2, lc.C_BEAT_T,
        'start', True, maxw=SW_ - 68, tag='ge:t')
FETCH = [('attn_metadata', 'dict / list 分支按层名取'),
         ('no_compile_layers[layer_name]', '层实例（含 kv_cache）'),
         ('slot_mapping.get(layer_name)', '本层槽位表')]
for i, (k, v) in enumerate(FETCH):
    fy = GEY + 46 + i * 24
    lc.text(S3X + 34, fy, '· ' + k, 7.8, '#334155', 'start', maxw=216, tag='fk' + str(i))
    lc.text(S3X + 254, fy, v, 7.4, lc.C_MUTE, 'start', maxw=136, tag='fv' + str(i))
lc.text(S3X + 20, GEY + 140, '取回后转调 self.impl.forward(self, q, k, v, kv_cache, …)', 7.9,
        lc.C_BEAT_T, 'start', maxw=SW_ - 40, tag='ge:note')

# ②→③ 回查箭头
lc.parrow([(S2X + SW_, GEY + 60), (S3X, GEY + 60)], lc.C_GPU_S, 1.6)
lc.text((S2X + SW_ + S3X) / 2, GEY + 50, '按 layer_name', 7.4, lc.C_GPU_S, 'middle', True,
        maxw=84, tag='c3')
lc.text((S2X + SW_ + S3X) / 2, GEY + 66, '回查通道', 7.4, lc.C_GPU_S, 'middle', True,
        maxw=84, tag='c3b')

# ---------------- 底部：代价条 + 新旧对照 ----------------
CY = SY + SH_ + 26
lc.rect(MX, CY, 700, 96, '#ffffff', lc.C_ABORT, rx=8, sw=1.3, dash=True)
lc.text(MX + 16, CY + 22, '代价：隐式全局状态', 9.5, lc.C_ABORT, 'start', True, maxw=200,
        tag='cost:t')
lc.text(MX + 16, CY + 42, '不设 context 直接调 attention → 当场 assert 崩：', 8.4,
        '#334155', 'start', maxw=660, tag='cost:l1')
lc.text(MX + 16, CY + 60, 'assert _forward_context is not None, "Forward context is not set.', 8.0,
        '#334155', 'start', maxw=660, tag='cost:l2')
lc.text(MX + 16, CY + 76, 'Please use `set_forward_context` …"（forward_context.py:L199-L205）', 8.0,
        '#334155', 'start', maxw=660, tag='cost:l3')

lc.rect(780, CY, BXR - 780, 96, '#ffffff', lc.C_MUTE, rx=8, sw=1.2)
lc.text(796, CY + 22, '新旧对照', 9.5, lc.C_TXT, 'start', True, maxw=140, tag='cmp:t')
lc.text(796, CY + 42, '旧：attn_metadata 从 model.forward 一路透传、绑架每个模型文件签名；', 8.4,
        '#334155', 'start', maxw=BXR - 812, tag='cmp:l1')
lc.text(796, CY + 60, '新：三段接力——模型定义（注册）与执行环境（注入/取回）解耦，', 8.4,
        '#334155', 'start', maxw=BXR - 812, tag='cmp:l2')
lc.text(796, CY + 78, 'forward 签名里从此不见 attn_metadata。', 8.4, '#334155', 'start',
        maxw=BXR - 812, tag='cmp:l3')

# ---------------- 页脚 ----------------
lc.text(MX, 748, '逐字锚 vllm/model_executor/layers/attention/attention.py:L443-L446（自注册+重名 raise）· L732-L772（get_attention_context 三支取回）· vllm/forward_context.py:L199-L205 · L229-L231 · L259-L344 · vllm/v1/worker/gpu_model_runner.py:L4432-L4456',
        8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 766, '行号基线 vLLM v0.27.1', 8.3, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch19-fig-forward-context-relay.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
