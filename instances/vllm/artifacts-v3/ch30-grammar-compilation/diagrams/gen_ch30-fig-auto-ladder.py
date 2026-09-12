#!/usr/bin/env python3
"""ch30 机制图 · m3 auto 降级阶梯（figure_spec ch30-fig-auto-ladder，模板 flow）

放大自 L0 采样列·结构化输出组的『前端校验期』段（L2 章图站 1，请求还没进引擎）——
auto 降级阶梯是这一段的决策展开，架构归属回指 L2 章图，不另立第二种架构画法
（FIGURE-SYSTEM §3）。

claim：auto 不是运行期试错而是校验期一次定终身：先试 xgrammar（试编+JSON 特性预检），
ValueError 降 guidance（两个 skip 判据：非 tekken Mistral 分词器 / guidance 特性预检，
可跳 outlines），LMFE 永不被 auto 选中。

数字全部取自 figure_spec.numbers（auto_ladder 三例、backend_conflict、
has_xgrammar_unsupported_json_features 四特性、阶梯与判据源码锚）；outlines 支按
『真实源码降 outlines』口径标注。坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 842
MX, BXR = 60, 1440
SBX = 972          # 侧栏左缘


def diamond(cx, cy, hw, hh, fill, stroke, sw=1.7):
    pts = f'{cx:.1f},{cy - hh:.1f} {cx + hw:.1f},{cy:.1f} {cx:.1f},{cy + hh:.1f} {cx - hw:.1f},{cy:.1f}'
    lc.ELEMS.append(((cx - hw - 2, cy - hh - 2, cx + hw + 2, cy + hh + 2),
                     f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'))


# ---------------- 标题区 ----------------
lc.text(MX, 34, 'auto 是校验期一次定终身，不是运行期试错', 16.5, lc.C_TXT, 'start', True,
        maxw=960, tag='title')
lc.text(MX, 58, '先试 xgrammar（试编 + JSON 特性预检）→ 编不动或不支持则 ValueError → 查两个 skip 判据降 guidance 或 outlines——'
               '全部发生在前端校验期（请求还没进引擎），请求到达引擎时 _backend 已定',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L0 采样列·结构化输出组 · L2 站 1『前端校验选后端』'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 左区：auto 决策阶梯 =================
# 入口
EY0, EH0 = 92, 78
lc.rect(300, EY0, 440, EH0, lc.C_API_F, lc.C_API_S, rx=9, sw=1.8)
lc.text(320, EY0 + 24, 'StructuredOutputsParams + backend="auto"', 11.5, lc.C_API_S, 'start', True,
        maxw=400, tag='ent:t')
lc.text(320, EY0 + 44, '六形态任一描述约束：json / json_object / regex /', 8.8, '#334155',
        'start', maxw=400, tag='ent:l1')
lc.text(320, EY0 + 60, 'choice / grammar / structural_tag（__post_init__ 强制六选一）', 8.8,
        '#334155', 'start', maxw=400, tag='ent:l2')
lc.seg(520, EY0 + EH0, 520, 196, lc.C_API_S, 1.8, 'dn')

# D1 菱形：试编 xgrammar
D1X, D1Y = 520, 252
diamond(D1X, D1Y, 170, 56, lc.C_API_F, lc.C_API_S)
lc.text(D1X, D1Y - 6, '试编 xgrammar', 11, lc.C_API_S, 'middle', True, maxw=280, tag='d1:t')
lc.text(D1X, D1Y + 12, 'validate_xgrammar_grammar', 8.2, '#334155', 'middle', maxw=300, tag='d1:s')
# D1 侧注（左，虚线）：试什么
lc.rect(92, 196, 240, 112, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
lc.text(104, 216, '试什么（四条全过才算过）', 9, lc.C_TXT, 'start', True, maxw=220, tag='d1n:t')
for j, ln in enumerate(['· choice → EBNF（原地改写：', '  生成 EBNF→choice=None→grammar=EBNF）',
                        '· lark 文法 → EBNF（无 ::= 判定）', '· regex 试编（ReDoS 超时护栏）',
                        '· json 特性预检：has_xgrammar_unsupported_', '  json_features（见下）']):
    lc.text(104, 234 + j * 14.5, ln, 7.8, '#334155', 'start', maxw=224, tag='d1n:l' + str(j))
lc.seg(332, 252, 350, 252, lc.C_MUTE, 1.2)

# D1 成功支（右）→ xgrammar 出口
lc.rect(740, 212, 210, 80, '#ffffff', lc.C_ENG_S, rx=8, sw=1.6)
lc.text(754, 234, '_backend = "xgrammar"', 10.5, lc.C_TXT, 'start', True, maxw=186, tag='o1:t')
lc.text(754, 252, '_backend_was_auto = True', 8.6, '#334155', 'start', maxw=186, tag='o1:l1')
lc.text(754, 268, '（记账：auto 选的，引擎定型', 8.2, lc.C_MUTE, 'start', maxw=186, tag='o1:l2')
lc.text(754, 282, '  后可复用放行）', 8.2, lc.C_MUTE, 'start', maxw=186, tag='o1:l3')
lc.seg(D1X + 170, D1Y, 740, D1Y, lc.C_API_S, 1.8, 'dn')
lc.text(845, 206, '编得动 + 预检过', 8.6, lc.C_API_S, 'middle', True, maxw=200, tag='b1:y')

# D1 失败支（下）→ ValueError → D2
lc.seg(D1X, D1Y + 56, D1X, 368, lc.C_ABORT, 1.8, 'ab')
lc.text(530, 340, 'ValueError（编不动 / 特性不支持）', 8.6, lc.C_ABORT, 'start', maxw=220, tag='b1:n')

# D2 菱形：skip_guidance 两判据
D2X, D2Y = 520, 430
diamond(D2X, D2Y, 180, 62, lc.C_API_F, lc.C_API_S)
lc.text(D2X, D2Y - 8, 'skip_guidance ?', 11, lc.C_API_S, 'middle', True, maxw=300, tag='d2:t')
lc.text(D2X, D2Y + 10, '两个判据查其一命中即跳过 guidance', 8.2, '#334155', 'middle', maxw=330, tag='d2:s')
# D2 侧注（左，虚线）：判据内容
lc.rect(92, 376, 240, 108, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
lc.text(104, 396, '两个 skip 判据', 9, lc.C_TXT, 'start', True, maxw=220, tag='d2n:t')
for j, ln in enumerate(['① _is_non_tekken_mistral(tokenizer)', '   非 tekken Mistral 分词器',
                        '② has_guidance_unsupported_json_', '   features(schema)',
                        '   guidance 不支持的 schema 特性']):
    lc.text(104, 414 + j * 14.5, ln, 7.8, '#334155', 'start', maxw=224, tag='d2n:l' + str(j))
lc.seg(332, 430, 340, 430, lc.C_MUTE, 1.2)

# D2 不命中（右）→ guidance
lc.rect(740, 390, 210, 80, '#ffffff', lc.C_ENG_S, rx=8, sw=1.6)
lc.text(754, 412, '_backend = "guidance"', 10.5, lc.C_TXT, 'start', True, maxw=186, tag='o2:t')
lc.text(754, 430, 'validate_guidance_grammar', 8.4, '#334155', 'start', maxw=186, tag='o2:l1')
lc.text(754, 446, '（llguidance 序列化 +', 8.2, lc.C_MUTE, 'start', maxw=186, tag='o2:l2')
lc.text(754, 460, '  _get_llg_tokenizer）', 8.2, lc.C_MUTE, 'start', maxw=186, tag='o2:l3')
lc.seg(D2X + 180, D2Y, 740, D2Y, lc.C_API_S, 1.8, 'dn')
lc.text(845, 384, '都不命中（默认降级）', 8.6, lc.C_API_S, 'middle', True, maxw=200, tag='b2:n')

# D2 命中（下）→ outlines
lc.seg(D2X, D2Y + 62, D2X, 536, lc.C_API_S, 1.8, 'dn')
lc.text(530, 512, '命中任一', 8.6, lc.C_API_S, 'start', maxw=100, tag='b2:y')
lc.rect(340, 536, 360, 66, '#ffffff', lc.C_ENG_S, rx=8, sw=1.6)
lc.text(354, 558, '_backend = "outlines"', 10.5, lc.C_TXT, 'start', True, maxw=330, tag='o3:t')
lc.text(354, 576, 'validate_structured_output_request_outlines', 8.2, '#334155', 'start', maxw=330, tag='o3:l1')
lc.text(354, 592, '（真实源码降 outlines；两条路都编不动则请求在此被拒）', 8.2, lc.C_MUTE,
        'start', maxw=336, tag='o3:l2')

# LMFE 永不被 auto 选中（虚线灰）
lc.rect(340, 620, 610, 54, '#ffffff', lc.C_MUTE, rx=8, sw=1.1, dash=True)
lc.text(354, 640, 'LMFE（lm-format-enforcer）永不被 auto 选中', 9.2, lc.C_TXT, 'start', True,
        maxw=580, tag='lmfe:t')
lc.text(354, 658, '无 rollback 能力（compile_grammar 对 max_rollback_tokens>0 显式拒投机）、能力面最窄——要它必须显式指定',
        8.2, '#334155', 'start', maxw=584, tag='lmfe:l1')

# 实测三例（chips）
lc.text(92, 706, '三例实测（本章驱动脚本走真实阶梯）：', 9, lc.C_TXT, 'start', True, maxw=240, tag='ex:t')
EX = [('choice ["yes","no"]', '→ xgrammar', '（was_auto=True）'),
      ('json multipleOf=5', '→ guidance', '（xgrammar 预检不过）'),
      ('json {"a":{"type":"integer"}}', '→ xgrammar', '（直接过）')]
ex_x = 92
for i, (a, b, c) in enumerate(EX):
    lc.rect(ex_x, 718, 286, 46, '#ffffff', lc.C_MUTE, rx=7, sw=1.2)
    lc.text(ex_x + 12, 736, a, 8.2, '#334155', 'start', maxw=264, tag='ex' + str(i) + 'a')
    lc.text(ex_x + 12, 754, b + '  ' + c, 8.6, lc.C_ENG_S, 'start', True, maxw=264, tag='ex' + str(i) + 'b')
    ex_x += 298

# ================= 右区：侧栏 =================
# ① 四显式后端
lc.rect(SBX, 92, BXR - SBX, 158, lc.C_API_F, lc.C_API_S, rx=9, sw=1.6)
lc.text(SBX + 16, 114, '四显式后端——指定即用，不经阶梯', 10.5, lc.C_API_S, 'start', True,
        maxw=430, tag='s1:t')
for j, ln in enumerate(['backend= "xgrammar" / "guidance" / "outlines" /',
                        '"lm-format-enforcer" 显式指定直接选定；',
                        'verify 期校验引擎支持集',
                        '（supported_backends，含引擎 --structured-outputs-config 限定）',
                        '· 与 auto 同一条入口：站 1 前端校验期一次定终身']):
    lc.text(SBX + 16, 134 + j * 17, ln, 8.4, '#334155', 'start', maxw=BXR - SBX - 30, tag='s1:l' + str(j))

# ② 引擎级单后端 + 冲突拒单（橙）
lc.rect(SBX, 266, BXR - SBX, 210, lc.C_ENG_F, lc.C_ENG_S, rx=9, sw=1.6)
lc.text(SBX + 16, 288, '引擎级单后端——首个结构化请求定型，全引擎共用', 10.5, lc.C_ENG_S,
        'start', True, maxw=430, tag='s2:t')
for j, ln in enumerate(['· grammar_init 惰性构造全引擎唯一后端，',
                        '  只认 params._backend（注释原话 "We do NOT support',
                        '  different backends on a per-request basis"）',
                        '· 显式冲突拒单：请求 guidance vs 引擎已定型 xgrammar',
                        '  → VLLMValidationError（"Request-level structured output',
                        '  backend selection is not supported…"）',
                        '· auto 记账放行：_backend_was_auto=True 时复用',
                        '  params 的请求照常进（_backend 对齐即可）',
                        '· 换后端 = 重启换引擎配置，不是每请求选项']):
    lc.text(SBX + 16, 308 + j * 17, ln, 8.4, '#334155', 'start', maxw=BXR - SBX - 30, tag='s2:l' + str(j))

# ③ 快速失败（虚线）
lc.rect(SBX, 492, BXR - SBX, 96, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(SBX + 16, 514, '快速失败谱系', 10, lc.C_TXT, 'start', True, maxw=430, tag='s3:t')
for j, ln in enumerate(['· 全部失败发生在校验期（拒单 / 降级），不是运行中段',
                        '· regex 编译的 ReDoS 护栏（嵌套量词超时拒单）同在此期',
                        '· 编译失败在引擎侧只杀单请求（时序见异步编译门一图）']):
    lc.text(SBX + 16, 534 + j * 17, ln, 8.4, '#334155', 'start', maxw=BXR - SBX - 30, tag='s3:l' + str(j))

# 判据特性清单（右栏底部）
lc.rect(SBX, 604, BXR - SBX, 70, '#ffffff', lc.C_MUTE, rx=8, sw=1.1)
lc.text(SBX + 16, 624, 'xgrammar 预检拦的 JSON 特性（backend_xgrammar.py:L225-L269）', 8.8,
        lc.C_TXT, 'start', True, maxw=430, tag='s4:t')
lc.text(SBX + 16, 642, 'multipleOf（数值步进）· uniqueItems/contains（数组）·', 8.2, '#334155',
        'start', maxw=430, tag='s4:l1')
lc.text(SBX + 16, 658, '非标 format（字符串）· patternProperties/propertyNames（对象）', 8.2,
        '#334155', 'start', maxw=430, tag='s4:l2')

# ================= 图例 + 页脚 =================
LY = 786
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, lc.C_API_F, lc.C_API_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '校验期（前端·请求还没进引擎）', 8.8, lc.C_TXT, 'start', maxw=240, tag='lg1')
lx0 += 21 + lc.tw('校验期（前端·请求还没进引擎）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, '#ffffff', lc.C_ENG_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '_backend 出口（定型后写回 params）', 8.8, lc.C_TXT, 'start', maxw=240, tag='lg2')
lx0 += 21 + lc.tw('_backend 出口（定型后写回 params）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '引擎级（EngineCore 侧）', 8.8, lc.C_TXT, 'start', maxw=180, tag='lg3')
lx0 += 21 + lc.tw('引擎级（EngineCore 侧）', 8.8) + 18
lc.seg(lx0 + 4, LY - 3, lx0 + 34, LY - 3, lc.C_ABORT, 1.8)
lc.text(lx0 + 40, LY + 1, 'ValueError 失败支', 8.8, lc.C_TXT, 'start', maxw=140, tag='lg4')

lc.text(MX, 814, 'vllm/sampling_params.py:L1043-L1086（阶梯主体）· L949-L966（引擎单后端冲突检查 + _backend_was_auto 记账）· '
                 'vllm/v1/structured_output/backend_xgrammar.py:L225-L269（JSON 特性预检）· L272-L303（validate_xgrammar_grammar）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 830, '三例后端选择与冲突报错 ＝ 本章驱动脚本实测（真实 get_structured_output_key / _validate_structured_outputs 链）· '
                 '行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch30-fig-auto-ladder.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
