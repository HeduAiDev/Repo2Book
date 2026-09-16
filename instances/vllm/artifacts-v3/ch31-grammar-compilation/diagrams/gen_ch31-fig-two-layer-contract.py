#!/usr/bin/env python3
"""ch30 机制图 · m5 两层 ABC 契约（figure_spec ch30-fig-two-layer-contract，模板 layout）

放大自 L0 采样列·结构化输出组的类型骨架——引擎级 Backend 与请求级 Grammar 两层 ABC
的展开（L2 章图站 3-4 之间的类型层），架构归属回指 L2 章图，不另立第二种架构画法
（FIGURE-SYSTEM §3）。

claim：两层 ABC 把『共享的重编译』与『独立的轻推进』类型化：引擎级 Backend
（三字段 vllm_config/tokenizer/vocab_size + 三方法 compile_grammar/allocate_token_bitmask/
destroy）全引擎一份、持 GrammarCompiler 与词表；请求级 Grammar（六方法）每请求一个、
各自推进 FSM。

数字全部取自 figure_spec.numbers（backend_types.py:L31-L95 六方法、L99-L136 三方法+三字段、
__init__.py:L114-L164 四后端同契约 + 单后端注释原话、dossier data_flow[3] 两层归属）。
坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 812
MX, BXR = 60, 1440
SBX = 1058          # 侧栏左缘（四后端名牌）

DEFS = lc.DEFS + (
    f'<marker id="sam" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    f'markerHeight="4.2" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{lc.C_SAM_S}"/></marker>')

# ---------------- 标题区 ----------------
lc.text(MX, 34, '共享的重编译在引擎级，独立的轻推进在请求级——两层 ABC 类型化', 16.5,
        lc.C_TXT, 'start', True, maxw=1020, tag='title')
lc.text(MX, 58, '引擎级 StructuredOutputBackend 全引擎一份（抱 GrammarCompiler 缓存与词表）；请求级 StructuredOutputGrammar 每请求一个'
               '（各自的 FSM 互不干扰）——四家后端同一套签名，这就是「换后端不动引擎」的接缝',
        10.5, lc.C_MUTE, 'start', maxw=1290, tag='subtitle')
_ch = '放大自 L0 采样列·结构化输出组 · L2 站 3-4 之间的类型层'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ================= 上层：引擎级 Backend（全引擎一份） =================
UY0, UH = 92, 296
lc.rect(MX, UY0, SBX - MX - 20, UH, lc.C_ENG_F, lc.C_ENG_S, rx=11, sw=2.2)
lc.text(MX + 18, UY0 + 24, '引擎级 StructuredOutputBackend（ABC）——全引擎一份', 12.5,
        lc.C_ENG_S, 'start', True, maxw=620, tag='u:t')
lc.text(SBX - 38, UY0 + 24, 'backend_types.py:L99-L136', 8.5, lc.C_FAINT, 'end', maxw=220, tag='u:file')
# 三字段
lc.text(MX + 18, UY0 + 48, '三字段：vllm_config · tokenizer · vocab_size（四个后端构造签名由此固定）',
        9, '#334155', 'start', maxw=640, tag='u:fields')
# 三方法行（三块 method chip）
METH_U = [('compile_grammar(request_type, grammar_spec)', '→ StructuredOutputGrammar（每请求一次调用）'),
          ('allocate_token_bitmask(max_num_seqs)', '→ [行, ceil(V/32)] int32 张量（一次性预算）'),
          ('destroy()', '后端清理')]
mw = (SBX - MX - 20 - 36 - 2 * 14) / 3
for i, (m, sub) in enumerate(METH_U):
    x = MX + 18 + i * (mw + 14)
    lc.rect(x, UY0 + 64, mw, 62, '#ffffff', lc.C_ENG_S, rx=7, sw=1.3)
    lc.text(x + 10, UY0 + 82, m, 8.2, lc.C_TXT, 'start', True, maxw=mw - 20, tag='um' + str(i))
    lc.text(x + 10, UY0 + 100, sub, 7.8, lc.C_MUTE, 'start', maxw=mw - 20, tag='ums' + str(i))
    lc.text(x + 10, UY0 + 116, '@abstractmethod', 7.2, lc.C_FAINT, 'start', maxw=mw - 20, tag='uma' + str(i))
# 内嵌共享资源两小箱
GW = (SBX - MX - 20 - 36 - 14) / 2
lc.rect(MX + 18, UY0 + 142, GW, 108, '#ffffff', lc.C_ENG_S, rx=7, sw=1.4)
lc.text(MX + 30, UY0 + 162, 'GrammarCompiler（xgrammar 侧）', 9.5, lc.C_TXT, 'start', True,
        maxw=GW - 24, tag='g1:t')
for j, ln in enumerate(['· 编译缓存：LRU + 字节预算', '  （VLLM_XGRAMMAR_CACHE_MB 默认 512MB',
                        '  ≈ 1000 个 JSON schema）', '· 跨请求共享——重编译全引擎只做一次']):
    lc.text(MX + 30, UY0 + 180 + j * 16, ln, 8, '#334155', 'start', maxw=GW - 24, tag='g1:l' + str(j))
lc.rect(MX + 18 + GW + 14, UY0 + 142, GW, 108, '#ffffff', lc.C_ENG_S, rx=7, sw=1.4)
lc.text(MX + 30 + GW + 14, UY0 + 162, 'tokenizer / 词表', 9.5, lc.C_TXT, 'start', True,
        maxw=GW - 24, tag='g2:t')
for j, ln in enumerate(['· 词表大小决定合法集挑选工作量', '  （编译期一次性算好）', '· 与编译器同属引擎级共享资源',
                        '· 首个结构化请求到达时惰性定型']):
    lc.text(MX + 30 + GW + 14, UY0 + 180 + j * 16, ln, 8, '#334155', 'start', maxw=GW - 24, tag='g2:l' + str(j))
lc.text(MX + 18, UY0 + 272, '注释原话："We do NOT support different backends on a per-request basis"——引擎级单后端，'
                            '首个结构化请求定型后全引擎共用',
        8.6, lc.C_ENG_S, 'start', True, maxw=SBX - MX - 56, tag='u:note')
lc.text(MX + 18, UY0 + 290, 'vllm/v1/structured_output/__init__.py:L114-L164（grammar_init 惰性构造四选一）',
        8, lc.C_FAINT, 'start', maxw=SBX - MX - 56, tag='u:file2')

# ================= 1→N 连线（三支扇出） =================
LY0, LH = 452, 226
NBOX = 3
bw = (SBX - MX - 20 - 2 * 16) / 3
for i in range(NBOX):
    cx = MX + i * (bw + 16) + bw / 2
    lc.seg(cx, UY0 + UH, cx, LY0, lc.C_ENG_S, 2.0, 'up')
lc.text(383, 436, 'compile_grammar（每请求一次；缓存命中在后端内部）', 9, lc.C_ENG_S, 'middle',
        True, maxw=256, tag='link')

# ================= 下层：请求级 Grammar（每请求一个） =================
REQS = [('gr-1', 'root ::= "yes" | "no"', '位置 1（只剩 EOS）'),
        ('gr-2', 'json schema A', '各自位置（互不影响）'),
        ('gr-3', 'json schema A（同键）', '各自位置（互不影响）')]
for i, (rid, spec, st) in enumerate(REQS):
    x = MX + i * (bw + 16)
    lc.rect(x, LY0, bw, LH, lc.C_SAM_F, lc.C_SAM_S, rx=9, sw=1.8)
    lc.text(x + 14, LY0 + 24, f'请求级 StructuredOutputGrammar', 10.5, lc.C_SAM_S, 'start', True,
            maxw=bw - 28, tag='r' + str(i) + ':t')
    lc.text(x + 14, LY0 + 42, f'请求 {rid} · {spec}', 8.2, '#334155', 'start', maxw=bw - 26,
            tag='r' + str(i) + ':id')
    for j, m in enumerate(['accept_tokens(request_id, tokens)', 'validate_tokens(tokens)  # 试走不推进',
                           'rollback(num_tokens)  # 悔棋', 'fill_bitmask(bitmask, index)',
                           'is_terminated()', 'reset()  # v0.27.1 全仓尚无调用者']):
        lc.text(x + 14, LY0 + 66 + j * 19, m, 8.4, '#334155', 'start', maxw=bw - 26,
                tag='r' + str(i) + 'm' + str(j))
    # 独立 FSM 圆点
    fx, fy = x + bw / 2, LY0 + 196
    lc.circle(fx, fy, 12, lc.C_SAM_S, 1.6, dash=False)
    lc.text(fx, fy + 3.5, 'FSM', 8, lc.C_SAM_S, 'middle', True, maxw=22, tag='r' + str(i) + 'fsm')
    lc.text(fx + 20, fy + 3.5, f'独立推进 · 当前态：{st}', 7.8, lc.C_MUTE, 'start', maxw=bw - 60,
            tag='r' + str(i) + ':st')
lc.text(MX, LY0 + LH + 24, '六方法契约：backend_types.py:L31-L95——accept 推进 / validate 试走（投机解码用）/ rollback 回退 / fill 出掩码 /'
                           ' is_terminated 判终 / reset 完整性预留',
        8.6, lc.C_MUTE, 'start', maxw=SBX - MX, tag='l:foot')

# ================= 右侧栏：四后端名牌 =================
lc.rect(SBX, UY0, BXR - SBX, LY0 + LH - UY0, '#f8fafc', lc.C_MUTE, rx=11, sw=1.6)
lc.text(SBX + 16, UY0 + 24, '四后端同契约', 12, lc.C_TXT, 'start', True, maxw=340, tag='s:t')
BACKENDS = [('XgrammarBackend', 'compile_json_schema / compile_grammar / compile_regex'),
            ('GuidanceBackend', 'llguidance LLMatcher 序列化'),
            ('OutlinesBackend', 'oc.Index + LRUCache(128) 可选 SQLite'),
            ('LMFormatEnforcerBackend', 'TokenEnforcer（无 rollback，拒投机）')]
by = UY0 + 44
for nm, sub in BACKENDS:
    lc.rect(SBX + 16, by, BXR - SBX - 32, 52, '#ffffff', lc.C_ENG_S, rx=7, sw=1.3)
    lc.text(SBX + 28, by + 20, nm, 9.5, lc.C_TXT, 'start', True, maxw=BXR - SBX - 56, tag='b:' + nm)
    lc.text(SBX + 28, by + 38, sub, 7.8, lc.C_MUTE, 'start', maxw=BXR - SBX - 56, tag='bs:' + nm)
    by += 62
lc.text(SBX + 16, by + 14, '· 引擎级各占一列：同一套三方法签名', 8.2, '#334155', 'start',
        maxw=BXR - SBX - 32, tag='s:l1')
lc.text(SBX + 16, by + 32, '· 请求级产物同实现六方法', 8.2, '#334155', 'start',
        maxw=BXR - SBX - 32, tag='s:l2')
lc.text(SBX + 16, by + 50, '· 编译器共享+缓存归引擎级；', 8.2, '#334155', 'start',
        maxw=BXR - SBX - 32, tag='s:l3')
lc.text(SBX + 16, by + 68, '  状态机独立推进归请求级', 8.2, '#334155', 'start',
        maxw=BXR - SBX - 32, tag='s:l4')

# ================= 图例 + 页脚 =================
LY = 724
lx0 = MX
lc.rect(lx0, LY - 9, 16, 11, lc.C_ENG_F, lc.C_ENG_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '引擎级（全引擎一份·共享重资源）', 8.8, lc.C_TXT, 'start', maxw=280, tag='lg1')
lx0 += 21 + lc.tw('引擎级（全引擎一份·共享重资源）', 8.8) + 18
lc.rect(lx0, LY - 9, 16, 11, lc.C_SAM_F, lc.C_SAM_S, rx=3, sw=1.4)
lc.text(lx0 + 21, LY + 1, '请求级（每请求一个·独立 FSM）', 8.8, lc.C_TXT, 'start', maxw=260, tag='lg2')

lc.text(MX, 754, 'vllm/v1/structured_output/backend_types.py:L99-L136（引擎级三方法+三字段）· L31-L95（请求级六方法）· '
                 '__init__.py:L114-L164（grammar_init 四选一 + 单后端注释）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, 770, '编译器跨请求共享+缓存归引擎级 · 状态机独立推进归请求级 ＝ 本章数据流走读结论 · '
                 '512MB / ≈1000 schema ＝ VLLM_XGRAMMAR_CACHE_MB 默认值 docstring · 行号基线 vLLM v0.27.1',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, 786, 'reset() 在 v0.27.1 全仓零 in-tree 调用者——契约完整性存在（勿杜撰使用场景）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

# ================= 装配输出 =================
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch30-fig-two-layer-contract.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
