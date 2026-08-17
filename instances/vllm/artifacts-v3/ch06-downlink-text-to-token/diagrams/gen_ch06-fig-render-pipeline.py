#!/usr/bin/env python3
"""ch06 机制图 1 · 渲染四步流水（explainer figure_spec ch06-fig-render-pipeline，模板 flow）

放大自 L0 蓝色 API 进程带（api_band · 本章 l0_zoom『API 进程下行泳道』）的 Renderer
渲染段——即本章 L2 章图 center 拍片行 ① chat 模板展开 / ② tokenize 下池 / ③ mm 预处理，
加 north『进门 · serving 层交棒』的机制展开。架构归属回指 L2/L0（FIGURE-SYSTEM §3.3）。

claim：四步流水 render_messages → tokenize → extras → process_for_engine 把『一段话』
变成带 'type' 的 EngineInput：进门口打一次 arrival_time 且批量中每路都携带同一时间戳，
批内可并行步骤全链 gather，chat 与 completion 两面同构（completion 的第 1 步是直通）。

数字全部取自 figure_spec.numbers（host 实测 trace + pin 锚点）；图内步骤名 step1/step3/
step4 与实测 trace 标记逐字同源；坐标由常量/循环计算；文本全 esc()。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1500, 960
MX = 60
BXR = 1440

C_BODY = '#334155'

# ---------------- 标题区 ----------------
lc.text(MX, 34, '文本变 token 的四道工序：render_messages → tokenize → extras → process_for_engine',
        16.5, lc.C_TXT, 'start', True, maxw=980, tag='title')
lc.text(MX, 58, '一段话进门先打一次总时钟（arrival_time），四步走完装成带 \'type\' 的 EngineInput；'
        '批内可并行的步骤全链 gather——chat 与 completion 两面同构',
        10.5, lc.C_MUTE, 'start', maxw=1000, tag='subtitle')
_ch = '放大自 L2 拍片 ①-③（渲染段）· L0：API 进程下行泳道'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_API_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 进门条 + 总时钟 ----------------
ENT_Y, ENT_H = 84, 54
lc.rect(MX, ENT_Y, BXR - MX, ENT_H, lc.C_API_F, lc.C_API_S, rx=8, sw=1.6)
lc.text(MX + 16, ENT_Y + 21, '进门 · serving 层交棒：_create_chat_completion 调 render_chat_request（serving.py:L252）'
        '→ await online_renderer.render_chat（L217）', 10.5, lc.C_TXT, 'start', True,
        maxw=960, tag='ent:t')
lc.text(MX + 16, ENT_Y + 41, '· engine_input 在进 generate 之前已诞生——OpenAI 路径的 tokenize 比引擎门面更靠前，'
        '切词不在引擎侧', 9, C_BODY, 'start', maxw=960, tag='ent:s')
# 总时钟图钉（贴进门条右段）：进门首行打点一次
CK_X, CK_Y = 1245, ENT_Y + ENT_H / 2
lc.circle(CK_X - 148, CK_Y, 9, lc.C_API_S, 1.6, dash=False)
lc.seg(CK_X - 148, CK_Y, CK_X - 148, CK_Y - 6, lc.C_API_S, 1.4)
lc.seg(CK_X - 148, CK_Y, CK_X - 143, CK_Y + 2, lc.C_API_S, 1.4)
lc.text(CK_X - 134, CK_Y - 4, 'arrival_time = time.time()', 9.5, lc.C_API_S, 'start', True,
        maxw=230, tag='ck:t')
lc.text(CK_X - 134, CK_Y + 12, '进门首行打点一次（base.py:L1080）', 8.5, lc.C_MUTE, 'start',
        maxw=230, tag='ck:s')

# ---------------- 四工序传送带 ----------------
CW = 280
GAP = (BXR - MX - 4 * CW) / 3
XS = [MX + i * (CW + GAP) for i in range(4)]      # 60 / 400 / 740 / 1080
CXS = [x + CW / 2 for x in XS]

STATIONS = [
    dict(title='step1 · render_messages', sub='chat 模板展开（L1093）· L2 ①',
         lines=['对话 messages → 标准文本 DictPrompt', '（渲染产物：还是文本，未切词）'],
         blackbox='chat 模板引擎（Jinja2）= 本章黑盒'),
    dict(title='step2 · tokenize', sub='下渲染线程池 · L2 ②',
         lines=['tokenizer(prompt) → input_ids', 'chat 面 add_special_tokens=False',
                "实测 'user: hello world' → [3,4,5]"]),
    dict(title='step3 · _apply_prompt_extras', sub='贴附加键（L651）',
         lines=['每批一次（同步、整批共享）', '本批无 extras 则直通']),
    dict(title='step4 · process_for_engine_async', sub='装盒（L962）',
         lines=["EngineInput：dict 带 'type'", '（token / multimodal / embeds）'],
         mmbox='mm 预处理支路 · L2 ③——仅带图的路进入'),
]
CONV_Y = 170
# 进门条 → step1 的交接箭头
lc.seg(XS[0] + CW / 2, ENT_Y + ENT_H, XS[0] + CW / 2, CONV_Y - 2, lc.C_API_S, 2.0, 'dn')
BOX_H = 128
for i, st in enumerate(STATIONS):
    x = XS[i]
    lc.rect(x, CONV_Y, CW, BOX_H, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
    lc.text(x + 14, CONV_Y + 20, st['title'], 11, lc.C_TXT, 'start', True, maxw=CW - 28,
            tag='st:t' + str(i))
    lc.text(x + 14, CONV_Y + 36, st['sub'], 8.5, lc.C_MUTE, 'start', maxw=CW - 28,
            tag='st:s' + str(i))
    for j, ln in enumerate(st['lines']):
        lc.text(x + 14, CONV_Y + 54 + j * 17, ln, 9, C_BODY, 'start', maxw=CW - 26,
                tag='st:l' + str(i) + str(j))
    if 'blackbox' in st:
        bb_h = 26
        lc.rect(x + 14, CONV_Y + BOX_H - bb_h - 10, CW - 28, bb_h, lc.C_API_F, lc.C_MUTE,
                rx=5, sw=1.1, dash=True)
        lc.text(x + CW / 2, CONV_Y + BOX_H - bb_h + 7, st['blackbox'], 8.5, lc.C_MUTE,
                'middle', maxw=CW - 40, tag='st:bb' + str(i))
    if 'mmbox' in st:
        mb_h = 26
        lc.rect(x + 14, CONV_Y + BOX_H - mb_h - 10, CW - 28, mb_h, '#ffffff', lc.C_ENG_S,
                rx=5, sw=1.2, dash=True)
        lc.text(x + CW / 2, CONV_Y + BOX_H - mb_h + 7, st['mmbox'], 8, lc.C_ENG_S, 'middle',
                maxw=CW - 34, tag='st:mm' + str(i))

ARROW_LABELS = ['DictPrompt', 'input_ids', 'TokPrompt']
ARROW_SUBS = ['（模板文本）', '（整数序列）', '（附加键贴毕）']
MID_Y = CONV_Y + BOX_H / 2
for i in range(3):
    x1, x2 = XS[i] + CW, XS[i + 1]
    lc.seg(x1, MID_Y, x2, MID_Y, lc.C_API_S, 2.0, 'dn')
    lc.text((x1 + x2) / 2, MID_Y - 14, ARROW_LABELS[i], 9, lc.C_API_S, 'middle', True,
            maxw=GAP - 6, tag='al' + str(i))
    lc.text((x1 + x2) / 2, MID_Y + 16, ARROW_SUBS[i], 8, lc.C_MUTE, 'middle', maxw=GAP - 4,
            tag='as' + str(i))
# 传送带 → EngineInput 出口感（右端）
lc.seg(XS[3] + CW, MID_Y, BXR, MID_Y, lc.C_API_S, 2.0, 'dn')
lc.text((XS[3] + CW + BXR) / 2, MID_Y - 14, "EngineInput", 9, lc.C_API_S, 'middle', True,
        maxw=GAP, tag='al:out')
lc.text((XS[3] + CW + BXR) / 2, MID_Y + 16, "带 'type'", 8, lc.C_MUTE, 'middle',
        maxw=GAP, tag='as:out')

# ---------------- 批量 3 路 gather 面板 ----------------
PNL_Y = CONV_Y + BOX_H + 44          # 342
PNL_H = 300
lc.rect(MX, PNL_Y, BXR - MX, PNL_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, PNL_Y + 22, '批量 3 路对话：可并行的工序全链 gather（host 实测步骤序）', 11.5,
        lc.C_TXT, 'start', True, maxw=700, tag='pnl:t')
lc.text(BXR - 16, PNL_Y + 22, '纯文本路：trace 恰 4 步、mm 工序不进入', 9, lc.C_MUTE, 'end',
        maxw=380, tag='pnl:note')

LANE_LBL_X = 76
COL_C = [350, 660, 940, 1200]
COL_HEAD = ['step1 render_messages ×3', 'tokenize ×3', 'step3 extras ×1',
            'step4 process_for_engine ×3']
COL_YH = PNL_Y + 44
for i, cx in enumerate(COL_C):
    lc.text(cx, COL_YH, COL_HEAD[i], 8.5, lc.C_API_S, 'middle', True, maxw=250,
            tag='ch' + str(i))

ROWS = [('路 1 · 纯文本', 0), ('路 2 · 带图', 1), ('路 3 · 纯文本', 2)]
ROW_Y = [COL_YH + 34 + k * 52 for k in range(3)]     # 节点中心 y
NODE_W, NODE_H = 176, 34
for (lbl, k), cy in zip(ROWS, ROW_Y):
    lc.text(LANE_LBL_X, cy + 3, lbl, 9, lc.C_TXT, 'start', True, maxw=110, tag='row' + str(k))
# col1 / col2：每路一节点
for k, cy in enumerate(ROW_Y):
    for col in (0, 1):
        x = COL_C[col] - NODE_W / 2
        lc.rect(x, cy - NODE_H / 2, NODE_W, NODE_H, lc.C_API_F, lc.C_API_S, rx=5, sw=1.2)
        lc.text(COL_C[col], cy + 3, ['render_messages', 'tokenize'][col], 9, lc.C_TXT,
                'middle', maxw=NODE_W - 10, tag='n%d%d' % (col, k))
# col3：一个整批节点跨三行（×1 每批一次）
c3_top = ROW_Y[0] - NODE_H / 2
c3_h = ROW_Y[2] - ROW_Y[0] + NODE_H
lc.rect(COL_C[2] - NODE_W / 2, c3_top, NODE_W, c3_h, '#ffffff', lc.C_API_S, rx=5, sw=1.4)
lc.text(COL_C[2], c3_top + c3_h / 2 - 6, 'extras ×1', 9.5, lc.C_TXT, 'middle', True,
        maxw=NODE_W - 10, tag='n3x')
lc.text(COL_C[2], c3_top + c3_h / 2 + 12, '（整批一次）', 8, lc.C_MUTE, 'middle',
        maxw=NODE_W - 10, tag='n3s')
# col4：每路一节点；路 2 的节点内嵌 mm 支路小框
for k, cy in enumerate(ROW_Y):
    x = COL_C[3] - NODE_W / 2
    if k == 1:
        h = 50
        lc.rect(x, cy - h / 2, NODE_W, h, lc.C_API_F, lc.C_API_S, rx=5, sw=1.2)
        lc.text(COL_C[3], cy - 8, 'process_for_engine', 8.5, lc.C_TXT, 'middle',
                maxw=NODE_W - 10, tag='n4m')
        lc.rect(x + 16, cy + 4, NODE_W - 32, 18, '#ffffff', lc.C_ENG_S, rx=4, sw=1.1, dash=True)
        lc.text(COL_C[3], cy + 16.5, '└ mm 预处理 ×1（仅此路）', 7.5, lc.C_ENG_S, 'middle',
                maxw=NODE_W - 40, tag='n4mm')
    else:
        lc.rect(x, cy - NODE_H / 2, NODE_W, NODE_H, lc.C_API_F, lc.C_API_S, rx=5, sw=1.2)
        lc.text(COL_C[3], cy + 3, 'process_for_engine', 9, lc.C_TXT, 'middle',
                maxw=NODE_W - 10, tag='n4' + str(k))
# 行内推进箭头（col k → col k+1）
for k, cy in enumerate(ROW_Y):
    for col in range(3):
        x1 = COL_C[col] + NODE_W / 2 + 3
        x2 = COL_C[col + 1] - NODE_W / 2 - 4
        lc.seg(x1, cy, x2, cy, lc.C_API_S, 1.1, 'std')
# gather 注记（列间、行带下方）
GATH_Y = ROW_Y[2] + NODE_H / 2 + 20
G = [('await asyncio.gather（L1092）', (COL_C[0] + COL_C[1]) / 2),
     ('逐条并行（tokenize_prompts_async 内 L646）', (COL_C[1] + COL_C[2]) / 2),
     ('await asyncio.gather（L1100）', (COL_C[2] + COL_C[3]) / 2)]
for txt, gx in G:
    lc.text(gx, GATH_Y, txt, 8, lc.C_MUTE, 'middle', maxw=280, tag='g' + str(gx))

# ---------------- 产物行：3 个 EngineInput + 共同时间戳 ----------------
PRD_Y = PNL_Y + PNL_H + 30
PRD_H = 58
BADGE_W = 172
badges = [('路 1 → type=token', 'arrival_time 同一打点'),
          ('路 2 → type=multimodal', 'arrival_time 同一打点'),
          ('路 3 → type=token', 'arrival_time 同一打点')]
BX = [770, 968, 1166]
for (t, s), bx in zip(badges, BX):
    lc.rect(bx - BADGE_W / 2, PRD_Y, BADGE_W, PRD_H, '#ffffff', lc.C_API_S, rx=7, sw=1.5)
    lc.text(bx, PRD_Y + 22, t, 9.5, lc.C_TXT, 'middle', True, maxw=BADGE_W - 12, tag='bd' + t)
    lc.text(bx, PRD_Y + 41, s, 8, lc.C_MUTE, 'middle', maxw=BADGE_W - 12, tag='bds' + t)
# 面板 col4 底边 → 分配轨 → 三盒（trunk + rail + drops，不穿任何框）
RAIL_Y = PRD_Y - 16
lc.seg(COL_C[3], PNL_Y + PNL_H, COL_C[3], RAIL_Y, lc.C_API_S, 2.0, 'std')
lc.seg(BX[0], RAIL_Y, COL_C[3], RAIL_Y, lc.C_API_S, 2.0)
lc.text(COL_C[3] - 10, RAIL_Y - 5, '3× EngineInput（同批三盒）', 8.5, lc.C_API_S, 'end',
        maxw=220, tag='fan')
for bx in BX:
    lc.seg(bx, RAIL_Y, bx, PRD_Y - 1.5, lc.C_API_S, 2.0, 'std')
# 时间戳图钉（左）→ 三盒逐链（虚线 = 单次打点随批携带）
ST_X, ST_Y = 350, PRD_Y + PRD_H / 2
lc.rect(ST_X - 120, PRD_Y + 6, 240, PRD_H - 12, lc.C_API_F, lc.C_API_S, rx=7, sw=1.4)
lc.circle(ST_X - 98, ST_Y, 8, lc.C_API_S, 1.4, dash=False)
lc.seg(ST_X - 98, ST_Y, ST_X - 98, ST_Y - 5, lc.C_API_S, 1.2)
lc.seg(ST_X - 98, ST_Y, ST_X - 94, ST_Y + 2, lc.C_API_S, 1.2)
lc.text(ST_X - 84, ST_Y - 3, '同一个 arrival_time', 9, lc.C_API_S, 'start', True,
        maxw=150, tag='stamp')
lc.text(ST_X - 84, ST_Y + 13, '打点一次 · 逐位相同', 8, lc.C_MUTE, 'start', maxw=150,
        tag='stamps')
seg_x = [ST_X + 120, BX[0] + BADGE_W / 2 + 3, BX[1] + BADGE_W / 2 + 3]
dst_x = [BX[0] - BADGE_W / 2 - 3, BX[1] - BADGE_W / 2 - 3, BX[2] - BADGE_W / 2 - 3]
for a, b in zip(seg_x, dst_x):
    lc.seg(a, ST_Y, b, ST_Y, lc.C_API_S, 1.2, dash=True)

# ---------------- completion 面条 ----------------
CMP_Y = PRD_Y + PRD_H + 26
CMP_H = 74
lc.rect(MX, CMP_Y, BXR - MX, CMP_H, '#ffffff', lc.C_MUTE, rx=8, sw=1.3)
lc.text(MX + 16, CMP_Y + 21, 'completion 面（render_cmpl · base.py:L985-L1006）：同一四步，step1 直通', 10.5,
        lc.C_TXT, 'start', True, maxw=760, tag='cmp:t')
mini = ['step1 render_prompt（直通——无模板可展开）', 'tokenize', 'step3 extras', 'step4 process_for_engine']
mw = [300, 110, 130, 230]
mx = MX + 16
for i, (m, w_) in enumerate(zip(mini, mw)):
    dash = (i == 0)
    lc.rect(mx, CMP_Y + 34, w_, 28, '#ffffff', lc.C_API_S, rx=5, sw=1.1, dash=dash)
    lc.text(mx + w_ / 2, CMP_Y + 51, m, 8.5, lc.C_TXT, 'middle', maxw=w_ - 8, tag='m' + str(i))
    if i < 3:
        lc.seg(mx + w_ + 2, CMP_Y + 48, mx + w_ + 24, CMP_Y + 48, lc.C_API_S, 1.3, 'std')
    mx += w_ + 26
lc.text(BXR - 16, CMP_Y + 21, '实测 trace 同四步', 9, lc.C_MUTE, 'end', maxw=300, tag='cmp:n')

# ---------------- 图例 + 页脚 ----------------
LEG_Y = CMP_Y + CMP_H + 30
lx = MX
items = [('arrow', None, '工序推进'), ('dashline', None, '同一 arrival_time 随批携带'),
         ('dashbox', None, '虚线框 = 支路 / 黑盒')]
for kind, _, name in items:
    if kind == 'arrow':
        lc.seg(lx, LEG_Y, lx + 30, LEG_Y, lc.C_API_S, 1.6, 'std')
    elif kind == 'dashline':
        lc.seg(lx, LEG_Y, lx + 30, LEG_Y, lc.C_API_S, 1.4, dash=True)
    else:
        lc.rect(lx, LEG_Y - 8, 22, 14, '#ffffff', lc.C_MUTE, rx=4, sw=1.1, dash=True)
    lc.text(lx + 36, LEG_Y + 3, name, 9.5, lc.C_TXT, 'start', maxw=260, tag='leg' + name)
    lx += 36 + lc.tw(name, 9.5) + 26
lc.text(MX, LEG_Y + 26, 'token id 与词切分（[3,4,5]）为确定性 seam 示意值（真实为 HF BPE）；'
        '工序顺序 / 产物判别键 / 时间戳语义为真代码路径 · 步骤序与计数 host 实测',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, LEG_Y + 44, '编排 verbatim：renderers/base.py:L1071-L1109（chat 面）/ L985-L1006（completion 面）· '
        '行号基线 vLLM v0.27.1', 9, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch06-fig-render-pipeline.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
