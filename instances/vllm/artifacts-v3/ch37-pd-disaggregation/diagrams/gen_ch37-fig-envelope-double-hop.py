#!/usr/bin/env python3
"""ch37 机制图 m2 · envelope-double-hop（figure_spec ch37-fig-envelope-double-hop，模板 swimlane·UML 时序图）

放大自 L0「双实例+KV 边界」北条 disaggregator 路由器（请求双跳）、L2 章图站 3/5。
架构归属回指 L0/L2（FIGURE-SYSTEM §3.3）：图右上角指北小签。

时序图种规约（WRITING-CONTRACT-v3 §8 + FIGURE-SYSTEM §0 硬规则 2）：参与者=竖直生命线
（顶部名牌）+ 共享时间轴（每条消息 y 一根网格线贯穿全部生命线）+ 消息=水平直线
（A 线某时刻 → B 线同一 y，禁一切折线/肘形）；瞬时动作=骑线标记+真实值标注。

claim：控制面的全部工作=一只 10 字段回执信封随请求/响应双跳搬运：跳 1 proxy 改写三项
（max_tokens=1、stream=False、摘 min_tokens）发单 P，P 只算 prefill；跳 2 从 P 响应掏出
kv_transfer_params 原样附加转交 D——信封字段从产出到挂载逐字不变。

数字全部取自 figure_spec.numbers（traces 实测 + pin 源码锚点，逐字核对）：
  · 跳 1 改写恰 3 项：max_tokens 7→1 · stream True→False · min_tokens 2 摘掉（发完放回）；入场 params 6 键
  · P 终局回执 10 键：remote_block_ids=[[0,1,2,3]] · remote_num_tokens=16 · tp_size=1 · engine_id/host/port
  · 跳 2：D 腿请求体 kv_transfer_params 与回执 == 为 True；D 挂载 10 键、7 个坐标字段全部相等
  · 可选省一次 tokenize：prompt_token_ids 复用（docs/features/disagg_prefill.md）
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
C_GRID = '#e2e8f0'

# ---------------- 标题区 ----------------
lc.text(MX, 36, 'disaggregator 不做分拣判断，只搬面单：一只 10 字段回执信封随请求双跳', 16.5,
        lc.C_TXT, 'start', True, maxw=1200, tag='title')
lc.text(MX, 60, '跳 1 改写三项发单 P（只算 prefill）· 跳 2 掏出 P 回执原样附加转交 D——信封字段从产出点到挂载点逐字不变',
        10.5, lc.C_MUTE, 'start', maxw=1200, tag='subtitle')
_ch = '放大自 L2 站 3/5（disaggregator 双跳）· L0：双实例+KV 边界北条'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_BEAT_T, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 参与者（竖直生命线 + 顶部名牌） ----------------
PARTS = [
    ('客户端', 'HTTP', lc.C_MUTE, 160),
    ('disaggregator', '路由器（proxy）', lc.C_API_S, 500),
    ('Prefill 引擎 P', 'kv_role=kv_producer', lc.C_ENG_S, 1010),
    ('Decode 引擎 D', 'kv_role=kv_consumer', lc.C_ENG_S, 1380),
]
NP_Y, NP_H = 88, 50
LF_TOP = NP_Y + NP_H
LF_BOT = 726


def part(i):
    return PARTS[i][3]


for nm, sub, color, cx in PARTS:
    w = 190 if nm != 'disaggregator' else 210
    lc.rect(cx - w / 2, NP_Y, w, NP_H, '#ffffff', color, rx=8, sw=1.8)
    lc.text(cx, NP_Y + 21, nm, 11.5, color, 'middle', True, maxw=w - 12, tag='np:' + nm)
    lc.text(cx, NP_Y + 39, sub, 8.5, lc.C_MUTE, 'middle', maxw=w - 12, tag='np:s' + nm)
    # 生命线（窄竖条：消息端点贴边）
    lc.rect(cx - 1.5, LF_TOP, 3, LF_BOT - LF_TOP, C_GRID, 'none', rx=0, sw=0)

# 共享时间轴：左侧竖直向下箭头
TA_X = 84
lc.seg(TA_X, LF_TOP + 10, TA_X, LF_BOT - 8, lc.C_MUTE, 1.4, 'std')
lc.text(TA_X, LF_TOP + 2, '时间', 9, lc.C_MUTE, 'middle', tag='time:t')


def gridline(y):
    lc.seg(104, y, 1500, y, C_GRID, 1.0)


def msg(y, a, b, color, marker, above, below=None):
    """水平消息直线：a→b（PARTS 下标）。above/below = 消息线标注行（居中骑线）。"""
    gridline(y)
    x1, x2 = part(a), part(b)
    lc.seg(x1, y, x2, y, color, 1.8, marker)
    mid = (x1 + x2) / 2
    for k, ln in enumerate(above):
        lc.text(mid, y - 26 + k * 14, ln, 9, lc.C_TXT, 'middle',
                maxw=abs(x2 - x1) - 30, tag='m%d_%d:%s' % (a, b, ln[:10]))
    for k, ln in enumerate(below or []):
        lc.text(mid, y + 15 + k * 14, ln, 8.5, lc.C_MUTE, 'middle',
                maxw=abs(x2 - x1) - 30, tag='mb%d_%d:%s' % (a, b, ln[:10]))


def act_bar(i, y0, y1, lines, side='right'):
    """生命线内活动条（EngineCore 橙）+ 侧注。"""
    cx = part(i)
    lc.rect(cx - 8, y0, 16, y1 - y0, lc.C_ENG_S, 'none', rx=3, sw=0)
    if side == 'right':
        for k, ln in enumerate(lines):
            lc.text(cx + 18, y0 + 16 + k * 14, ln, 8.5, lc.C_ENG_S, 'start',
                    maxw=380, tag='act:%s' % ln[:10])
    else:
        for k, ln in enumerate(lines):
            lc.text(cx - 18, y0 + 16 + k * 14, ln, 8.5, lc.C_ENG_S, 'end',
                    maxw=380, tag='act:%s' % ln[:10])


# ---------------- 消息序列 ----------------
# ① 客户端 → proxy：原始请求
msg(186, 0, 1, lc.C_MUTE, 'std',
    ['原始请求（D 才需要的参数躺在原始请求体里）'],
    ['max_tokens=7 · min_tokens=2 · stream=True'])

# ② proxy → P：跳 1 发单（改写三项 + 6 键入场）
msg(254, 1, 2, lc.C_API_S, 'dn',
    ['跳 1 · 发单 P：改写恰 3 项——max_tokens 7→1 · stream True→False · 摘 min_tokens（发完放回）'],
    ['入场 kv_transfer_params 6 键（remote_*=None）· 旗标 do_remote_decode=True（P 腿：decode 在远端）'])

# P 活动条：prefill 16 token
act_bar(2, 272, 348, ['P 只算 prefill：16 token', '终局 LENGTH_CAPPED', '（max_tokens=1 收工交块）'])

# ③ P → proxy：终局回执（信封骑线）
y_ret = 404
gridline(y_ret)
lc.seg(part(2), y_ret, part(1), y_ret, lc.C_ENG_S, 1.8, 'up')
# 信封（骑在回程消息线上的小矩形，KV 青）
ENV_X0, ENV_X1 = 580, 960
lc.rect(ENV_X0, y_ret - 82, ENV_X1 - ENV_X0, 92, lc.C_KV_F, lc.C_KV_S, rx=6, sw=1.6)
lc.text((ENV_X0 + ENV_X1) / 2, y_ret - 66, '回执信封 · kv_transfer_params（10 键）', 9.5,
        lc.C_KV_S, 'middle', True, maxw=ENV_X1 - ENV_X0 - 16, tag='env:t')
lc.text((ENV_X0 + ENV_X1) / 2, y_ret - 50, '坐标 7：remote_block_ids=[[0,1,2,3]] · remote_num_tokens=16 · tp_size=1',
        8.5, lc.C_TXT, 'middle', maxw=ENV_X1 - ENV_X0 - 16, tag='env:l1')
lc.text((ENV_X0 + ENV_X1) / 2, y_ret - 36, 'remote_engine_id/host/port · remote_request_id=req-1',
        8.5, lc.C_TXT, 'middle', maxw=ENV_X1 - ENV_X0 - 16, tag='env:l2')
lc.text((ENV_X0 + ENV_X1) / 2, y_ret - 22, '旗标 2：do_remote_prefill=True / do_remote_decode=False',
        8.5, lc.C_TXT, 'middle', maxw=ENV_X1 - ENV_X0 - 16, tag='env:l3')
lc.text((ENV_X0 + ENV_X1) / 2, y_ret - 8, '到期占位 1：remote_blocks_expiry_time=None',
        8.5, lc.C_TXT, 'middle', maxw=ENV_X1 - ENV_X0 - 16, tag='env:l4')
lc.text((part(1) + part(2)) / 2, y_ret + 14, 'P 终局回执随 P 响应出引擎（request_finished 一次性构造）', 8.5,
        lc.C_MUTE, 'middle', maxw=part(2) - part(1) - 30, tag='m3:below')

# ④ proxy → D：跳 2 转交（信封原样附加）
msg(486, 1, 3, lc.C_API_S, 'dn',
    ['跳 2 · 转交 D：信封原样附加（整体复制 · 不做字段级改写）——D 腿请求体与回执 == 为 True',
     'min_tokens=2 放回 · stream=True 恢复 · 旗标 do_remote_prefill=True（D 腿：prefill 在远端）'],
    ['坐标字段无任何翻转点：7 个坐标字段与 P 产出逐字相等'])

# D 活动条：异步拉 + decode
act_bar(3, 512, 588, ['D：异步拉 KV（READ 直读 P 显存）',
                      '→ decode 续算 7 token',
                      'Request 挂载 10 键'], side='left')

# ⑤ D → proxy：流式回包
msg(622, 3, 1, lc.C_ENG_S, 'up',
    ['D 流式回包'])

# ⑥ proxy → 客户端：SSE
msg(662, 1, 0, lc.C_API_S, 'dn',
    ['SSE 流转发（stream=True）'])

# 两条腿的旗标翻转高亮（骑线小签，KV 青）
FLAG_Y = 700
lc.rect(part(1) + 30, FLAG_Y - 14, 300, 20, lc.C_KV_F, lc.C_KV_S, rx=9, sw=1.1)
lc.text(part(1) + 180, FLAG_Y, 'P 腿翻这面：do_remote_decode=True', 8.5, lc.C_KV_S, 'middle', True,
        maxw=290, tag='flag:1')
lc.rect(part(1) + 350, FLAG_Y - 14, 320, 20, lc.C_KV_F, lc.C_KV_S, rx=9, sw=1.1)
lc.text(part(1) + 510, FLAG_Y, 'D 腿翻这面：do_remote_prefill=True', 8.5, lc.C_KV_S, 'middle', True,
        maxw=310, tag='flag:2')

# ---------------- 页脚 ----------------
FY = LF_BOT + 26
lc.text(MX, FY, '双跳 = 2 次 HTTP 过境 + 2 次引擎过境 · 信封 10 键 = 7 坐标字段 + 2 旗标 + 1 到期时刻占位 None · '
                '可选省一次 tokenize：prompt_token_ids 复用（docs/features/disagg_prefill.md）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot1')
lc.text(MX, FY + 15, '逐字锚 toy_proxy_server.py:L155-L197（跳 1 改写）/ L219-L237（跳 2 附加）· '
                     'nixl/pull_scheduler.py:L269-L280（回执一次性构造）· '
                     'vllm/v1/core/sched/scheduler.py:L1901-L1935（回执随 P 响应出引擎 · EngineCoreOutput.kv_transfer_params）· '
                     'vllm/v1/request.py:L100-L122（D 挂载）',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot2')
lc.text(MX, FY + 30, '行号基线 vLLM v0.27.1 · 角色色即身份：蓝=proxy（API 面）/ 橙=引擎 / 青=信封（KV 坐标），与全书 L0/L2 同源',
        8.5, lc.C_FAINT, 'start', maxw=BXR - MX, tag='foot3')

H = FY + 48

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch37-fig-envelope-double-hop.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
