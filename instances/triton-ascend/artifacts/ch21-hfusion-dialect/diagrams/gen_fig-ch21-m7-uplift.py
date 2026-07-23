#!/usr/bin/env python3
"""swimlane 变体：LinalgToHFusion 上抬——4 个 pattern 各认一种昇腾语义戳，
把匹配的 linalg 输入模式换成 hfusion 输出 op（partial conversion）。
每道泳道：左=linalg 输入模式，中间=pattern 名+识别的戳，右=hfusion 输出 op。
LinalgMapToHFusionPattern 道内两例（一元/二元）。底部合法性框：
map/generic 恒 illegal，reduce 仅带 reduce_mode 才 illegal，legal 方言 7 个。
数值/夹具锚全部来自 explainer numbers/worked_example，全坐标计算，零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def est_w(s, size=11):
    """粗估纯 ASCII 粗体文本宽度（无 CJK），按经验系数 0.66*size/字符。"""
    return len(s) * size * 0.66

CJK_RANGE = ('㐀', '鿿')

def char_w(c, size):
    """与 lint_diagram_geometry 同口径的字符宽度估算：CJK≈size，其余≈size*0.55。"""
    return size if CJK_RANGE[0] <= c <= CJK_RANGE[1] else size * 0.55

def wrap_text(s, max_w, size):
    """按 lint 口径贪心换行，返回多行列表，逐行宽度 <= max_w；
    超宽时优先回退到当前行内最后一个空格处断行，避免把英文标识符切断在词中间。"""
    lines, cur, cur_w, last_space = [], "", 0.0, -1
    for ch in s:
        cw = char_w(ch, size)
        if cur and cur_w + cw > max_w:
            if last_space >= 0:
                lines.append(cur[:last_space])
                rest = cur[last_space + 1:]
            else:
                lines.append(cur)
                rest = ""
            cur, cur_w, last_space = "", 0.0, -1
            for rc in rest:
                cur += rc
                cur_w += char_w(rc, size)
                if rc == " ":
                    last_space = len(cur) - 1
        cur += ch
        cur_w += cw
        if ch == " ":
            last_space = len(cur) - 1
    if cur:
        lines.append(cur)
    return lines

TITLE = "LinalgToHFusion：4 pattern 上抬，普通 linalg 原样放行"
SUBTITLE = "-convert-linalg-to-hfusion（applyPartialConversion，LinalgToHFusion.cpp:L465-L466 注册 4 个 pattern）"

LANES = [
    {
        "pattern": "LinalgMapToHFusionPattern",
        "color": ("#eff6ff", "#1d4ed8", "#1e3a8a"),
        "rows": [
            ("linalg.map{__hmf_relu}", "callee 名 __hmf_relu", "hfusion.elemwise_unary<relu>"),
            ("linalg.map{__hmf_ldexp}（2 输入）", "callee 名 __hmf_ldexp", "hfusion.elemwise_binary<ldexp>"),
        ],
    },
    {
        "pattern": "LinalgGenericToHFusionArangePattern",
        "color": ("#f5f3ff", "#7c3aed", "#4c1d95"),
        "rows": [
            ("linalg.generic yield index_cast(linalg.index)", "1D yield=index 模式", "hfusion.arange offset[%c0] strides[%c1]"),
        ],
    },
    {
        "pattern": "AtomicLinalgGenericToHFusionStorePattern",
        "color": ("#fdf2f8", "#be185d", "#831843"),
        "rows": [
            ("linalg.generic{GenericAtomicRMW=\"fadd\"}", "GenericAtomicRMW 属性", "hfusion.atomic_rmw atomic_kind=<add>"),
        ],
    },
    {
        "pattern": "LinalgToHFusionReduceWithIndex",
        "color": ("#f0fdf4", "#16a34a", "#14532d"),
        "rows": [
            ("linalg.reduce{reduce_mode=\"max_with_index\"}", "reduce_mode 属性", "hfusion.reduce_with_index<max>"),
        ],
    },
]

PAD, TOP = 50, 110
IN_W, OUT_W = 300, 280
ARROW_W = 190
ROW_H = 40
ROW_GAP = 6
HEADER_H = 24
LANE_GAP = 20

w = PAD * 2 + IN_W + ARROW_W + OUT_W

CO_HEAD = "partial conversion 合法性（LinalgToHFusion.cpp:L479-L493）:"
CO1 = "linalg.map / linalg.generic 恒 illegal（addIllegalOp，必被消费）；linalg.reduce 仅带 reduce_mode 属性时 illegal（addDynamicallyLegalOp）。"
CO2 = "legal 方言 7 个：memref / linalg / bufferization / tensor / hfusion / arith / math——普通 linalg op（无戳）从不进 illegal 集，原样放行。"
CO3 = "每命中一次 pattern，illegal op 计数严格 -1；4 个 pattern 各自 replaceOpWithNewOp，单调收敛到 0 即转换成功。"
CO_FS = 11.5
CO_MAX_W = w - 2 * (PAD + 14)
CO_LINES = []
for note in (CO1, CO2, CO3):
    CO_LINES.extend(wrap_text(note, CO_MAX_W, CO_FS))
CALLOUT_H = 30 + len(CO_LINES) * 20 + 10

y = TOP
lane_ys = []
for lane in LANES:
    n = len(lane["rows"])
    lane_h = HEADER_H + n * ROW_H + (n - 1) * ROW_GAP
    lane_ys.append((y, lane_h))
    y += lane_h + LANE_GAP
content_bottom = y - LANE_GAP
h = content_bottom + CALLOUT_H + PAD + 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-26}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD-8}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

in_x = PAD
arrow_x0 = in_x + IN_W
out_x = arrow_x0 + ARROW_W

for lane, (ly, lane_h) in zip(LANES, lane_ys):
    fill, stroke, tf = lane["color"]
    L.append(f'<rect x="{in_x}" y="{ly}" width="{IN_W+ARROW_W+OUT_W}" height="{lane_h}" rx="6" '
              f'fill="{fill}" opacity="0.35"/>')
    header_w = est_w(lane["pattern"]) + 16
    L.append(f'<rect x="{in_x}" y="{ly}" width="{header_w}" height="{HEADER_H}" rx="4" '
              f'fill="{stroke}"/>')
    L.append(f'<text x="{in_x+8}" y="{ly+HEADER_H-7}" font-family="sans-serif" font-size="11" '
              f'font-weight="bold" fill="white">{esc(lane["pattern"])}</text>')
    row_top = ly + HEADER_H + 4
    for i, (in_txt, tag, out_txt) in enumerate(lane["rows"]):
        ry = row_top + i * (ROW_H + ROW_GAP)
        L.append(f'<rect x="{in_x}" y="{ry}" width="{IN_W-10}" height="{ROW_H-6}" rx="5" '
                  f'fill="white" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{in_x+10}" y="{ry+(ROW_H-6)/2+4}" font-family="sans-serif" '
                  f'font-size="11" fill="#1e293b">{esc(in_txt)}</text>')
        ax1 = in_x + IN_W - 10
        ax2 = out_x
        ay = ry + (ROW_H - 6) / 2
        L.append(f'<line x1="{ax1}" y1="{ay}" x2="{ax2}" y2="{ay}" '
                  f'stroke="{stroke}" stroke-width="1.6" marker-end="url(#a)"/>')
        L.append(f'<text x="{(ax1+ax2)/2}" y="{ay-6}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10" fill="{tf}">识别: {esc(tag)}</text>')
        L.append(f'<rect x="{out_x}" y="{ry}" width="{OUT_W-10}" height="{ROW_H-6}" rx="5" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        L.append(f'<text x="{out_x+10}" y="{ry+(ROW_H-6)/2+4}" font-family="sans-serif" '
                  f'font-size="11" font-weight="bold" fill="{tf}">{esc(out_txt)}</text>')

# 顶部列标
L.append(f'<text x="{in_x}" y="{TOP-4}" font-family="sans-serif" font-size="10" '
          f'fill="#94a3b8">linalg 输入模式</text>')
L.append(f'<text x="{out_x}" y="{TOP-4}" font-family="sans-serif" font-size="10" '
          f'fill="#94a3b8">hfusion 输出 op</text>')

# 底部合法性 callout
co_y = content_bottom + 20
L.append(f'<rect x="{PAD}" y="{co_y}" width="{w-2*PAD}" height="{CALLOUT_H}" rx="6" '
          'fill="#f8fafc" stroke="#475569" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+14}" y="{co_y+22}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#334155">{esc(CO_HEAD)}</text>')
for i, line in enumerate(CO_LINES):
    L.append(f'<text x="{PAD+14}" y="{co_y+42+i*20}" font-family="sans-serif" font-size="{CO_FS}" '
              f'fill="#334155">{esc(line)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch21-m7-uplift.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
