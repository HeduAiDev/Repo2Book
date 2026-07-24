#!/usr/bin/env python3
"""fig-ch31-dispatch-flow — flow 模板(决策链 + 底部双落点,风格参照
ch04 fig-ch04-m2-fourth-branch-decision)。get_backend_func 三级解析
(缓存命中? -> env 显式指定? -> 自动探测)全部汇入一个全局缓存写入点,
再经 execute_func 两级查表分派到两框架终态 C++ 片段。
数字/字面量取自 traces/registry_trace.json 与 dossier embed_excerpts,逐字核对。
全部坐标由循环/常量计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def text_w(s, fs):
    return fs * sum((0.98 if '一' <= c <= '鿿' else 0.58) for c in s)


TITLE = "get_backend_func：解析活动框架 → 两级查表分派"
SUBTITLE = "同一句 header_file 请求，在 mindspore / torch_npu 下产出零共享的两段 C++"
ENTRY_LABEL = "入口：get_backend_func(\"header_file\", True)"
EXEC_LABEL = "execute_func(backend_policy, \"header_file\", True)"
EXEC_SUBLABEL = "两级查表 strategies[backend_policy][\"header_file\"]"

PAD = 44
TOP = 108

# 主链三个判定框:(标题, 条件行, 源码位置, 继续标签, 旁路标签(到 merge))
CHAIN = [
    ("① 缓存命中？", "backend_policy is None", "utils.py:L42",
     "是，继续解析", "否 → 直接复用缓存（粘滞，不再解析）"),
    ("② 环境变量显式指定？", "os.getenv(\"TRITON_BACKEND\") ∈ {torch_npu, mindspore}", "utils.py:L43-L45",
     "否，继续探测", "命中 → backend_policy = env 值"),
    ("③ 自动探测", "import torch; import torch_npu 是否成功", "utils.py:L46-L52",
     None, None),
]

longest = max(text_w(cond, 11.5) for _, cond, *_ in CHAIN)
MAIN_W = int(longest) + 64
BOX_H = 74
GAP_V = 92

row_y = [TOP + i * (BOX_H + GAP_V) for i in range(len(CHAIN))]

# ③ 之后直接是两个探测结果框(成功=torch_npu / ImportError=mindspore)，再共同汇入 merge
PROBE_W, PROBE_H = 240, 64
probe_gap = 60
probe_y = row_y[2] + BOX_H + 56
probe_total = PROBE_W * 2 + probe_gap

# merge(写入全局缓存)与 execute_func 框
MERGE_W, MERGE_H = MAIN_W + 40, 74

EXEC_W, EXEC_H = MAIN_W, 60

# 底部两个终态 C++ 框(term_gap 须能放下中间「共享 include」标注框且不覆到两侧文字)
TERM_W, TERM_H = 430, 168
term_gap = 210
term_total = TERM_W * 2 + term_gap

# 主链①②右侧旁路列(线 + 标签)需要的额外宽度
BYPASS_COL_GAP = 40    # 主链右边缘到旁路竖线的距离
BYPASS_LABEL_W = 300   # 旁路标签预留宽度

# 统一以一个画布中心 center_x 摆放每一行(主链/探测/merge/execute/终态)，
# 核心宽度取各行最宽者,并保证主链右侧有足够空间放旁路列 + 标签。
core_w = max(MAIN_W + 2 * (BYPASS_COL_GAP + BYPASS_LABEL_W), probe_total, term_total)
center_x = PAD + core_w / 2

main_x = center_x - MAIN_W / 2
probe_left_x = center_x - probe_total / 2
probe_right_x = probe_left_x + PROBE_W + probe_gap
merge_x = center_x - MERGE_W / 2
exec_x = center_x - EXEC_W / 2
term_left_x = center_x - term_total / 2
term_right_x = term_left_x + TERM_W + term_gap

merge_y = probe_y + PROBE_H + 92
exec_y = merge_y + MERGE_H + 56
term_y = exec_y + EXEC_H + 78

w = PAD * 2 + core_w
h = term_y + TERM_H + 100

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="p" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w / 2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>',
     f'<text x="{main_x}" y="{TOP - 24}" font-family="sans-serif" font-size="12.5" '
     f'font-weight="bold" fill="#7c3aed">{esc(ENTRY_LABEL)}</text>']

# 主链三个判定框
for i, (name, cond, loc, cont_lbl, bypass_lbl) in enumerate(CHAIN):
    y = row_y[i]
    L.append(f'<rect x="{main_x}" y="{y}" width="{MAIN_W}" height="{BOX_H}" rx="10" '
              f'fill="#e0f2fe" stroke="#0369a1" stroke-width="1.6"/>')
    L.append(f'<text x="{main_x + 16}" y="{y + 24}" font-family="sans-serif" font-size="13.5" '
              f'font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{main_x + 16}" y="{y + 44}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(cond)}</text>')
    L.append(f'<text x="{main_x + MAIN_W - 14}" y="{y + BOX_H - 9}" text-anchor="end" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
              f'fill="#0369a1">{esc(loc)}</text>')

# 主链竖向连接(①→②→③, "继续"边)
for i in range(2):
    y1 = row_y[i] + BOX_H
    y2 = row_y[i + 1]
    cx = main_x + MAIN_W / 2
    L.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" '
              f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
    L.append(f'<text x="{cx + 10}" y="{(y1 + y2) / 2 + 4}" font-family="sans-serif" font-size="11" '
              f'fill="#334155">{esc(CHAIN[i][3])}</text>')

# ①②的旁路(直接汇入 merge，标注粘滞/命中)——先画到右侧转折点，稍后统一连到 merge 顶边
bypass_anchor_x = main_x + MAIN_W + BYPASS_COL_GAP
bypass_points = []
for i in range(2):
    y = row_y[i] + BOX_H / 2
    cx1 = main_x + MAIN_W
    L.append(f'<line x1="{cx1}" y1="{y}" x2="{bypass_anchor_x}" y2="{y}" '
              f'stroke="#7c3aed" stroke-width="1.6" stroke-dasharray="5,4"/>')
    L.append(f'<text x="{cx1 + 12}" y="{y - 8}" font-family="sans-serif" font-size="11" '
              f'font-weight="bold" fill="#7c3aed">{esc(CHAIN[i][4])}</text>')
    bypass_points.append((bypass_anchor_x, y))

# ③ 之后:两个探测结果框
probe_cx_l = probe_left_x + PROBE_W / 2
probe_cx_r = probe_right_x + PROBE_W / 2
node3_cx = main_x + MAIN_W / 2
node3_by = row_y[2] + BOX_H
mid_y = (node3_by + probe_y) / 2
L.append(f'<path d="M {node3_cx} {node3_by} L {node3_cx} {mid_y} L {probe_cx_l} {mid_y} L {probe_cx_l} {probe_y}" '
          f'fill="none" stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<path d="M {node3_cx} {node3_by} L {node3_cx} {mid_y} L {probe_cx_r} {mid_y} L {probe_cx_r} {probe_y}" '
          f'fill="none" stroke="#1d4ed8" stroke-width="2" marker-end="url(#b)"/>')
L.append(f'<text x="{probe_cx_l}" y="{mid_y - 10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#15803d">{esc("成功")}</text>')
L.append(f'<text x="{probe_cx_r}" y="{mid_y - 10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#1d4ed8">{esc("ImportError")}</text>')

L.append(f'<rect x="{probe_left_x}" y="{probe_y}" width="{PROBE_W}" height="{PROBE_H}" rx="9" '
          f'fill="#dcfce7" stroke="#15803d" stroke-width="2"/>')
L.append(f'<text x="{probe_cx_l}" y="{probe_y + 26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#14532d">{esc("backend_policy = torch_npu")}</text>')
L.append(f'<text x="{probe_cx_l}" y="{probe_y + 46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#166534">{esc("import torch, torch_npu 成功")}</text>')

L.append(f'<rect x="{probe_right_x}" y="{probe_y}" width="{PROBE_W}" height="{PROBE_H}" rx="9" '
          f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>')
L.append(f'<text x="{probe_cx_r}" y="{probe_y + 26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#1e3a8a">{esc("backend_policy = mindspore")}</text>')
L.append(f'<text x="{probe_cx_r}" y="{probe_y + 46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#1e40af">{esc("ImportError（未装 torch_npu）")}</text>')

# merge 框(全部路径汇入):①②旁路 + 两个探测结果
merge_cx = merge_x + MERGE_W / 2
for bx, by in bypass_points:
    L.append(f'<path d="M {bx} {by} L {bx} {merge_y - 20} L {merge_cx + 60} {merge_y - 20} '
              f'L {merge_cx + 60} {merge_y}" fill="none" stroke="#7c3aed" stroke-width="1.6" '
              f'stroke-dasharray="5,4" marker-end="url(#p)"/>')
for px, lbl_x in ((probe_cx_l, -60), (probe_cx_r, 60)):
    L.append(f'<line x1="{px}" y1="{probe_y + PROBE_H}" x2="{px}" y2="{merge_y}" '
              f'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

L.append(f'<rect x="{merge_x}" y="{merge_y}" width="{MERGE_W}" height="{MERGE_H}" rx="10" '
          f'fill="#fef9c3" stroke="#a16207" stroke-width="2"/>')
L.append(f'<text x="{merge_cx}" y="{merge_y + 28}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#713f12">'
          f'{esc("backend_policy 写入全局缓存")}</text>')
L.append(f'<text x="{merge_cx}" y="{merge_y + 50}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#854d0e">{esc("粘滞：本进程内之后一律复用，不再重新解析")}</text>')

# merge -> execute_func
L.append(f'<line x1="{merge_cx}" y1="{merge_y + MERGE_H}" x2="{merge_cx}" y2="{exec_y}" '
          f'stroke="#334155" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<rect x="{exec_x}" y="{exec_y}" width="{EXEC_W}" height="{EXEC_H}" rx="10" '
          f'fill="#ede9fe" stroke="#6d28d9" stroke-width="2"/>')
L.append(f'<text x="{exec_x + EXEC_W / 2}" y="{exec_y + 26}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#4c1d95">'
          f'{esc(EXEC_LABEL)}</text>')
L.append(f'<text x="{exec_x + EXEC_W / 2}" y="{exec_y + 46}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#5b21b6">'
          f'{esc(EXEC_SUBLABEL)}</text>')

# execute_func -> 两个终态框
exec_cx = exec_x + EXEC_W / 2
term_cx_l = term_left_x + TERM_W / 2
term_cx_r = term_right_x + TERM_W / 2
exec_by = exec_y + EXEC_H
mid2_y = (exec_by + term_y) / 2
L.append(f'<path d="M {exec_cx} {exec_by} L {exec_cx} {mid2_y} L {term_cx_l} {mid2_y} L {term_cx_l} {term_y}" '
          f'fill="none" stroke="#15803d" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<path d="M {exec_cx} {exec_by} L {exec_cx} {mid2_y} L {term_cx_r} {mid2_y} L {term_cx_r} {term_y}" '
          f'fill="none" stroke="#1d4ed8" stroke-width="2" marker-end="url(#b)"/>')
L.append(f'<text x="{term_cx_l}" y="{mid2_y - 10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#15803d">{esc("mindspore")}</text>')
L.append(f'<text x="{term_cx_r}" y="{mid2_y - 10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#1d4ed8">{esc("torch_npu")}</text>')

MS_LINE1 = '#include "include/utils/device_manager_conf.h"'
TN_LINE1 = "#include <ATen/ATen.h>"

L.append(f'<rect x="{term_left_x}" y="{term_y}" width="{TERM_W}" height="{TERM_H}" rx="10" '
          f'fill="#f0fdf4" stroke="#15803d" stroke-width="2.2"/>')
L.append(f'<text x="{term_left_x + 16}" y="{term_y + 26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#14532d">{esc("mindspore 版 header_file — 5 行 C++")}</text>')
L.append(f'<text x="{term_left_x + 16}" y="{term_y + 50}" font-family="monospace" font-size="10.5" '
          f'fill="#166534">{esc(MS_LINE1)}</text>')
L.append(f'<text x="{term_left_x + 16}" y="{term_y + 68}" font-family="monospace" font-size="10.5" '
          f'fill="#166534">{esc("… device_context_manager.h / aclnn_utils.h")}</text>')
L.append(f'<text x="{term_left_x + 16}" y="{term_y + 86}" font-family="monospace" font-size="10.5" '
          f'fill="#166534">{esc("… op_executor.h / pipeline.h（共 5 行）")}</text>')
L.append(f'<text x="{term_left_x + 16}" y="{term_y + 112}" font-family="sans-serif" font-size="10.5" '
          f'fill="#15803d" font-weight="bold">{esc("backend_register.py:L276-L282")}</text>')
L.append(f'<text x="{term_left_x + 16}" y="{term_y + 134}" font-family="sans-serif" font-size="10.5" '
          f'fill="#166534">{esc("同一能力，两框架各一套 C++ include")}</text>')

L.append(f'<rect x="{term_right_x}" y="{term_y}" width="{TERM_W}" height="{TERM_H}" rx="10" '
          f'fill="#eff6ff" stroke="#1d4ed8" stroke-width="2.2"/>')
L.append(f'<text x="{term_right_x + 16}" y="{term_y + 26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#1e3a8a">{esc("torch_npu 版 header_file — 3 行 C++")}</text>')
L.append(f'<text x="{term_right_x + 16}" y="{term_y + 50}" font-family="monospace" font-size="10.5" '
          f'fill="#1e40af">{esc(TN_LINE1)}</text>')
L.append(f'<text x="{term_right_x + 16}" y="{term_y + 68}" font-family="monospace" font-size="10.5" '
          f'fill="#1e40af">{esc("#include <torch_npu/.../NPUWorkspaceAllocator.h>")}</text>')
L.append(f'<text x="{term_right_x + 16}" y="{term_y + 86}" font-family="monospace" font-size="10.5" '
          f'fill="#1e40af">{esc("#include <torch_npu/.../OpCommand.h>（共 3 行）")}</text>')
L.append(f'<text x="{term_right_x + 16}" y="{term_y + 112}" font-family="sans-serif" font-size="10.5" '
          f'fill="#1d4ed8" font-weight="bold">{esc("backend_register.py:L285-L289")}</text>')
L.append(f'<text x="{term_right_x + 16}" y="{term_y + 134}" font-family="sans-serif" font-size="10.5" '
          f'fill="#1e40af">{esc("同一能力，两框架各一套 C++ include")}</text>')

# 两终端框之间的「共享 include = 0 行」标注
mid3_x = (term_left_x + TERM_W + term_right_x) / 2
L.append(f'<rect x="{mid3_x - 66}" y="{term_y + TERM_H / 2 - 20}" width="132" height="40" rx="8" '
          f'fill="#fef2f2" stroke="#b91c1c" stroke-width="1.6"/>')
L.append(f'<text x="{mid3_x}" y="{term_y + TERM_H / 2 - 3}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#7f1d1d">'
          f'{esc("共享 include")}</text>')
L.append(f'<text x="{mid3_x}" y="{term_y + TERM_H / 2 + 13}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" fill="#b91c1c">'
          f'{esc("= 0 行")}</text>')

foot_y1 = h - 40
foot_y2 = h - 18
FOOT1 = ("缓存粘滞实测：先在 TRITON_BACKEND=mindspore 下调用一次并解析出 mindspore，"
         "再把环境变量改成 torch_npu 后重复调用同一进程——backend_policy 仍返回 mindspore，不因环境变量事后改变而重新解析。")
FOOT2 = "对照：allocate_memory 同样两框架各一套 C++（mindspore 走 pyboost::MemBlock，torch_npu 走 at::empty(kPrivateUse1)）。"
L.append(f'<text x="{PAD}" y="{foot_y1}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOT1)}</text>')
L.append(f'<text x="{PAD}" y="{foot_y2}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc(FOOT2)}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch31-dispatch-flow.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w:.0f}x{h:.0f})')
