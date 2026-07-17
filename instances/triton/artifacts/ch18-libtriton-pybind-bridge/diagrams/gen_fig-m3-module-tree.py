#!/usr/bin/env python3
"""fig-m3-module-tree: flow(树状装配)模板。
claim: PYBIND11_MODULE(libtriton) 一次 import 就把 4 个核心子模块(ir/passes/
interpreter/llvm)加最多 4 个后端子模块装配成 Python 侧的 triton._C.libtriton 树。
改造点：ROOT 文本 + CHILDREN(标题/细节行列表，长代码行按逻辑断点预先拆行避免溢出)。
全坐标由循环/常量计算，零魔数;仅文本内容的换行点是人工选定(代码语义断点，
非坐标魔数)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROOT_LINES = ['PYBIND11_MODULE(libtriton, m)', '入口宏 · main.cc:L46']

# 每个 child: title(标题行) / detail(细节行列表，已按语义断点预拆) / badge(数字徽标或 None)
CHILDREN = [
    {
        "title": "辅助初始化",
        "detail": ["init_triton_stacktrace_hook(m)",
                   "init_triton_env_vars(m)",
                   "main.cc:L48-L49"],
        "badge": None,
    },
    {
        "title": "→ .ir",
        "detail": ["init_triton_ir(",
                   "m.def_submodule(\"ir\"))",
                   "承载 builder + 129 个 create_*"],
        "badge": None,
    },
    {
        "title": "→ .passes",
        "detail": ["init_triton_passes(",
                   "m.def_submodule(\"passes\"))",
                   "下辖 6 子分组 add_*"],
        "badge": "×6",
    },
    {
        "title": "→ .interpreter",
        "detail": ["init_triton_interpreter(",
                   "m.def_submodule(\"interpreter\"))",
                   "load / store / atomic_*"],
        "badge": None,
    },
    {
        "title": "→ .llvm",
        "detail": ["init_triton_llvm(",
                   "m.def_submodule(\"llvm\"))"],
        "badge": None,
    },
    {
        "title": "→ .<backend>",
        "detail": ["FOR_EACH_P(INIT_BACKEND,",
                   "TRITON_BACKENDS_TUPLE)",
                   "CMake 注入，main.cc:L44,L54",
                   "例:nvidia → init_triton_nvidia(...)"],
        "badge": "×N (N≤4)",
    },
]

N = len(CHILDREN)
CHILD_W, HGAP = 280, 24
PAD = 40
ROOT_W, ROOT_H = 420, 62
TOP = 90
BUS_Y_GAP = 46        # root 底 → 汇流线
DROP_GAP = 40         # 汇流线 → 各 child 顶
TITLE_TO_DETAIL_GAP = 20
DETAIL_LINE_H = 16
BOX_BOTTOM_PAD = 16

max_detail_lines = max(len(c["detail"]) for c in CHILDREN)
CHILD_H = 22 + TITLE_TO_DETAIL_GAP + max_detail_lines * DETAIL_LINE_H + BOX_BOTTOM_PAD

w = PAD * 2 + N * CHILD_W + (N - 1) * HGAP
root_x = w / 2
root_top = TOP
root_bottom = root_top + ROOT_H
bus_y = root_bottom + BUS_Y_GAP
child_top = bus_y + DROP_GAP
child_bottom = child_top + CHILD_H
badge_h = 22
h = child_bottom + 20 + badge_h + PAD + 34  # 底部留徽标行 + 脚注行

child_x = {i: PAD + i * (CHILD_W + HGAP) + CHILD_W / 2 for i in range(N)}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>'
          '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
          '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
          f'font-size="17" font-weight="bold" fill="#0f172a">'
          f'{esc("PYBIND11_MODULE 把 _C 扩展装配成一棵子模块树")}</text>')

# root 框
rx0 = root_x - ROOT_W / 2
L.append(f'<rect x="{rx0}" y="{root_top}" width="{ROOT_W}" height="{ROOT_H}" rx="8" '
          'fill="#dbeafe" stroke="#2563eb" stroke-width="1.6"/>')
for li, txt in enumerate(ROOT_LINES):
    fs = 14 if li == 0 else 12
    fw = "bold" if li == 0 else "normal"
    fill = "#1e3a8a" if li == 0 else "#3b5b8c"
    ty = root_top + 24 + li * 20
    L.append(f'<text x="{root_x}" y="{ty}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="{fs}" font-weight="{fw}" fill="{fill}">{esc(txt)}</text>')

# root → 汇流线(竖线) → 汇流横线 → 各 child 竖线(带箭头)
L.append(f'<line x1="{root_x}" y1="{root_bottom}" x2="{root_x}" y2="{bus_y}" '
          'stroke="#334155" stroke-width="1.6"/>')
xs_all = [child_x[i] for i in range(N)]
L.append(f'<line x1="{min(xs_all)}" y1="{bus_y}" x2="{max(xs_all)}" y2="{bus_y}" '
          'stroke="#334155" stroke-width="1.6"/>')
for i in range(N):
    x = child_x[i]
    L.append(f'<line x1="{x}" y1="{bus_y}" x2="{x}" y2="{child_top}" '
              'stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')

# child 框(等高 CHILD_H,按最长 detail 行数统一算出,避免逐框手调)
for i, c in enumerate(CHILDREN):
    x = child_x[i]
    cx0 = x - CHILD_W / 2
    is_special = c["badge"] is not None
    fill = "#fef3c7" if is_special else "#e2e8f0"
    stroke = "#d97706" if is_special else "#64748b"
    L.append(f'<rect x="{cx0}" y="{child_top}" width="{CHILD_W}" height="{CHILD_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    ty = child_top + 22
    L.append(f'<text x="{x}" y="{ty}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="#0f172a">{esc(c["title"])}</text>')
    for li, line in enumerate(c["detail"]):
        dty = ty + TITLE_TO_DETAIL_GAP + li * DETAIL_LINE_H
        L.append(f'<text x="{x}" y="{dty}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="10.5" fill="#334155">{esc(line)}</text>')
    if c["badge"]:
        by = child_bottom + 20
        L.append(f'<rect x="{x-38}" y="{by-16}" width="76" height="{badge_h}" rx="11" '
                  'fill="#fde68a" stroke="#d97706" stroke-width="1"/>')
        L.append(f'<text x="{x}" y="{by}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="11" font-weight="bold" fill="#78350f">{esc(c["badge"])}</text>')

# 底部脚注:核心数字汇总
foot_y = h - 20
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#475569">'
          f'{esc("1 个入口宏 → 4 个核心子模块(ir/passes/interpreter/llvm) + 至多 4 个后端子模块 —— 一次 import 装配完成")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m3-module-tree.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
