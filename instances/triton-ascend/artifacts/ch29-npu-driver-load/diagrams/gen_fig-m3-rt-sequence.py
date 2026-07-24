#!/usr/bin/env python3
"""fig-m3-rt-sequence：registerKernel 装一个 kernel 恰好按序调三个 CANN rt* 接口
(rtSetDevice→rtDevBinaryRegister→rtFunctionRegister)，每步失败即 printf 错误码并
返回 {nullptr,nullptr}，全过关才产出两个句柄。主干纵向下行，每个 rt* 调用步右侧
挂一个「失败早退」红色侧支(不汇回主干,终止于 return)。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "registerKernel：3 个 rt* 调用逐级注册，任一失败即早退双 nullptr"
SUBTITLE = "third_party/ascend/backend/npu_utils.cpp:L40-L82 —— 设备→二进制→函数，逐级注册"

PAD = 40
TRUNK_W = 560
SIDE_W = 340
SIDE_GAP = 60
TOP = 100
GAP = 26

trunk_cx = PAD + TRUNK_W / 2
side_cx = PAD + TRUNK_W + SIDE_GAP + SIDE_W / 2
w = PAD + TRUNK_W + SIDE_GAP + SIDE_W + PAD

elems = []
def add(s): elems.append(s)

def trunk_box(y, lines, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e", bold_first=True):
    n = len(lines)
    box_h = 34 + 20 * (n - 1) + 34
    bx = trunk_cx - TRUNK_W / 2
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{TRUNK_W}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    y0 = y + box_h / 2 - (n - 1) * 10 + 5
    for k, line in enumerate(lines):
        fw = 'font-weight="bold" ' if (bold_first and k == 0) else ''
        fs = 13 if k == 0 else 11.5
        fc = text_fill if k == 0 else "#334155"
        add(f'<text x="{trunk_cx:.0f}" y="{y0+k*20:.0f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="{fs}" {fw}fill="{fc}">{esc(line)}</text>')
    return box_h

def side_box(y, lines):
    n = len(lines)
    box_h = 20 * n + 20
    bx = side_cx - SIDE_W / 2
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{SIDE_W}" height="{box_h}" rx="8" '
        'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.5"/>')
    y0 = y + box_h / 2 - (n - 1) * 10 + 4
    for k, line in enumerate(lines):
        add(f'<text x="{side_cx:.0f}" y="{y0+k*20:.0f}" text-anchor="middle" '
            f'font-family="monospace" font-size="11.5" fill="#991b1b">{esc(line)}</text>')
    return box_h

def arrow(x, y1, y2, color="#334155", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2"{d} marker-end="url(#a)"/>')

def side_link(trunk_y_mid, box_top):
    # 从主干节点右边缘引一条细红线到侧支框顶部
    x1 = trunk_cx + TRUNK_W / 2
    x2 = side_cx - SIDE_W / 2
    add(f'<line x1="{x1:.0f}" y1="{trunk_y_mid:.0f}" x2="{x2-14:.0f}" y2="{trunk_y_mid:.0f}" '
        'stroke="#b91c1c" stroke-width="1.5" stroke-dasharray="4,3"/>')
    add(f'<line x1="{x2-14:.0f}" y1="{trunk_y_mid:.0f}" x2="{x2-14:.0f}" y2="{box_top:.0f}" '
        'stroke="#b91c1c" stroke-width="1.5" stroke-dasharray="4,3"/>')
    add(f'<line x1="{x2-14:.0f}" y1="{box_top:.0f}" x2="{x2:.0f}" y2="{box_top:.0f}" '
        'stroke="#b91c1c" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#c)"/>')

y = TOP
bh = trunk_box(y, [
    "填 rtDevBinary_t：data=blob, length=data_size",
    "magic = (kernel_mode==\"aiv\") ? RT_DEV_BINARY_MAGIC_ELF_AIVEC",
    "                              : RT_DEV_BINARY_MAGIC_ELF",
    "version = 0  (npu_utils.cpp:L36-L53)",
])
y += bh + GAP

steps = [
    (["rtSetDevice(device)", "(npu_utils.cpp:L55)"],
     ["失败 → printf(\"rtSetDevice failed, 0x%x\")", "→ return {nullptr, nullptr}", "(L57-L58)"]),
    (["rtDevBinaryRegister(&devbin, &devbinHandle)", "拿 devbinHandle (npu_utils.cpp:L62)"],
     ["失败 → printf(\"rtDevBinaryRegister failed, 0x%x\")", "→ return {nullptr, nullptr}", "(L64-L65)"]),
    (["stubName = name + \"_\" + registered_names[name]；", "registered_names[name]++ (L68-L70)"],
     None),
    (["func_stubs.emplace(stubName, make_unique<size_t>(0))",
      "func_stub_handle = registered.first->second.get() (L71-L72)"],
     None),
    (["rtFunctionRegister(devbinHandle, func_stub_handle,",
      "stubName, name, funcMode=0) (npu_utils.cpp:L73-L74)"],
     ["失败 → printf(\"rtFunctionRegister failed, ...\")", "→ return {nullptr, nullptr}", "(L76-L78)"]),
]

for i, (trunk_lines, err_lines) in enumerate(steps):
    arrow(trunk_cx, y - GAP, y)
    bh = trunk_box(y, trunk_lines)
    if err_lines is not None:
        trunk_mid = y + bh / 2
        err_h = side_box(y, err_lines)
        side_link(trunk_mid, y)
    y += bh + GAP

arrow(trunk_cx, y - GAP, y)
bh = trunk_box(y, [
    "return (devbinHandle, func_stub_handle)",
    "—— 全过关才产出 2 个句柄 (npu_utils.cpp:L81)",
], fill="#dcfce7", stroke="#15803d", text_fill="#14532d")
y += bh

tail_bottom = y

def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)

note_lines = [
    "aiv/aic 是二分不是三分：kernel_mode==\"aiv\" 才走 AIVEC 魔数，其余(aic/mix)统一走",
    "ELF 魔数——不是 aiv/aic 各自专属一个魔数的对称三分。3 次 rt* 调用步步把关，",
    "任一非 RT_ERROR_NONE 立即短路吐出双 nullptr，成功路径唯一产出 2 个稳定句柄。",
]
note_top = tail_bottom + 30
note_h = 22 * len(note_lines) + 22
w = max(w, PAD * 2 + max(cjk_w(s, 12) for s in note_lines) + 32)
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*22:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="c" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
     '</defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-m3-rt-sequence.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
