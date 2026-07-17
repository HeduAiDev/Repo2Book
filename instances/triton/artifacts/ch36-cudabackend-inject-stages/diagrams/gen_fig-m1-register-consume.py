#!/usr/bin/env python3
"""fig-m1-register-consume: flow 模板。
add_stages 把五段编译函数按序登记进 stages 字典(上排) ->
compile() 的 for 循环按插入序逐段消费、每段落缓存(下排)。
"注册->消费"闭环。全部坐标由循环/常量计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


# 五段:(键名, 驱动函数, 产出类型, 是否末段)
STAGES = [
    ("ttir", "make_ttir", "TTIR (str)", False),
    ("ttgir", "make_ttgir", "TTGIR (str)", False),
    ("llir", "make_llir", "LLIR (str)", False),
    ("ptx", "make_ptx", "PTX (str)", False),
    ("cubin", "make_cubin", "cubin (bytes)", True),
]

BOX_W, BOX_H = 128, 58
GAP = 46
PAD = 46

TITLE_Y = 26
SUB_Y = 44
LABEL1_Y = 68
LABEL2_Y = 88
ROW1_Y = 142
CALLOUT_TOP = 192
CALLOUT_H = 34
ROW2_Y = 292
CACHE_LINE_TOP = ROW2_Y + BOX_H / 2
CACHE_Y = CACHE_LINE_TOP + 66

n = len(STAGES)
w = PAD * 2 + n * BOX_W + (n - 1) * GAP
h = CACHE_Y + 44

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
    '<marker id="ah" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
    '<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#c2410c"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(
    f'<text x="{w/2}" y="{TITLE_Y}" text-anchor="middle" font-family="sans-serif" font-size="17" '
    f'font-weight="bold" fill="#0f172a">{esc("注册→消费闭环:add_stages 登记 stages 字典,compile() 按插入序逐段消费")}</text>'
)
L.append(
    f'<text x="{w/2}" y="{SUB_Y}" text-anchor="middle" font-family="sans-serif" font-size="12" '
    f'fill="#475569">{esc("third_party/nvidia/backend/compiler.py:L384-L389 → python/triton/compiler/compiler.py:L261-L292")}</text>'
)

centers = [PAD + BOX_W / 2 + i * (BOX_W + GAP) for i in range(n)]

# --- 行标签:两条都放在 row1 之上(标题区正下方),避免任何竖直连线穿字 ---
L.append(
    f'<text x="{PAD}" y="{LABEL1_Y}" font-family="sans-serif" font-size="13" '
    f'font-weight="bold" fill="#1d4ed8">{esc("① 登记 add_stages(stages, options):5 个键按序 insert")}</text>'
)
L.append(
    f'<text x="{PAD}" y="{LABEL2_Y}" font-family="sans-serif" font-size="13" '
    f'font-weight="bold" fill="#1d4ed8">{esc("② 消费 compile():list(stages.items())[first_stage:] 顺序遍历")}</text>'
)

# --- 第 1 行:登记(蓝色主线,水平箭头表插入顺序) ---
for i in range(n - 1):
    x1 = centers[i] + BOX_W / 2
    x2 = centers[i + 1] - BOX_W / 2
    L.append(f'<line x1="{x1}" y1="{ROW1_Y}" x2="{x2}" y2="{ROW1_Y}" stroke="#3b82f6" '
              f'stroke-width="2" marker-end="url(#ah)"/>')

for i, (key, fn, out, is_last) in enumerate(STAGES):
    cx = centers[i]
    x = cx - BOX_W / 2
    y = ROW1_Y - BOX_H / 2
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              f'fill="#dbeafe" stroke="#2563eb" stroke-width="1.8"/>')
    # 序号徽标
    L.append(f'<circle cx="{x+15}" cy="{y+15}" r="11" fill="#2563eb"/>')
    L.append(f'<text x="{x+15}" y="{y+19}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="white">{i+1}</text>')
    key_label = '"' + key + '"'
    L.append(f'<text x="{cx}" y="{ROW1_Y-6}" text-anchor="middle" font-family="monospace" '
              f'font-size="14" font-weight="bold" fill="#0f172a">{esc(key_label)}</text>')
    L.append(f'<text x="{cx}" y="{ROW1_Y+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" fill="#475569">{esc(f"→ {fn}(...)")}</text>')

# --- 竖直连接:登记 -> 消费(中间三段直连;首末两段经 callout 断成两截) ---
row1_bottom = ROW1_Y + BOX_H / 2
row2_top = ROW2_Y - BOX_H / 2
for i in range(1, n - 1):
    cx = centers[i]
    L.append(f'<line x1="{cx}" y1="{row1_bottom}" x2="{cx}" y2="{row2_top}" stroke="#94a3b8" '
              f'stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# 首段 callout: first_stage=0
fs_cx = centers[0]
fs_w = 158
L.append(f'<line x1="{fs_cx}" y1="{row1_bottom}" x2="{fs_cx}" y2="{CALLOUT_TOP}" '
          f'stroke="#2563eb" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#a)"/>')
L.append(f'<rect x="{fs_cx-fs_w/2}" y="{CALLOUT_TOP}" width="{fs_w}" height="{CALLOUT_H}" rx="7" '
          f'fill="#eff6ff" stroke="#2563eb" stroke-width="1.4"/>')
L.append(f'<text x="{fs_cx}" y="{CALLOUT_TOP+15}" text-anchor="middle" font-family="monospace" '
          f'font-size="11" font-weight="bold" fill="#1d4ed8">{esc("first_stage = 0")}</text>')
L.append(f'<text x="{fs_cx}" y="{CALLOUT_TOP+29}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="9.5" fill="#1d4ed8">{esc("(AST 源,ext=ttir)")}</text>')
L.append(f'<line x1="{fs_cx}" y1="{CALLOUT_TOP+CALLOUT_H}" x2="{fs_cx}" y2="{row2_top}" '
          f'stroke="#2563eb" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#a)"/>')

# 末段 callout: 返回 bytes
b_cx = centers[-1]
b_w = 140
L.append(f'<line x1="{b_cx}" y1="{row1_bottom}" x2="{b_cx}" y2="{CALLOUT_TOP}" '
          f'stroke="#c2410c" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#a)"/>')
L.append(f'<rect x="{b_cx-b_w/2}" y="{CALLOUT_TOP}" width="{b_w}" height="{CALLOUT_H}" rx="7" '
          f'fill="#fff7ed" stroke="#c2410c" stroke-width="1.4"/>')
L.append(f'<text x="{b_cx}" y="{CALLOUT_TOP+15}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" font-weight="bold" fill="#c2410c">{esc("末段:返回")}</text>')
L.append(f'<text x="{b_cx}" y="{CALLOUT_TOP+29}" text-anchor="middle" font-family="monospace" '
          f'font-size="11" font-weight="bold" fill="#c2410c">{esc("bytes")}</text>')
L.append(f'<line x1="{b_cx}" y1="{CALLOUT_TOP+CALLOUT_H}" x2="{b_cx}" y2="{row2_top}" '
          f'stroke="#c2410c" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#ao)"/>')

# --- 第 2 行:消费(蓝色水平箭头表 for 循环顺序,末段橙色高亮) ---
for i in range(n - 1):
    x1 = centers[i] + BOX_W / 2
    x2 = centers[i + 1] - BOX_W / 2
    hot_last = (i == n - 2)
    color = "#c2410c" if hot_last else "#3b82f6"
    marker = "ao" if hot_last else "ah"
    L.append(f'<line x1="{x1}" y1="{ROW2_Y}" x2="{x2}" y2="{ROW2_Y}" stroke="{color}" '
              f'stroke-width="2" marker-end="url(#{marker})"/>')

for i, (key, fn, out, is_last) in enumerate(STAGES):
    cx = centers[i]
    x = cx - BOX_W / 2
    y = ROW2_Y - BOX_H / 2
    fill = "#ffedd5" if is_last else "#e0f2fe"
    stroke = "#c2410c" if is_last else "#0284c7"
    L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2.4 if is_last else 1.8}"/>')
    L.append(f'<text x="{cx}" y="{ROW2_Y-6}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(fn)}</text>')
    color2 = "#c2410c" if is_last else "#0369a1"
    L.append(f'<text x="{cx}" y="{ROW2_Y+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10.5" font-weight="{"bold" if is_last else "normal"}" '
              f'fill="{color2}">{esc(out)}</text>')

# --- 落缓存条:每段消费后写入缓存 ---
band_x0 = centers[0] - BOX_W / 2
band_x1 = centers[-1] + BOX_W / 2
for i in range(n):
    cx = centers[i]
    L.append(f'<line x1="{cx}" y1="{CACHE_LINE_TOP}" x2="{cx}" y2="{CACHE_Y}" '
              f'stroke="#16a34a" stroke-width="1.4" stroke-dasharray="3,3" marker-end="url(#ag)"/>')
L.append(f'<rect x="{band_x0}" y="{CACHE_Y}" width="{band_x1-band_x0}" height="30" rx="8" '
          f'fill="#ecfdf5" stroke="#22c55e" stroke-width="1.6"/>')
L.append(f'<text x="{(band_x0+band_x1)/2}" y="{CACHE_Y+20}" text-anchor="middle" '
          f'font-family="monospace" font-size="11.5" fill="#15803d">'
          f'{esc("metadata_group[ir_filename] = fn_cache_manager.put(next_module, ir_filename)  ×5")}</text>')

L.append('</svg>')
out_path = Path(__file__).with_name("fig-m1-register-consume.svg")
out_path.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out_path}")
