#!/usr/bin/env python3
"""fig-ch26-hacc-target-patch：hacc.target 不是编译 stage，而是靠幂等 monkey-patch
包住 ASTSource.make_ir——module 生成后立刻 set_attr 目标型号。两条独立判定：
①patch 守卫(只包一次) ②编译时按 options.arch 是否为空决定是否真正注入。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "hacc.target 注入：幂等 monkey-patch 包住 ASTSource.make_ir"
SUBTITLE = "third_party/ascend/backend/__init__.py:L27-52 —— 不是 add_stages 里的一段 stage"

PAD = 40
TOP = 92
GAP = 26
BOX_W = 680

elems = []


def add(s):
    elems.append(s)


def box(cx, y, lines, w=BOX_W, fill="#e0f2fe", stroke="#0369a1", text_fill="#0c4a6e",
        bold=False, fs=13, mono=False):
    n = len(lines)
    box_h = 30 + 20 * (n - 1) + 34
    bx = cx - w / 2
    add(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{w:.0f}" height="{box_h:.0f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    y0 = y + box_h / 2 - (n - 1) * 10 + 5
    fw = 'font-weight="bold" ' if bold else ''
    ff = 'monospace' if mono else 'sans-serif'
    for k, line in enumerate(lines):
        add(f'<text x="{cx:.0f}" y="{y0+k*20:.0f}" text-anchor="middle" '
            f'font-family="{ff}" font-size="{fs}" {fw}fill="{text_fill}">{esc(line)}</text>')
    return box_h


def varrow(x, y1, y2, color="#334155"):
    add(f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2:.0f}" '
        f'stroke="{color}" stroke-width="2" marker-end="url(#a)"/>')


x_center = PAD + BOX_W / 2
w = PAD * 2 + BOX_W

# --- 第一部分：patch 守卫（跨调用只包一次） ---
y = TOP
add(f'<text x="{PAD}" y="{y-10:.0f}" font-family="sans-serif" font-size="13" '
    f'font-weight="bold" fill="#334155">① 每次 get_codegen_implementation() 都会走到的守卫</text>')
bh = box(x_center, y, ["get_codegen_implementation() → _apply_ascend_patch()"],
         fill="#dbeafe", stroke="#1d4ed8", text_fill="#1e3a5f", bold=True, mono=True, fs=12.5)
y += bh

varrow(x_center, y, y + GAP)
y += GAP
judge_h = 60
add(f'<rect x="{x_center-BOX_W/2:.0f}" y="{y:.0f}" width="{BOX_W}" height="{judge_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_center:.0f}" y="{y+judge_h/2+5:.0f}" text-anchor="middle" font-family="monospace" '
    f'font-size="12.5" font-weight="bold" fill="#78350f">ASTSource._ascend_patch_applied ？</text>')
judge_bottom = y + judge_h

fork_y = judge_bottom + 22
BRANCH_W = 320
gap2 = 40
x_l = x_center - BRANCH_W / 2 - gap2 / 2
x_r = x_center + BRANCH_W / 2 + gap2 / 2
add(f'<line x1="{x_center:.0f}" y1="{judge_bottom:.0f}" x2="{x_center:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{x_l:.0f}" y1="{fork_y:.0f}" x2="{x_r:.0f}" y2="{fork_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
varrow(x_l, fork_y, fork_y + GAP)
varrow(x_r, fork_y, fork_y + GAP)
add(f'<text x="{x_l:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" font-weight="bold" fill="#334155">False(首次调用)</text>')
add(f'<text x="{x_r:.0f}" y="{fork_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" font-weight="bold" fill="#334155">True(第 2+ 次调用)</text>')

y2 = fork_y + GAP
lh = box(x_l, y2, ["替换 ASTSource.make_ir", "= _patched_make_ir",
                    "置 _ascend_patch_applied = True"],
         w=BRANCH_W, fill="#dcfce7", stroke="#15803d", text_fill="#14532d", fs=11.5)
rh = box(x_r, y2, ["no-op —— 函数体整体跳过", "不再二次包裹"],
         w=BRANCH_W, fill="#f1f5f9", stroke="#64748b", text_fill="#334155", fs=11.5)
part1_bottom = y2 + max(lh, rh)

tag_y = part1_bottom + 14
add(f'<rect x="{x_center-BOX_W/2:.0f}" y="{tag_y:.0f}" width="{BOX_W}" height="34" rx="8" '
    'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1"/>')
add(f'<text x="{x_center:.0f}" y="{tag_y+22:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#1e3a5f">N 次调用 ⇒ 恰 1 次实际 patch（幂等，绝不嵌套）</text>')
part1_end = tag_y + 34

# --- 第二部分：编译时 _patched_make_ir 的注入判定 ---
y3 = part1_end + 56
add(f'<text x="{PAD}" y="{y3-10:.0f}" font-family="sans-serif" font-size="13" '
    f'font-weight="bold" fill="#334155">② 每次编译经过 _patched_make_ir 时的独立判定</text>')
bh3 = box(x_center, y3, ["module = 原 make_ir(...)  —— 先生成 Triton IR module"],
          fill="#ede9fe", stroke="#6d28d9", text_fill="#3730a3", bold=True, mono=True, fs=12.5)
y3 += bh3

varrow(x_center, y3, y3 + GAP)
y3 += GAP
judge2_h = 56
add(f'<rect x="{x_center-BOX_W/2:.0f}" y="{y3:.0f}" width="{BOX_W}" height="{judge2_h}" rx="14" '
    'fill="#fef3c7" stroke="#b45309" stroke-width="2"/>')
add(f'<text x="{x_center:.0f}" y="{y3+judge2_h/2+5:.0f}" text-anchor="middle" font-family="monospace" '
    f'font-size="12.5" font-weight="bold" fill="#78350f">hasattr(options, "arch") and options.arch ？</text>')
judge2_bottom = y3 + judge2_h

fork2_y = judge2_bottom + 22
add(f'<line x1="{x_center:.0f}" y1="{judge2_bottom:.0f}" x2="{x_center:.0f}" y2="{fork2_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
add(f'<line x1="{x_l:.0f}" y1="{fork2_y:.0f}" x2="{x_r:.0f}" y2="{fork2_y:.0f}" '
    'stroke="#334155" stroke-width="2"/>')
varrow(x_l, fork2_y, fork2_y + GAP)
varrow(x_r, fork2_y, fork2_y + GAP)
add(f'<text x="{x_l:.0f}" y="{fork2_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" font-weight="bold" fill="#334155">True(arch 非空，如 Ascend910B)</text>')
add(f'<text x="{x_r:.0f}" y="{fork2_y-8:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" font-weight="bold" fill="#334155">False(arch 为空)</text>')

y4 = fork2_y + GAP
lh2 = box(x_l, y4, ["builder = ascendnpu_ir_builder(...)",
                     "module.set_attr(\"hacc.target\",",
                     "  builder.parse_attr('#hacc.target<\"arch\">'))"],
          w=BRANCH_W, fill="#dcfce7", stroke="#15803d", text_fill="#14532d", fs=10.5, mono=True)
rh2 = box(x_r, y4, ["跳过注入", "（module 上无 hacc.target 属性）"],
          w=BRANCH_W, fill="#fee2e2", stroke="#b91c1c", text_fill="#7f1d1d", fs=11.5)
part2_bottom = y4 + max(lh2, rh2)

tag2_y = part2_bottom + 14
add(f'<rect x="{x_center-BOX_W/2:.0f}" y="{tag2_y:.0f}" width="{BOX_W}" height="34" rx="8" '
    'fill="#ede9fe" stroke="#6d28d9" stroke-width="1"/>')
tag2_text = '注入的属性：#hacc.target<"Ascend910B">  ｜  arch 空时注入次数 = 0'
add(f'<text x="{x_center:.0f}" y="{tag2_y+22:.0f}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12.5" font-weight="bold" fill="#3730a3">{esc(tag2_text)}</text>')
content_bottom = tag2_y + 34

# --- 底部注解 ---
def cjk_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


note_lines = [
    "两条判定相互独立：①决定 make_ir 是否被替换成 _patched_make_ir(跨调用只发生一次)；",
    "②决定被替换后的 make_ir 每次执行时是否真的贴上 hacc.target(取决于当次 options.arch)。",
    "注入失败仅降级为 logging.warning，不阻断编译(__init__.py:L46-47)。",
]
note_top = content_bottom + 34
note_w_needed = max(cjk_w(s, 12) for s in note_lines) + 32
w2 = max(w, note_w_needed + 2 * PAD)
note_h = 22 * len(note_lines) + 22
add(f'<rect x="{PAD}" y="{note_top:.0f}" width="{w2-2*PAD:.0f}" height="{note_h}" rx="8" '
    'fill="#eff6ff" stroke="#93c5fd"/>')
for i, line in enumerate(note_lines):
    add(f'<text x="{PAD+16}" y="{note_top+22+i*22:.0f}" font-family="sans-serif" '
        f'font-size="12" fill="#1e3a5f">{esc(line)}</text>')

h = note_top + note_h + PAD
w = w2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>',
     f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>'] + elems + ['</svg>']

out = Path(__file__).with_name("fig-ch26-hacc-target-patch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w:.0f} h={h:.0f}")
