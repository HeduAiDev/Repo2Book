#!/usr/bin/env python3
"""flow 模板：visit_With 的分派路径——scope 类对象当字典键，命中/未命中两分支。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W, PAD = 980, 40
BOX_W, BOX_H = 620, 50
CX = PAD + BOX_W / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 645">',
     '<defs>'
     '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
     '<marker id="ao" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#c2410c"/></marker>'
     '</defs>',
     f'<rect width="{W}" height="645" fill="white"/>',
     f'<text x="{PAD}" y="32" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("visit_With 分派：scope 类对象本身就是字典键")}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12.5" fill="#64748b">'
     f'{esc("with scope(...) 不走 Python 上下文管理器协议——__enter__/__exit__ 不参与")}</text>']

def box(y, h, text_lines, fill, stroke, bw=BOX_W, x=None, tsize=13.5, bold=True):
    bx = PAD if x is None else x
    L.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="{h}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(text_lines)
    y0 = y + h / 2 - (n - 1) * 9 + 5
    for k, (line, small) in enumerate(text_lines):
        fs = 11.5 if small else tsize
        fw = 'font-weight="bold" ' if (bold and not small) else ''
        L.append(f'<text x="{bx+bw/2}" y="{y0+k*17}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" {fw}'
                  f'fill="#0f172a">{esc(line)}</text>')
    return bx, y, bw, h

def varrow(x, y1, y2, marker="a", dash=False):
    d = ' stroke-dasharray="5,4"' if dash else ''
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#334155" '
              f'stroke-width="1.6"{d} marker-end="url(#{marker})"/>')

def sidelabel(x, y, text, color="#475569", anchor="start"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" '
              f'font-size="11.5" fill="{color}">{esc(text)}</text>')

y = 78
# 1. entry
b1 = box(y, 44, [("visit_With(node)", False), ("assert len(node.items) == 1", True)],
         "#e2e8f0", "#64748b")
y2 = y + 44
varrow(CX, y2, y2 + 26)
sidelabel(CX + 18, y2 + 18, "python/triton/compiler/code_generator.py:L803")

# 2. isinstance check (diamond-ish rect)
y = y2 + 26
b2 = box(y, 46, [("context = node.items[0].context_expr", True),
                 ("isinstance(context, ast.Call) ?", False)], "#eef2ff", "#6366f1")
y2 = y + 46
varrow(CX, y2, y2 + 26)
sidelabel(CX + 18, y2 + 18, "是 → 继续；否 → 无 with 语义（直接跳最终兜底）")

# 3. visit(context.func) -> class object
y = y2 + 26
b3 = box(y, 46, [("withitemClass = self.visit(context.func)", False),
                 ("查表键 = scope 类对象本身（不是字符串）", True)], "#eef2ff", "#6366f1")
y2 = y + 46
varrow(CX, y2, y2 + 26)
sidelabel(CX + 18, y2 + 18, "third_party/ascend/language/cann/extension/dispatch.py:L32")

# 4. dict lookup box, showing WITH_DISPATCH contents
y = y2 + 26
dict_h = 90
bx = PAD + BOX_W / 2 - 300
L.append(f'<rect x="{bx}" y="{y}" width="600" height="{dict_h}" rx="9" '
          'fill="#fefce8" stroke="#ca8a04" stroke-width="1.6"/>')
L.append(f'<text x="{bx+300}" y="{y+20}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#713f12">'
          f'{esc("WITH_DISPATCH.get(withitemClass)")}</text>')
L.append(f'<text x="{bx+300}" y="{y+38}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#854d0e">'
          f'{esc("基座初值 {}（code_generator.py:L26）+ ASCEND_WITH_DISPATCH 注入 2 项")}</text>')
dict_repr = '{ scope 类: handle_scope_with,  "mangle_ty": mangle_ty }'
L.append(f'<text x="{bx+300}" y="{y+58}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12" fill="#713f12">'
          f'{esc(dict_repr)}</text>')
L.append(f'<text x="{bx+300}" y="{y+76}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#a16207">'
          f'{esc("dispatch.py:L31-L34——scope 类本身当 key")}</text>')
y2 = y + dict_h

# branch split
by = y2 + 40
L.append(f'<line x1="{CX}" y1="{y2}" x2="{CX}" y2="{y2+18}" stroke="#334155" stroke-width="1.6"/>')
hit_x = PAD + 40
miss_x = PAD + BOX_W - 300 + 40
L.append(f'<line x1="{hit_x+150}" y1="{y2+18}" x2="{hit_x+150}" y2="{by}" '
          'stroke="#15803d" stroke-width="1.8" marker-end="url(#ag)"/>')
L.append(f'<line x1="{miss_x+150}" y1="{y2+18}" x2="{miss_x+150}" y2="{by}" '
          'stroke="#c2410c" stroke-width="1.8" marker-end="url(#ao)"/>')
L.append(f'<line x1="{hit_x+150}" y1="{y2+18}" x2="{miss_x+150}" y2="{y2+18}" '
          'stroke="#334155" stroke-width="1.6"/>')
L.append(f'<text x="{hit_x+150}" y="{y2+13}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#15803d">{esc("命中")}</text>')
L.append(f'<text x="{miss_x+150}" y="{y2+13}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" font-weight="bold" fill="#c2410c">{esc("未命中")}</text>')

# hit branch box
hb_h = 78
L.append(f'<rect x="{hit_x}" y="{by}" width="300" height="{hb_h}" rx="9" '
          'fill="#f0fdf4" stroke="#15803d" stroke-width="1.8"/>')
L.append(f'<text x="{hit_x+150}" y="{by+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#14532d">'
          f'{esc("handler(self, node)")}</text>')
L.append(f'<text x="{hit_x+150}" y="{by+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#166534">'
          f'{esc("= handle_scope_with(self, node)")}</text>')
L.append(f'<text x="{hit_x+150}" y="{by+58}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#166534">'
          f'{esc("传的是整条 with 的 AST 节点")}</text>')
L.append(f'<text x="{hit_x+150}" y="{by+74}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#4d7c0f">'
          f'{esc("code_generator.py:L809-L811")}</text>')

# miss branch box
L.append(f'<rect x="{miss_x}" y="{by}" width="300" height="{hb_h}" rx="9" '
          'fill="#fff7ed" stroke="#c2410c" stroke-width="1.8"/>')
L.append(f'<text x="{miss_x+150}" y="{by+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#7c2d12">'
          f'{esc("visit_compound_statement(node.body)")}</text>')
L.append(f'<text x="{miss_x+150}" y="{by+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#9a3412">'
          f'{esc("等价于把 with 当透明壳")}</text>')
L.append(f'<text x="{miss_x+150}" y="{by+58}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#9a3412">'
          f'{esc("__enter__/__exit__ 全程不参与、无 IR")}</text>')
L.append(f'<text x="{miss_x+150}" y="{by+74}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#c2410c">'
          f'{esc("code_generator.py:L813-L814")}</text>')

foot_y = by + hb_h + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("ch04 讲过的“按 is_builtin 路由”同一思路的第二个入口：这次分派发生在 with 语句上。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m1-with-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
