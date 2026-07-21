#!/usr/bin/env python3
"""state-table 模板：handle_scope_with 两趟 visit 的状态演化——
列=4 个阶段（试跑/建 scope op/正式重跑/封口回填），行=插入点/builder 动作/SSA 产物/结果。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "handle_scope_with 两趟 visit：先试跑数变量，再正式建 IR"
SUBTITLE = "示例：with scope(core_mode=\"vector\", disable_auto_sync=True, my_hint=3): a=a+1; c=a*2（scope_defs={a,c}，k=2）"
COLS = ["① 试跑（第 1 趟）", "② 建 scope.scope",
        "③ 正式重跑（第 2 趟）", "④ 封口回填"]
ROW_LABELS = ["插入点", "builder 动作", "SSA 产物", "local_defs / 结果"]
CELLS = {
    "插入点": [
        "func.entry → block1（dummy）",
        "回到 func.entry → region#0",
        "block2:start",
        "block2:end → func.entry",
    ],
    "builder 动作": [
        "create_block()\nvisit_compound_statement ×1\ndummy.erase()",
        "create_scope_op(attrs, 2 个结果类型)\ncreate_block_with_parent(region#0, [])",
        "lscope ← liveins 副本\nvisit_compound_statement ×2",
        "scope_return([%3, %4])",
    ],
    "SSA 产物": [
        "%1=a, %2=c",
        "—",
        "%3=a, %4=c",
        "2 个操作数",
    ],
    "local_defs / 结果": [
        "names=[a,c] 冻结\n（随 block 一起作废）",
        "attrs={noinline, tcore_type<VECTOR>,\nhivm.disable_auto_sync, my_hint}\n区入口块参数=0",
        "local_defs 含残留\n{a:%1,c:%2}（不影响 names）",
        "a→%scope_res0, c→%scope_res1\nn 仍是 %n_outer",
    ],
}
HIGHLIGHT_ROW = "SSA 产物"
STATUS = {"SSA 产物": ["discarded", None, "kept", None]}
COLOR = {"discarded": ("#fee2e2", "#b91c1c"), "kept": ("#ecfdf5", "#047857")}

LABEL_W, COL_W, PAD, TOP, HEADER_H = 150, 315, 34, 108, 40
ROW_H = [46, 78, 46, 78]
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + sum(ROW_H) + 56
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = []
acc = TOP + HEADER_H
for rh in ROW_H:
    row_y.append(acc)
    acc += rh

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="11.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

for j, name in enumerate(COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    rh = ROW_H[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+rh/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    statuses = STATUS.get(row)
    for j in range(len(COLS)):
        cx = col_x[j]
        lines = CELLS[row][j].split("\n")
        status = statuses[j] if statuses else None
        if status:
            fill, stroke = COLOR[status]
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{rh-8}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        else:
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{rh-8}" rx="4" '
                      f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
        text_fill = COLOR[status][1] if status else "#374151"
        weight_attr = 'font-weight="bold" ' if status else ''
        n = len(lines)
        y0 = ry + rh / 2 - (n - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" fill="{text_fill}" '
                      f'{weight_attr}>{esc(line)}</text>')

foot_y = h - 24
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">'
          f'{esc("红=第 1 趟随 dummy block 一起作废（%1,%2）；绿=第 2 趟真正进 region、经 scope.return 回填外层（%3,%4）——两趟共造 4 个 SSA 值，2 个被丢。")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m3-two-pass-ssa.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
