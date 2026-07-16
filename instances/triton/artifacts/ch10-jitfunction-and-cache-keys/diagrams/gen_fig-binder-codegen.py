#!/usr/bin/env python3
"""flow 模板:create_function_from_signature 把固定签名一次性 exec 成 dynamic_func(binder),
此后每次发射只调它一次即得缓存键原料五元组。三段横排:签名 → exec(仅首次)→ 返回五元组,
高亮第 2 槽 sig_and_spec 并引出到下一图 fig-launch-cache-key。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "create_function_from_signature —— 把签名一次性 exec 成专属 binder"
SUBTITLE = "add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)；此函数仅在首次发射（self.binder is None）触发一次"

# --- 左：signature 盒 ---
SIG_PARAMS = [
    ("x_ptr", False),
    ("y_ptr", False),
    ("output_ptr", False),
    ("n_elements", False),
    ("BLOCK_SIZE", True),  # constexpr
]

# --- 右：五元组槽位 ---
SLOTS = [
    ("1", "bound_args", "参数名 → 实参 dict", False),
    ("2", "sig_and_spec", "4 dtype 签名 + 4 D/1/N 特化", True),
    ("3", "constexpr_vals", "(BLOCK_SIZE,)", False),
    ("4", "non_constexpr_vals", "(x_ptr, y_ptr, output_ptr, n_elements)", False),
    ("5", "excess_kwargs", "{}", False),
]

NUMS = [
    ("参数数", "5"),
    ("dtype 签名项", "4"),
    ("D/1/N 特化项", "4"),
    ("返回元组槽位", "5"),
]

# ---------- 版式常量 ----------
PAD = 40
TOP = 108
STAGE_W = 300
STAGE_GAP = 70
STAGE_H = 460
W = PAD * 2 + STAGE_W * 3 + STAGE_GAP * 2
H = TOP + STAGE_H + 96

STAGE_X = [PAD + i * (STAGE_W + STAGE_GAP) for i in range(3)]
STAGE_LABELS = ["① 装饰期已定：签名", "② 首次发射：exec 生成", "③ 此后每次发射：五元组产出"]
STAGE_COLORS = ["#eef2ff", "#fef3c7", "#ecfdf5"]
STAGE_BORDERS = ["#6366f1", "#d97706", "#059669"]

L = []
L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">')
L.append(
    '<defs><marker id="arrow" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
    '<marker id="arrow-hi" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker></defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(
    f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="18" '
    f'font-weight="bold" fill="#1e293b">{esc(TITLE)}</text>'
)
L.append(
    f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="12.5" '
    f'fill="#64748b">{esc(SUBTITLE)}</text>'
)

# 阶段外框 + 标题
for i in range(3):
    x = STAGE_X[i]
    L.append(
        f'<rect x="{x}" y="{TOP}" width="{STAGE_W}" height="{STAGE_H}" rx="10" '
        f'fill="{STAGE_COLORS[i]}" stroke="{STAGE_BORDERS[i]}" stroke-width="1.5"/>'
    )
    L.append(
        f'<text x="{x + STAGE_W/2}" y="{TOP+26}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13.5" font-weight="bold" '
        f'fill="{STAGE_BORDERS[i]}">{esc(STAGE_LABELS[i])}</text>'
    )

# ---------- 阶段 1：signature 参数卡 ----------
x1 = STAGE_X[0]
card_top = TOP + 46
card_h = 40
card_gap = 8
for idx, (name, is_cx) in enumerate(SIG_PARAMS):
    cy = card_top + idx * (card_h + card_gap)
    fill = "#fde68a" if is_cx else "#dbeafe"
    stroke = "#b45309" if is_cx else "#2563eb"
    L.append(
        f'<rect x="{x1+20}" y="{cy}" width="{STAGE_W-40}" height="{card_h}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    label = f"{name}" + ("  (tl.constexpr)" if is_cx else "")
    L.append(
        f'<text x="{x1+STAGE_W/2}" y="{cy+card_h/2+5}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
        f'fill="#1e293b">{esc(label)}</text>'
    )
sig_bottom = card_top + len(SIG_PARAMS) * (card_h + card_gap) - card_gap
L.append(
    f'<text x="{x1+20}" y="{sig_bottom+26}" font-family="sans-serif" font-size="11.5" '
    f'fill="#475569">inspect.signature 解析出 5 个 KernelParam</text>'
)
L.append(
    f'<text x="{x1+20}" y="{sig_bottom+44}" font-family="sans-serif" font-size="11.5" '
    f'fill="#475569">(4 非 constexpr + 1 constexpr)</text>'
)

# ---------- 阶段 2：exec 说明卡 ----------
x2 = STAGE_X[1]
exec_top = TOP + 46
exec_box_h = 150
L.append(
    f'<rect x="{x2+20}" y="{exec_top}" width="{STAGE_W-40}" height="{exec_box_h}" rx="8" '
    f'fill="white" stroke="#d97706" stroke-width="1.5"/>'
)
L.append(
    f'<text x="{x2+STAGE_W/2}" y="{exec_top+26}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="13" font-weight="bold" '
    f'fill="#92400e">create_function_from_signature</text>'
)
exec_lines = [
    "exec 生成 dynamic_func(binder)：",
    "把 mangle_type / compute_spec_key",
    "逐参数字面内联成直线代码，",
    "无循环、无查表。",
]
for i, line in enumerate(exec_lines):
    L.append(
        f'<text x="{x2+36}" y="{exec_top+52+i*20}" font-family="sans-serif" '
        f'font-size="12" fill="#374151">{esc(line)}</text>'
    )
L.append(
    f'<rect x="{x2+20}" y="{exec_top+exec_box_h+16}" width="{STAGE_W-40}" height="34" rx="6" '
    f'fill="#fee2e2" stroke="#b91c1c" stroke-width="1.5"/>'
)
L.append(
    f'<text x="{x2+STAGE_W/2}" y="{exec_top+exec_box_h+16+22}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="12" font-weight="bold" '
    f'fill="#b91c1c">仅首次发射触发（self.binder is None）</text>'
)
# 记忆化说明
memo_y = exec_top + exec_box_h + 16 + 34 + 26
L.append(
    f'<text x="{x2+20}" y="{memo_y}" font-family="sans-serif" font-size="11.5" '
    f'fill="#475569">binder 存进 self.binder，N 次发射</text>'
)
L.append(
    f'<text x="{x2+20}" y="{memo_y+18}" font-family="sans-serif" font-size="11.5" '
    f'fill="#475569">摊销到 1 次 exec 生成。</text>'
)

# ---------- 阶段 3：五元组槽位 ----------
x3 = STAGE_X[2]
slot_top = TOP + 46
slot_h = 62
slot_gap = 10
for idx, (num, name, detail, hi) in enumerate(SLOTS):
    sy = slot_top + idx * (slot_h + slot_gap)
    fill = "#fecaca" if hi else "white"
    stroke = "#b91c1c" if hi else "#059669"
    sw = "2.5" if hi else "1.5"
    L.append(
        f'<rect x="{x3+20}" y="{sy}" width="{STAGE_W-40}" height="{slot_h}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )
    L.append(
        f'<circle cx="{x3+42}" cy="{sy+18}" r="11" fill="{stroke}"/>'
    )
    L.append(
        f'<text x="{x3+42}" y="{sy+22}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="white">{esc(num)}</text>'
    )
    name_fill = "#7f1d1d" if hi else "#065f46"
    L.append(
        f'<text x="{x3+62}" y="{sy+22}" font-family="sans-serif" font-size="12.5" '
        f'font-weight="bold" fill="{name_fill}">{esc(name)}</text>'
    )
    L.append(
        f'<text x="{x3+34}" y="{sy+42}" font-family="sans-serif" font-size="10.5" '
        f'fill="#475569">{esc(detail)}</text>'
    )

# 阶段间箭头（跨阶段外框，锚在阶段中纵向中点）
mid_y = TOP + STAGE_H / 2 - 40
for i in range(2):
    xA = STAGE_X[i] + STAGE_W
    xB = STAGE_X[i + 1]
    L.append(
        f'<line x1="{xA}" y1="{mid_y}" x2="{xB}" y2="{mid_y}" '
        f'stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>'
    )

# 引出线：第2槽 sig_and_spec 高亮框右侧 → 底部说明条，指向下一图
hi_idx = [i for i, s in enumerate(SLOTS) if s[3]][0]
hi_sy = slot_top + hi_idx * (slot_h + slot_gap)
hi_right_x = x3 + STAGE_W - 20
hi_right_y = hi_sy + slot_h / 2
callout_y = TOP + STAGE_H + 34
L.append(
    f'<line x1="{hi_right_x}" y1="{hi_right_y}" x2="{hi_right_x+18}" y2="{hi_right_y}" '
    f'stroke="#b91c1c" stroke-width="2"/>'
)
L.append(
    f'<line x1="{hi_right_x+18}" y1="{hi_right_y}" x2="{hi_right_x+18}" y2="{callout_y}" '
    f'stroke="#b91c1c" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow-hi)"/>'
)
callout_w = 340
callout_x = min(hi_right_x + 18 - callout_w + 20, W - PAD - callout_w)
L.append(
    f'<rect x="{callout_x}" y="{callout_y}" width="{callout_w}" height="40" rx="6" '
    f'fill="#fef2f2" stroke="#b91c1c" stroke-width="1.5"/>'
)
L.append(
    f'<text x="{callout_x+callout_w/2}" y="{callout_y+17}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" font-weight="bold" '
    f'fill="#b91c1c">第 2 槽 sig_and_spec 就是拼缓存键的原料</text>'
)
L.append(
    f'<text x="{callout_x+callout_w/2}" y="{callout_y+33}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11" fill="#7f1d1d">下一图：缓存键怎么由它拼成</text>'
)

# 底部数字条
foot_y = H - 14
foot_items = [f"{label}={value}" for label, value in NUMS]
L.append(
    f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
    f'fill="#334155">{esc("  ·  ".join(foot_items))}</text>'
)

L.append("</svg>")
out = Path(__file__).with_name("fig-binder-codegen.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
