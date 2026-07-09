#!/usr/bin/env python3
"""paper-fig-eagle-fig5: 论文精髓图重绘。
重绘自 arXiv:2401.15077 Fig.5（§3.1 Drafting phase）——2x2 并排对比标准投机采样/
Lookahead/Medusa/EAGLE 四种方法各自怎样起草第 4、5 个 token(t4、t5)。布局与原图一致
(2x2 网格,每格 1-2 行"输入 → 模型 → 输出"流程,红框=草稿模型的预测);配色/字体套本书
视觉语言,文字译中。全部坐标由循环计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TOKEN_W, TOKEN_H = 46, 40
GAP = 6
MODEL_W, MODEL_H = 250, 40
ARROW_GAP = 26

# 每个 panel: 标题, 模型框文案, 两行(每行: 输入 token 列表[(text,predicted?)], 输出 token)
PANELS = [
    ("标准投机采样（Speculative Sampling）", "Smaller LLM 小模型", [
        (["t1", "t2", "t3"], "t4"),
        (["t1", "t2", "t3", "t4*"], "t5"),
    ]),
    ("Lookahead", "2-Gram, Jacobi", [
        (["t3"], "t4"),
        (["t4*"], "t5"),
    ]),
    ("Medusa", "Medusa Head1 / Head2", [
        (["f2"], "t4"),
        (["f2"], "t5"),
    ]),
    ("EAGLE", "Embedding + Autoregression Head", [
        (["t2/f1", "t3/f2"], "f3→t4"),
        (["t2/f1", "t3/f2", "t4*/f3*"], "f4→t5"),
    ]),
]

COL_W, ROW_H = 660, 320
PAD, TOP = 40, 130
W = PAD * 2 + COL_W * 2
H = TOP + ROW_H * 2 + 40

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="38" text-anchor="middle" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#0f172a">{esc("四种方法起草第 4、5 个 token：结构差在哪")}</text>',
     f'<text x="{W/2}" y="60" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc("重绘自 arXiv:2401.15077 Fig.5：红框=草稿模型的预测；EAGLE 输入同时含 token 与特征、逐步把预测结果并入下一步输入")}</text>',
     f'<text x="{W/2}" y="80" text-anchor="middle" font-family="sans-serif" font-size="11" '
     f'fill="#94a3b8">{esc("(标 * 者表示该 token/特征是上一步刚产出的预测结果，本步随其余输入一并喂入)")}</text>']

for pi, (title, model_label, rows) in enumerate(PANELS):
    ci, ri = pi % 2, pi // 2
    x0 = PAD + ci * COL_W
    y0 = TOP + ri * ROW_H
    L.append(f'<rect x="{x0+8}" y="{y0-6}" width="{COL_W-40}" height="{ROW_H-30}" rx="10" '
              f'fill="none" stroke="#e2e8f0" stroke-width="1.5"/>')
    L.append(f'<text x="{x0+COL_W/2-16}" y="{y0+16}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#1e40af">{esc(title)}</text>')
    for ridx, (inputs, output) in enumerate(rows):
        ry = y0 + 42 + ridx * 108
        ix = x0 + 30
        # input tokens stacked/row
        tok_xs = []
        for tok in inputs:
            is_pred = tok.endswith("*")
            disp = tok.rstrip("*")
            fill = "#fde8d7" if is_pred else "#dbeafe"
            stroke = "#c2410c" if is_pred else "#1e40af"
            L.append(f'<rect x="{ix}" y="{ry}" width="{TOKEN_W}" height="{TOKEN_H}" rx="5" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if is_pred else 1.3}"/>')
            L.append(f'<text x="{ix+TOKEN_W/2}" y="{ry+TOKEN_H/2+5}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="{stroke}">{esc(disp)}</text>')
            tok_xs.append(ix)
            ix += TOKEN_W + GAP
        model_x = ix + ARROW_GAP
        mid_in_y = ry + TOKEN_H / 2
        L.append(f'<line x1="{ix-GAP+4}" y1="{mid_in_y}" x2="{model_x-4}" y2="{mid_in_y}" '
                  'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')
        L.append(f'<rect x="{model_x}" y="{ry-4}" width="{MODEL_W}" height="{TOKEN_H+8}" rx="6" '
                  'fill="#fef3c7" stroke="#92400e" stroke-width="1.3"/>')
        L.append(f'<text x="{model_x+MODEL_W/2}" y="{ry+TOKEN_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="#78350f">{esc(model_label)}</text>')
        out_x = model_x + MODEL_W + ARROW_GAP
        L.append(f'<line x1="{model_x+MODEL_W}" y1="{mid_in_y}" x2="{out_x-4}" y2="{mid_in_y}" '
                  'stroke="#64748b" stroke-width="1.3" marker-end="url(#a)"/>')
        out_w = TOKEN_W + (18 if "→" in output else 0)
        L.append(f'<rect x="{out_x}" y="{ry}" width="{out_w}" height="{TOKEN_H}" rx="5" '
                  'fill="#fde8d7" stroke="#b91c1c" stroke-width="2"/>')
        L.append(f'<text x="{out_x+out_w/2}" y="{ry+TOKEN_H/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="#b91c1c">{esc(output)}</text>')
        rowlabel = "起草 t4" if ridx == 0 else "起草 t5（用上一步产出）"
        L.append(f'<text x="{x0+30}" y="{ry-10}" font-family="sans-serif" font-size="10" '
                  f'fill="#94a3b8">{esc(rowlabel)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-eagle-fig5.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
