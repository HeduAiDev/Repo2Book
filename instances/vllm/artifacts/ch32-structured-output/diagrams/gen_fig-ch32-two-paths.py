#!/usr/bin/env python3
"""fig-ch32-two-paths: worker 侧有两条并存路径,默认走 xgrammar 库函数那条;
vLLM 自写的 Triton kernel 要显式打开 VLLM_USE_V2_MODEL_RUNNER 才生效,两者结果逐元素相同。
template: before-after(左=默认,右=opt-in)——路线前提图,必须让读者一眼认出哪条是默认。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

W, H = 1300, 640
SUBTITLE = ('分叉点:gpu_worker.py:316  if self.use_v2_model_runner:  '
            '——env 读取 int(os.getenv("VLLM_USE_V2_MODEL_RUNNER", "0"))')

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="17" '
          f'font-weight="bold" fill="#0f172a">'
          f'{esc("一件事,两条路:默认走 xgrammar 库函数,vLLM 自写 kernel 需显式开关")}</text>')
L.append(f'<text x="{W/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(SUBTITLE)}</text>')

PAD = 50
PANEL_W = 560
GAP = 80
TOP = 100
BOX_W = 460
BH = 46
VGAP = 20

PANELS = [
    {
        "badge": "默认部署",
        "badge_color": "#16a34a",
        "title": 'VLLM_USE_V2_MODEL_RUNNER = False(默认)',
        "steps": [
            "GPUModelRunnerV1",
            "gpu_model_runner.py:4245 apply_grammar_bitmask",
            "structured_output/utils.py:44 重排",
            "sorted_bitmask 5x3 + indices[4](本例)",
            "xgr.apply_token_bitmask_inplace(xgrammar 库函数)",
        ],
    },
    {
        "badge": "需手动开启",
        "badge_color": "#d97706",
        "title": "VLLM_USE_V2_MODEL_RUNNER = 1(opt-in)",
        "steps": [
            "GPUModelRunnerV2",
            "StructuredOutputsWorker",
            "紧凑掩码 4x3 直接 H2D",
            "行映射 4 个 int32(不分配 sorted_bitmask)",
            "_apply_grammar_bitmask_kernel(vLLM 自写 Triton kernel)",
        ],
    },
]

for p, panel in enumerate(PANELS):
    px = PAD + p * (PANEL_W + GAP)
    cx = px + PANEL_W / 2
    is_default = (p == 0)
    bw = 130
    L.append(f'<rect x="{cx-bw/2}" y="{TOP-38}" width="{bw}" height="26" rx="13" '
              f'fill="{panel["badge_color"]}"/>')
    L.append(f'<text x="{cx}" y="{TOP-20}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" fill="white">{esc(panel["badge"])}</text>')
    L.append(f'<text x="{cx}" y="{TOP-2}" text-anchor="middle" font-family="monospace" '
              f'font-size="11.5" fill="#1e293b">{esc(panel["title"])}</text>')
    panel_border = "#16a34a" if is_default else "#d97706"
    L.append(f'<rect x="{px}" y="{TOP+14}" width="{PANEL_W}" height="{len(panel["steps"])*(BH+VGAP)+10}" '
              f'rx="10" fill="none" stroke="{panel_border}" stroke-width="1.5" stroke-dasharray="6,4"/>')
    for i, step in enumerate(panel["steps"]):
        y = TOP + 30 + i * (BH + VGAP)
        last = (i == len(panel["steps"]) - 1)
        fill = "#dcfce7" if (is_default and last) else ("#fef3c7" if (not is_default and last) else "#e2e8f0")
        stroke = panel_border if last else "#64748b"
        L.append(f'<rect x="{cx-BOX_W/2}" y="{y}" width="{BOX_W}" height="{BH}" rx="8" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if last else 1}"/>')
        fs = 11.5 if len(step) < 40 else 10
        L.append(f'<text x="{cx}" y="{y+BH/2+4}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{fs}" fill="#0f172a">{esc(step)}</text>')
        if i < len(panel["steps"]) - 1:
            L.append(f'<line x1="{cx}" y1="{y+BH}" x2="{cx}" y2="{y+BH+VGAP-4}" '
                      f'stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')

midy = TOP + 30 + (len(PANELS[0]["steps"]) * (BH + VGAP) - VGAP) / 2
L.append(f'<line x1="{PAD+PANEL_W+14}" y1="{midy}" x2="{PAD+PANEL_W+GAP-14}" y2="{midy}" '
          f'stroke="#94a3b8" stroke-width="2" stroke-dasharray="5,3"/>')
L.append(f'<text x="{PAD+PANEL_W+GAP/2}" y="{midy-10}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#64748b">{esc("同一批输入")}</text>')

FOOT_Y = TOP + 30 + len(PANELS[0]["steps"]) * (BH + VGAP) + 30
L.append(f'<rect x="{PAD}" y="{FOOT_Y}" width="{W-2*PAD}" height="70" rx="8" '
          f'fill="#eef2ff" stroke="#6366f1"/>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+26}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("两路径结果逐元素相同(True)——本例 logits 行 1 两条路都只剩 token 5、7 存活")}</text>')
L.append(f'<text x="{W/2}" y="{FOOT_Y+48}" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
          f'fill="#3730a3">{esc("kernel(右)是演进方向,不是当前的默认行为——本章解读时先讲左边,kernel 深挖在后")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch32-two-paths.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
