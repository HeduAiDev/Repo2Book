#!/usr/bin/env python3
"""fig-ch26-basebackend-contract：BaseBackend 是后端无关契约——6 个 @abstractmethod，
AscendBackend 逐一落地。layout 模板改造为单列填空表（本书聚焦昇腾侧）。
坐标全部由循环/常量计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "BaseBackend 契约面：6 个抽象方法，AscendBackend 逐一落地"
SUBTITLE = "python/triton/backends/compiler.py:L226-290 —— Triton 编译驱动只认这 6 个签名，不认识具体硬件"

ROWS = [
    ("supports_target", "target.backend == \"npu\""),
    ("hash", "str(self.target)  —— GPUTarget 的 repr"),
    ("parse_options", "kwargs 过滤成 NPUOptions.__dataclass_fields__ → NPUOptions(**args)"),
    ("add_stages", "注册 ttir → ttadapter(linalg) → npubin 下降管线"),
    ("load_dialects", "ascend.load_dialects(ctx)"),
    ("get_module_map", "return {}  —— 无接口→设备实现映射需要覆写"),
]

NAME_W, VAL_W, ROW_H, HEADER_H = 210, 560, 40, 36
PAD, TOP = 40, 128
LEGEND_H = 56

n = len(ROWS)
table_w = NAME_W + VAL_W
w = PAD * 2 + table_w
group_h = HEADER_H + n * ROW_H
h = TOP + group_h + LEGEND_H + PAD + 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

header_y = TOP
body_top = TOP + HEADER_H

# 表头：契约方法名 | AscendBackend 实现
L.append(f'<rect x="{PAD}" y="{header_y}" width="{NAME_W-6}" height="{HEADER_H-6}" rx="4" '
          'fill="#334155" stroke="#1e293b" stroke-width="1"/>')
L.append(f'<text x="{PAD+(NAME_W-6)/2}" y="{header_y+(HEADER_H-6)/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="white" '
          f'font-weight="bold">{esc("@abstractmethod × 6")}</text>')
vx = PAD + NAME_W
L.append(f'<rect x="{vx}" y="{header_y}" width="{VAL_W-6}" height="{HEADER_H-6}" rx="4" '
          'fill="#1d4ed8" stroke="#1e3a8a" stroke-width="1"/>')
L.append(f'<text x="{vx+(VAL_W-6)/2}" y="{header_y+(HEADER_H-6)/2+5}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12.5" fill="white" '
          f'font-weight="bold">{esc("AscendBackend 落地")}</text>')

for i, (name, impl) in enumerate(ROWS):
    ry = body_top + i * ROW_H
    row_fill = "#f8fafc" if i % 2 == 0 else "white"
    L.append(f'<rect x="{PAD}" y="{ry}" width="{table_w}" height="{ROW_H}" '
              f'fill="{row_fill}" stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{PAD+14}" y="{ry+ROW_H/2+4}" font-family="monospace" '
              f'font-size="12.5" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<text x="{vx+14}" y="{ry+ROW_H/2+4}" font-family="monospace" '
              f'font-size="11.5" fill="#1e293b">{esc(impl)}</text>')

bottom = body_top + n * ROW_H
legend_y = bottom + 26
L.append(f'<text x="{PAD}" y="{legend_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("同一份 6 方法契约，nvidia/amd 也各自实现一份——Triton 编译总控代码不因新增后端而改动。")}</text>')
L.append(f'<text x="{PAD}" y="{legend_y+18}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc("BaseBackend 另有 2 个可覆写钩子（get_attrs_descriptor / compute_spec_key），本图只列必填契约。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch26-basebackend-contract.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  w={w} h={h}")
