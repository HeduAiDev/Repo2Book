#!/usr/bin/env python3
"""ch32 图3：同一个 generate 生成器，分流为流式(DELTA) vs 非流式(FINAL_ONLY) 的对照表。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)

rows = [
    ("维度", "流式 stream=True (DELTA)", "非流式 stream=False (FINAL_ONLY)"),
    ("同源生成器", "engine_client.generate(...)  —  完全同一个 AsyncGenerator", ""),
    ("消费方式", "逐个 RequestOutput 立即 yield", "async for 取到末个 RequestOutput"),
    ("首块", "单独发只含 role 的空 delta", "无（一次成型）"),
    ("增量载体", "DeltaMessage（role / content / tool_calls）", "ChatMessage（完整 message）"),
    ("用量 usage", "可选：include_usage 时末尾单独一块", "response.usage 内嵌"),
    ("错误处理", "写成 error data 帧（200 已发，改不了状态码）", "ErrorResponse → 可设 4xx/5xx 状态码"),
    ("终止信号", "data: [DONE]\\n\\n 哨兵", "直接 return 整个 JSON"),
    ("FastAPI 出口", "StreamingResponse(text/event-stream)", "JSONResponse"),
    ("延迟特征", "TTFT 低，逐 token 推送", "一次性返回，聚合 usage/logprobs"),
]

ncol = 3
colw = [180, 405, 405]
rowh = 46
hh = 50
W = sum(colw) + 40
H = hh + (len(rows) - 1) * rowh + 40
x0, y0 = 20, 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# header
hx = x0
hys = y0
xs_pos = [x0]
for w in colw:
    xs_pos.append(xs_pos[-1] + w)

hdr = rows[0]
fills_hdr = ["#334155", "#0e7490", "#7c3aed"]
for c in range(ncol):
    L.append(f'<rect x="{xs_pos[c]}" y="{y0}" width="{colw[c]}" height="{hh}" '
             f'fill="{fills_hdr[c]}"/>')
    cx = xs_pos[c] + colw[c] / 2
    L.append(f'<text x="{cx:.0f}" y="{y0 + hh / 2 + 5:.0f}" text-anchor="middle" '
             f'font-size="15" font-weight="bold" fill="white">{esc(hdr[c])}</text>')

y = y0 + hh
for r in range(1, len(rows)):
    label, a, b = rows[r]
    same = (b == "")
    bg = "#f8fafc" if r % 2 else "#eef2f7"
    # label cell
    L.append(f'<rect x="{xs_pos[0]}" y="{y}" width="{colw[0]}" height="{rowh}" '
             f'fill="#f1f5f9" stroke="#cbd5e1" stroke-width="0.8"/>')
    L.append(f'<text x="{xs_pos[0] + 12}" y="{y + rowh / 2 + 5:.0f}" font-size="13" '
             f'font-weight="bold" fill="#0f172a">{esc(label)}</text>')
    if same:
        # span across both columns
        L.append(f'<rect x="{xs_pos[1]}" y="{y}" width="{colw[1] + colw[2]}" height="{rowh}" '
                 f'fill="#dcfce7" stroke="#cbd5e1" stroke-width="0.8"/>')
        cx = xs_pos[1] + (colw[1] + colw[2]) / 2
        L.append(f'<text x="{cx:.0f}" y="{y + rowh / 2 + 5:.0f}" text-anchor="middle" '
                 f'font-size="13.5" font-weight="bold" fill="#14532d" font-family="monospace">{esc(a)}</text>')
    else:
        for c, txt, col in [(1, a, "#0e7490"), (2, b, "#7c3aed")]:
            L.append(f'<rect x="{xs_pos[c]}" y="{y}" width="{colw[c]}" height="{rowh}" '
                     f'fill="{bg}" stroke="#cbd5e1" stroke-width="0.8"/>')
            fam = 'monospace' if ('[DONE]' in txt or 'Response' in txt or '(' in txt) else 'sans-serif'
            L.append(f'<text x="{xs_pos[c] + 12}" y="{y + rowh / 2 + 5:.0f}" font-size="12.5" '
                     f'fill="#1e293b" font-family="{fam}">{esc(txt)}</text>')
    y += rowh

L.append('</svg>')
open("ch32-stream-vs-final.svg", "w").write('\n'.join(L))
print("wrote ch32-stream-vs-final.svg")
