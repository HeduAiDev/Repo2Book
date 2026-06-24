#!/usr/bin/env python3
"""ch32 图1：一条 /v1/chat/completions 请求从 HTTP 到 AsyncLLM 再分叉出 SSE / JSON 的生命周期。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 980, 920
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="arG" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#0e7490"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

CX = W / 2
boxw = 560
boxh = 56
gap = 30


def box(cx, y, w, h, fill, stroke, lines, sub=None, tcol="#0f172a"):
    x = cx - w / 2
    L.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{w}" height="{h}" rx="9" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    ty = y + h / 2 - (n - 1) * 9 + (4 if sub is None else -4)
    for i, t in enumerate(lines):
        fw = 'bold' if i == 0 else 'normal'
        fs = 15 if i == 0 else 13
        L.append(f'<text x="{cx:.0f}" y="{ty + i * 18:.0f}" text-anchor="middle" '
                 f'font-size="{fs}" font-weight="{fw}" fill="{tcol}">{esc(t)}</text>')
    if sub:
        L.append(f'<text x="{cx:.0f}" y="{y + h - 9:.0f}" text-anchor="middle" '
                 f'font-size="11" fill="#64748b" font-family="monospace">{esc(sub)}</text>')


def varrow(y0, y1, cx=CX, color="#475569", label=None):
    L.append(f'<line x1="{cx:.0f}" y1="{y0:.0f}" x2="{cx:.0f}" y2="{y1:.0f}" '
             f'stroke="{color}" stroke-width="1.8" marker-end="url(#{"arG" if color!="#475569" else "ar"})"/>')
    if label:
        L.append(f'<text x="{cx + 12:.0f}" y="{(y0 + y1) / 2 + 4:.0f}" font-size="11" '
                 f'fill="{color}">{esc(label)}</text>')


y = 24
nodes = [
    ("#e0f2fe", "#0284c7", ["HTTP  POST /v1/chat/completions"], None),
    ("#f1f5f9", "#94a3b8", ["FastAPI 中间件 + 路由依赖",
                            "Auth(Bearer) · X-Request-Id · validate_json_request"],
     "server_utils.py · api_router.py"),
    ("#f1f5f9", "#94a3b8", ["create_chat_completion (handler)",
                            "with_cancellation · load_aware_call 装饰"],
     "chat_completion/api_router.py:L40"),
    ("#fef9c3", "#ca8a04", ["render_chat_request",
                            "_check_model(404) · engine.errored → 抛 dead_error"],
     "chat_completion/serving.py:L202"),
    ("#fef9c3", "#ca8a04", ["OpenAIServingRender.render_chat → preprocess_chat",
                            "chat template → prompt token_ids (engine_inputs)"],
     "serve/render/serving.py:L184"),
    ("#fef9c3", "#ca8a04", ["request_id = 'chatcmpl-' + (X-Request-Id | uuid)",
                            "SamplingParams · LoRA 适配"],
     "engine/serving.py:L592"),
]
ys = []
for fill, st, lines, sub in nodes:
    ys.append(y)
    box(CX, y, boxw, boxh + (10 if sub else 0), fill, st, lines, sub)
    bot = y + boxh + (10 if sub else 0)
    y = bot + gap
    varrow(bot, y - gap + gap, CX) if False else None

# redo arrows precisely
# recompute bottoms
y = 24
bottoms = []
for fill, st, lines, sub in nodes:
    h = boxh + (10 if sub else 0)
    bottoms.append((y, y + h))
    y += h + gap
for i in range(len(nodes) - 1):
    varrow(bottoms[i][1], bottoms[i + 1][0])

# the seam node: engine_client.generate
seam_y = bottoms[-1][1] + gap
varrow(bottoms[-1][1], seam_y)
sh = 64
box(CX, seam_y, boxw, sh, "#dcfce7", "#16a34a",
    ["engine_client.generate(...)  →  AsyncGenerator[RequestOutput]",
     "与 ch04 的接缝：AsyncLLM + output_handler 多路分发"],
    "v1/engine/async_llm.py", tcol="#14532d")
seam_bot = seam_y + sh

# fork
fork_y = seam_bot + 46
lx = CX - 230
rx = CX + 230
# vertical down then split
L.append(f'<line x1="{CX:.0f}" y1="{seam_bot:.0f}" x2="{CX:.0f}" y2="{seam_bot + 22:.0f}" '
         f'stroke="#0e7490" stroke-width="1.8"/>')
L.append(f'<line x1="{lx:.0f}" y1="{seam_bot + 22:.0f}" x2="{rx:.0f}" y2="{seam_bot + 22:.0f}" '
         f'stroke="#0e7490" stroke-width="1.8"/>')
L.append(f'<text x="{lx - 8:.0f}" y="{seam_bot + 17:.0f}" text-anchor="end" font-size="12" '
         f'font-weight="bold" fill="#0e7490">request.stream = True</text>')
L.append(f'<text x="{rx + 8:.0f}" y="{seam_bot + 17:.0f}" font-size="12" '
         f'font-weight="bold" fill="#0e7490">request.stream = False</text>')
varrow(seam_bot + 22, fork_y, cx=lx, color="#0e7490")
varrow(seam_bot + 22, fork_y, cx=rx, color="#0e7490")

fbw = 380
fbh = 92
box(lx, fork_y, fbw, fbh, "#cffafe", "#0e7490",
    ["chat_completion_stream_generator",
     "首块 role 空 delta → 逐 output 推 DeltaMessage",
     "SSE: data:{…}\\n\\n  …  data:[DONE]\\n\\n"],
    "serving.py:L408")
box(rx, fork_y, fbw, fbh, "#cffafe", "#0e7490",
    ["chat_completion_full_generator",
     "async for 聚合到末个 RequestOutput",
     "组装 choices + UsageInfo 一次返回"],
    "serving.py:L1148")

out_y = fork_y + fbh + 40
varrow(fork_y + fbh, out_y, cx=lx, color="#0e7490")
varrow(fork_y + fbh, out_y, cx=rx, color="#0e7490")
obh = 50
box(lx, out_y, fbw, obh, "#e0f2fe", "#0284c7",
    ["StreamingResponse", "media_type = text/event-stream"])
box(rx, out_y, fbw, obh, "#e0f2fe", "#0284c7",
    ["JSONResponse", "ChatCompletionResponse"])

L.append('</svg>')
open("ch32-request-lifecycle.svg", "w").write('\n'.join(L))
print("wrote ch32-request-lifecycle.svg")
