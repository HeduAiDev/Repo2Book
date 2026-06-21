"""06-get-outputs-streaming-vs-final: 双泳道对照 streaming vs FINAL_ONLY 归并。"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(s)

w, h = 1120, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append('<defs>'
         '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(f'<text x="{w/2}" y="32" text-anchor="middle" font-size="19" font-weight="bold" fill="#0f172a">get_outputs：流式逐条转发  vs  FINAL_ONLY 按 index 聚合</text>')

def lane(y0, lh, fill, stroke, title):
    L.append(f'<rect x="20" y="{y0}" width="{w-40}" height="{lh}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    L.append(f'<text x="40" y="{y0+28}" font-size="16" font-weight="bold" fill="{stroke}">{esc(title)}</text>')

def step(x, y, bw, bh, fill, stroke, lines, fs=13):
    L.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    n = len(lines)
    start = y + bh/2 - (n-1)*(fs+3)/2 + fs/2 - 2
    for i, ln in enumerate(lines):
        wt = "bold" if i == 0 else "normal"
        L.append(f'<text x="{x+bw/2}" y="{start+i*(fs+3)}" text-anchor="middle" font-size="{fs}" font-weight="{wt}" fill="#0f172a">{esc(ln)}</text>')

def arrow(x1, y, x2):
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="#475569" stroke-width="1.6" marker-end="url(#a)"/>')

# ---- Lane 1: STREAMING ----
y0 = 60
lane(y0, 240, "#eff6ff", "#2563eb", "STREAMING（output_kind ≠ FINAL_ONLY）：每路增量到达即转发")
sy = y0 + 70
step(50, sy, 200, 80, "#dbeafe", "#2563eb",
     ["child i 增量到达", "CompletionOutput(index=i)"])
arrow(250, sy+40, 300)
step(300, sy, 230, 80, "#fef9c3", "#ca8a04",
     ["完成则移出 child_requests", "已返还过的 child → 去重"])
arrow(530, sy+40, 580)
step(580, sy, 230, 80, "#dcfce7", "#16a34a",
     ["return [completion_output]", "（去重命中则 return []）"])
arrow(810, sy+40, 860)
step(860, sy, 220, 80, "#ede9fe", "#7c3aed",
     ["finished = child_requests 空", "客户端自行拼接 n 路"])
L.append(f'<text x="{w/2}" y="{y0+225}" text-anchor="middle" font-size="13" fill="#1e40af">任一路先完成即可立刻提前返还，无需等齐——粒度更细</text>')

# ---- Lane 2: FINAL_ONLY ----
y1 = 330
lane(y1, 270, "#fefce8", "#ca8a04", "FINAL_ONLY：output_aggregator[index] 归位，攒齐 n 路才一次性吐")
# aggregator slots evolving
ay = y1 + 60
labels = [
    ("child2 到达", ["[ _ , _ , c ]", "child_requests 非空 → return []"], "#fee2e2", "#dc2626"),
    ("child0 到达", ["[ a , _ , c ]", "仍非空 → return []"], "#fee2e2", "#dc2626"),
    ("child1 到达", ["[ a , b , c ]", "child_requests 空 → 一次性吐出"], "#dcfce7", "#16a34a"),
]
bx = 60
for i, (cap, lines, fill, stroke) in enumerate(labels):
    x = bx + i * 330
    L.append(f'<text x="{x+140}" y="{ay-8}" text-anchor="middle" font-size="13" font-weight="bold" fill="#854d0e">{esc(cap)}（乱序到达）</text>')
    step(x, ay, 280, 86, fill, stroke, lines, fs=14)
    if i < 2:
        arrow(x+280, ay+43, x+330)
L.append(f'<text x="{w/2}" y="{y1+250}" text-anchor="middle" font-size="13" fill="#854d0e">归位按 CompletionOutput.index，结果顺序 [out0,out1,out2] 与到达顺序无关，finished=True</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch06-input-processor/diagrams/06-get-outputs-streaming-vs-final.svg", "w").write('\n'.join(L))
print("wrote getoutputs svg")
