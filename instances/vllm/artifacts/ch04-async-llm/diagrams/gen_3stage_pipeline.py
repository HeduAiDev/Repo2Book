#!/usr/bin/env python3
"""fig-3stage-pipeline: AsyncLLM 三段式异步解耦总览（进程边界 + 数据流）。"""
import re, sys, os, subprocess
from pathlib import Path
import xml.sax.saxutils as xs


def esc(s): return xs.escape(s)


_CJK = re.compile(r'[㐀-䶿一-鿿　-〿＀-￯]')


_CJK_FONT = "Droid Sans Fallback"  # host 上唯一带 CJK 字形的字体


def cjk_font(svg: str) -> str:
    """PNG 渲染器（ImageMagick）对 sans-serif/monospace 的中文不做逐字回退 → 中文丢字形。
    凡 <text> 内容含中文的，把其 font-family 强制为单独的 "Droid Sans Fallback"
    （绝不与 latin 族逗号并列，否则又会丢字），让中文在 PNG 里正确渲染。
    latin/数字文本（不含 CJK）保持原 font-family 不变。"""
    def repl(m):
        attrs, content = m.group(1), m.group(2)
        if not _CJK.search(content):
            return m.group(0)
        new_attrs, n = re.subn(r'font-family="[^"]*"', f'font-family="{_CJK_FONT}"', attrs)
        if n == 0:
            new_attrs = attrs + f' font-family="{_CJK_FONT}"'
        return f'<text{new_attrs}>{content}</text>'
    return re.sub(r'<text\b([^>]*)>(.*?)</text>', repl, svg, flags=re.S)


# 颜色
C_FRONT = "#eef2ff"   # 本进程底色
C_FRONT_BD = "#6366f1"
C_CORE = "#e5e7eb"    # EngineCore 灰显
C_CORE_BD = "#9ca3af"
C_REQ = "#2563eb"     # EngineCoreRequest 进
C_OUT = "#dc2626"     # EngineCoreOutput 出
C_BOX = "#ffffff"
C_BD = "#475569"
C_TXT = "#1e293b"
C_MUT = "#64748b"
C_QUEUE = "#fef3c7"
C_QUEUE_BD = "#d97706"


def box(L, x, y, w, h, title, sub="", fill=C_BOX, bd=C_BD, tcol=C_TXT, fs=13):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{bd}" stroke-width="2"/>')
    cy = y + (h // 2 if not sub else h // 2 - 7)
    L.append(f'<text x="{x+w//2}" y="{cy+5}" text-anchor="middle" font-family="sans-serif" font-size="{fs}" font-weight="bold" fill="{tcol}">{esc(title)}</text>')
    if sub:
        L.append(f'<text x="{x+w//2}" y="{cy+22}" text-anchor="middle" font-family="monospace" font-size="10" fill="{C_MUT}">{esc(sub)}</text>')


def arrow(L, x1, y1, x2, y2, color, mid="", dash=False, mid_dy=-6, mid_anchor="middle"):
    da = ' stroke-dasharray="6,4"' if dash else ''
    mk = "ar" if color == C_REQ else ("ao" if color == C_OUT else "ag")
    L.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2.2"{da} marker-end="url(#{mk})"/>')
    if mid:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2 + mid_dy
        L.append(f'<text x="{mx}" y="{my}" text-anchor="{mid_anchor}" font-family="monospace" font-size="10" fill="{color}">{esc(mid)}</text>')


def vrect(L, x, y, w, h, fill, bd, op="1"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" fill-opacity="{op}" stroke="{bd}" stroke-width="1.5" stroke-dasharray="2,3"/>')


def build():
    w, h = 1180, 720
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs>')
    for mk, col in [("ar", C_REQ), ("ao", C_OUT), ("ag", C_MUT)]:
        L.append(f'<marker id="{mk}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{col}"/></marker>')
    L.append('</defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

    L.append(f'<text x="30" y="34" font-family="sans-serif" font-size="20" font-weight="bold" fill="{C_TXT}">{esc("AsyncLLM 三段式异步解耦：进程边界与数据流")}</text>')

    # 泳道区域
    front_x, front_w = 20, 560
    bound_x, bound_w = 600, 120
    core_x, core_w = 740, 420
    lane_y, lane_h = 60, 630

    vrect(L, front_x, lane_y, front_w, lane_h, C_FRONT, C_FRONT_BD, op="0.5")
    vrect(L, core_x, lane_y, core_w, lane_h, C_CORE, C_CORE_BD, op="0.6")
    # IPC 接缝（虚线进程边界）
    L.append(f'<rect x="{bound_x}" y="{lane_y}" width="{bound_w}" height="{lane_h}" fill="none" stroke="{C_MUT}" stroke-width="2" stroke-dasharray="8,6"/>')

    L.append(f'<text x="{front_x+12}" y="{lane_y+22}" font-family="sans-serif" font-size="13" font-weight="bold" fill="{C_FRONT_BD}">{esc("本进程  Frontend（前端）")}</text>')
    L.append(f'<text x="{bound_x+bound_w//2}" y="{lane_y+18}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold" fill="{C_MUT}">{esc("IPC 接缝")}</text>')
    L.append(f'<text x="{bound_x+bound_w//2}" y="{lane_y+34}" text-anchor="middle" font-family="monospace" font-size="9" fill="{C_MUT}">{esc("f2: ch07")}</text>')
    L.append(f'<text x="{core_x+12}" y="{lane_y+22}" font-family="sans-serif" font-size="13" font-weight="bold" fill="#6b7280">{esc("独立进程  EngineCore")}</text>')

    # ---- 进入方向（蓝） ----
    # Stage1 InputProcessor
    box(L, 40, 95, 200, 56, "Stage1  InputProcessor", "prompt → EngineCoreRequest", fill="#ffffff", bd=C_FRONT_BD)
    L.append(f'<text x="248" y="108" font-family="monospace" font-size="9" fill="{C_MUT}">{esc("(ch05)")}</text>')
    # add_request 扇出节点
    box(L, 90, 190, 110, 50, "add_request", "扇出点", fill="#dbeafe", bd=C_REQ)
    arrow(L, 140, 151, 140, 190, C_MUT)
    # 上路：OutputProcessor 登记
    box(L, 300, 175, 230, 56, "OutputProcessor.add_request", "登记 req_id → queue（本进程）", fill="#ffffff", bd=C_FRONT_BD)
    arrow(L, 200, 205, 300, 200, C_REQ, "this process")
    # 下路：add_request_async 穿边界到 EngineCore
    box(L, 760, 300, 200, 60, "EngineCore", "调度 + 执行（ch03 / ch07）", fill=C_CORE, bd=C_CORE_BD, tcol="#6b7280")
    arrow(L, 145, 240, 660, 330, C_REQ, "add_request_async", dash=False, mid_dy=-8)
    arrow(L, 660, 330, 760, 330, C_REQ, "separate process", mid_dy=-8)

    # f2 anchor 标在边界
    L.append(f'<rect x="608" y="300" width="20" height="20" rx="10" fill="{C_REQ}"/>')
    L.append(f'<text x="618" y="314" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="bold" fill="white">{esc("2")}</text>')

    # EngineCoreRequest 消息标签
    L.append(f'<text x="400" y="288" text-anchor="middle" font-family="monospace" font-size="11" font-weight="bold" fill="{C_REQ}">{esc("EngineCoreRequest →（已 tokenize 的请求）")}</text>')

    # ---- 返回方向（红） ----
    # EngineCore -> get_output_async -> output_handler
    box(L, 740, 470, 220, 60, "output_handler", "背景 asyncio.Task（生产者）", fill="#fee2e2", bd=C_OUT)
    arrow(L, 870, 360, 870, 470, C_OUT, "get_output_async", mid_dy=4, mid_anchor="start")
    # f3 anchor
    L.append(f'<rect x="954" y="488" width="20" height="20" rx="10" fill="{C_OUT}"/>')
    L.append(f'<text x="964" y="502" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="bold" fill="white">{esc("3")}</text>')
    # EngineCoreOutput 消息标签
    L.append(f'<text x="610" y="450" text-anchor="middle" font-family="monospace" font-size="11" font-weight="bold" fill="{C_OUT}">{esc("← EngineCoreOutput（new_token_ids + finished）")}</text>')

    # output_handler -> process_outputs -> 多个队列
    box(L, 300, 470, 230, 56, "process_outputs", "按 req_id 解多路复用", fill="#fff", bd=C_OUT)
    arrow(L, 740, 500, 530, 500, C_OUT, "process_outputs")

    # per-request 队列（f1）
    qx = 60
    for i, lab in enumerate(["queue A", "queue B", "queue C"]):
        qy = 560 + i * 50
        box(L, qx, qy, 150, 40, f"RequestOutputCollector", lab, fill=C_QUEUE, bd=C_QUEUE_BD, fs=11)
        arrow(L, 300, 500 + (i - 1) * 6, 210, qy + 20, C_OUT, "" )
    # f1 anchor
    L.append(f'<rect x="200" y="552" width="20" height="20" rx="10" fill="{C_QUEUE_BD}"/>')
    L.append(f'<text x="210" y="566" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="bold" fill="white">{esc("1")}</text>')

    # 队列 -> generate -> 客户端
    box(L, 270, 600, 130, 44, "generate() 协程", "get_nowait / get", fill="#ffffff", bd=C_FRONT_BD, fs=11)
    arrow(L, 210, 600, 270, 620, C_OUT, "")
    box(L, 440, 600, 110, 44, "客户端", "yield RequestOutput", fill="#f1f5f9", bd=C_BD, fs=11)
    arrow(L, 400, 622, 440, 622, C_OUT, "")

    # 图例
    lx, ly = 740, 600
    L.append(f'<rect x="{lx}" y="{ly}" width="410" height="100" rx="6" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
    L.append(f'<text x="{lx+12}" y="{ly+22}" font-family="sans-serif" font-size="12" font-weight="bold" fill="{C_TXT}">{esc("图例")}</text>')
    L.append(f'<rect x="{lx+12}" y="{ly+32}" width="24" height="4" fill="{C_REQ}"/>')
    L.append(f'<text x="{lx+44}" y="{ly+39}" font-family="sans-serif" font-size="11" fill="{C_TXT}">{esc("EngineCoreRequest（进入引擎方向）")}</text>')
    L.append(f'<rect x="{lx+12}" y="{ly+52}" width="24" height="4" fill="{C_OUT}"/>')
    L.append(f'<text x="{lx+44}" y="{ly+59}" font-family="sans-serif" font-size="11" fill="{C_TXT}">{esc("EngineCoreOutput（返回前端方向）")}</text>')
    for i, (n, t) in enumerate([("1", "per-request 队列 → ch08"), ("2", "进程边界/IPC → ch07"), ("3", "生产者-消费者 → ch08")]):
        ix = lx + 12 + i * 0
        L.append(f'<rect x="{lx+12}" y="{ly+72+ (i//3)*0}" width="0" height="0"/>')
    # 三个伏笔锚一行
    fy = ly + 80
    for i, (n, t) in enumerate([("1", "f1 队列→ch08"), ("2", "f2 边界→ch07"), ("3", "f3 生产消费→ch08")]):
        cx = lx + 24 + i * 135
        col = [C_QUEUE_BD, C_REQ, C_OUT][i]
        L.append(f'<rect x="{cx-12}" y="{fy-12}" width="18" height="18" rx="9" fill="{col}"/>')
        L.append(f'<text x="{cx-3}" y="{fy+1}" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="bold" fill="white">{esc(n)}</text>')
        L.append(f'<text x="{cx+12}" y="{fy+2}" font-family="sans-serif" font-size="9.5" fill="{C_TXT}">{esc(t)}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == '__main__':
    base = sys.argv[1]
    svg = cjk_font(build())
    Path(base + '.svg').write_text(svg)
    print(f"SVG {len(svg)}B")
    assert subprocess.run(['xmllint', '--noout', base + '.svg']).returncode == 0
    vs = Path('/mnt/e/Laboratory/Repo2Book/.claude/skills/svg-diagram/scripts/validate_svg.py')
    r = subprocess.run([sys.executable, str(vs), base + '.svg'], capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stdout
    subprocess.run(['convert', '-density', '150', base + '.svg', base + '.png'], check=True)
    print(f"PNG {os.path.getsize(base+'.png')//1024}KB")
