#!/usr/bin/env python3
"""全书地图（无高亮）：8 Part / 33 章。每 Part 一栏组，列出本 Part 各章短中文标题。"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(s)

# (Part 标题, 副标, [ (ch_id, 短中文标题) ... ])
PARTS = [
    ("Part I", "全局图景：一个请求从头到尾", [
        ("ch01", "导读：v1 心智模型 + 本书读法"),
        ("ch02", "请求生命周期（鸟瞰）"),
        ("ch03", "EngineArgs → VllmConfig 装配"),
    ]),
    ("Part II", "异步三段式解耦（旗舰）", [
        ("ch04", "AsyncLLM 三段式门面"),
        ("ch05", "段一 · 输入处理"),
        ("ch06", "并行采样扇出 n>1"),
        ("ch07", "IPC 边界：ZMQ/msgpack"),
        ("ch08", "段三 · 输出处理"),
        ("ch09", "增量去 token 化与 stop string"),
        ("ch10", "Logprobs 装配与字节回退"),
    ]),
    ("Part III", "EngineCore 内部：忙循环", [
        ("ch11", "EngineCore 与忙循环"),
        ("ch12", "流水线并行：batch queue"),
    ]),
    ("Part IV", "调度与 KV Cache", [
        ("ch13", "连续批处理与 token 预算"),
        ("ch14", "抢占 · 等待队列 · 回填"),
        ("ch15", "分页 KV cache：块池/前缀复用"),
        ("ch16", "分配与多注意力协调"),
    ]),
    ("Part V", "执行：Worker/Runner/算子", [
        ("ch17", "Executor 与 Worker 生命周期"),
        ("ch18", "持久批次与 _prepare_inputs"),
        ("ch19", "前向与采样解耦"),
        ("ch20", "分布式并行：组与集合通信"),
        ("ch21", "异步通信与数据并行"),
    ]),
    ("Part VI", "模型/算子/注意力/采样", [
        ("ch22", "模型定义与权重加载（Llama）"),
        ("ch23", "自定义算子与 torch.compile"),
        ("ch24", "注意力后端与元数据"),
        ("ch25", "读整模型：DeepSeek-V4"),
        ("ch26", "从模型代码到架构图"),
        ("ch27", "采样流水线"),
        ("ch28", "投机解码：提议与拒绝采样"),
    ]),
    ("Part VII", "Prefill/Decode 分离", [
        ("ch29", "PD 分离 I：KV Connector 契约"),
        ("ch30", "PD 分离 II：Worker 执行与后端"),
    ]),
    ("Part VIII", "服务接口", [
        ("ch31", "离线 LLM API"),
        ("ch32", "OpenAI 兼容服务"),
        ("ch33", "弹性扩缩与多轮"),
    ]),
]

# 布局：两列 Part，竖排
COL_W = 470
COL_GAP = 30
PAD = 24
PART_HEAD_H = 40
CH_H = 26
PART_GAP = 16
TITLE_H = 56

# 分成两列，尽量均衡高度
def part_h(p):
    return PART_HEAD_H + len(p[2]) * CH_H + 14

col_assign = [0, 0, 0, 0, 1, 1, 1, 1]  # Part I-IV 左，V-VIII 右
cols = {0: [], 1: []}
for i, p in enumerate(PARTS):
    cols[col_assign[i]].append(p)

def col_height(parts):
    return sum(part_h(p) + PART_GAP for p in parts) - PART_GAP

H = TITLE_H + PAD + max(col_height(cols[0]), col_height(cols[1])) + PAD
W = PAD + COL_W + COL_GAP + COL_W + PAD

ACCENT = {0: "#2563eb", 1: "#0891b2", 2: "#7c3aed", 3: "#059669",
          4: "#d97706", 5: "#db2777", 6: "#4f46e5", 7: "#0d9488"}

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="34" text-anchor="middle" font-size="22" font-weight="bold" fill="#0f172a">vLLM v1 源码解读 · 全书地图（8 Part / 33 章）</text>')

part_index = {id(p): i for i, p in enumerate(PARTS)}
for c in (0, 1):
    x0 = PAD + c * (COL_W + COL_GAP)
    y = TITLE_H + PAD
    for p in cols[c]:
        idx = part_index[id(p)]
        accent = ACCENT[idx]
        ph = part_h(p)
        flagship = "旗舰" in p[1]
        # Part 容器
        bg = "#fffbeb" if flagship else "#f8fafc"
        bd = "#f59e0b" if flagship else "#e2e8f0"
        sw = 2.5 if flagship else 1
        L.append(f'<rect x="{x0}" y="{y}" width="{COL_W}" height="{ph}" rx="10" fill="{bg}" stroke="{bd}" stroke-width="{sw}"/>')
        # Part 标题条
        L.append(f'<rect x="{x0}" y="{y}" width="6" height="{ph}" rx="3" fill="{accent}"/>')
        L.append(f'<text x="{x0+18}" y="{y+25}" font-size="16" font-weight="bold" fill="{accent}">{esc(p[0])}</text>')
        sub = p[1]
        # 副标起点右移到 x0+104：让位给最宽的罗马数字(Part VIII 粗体到 x0+97)，避免压标题
        L.append(f'<text x="{x0+104}" y="{y+25}" font-size="13" fill="#475569">{esc(sub)}</text>')
        if flagship:
            L.append(f'<rect x="{x0+COL_W-58}" y="{y+9}" width="46" height="20" rx="10" fill="#f59e0b"/>')
            L.append(f'<text x="{x0+COL_W-35}" y="{y+23}" text-anchor="middle" font-size="12" font-weight="bold" fill="white">旗舰</text>')
        # 章列表
        cy = y + PART_HEAD_H
        for ch_id, title in p[2]:
            L.append(f'<text x="{x0+20}" y="{cy+17}" font-size="12.5" font-weight="bold" fill="{accent}">{esc(ch_id)}</text>')
            L.append(f'<text x="{x0+62}" y="{cy+17}" font-size="12.5" fill="#1e293b">{esc(title)}</text>')
            cy += CH_H
        y += ph + PART_GAP

L.append('</svg>')
import sys
out = sys.argv[1] if len(sys.argv) > 1 else "book-map.svg"
open(out, "w", encoding="utf-8").write("\n".join(L))
print("wrote", out, f"({W}x{H})")
