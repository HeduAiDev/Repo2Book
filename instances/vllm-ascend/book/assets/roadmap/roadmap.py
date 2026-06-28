#!/usr/bin/env python3
"""Book roadmap master — renders the vLLM-Ascend 书脊（7 Part 主线）as an SVG,
highlighting the current chapter's Part. Reused as each chapter's Roadmap ("你在这里").

姊妹篇说明：本书是 vLLM 书的昇腾对位篇，主线讲「昇腾如何顶替 vLLM 执行链」。
7 个 Part 作横向泳道（左→右），即「插件接入 → 顶替设备/显存 → 顶替并行/KV解耦 →
顶替执行主线 → 顶替注意力/KV → 顶替算子/编译 → 顶替量化/采样/投机/模型」这条接管链。

Usage:
  python3 roadmap.py --highlight attach        --out roadmap.svg   # Part 键: 高亮一个 Part 泳道
  python3 roadmap.py --highlight P4            --out roadmap.svg   # 等价 Part 别名 P1..P7
  python3 roadmap.py --highlight worker        --out roadmap.svg   # 子系统键: 高亮所属 Part + "本章深入"框
  python3 roadmap.py --highlight ch20          --out roadmap.svg   # 章号键: 同上(直接按 chapter_id 定位)
  python3 roadmap.py                            --out roadmap.svg   # 空: 全书总览(meta 章)

Part 键见 STAGES（每个键=一个 Part，配 P1..P7 别名）；子系统/章号键见 ALIASES
（映射到所属 Part 阶段 + 中文「本章深入」标签）。错键报错并列出全部可用键。
Coordinates are computed (svg-diagram skill convention); text is escaped.
"""
import argparse
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# (Part 键, 标题, 副标题) — vllm-ascend 书脊 7 Part 作主线阶段
# ⚠️ 键名即语义：每个键对应它所高亮的那个 Part 框。每个 Part 另有 P1..P7 别名（见 ALIASES）。
STAGES = [
    ("attach",      "I 接入机制",     "entry points · NPUPlatform · patch"),
    ("device-mem",  "II 设备与显存",  "通信器 · camem sleep-mode"),
    ("parallel-kv", "III 并行/KV解耦", "昇腾并行组 · eplb · PD/池化"),
    ("exec",        "IV 执行主线",    "NPUWorker · NPUModelRunner"),
    ("attention",   "V 注意力与KV",   "MHA · MLA · 稀疏 · KV管理"),
    ("ops-compile", "VI 算子与编译",  "CustomOp · torch.library · ACLGraph"),
    ("models",      "VII 量化/采样/投机/模型", "注册范式四连"),
]

# 别名 → Part 键。三类别名都映射到所属 Part：
#   (a) P1..P7 简记；
#   (b) outline 里每章的 subsystem 值（注意 'distributed' 跨 P2/P3、'worker' 在 P4、
#       'attention'/'core' 在 P5——故 subsystem 单独不足以定位，子系统别名取「最常见归属」，
#       精确定位请用章号键）；
#   (c) 每个 chapter_id（ch01..ch30）直接定位到其 Part，工作流按章发车最稳。
# 值形如 (Part键, 中文"本章深入"标签)；标签=None 表示纯 Part 高亮、不画 callout。
ALIASES = {
    # (a) P1..P7 简记别名 —— 纯 Part 高亮
    "P1": ("attach", None), "P2": ("device-mem", None), "P3": ("parallel-kv", None),
    "P4": ("exec", None), "P5": ("attention", None), "P6": ("ops-compile", None),
    "P7": ("models", None),
    # (b) outline subsystem 值（取最常见归属 Part；歧义见上注）
    "overview": ("attach", None),
    # (c) 逐章 chapter_id → Part + 本章深入标签
    "ch01": ("attach", None),                          # meta 鸟瞰：纯 Part I 高亮
    "ch02": ("attach", "entry points 与 NPUPlatform"),
    "ch03": ("attach", "两段式 monkey-patch"),
    "ch04": ("attach", "引擎核心 patch（KV-cache）"),
    "ch05": ("attach", "check_and_update_config"),
    "ch06": ("device-mem", "NPUCommunicator / pyhccl"),
    "ch07": ("device-mem", "camem sleep-mode 分配器"),
    "ch08": ("parallel-kv", "昇腾并行组 / CP"),
    "ch09": ("parallel-kv", "eplb 专家负载均衡"),
    "ch10": ("parallel-kv", "PD 分离 / mooncake P2P"),
    "ch11": ("parallel-kv", "KV 池化 / ascend_store"),
    "ch12": ("parallel-kv", "KV host/CPU 卸载"),
    "ch13": ("exec", "NPUWorker 重写"),
    "ch14": ("exec", "NPUModelRunner 继承+猴补"),
    "ch15": ("exec", "execute_model / forward context"),
    "ch16": ("exec", "KV cache 在昇腾落地"),
    "ch17": ("exec", "310P 全栈特化"),
    "ch18": ("attention", "注意力后端选择/注册"),
    "ch19": ("attention", "AscendAttention MHA"),
    "ch20": ("attention", "MLA 在 NPU 上"),
    "ch21": ("attention", "稀疏注意力 SFA/DSA"),
    "ch22": ("attention", "KV 管理与调度器"),
    "ch23": ("ops-compile", "CustomOp OOT 顶替"),
    "ch24": ("ops-compile", "torch.library / meta 注册"),
    "ch25": ("ops-compile", "AscendCompiler / ACLGraph"),
    "ch26": ("ops-compile", "FusedMoE / batch-invariant"),
    "ch27": ("models", "昇腾量化框架"),
    "ch28": ("models", "采样的 NPU 对位"),
    "ch29": ("models", "投机解码 proposer"),
    "ch30": ("models", "模型 / LoRA / netloader"),
}


def build(highlight: str) -> str:
    spine_keys = [k for k, _, _ in STAGES]
    sub_label = None
    hl_key = highlight
    if highlight in ALIASES:
        hl_key, sub_label = ALIASES[highlight]
    elif highlight and highlight not in spine_keys:
        # 章号键单独成组列出，便于工作流/读者对号入座
        chs = [k for k in ALIASES if k.startswith("ch")]
        ps = [k for k in ALIASES if k.startswith("P") and len(k) == 2]
        others = [k for k in ALIASES if k not in chs and k not in ps]
        raise SystemExit(
            f"未知 --highlight {highlight!r}。\n"
            f"  Part 键: {', '.join(spine_keys)}\n"
            f"  Part 别名: {', '.join(ps)}\n"
            f"  章号键: {', '.join(chs)}\n"
            f"  子系统键: {', '.join(others)}\n"
            "（Part 章用 Part 键/别名；按章发车用章号键 chNN；meta 总览用 ''。）"
        )

    # 7 个框较宽 → 框窄一点、间隙小一点，整图横向铺开。
    bw, bh, gap, x0, y0 = 156, 76, 26, 30, 96
    w = x0 * 2 + len(STAGES) * bw + (len(STAGES) - 1) * gap
    h = 290 if sub_label else 210
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
             'markerWidth="7" markerHeight="5" orient="auto">'
             '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="{w // 2}" y="40" text-anchor="middle" font-size="20" '
             f'font-weight="bold" fill="#0f172a">vLLM-Ascend 接管链 · 全书地图（7 Part 书脊）</text>')
    # 无 highlight（ch01 meta 鸟瞰章）：图是全局鸟瞰，不承诺"你在这里"高亮。
    subtitle = ("你在这里（高亮处为本章所在 Part；昇腾在此顶替 vLLM 对应一站）" if hl_key
                else "全书总览：插件如何逐站顶替 vLLM 执行链（后续各章逐 Part 放大）")
    L.append(f'<text x="{w // 2}" y="66" text-anchor="middle" font-size="13" '
             f'fill="#64748b">{esc(subtitle)}</text>')
    hl_x = None
    for i, (key, label, sub) in enumerate(STAGES):
        x = x0 + i * (bw + gap)
        on = (key == hl_key)
        if on:
            hl_x = x
        fill = "#2563eb" if on else "#f1f5f9"
        stroke = "#1d4ed8" if on else "#cbd5e1"
        tcol = "white" if on else "#0f172a"
        scol = "#dbeafe" if on else "#64748b"
        sw = 3 if on else 2
        L.append(f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="10" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{x + bw // 2}" y="{y0 + 32}" text-anchor="middle" '
                 f'font-size="14" font-weight="bold" fill="{tcol}">{esc(label)}</text>')
        L.append(f'<text x="{x + bw // 2}" y="{y0 + 55}" text-anchor="middle" '
                 f'font-size="10" fill="{scol}">{esc(sub)}</text>')
        if i < len(STAGES) - 1:
            ax = x + bw
            ax2 = x + bw + gap
            L.append(f'<line x1="{ax}" y1="{y0 + bh // 2}" x2="{ax2 - 3}" '
                     f'y2="{y0 + bh // 2}" stroke="#64748b" stroke-width="2" '
                     f'marker-end="url(#a)"/>')

    # off-Part: 在高亮 Part 框下方画「本章深入：<子系统>」标注框 + 连线
    if sub_label and hl_x is not None:
        cy = y0 + bh + 38          # callout 顶
        ch = 54
        # callout 宽度自适应标签：估算 13.5px 字宽(CJK≈1em/拉丁数字≈0.58em)+左右内边距，
        # 取「Part 框宽 + 16」与「标签所需宽」的较大者，长标签也不裁切。
        lbl_w = 13.5 * sum((1.0 if ('⺀' <= c <= '鿿' or '＀' <= c <= '￯')
                            else 0.58 if (c.isascii() and c.isalnum()) else 0.5)
                           for c in sub_label)
        cw = max(bw + 16, int(lbl_w) + 32)
        cx = hl_x + bw // 2 - cw // 2   # 以 Part 框中心对齐，向两侧对称展开
        L.append(f'<line x1="{hl_x + bw // 2}" y1="{y0 + bh}" '
                 f'x2="{hl_x + bw // 2}" y2="{cy}" stroke="#7c3aed" '
                 f'stroke-width="2" stroke-dasharray="4 3" marker-end="url(#a)"/>')
        L.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="9" '
                 f'fill="#f5f3ff" stroke="#7c3aed" stroke-width="2.5"/>')
        L.append(f'<text x="{cx + cw // 2}" y="{cy + 22}" text-anchor="middle" '
                 f'font-size="11" fill="#7c3aed">本章深入</text>')
        L.append(f'<text x="{cx + cw // 2}" y="{cy + 43}" text-anchor="middle" '
                 f'font-size="13.5" font-weight="bold" fill="#6d28d9">{esc(sub_label)}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlight", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build(a.highlight))
    print("wrote", a.out)
