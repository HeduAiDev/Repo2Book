#!/usr/bin/env python3
"""Book map master — renders the Triton 书脊（9 Part 降级阶梯）as a narrow
horizontal strip (breadcrumb), highlighting the current chapter's Part.
Reused as each chapter's 开篇「你在这里」横幅。

本书主线命题：一门 GPU DSL（@triton.jit 里的一行 Python）一路编译成 PTX/cubin，
每一层（追踪 → TTIR → TTGIR+布局 → 优化 pass → LLVM → PTX）都是一次带理由的降级。
9 个 Part 就是这条降级阶梯的 9 级台阶——排成**单行面包屑**，从左到右（I→IX），
中间用 → 箭头连成一条链，直观体现「一路降级」的单向主线。当前章所在 Part 高亮
（填色加粗），其余淡置；高亮 chip 下方挂一个细小的「本章深入」小标。
版式为**窄长条**（宽 ≈1500、高 ≈240，宽高比 ≈6:1），只占很少版面。

Usage:
  python3 roadmap.py --highlight orientation   --out roadmap.svg   # Part 键: 高亮一个 Part
  python3 roadmap.py --highlight P5             --out roadmap.svg   # 等价 Part 别名 P1..P9
  python3 roadmap.py --highlight ch23           --out roadmap.svg   # 章号键: 同上 + "本章深入" 小标
  python3 roadmap.py                             --out roadmap.svg   # 空: 全书总览（meta 章）

Part 键见 STAGES（每个键=一个 Part，配 P1..P9 别名，且与 outline 的 subsystem 字段
一一同名——本书每个 subsystem 恰好落在唯一一个 Part，不存在 vllm-ascend 那种子系统
跨 Part 的歧义，故子系统键与 Part 键相同、无需单列）。章号键见 ALIASES。
错键报错并列出全部可用键（Part 键/Part 别名/章号键）。
Coordinates are computed (svg-diagram skill convention); text is escaped.
"""
import argparse
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


# (Part 键, 标题, 副标题) — Triton 书脊 9 Part 作降级阶梯的 9 级台阶
# ⚠️ 键名即语义：每个键对应它所高亮的那个 Part chip，且与 outline.json 里每章的
# subsystem 字段原样同名（本书里 subsystem 与 Part 一一对应，无跨 Part 情形）。
# 横条版式只显示 label（已含罗马数字 + 极短标签）；sub 副标题保留在数据里备用。
STAGES = [
    ("orientation",          "I 起步",          "DSL 与目标机器 · 五级降级阶梯"),
    ("dsl-language",         "II 领域语言 tl.*", "追踪期把 Python 翻成 IR"),
    ("jit-runtime",          "III 宿主运行时",   "fn[grid]→发射 · driver/autotune"),
    ("compiler-driver",      "IV 编译前端",      "AST→TTIR · compile() 主循环"),
    ("ir-dialects",          "V IR 与布局",      "tt→ttg→ttng · LinearLayout"),
    ("analysis-transforms",  "VI 优化 pass",     "AxisInfo·Coalesce·流水线"),
    ("conversion-lowering",  "VII 降级",         "TTGIR→LLVM→PTX"),
    ("backends-hw",          "VIII 硬件后端",    "CUDA/HIP·新卡怎么接进来"),
    ("tooling-ecosystem",    "IX 工具生态",      "proton·AOT·tutorials"),
]

# 别名 → (Part 键, 中文"本章深入"标签)。三类别名：
#   (a) P1..P9 简记 —— 纯 Part 高亮；
#   (b) 每个 chapter_id（ch01..ch43，无缺号）直接定位到其 Part + 本章一句话深入标签；
#       meta 鸟瞰章（ch01，扮演与全书总览同等的定向角色）比照 vllm-ascend 惯例
#       标签留 None，只做纯 Part 高亮。
ALIASES = {
    # (a) P1..P9 简记别名 —— 纯 Part 高亮
    "P1": ("orientation", None), "P2": ("dsl-language", None),
    "P3": ("jit-runtime", None), "P4": ("compiler-driver", None),
    "P5": ("ir-dialects", None), "P6": ("analysis-transforms", None),
    "P7": ("conversion-lowering", None), "P8": ("backends-hw", None),
    "P9": ("tooling-ecosystem", None),
    # (b) 逐章 chapter_id → Part + 本章深入标签
    "ch01": ("orientation", None),                                    # 全书心智模型：纯 Part I 高亮
    "ch02": ("orientation", "原理篇：GPU 执行模型 SIMT/占用率"),
    "ch03": ("orientation", "鸟瞰：@jit→PTX→launch 全链路"),
    "ch04": ("dsl-language", "tl.* 两层结构与 @builtin/constexpr"),
    "ch05": ("dsl-language", "类型系统与 tensor：三层类型"),
    "ch06": ("dsl-language", "类型提升与隐式广播"),
    "ch07": ("dsl-language", "造块/形状变换/访存与原子"),
    "ch08": ("dsl-language", "dot、归约与扫描"),
    "ch09": ("dsl-language", "自举标准库/数学·extern/随机数"),
    "ch10": ("jit-runtime", "JITFunction 与缓存键"),
    "ch11": ("jit-runtime", "run()：缓存→编译→发射"),
    "ch12": ("jit-runtime", "driver 抽象/后端发现/autotune"),
    "ch13": ("jit-runtime", "TRITON_INTERPRET 无 GPU 替身执行"),
    "ch14": ("compiler-driver", "compile() 驱动主循环与后端契约"),
    "ch15": ("compiler-driver", "原理篇：SSA 与结构化控制流"),
    "ch16": ("compiler-driver", "CodeGenerator：AST→TTIR"),
    "ch17": ("compiler-driver", "控制流下降到 scf（皇冠明珠）"),
    "ch18": ("compiler-driver", "双语桥：libtriton pybind11 绑定"),
    "ch19": ("ir-dialects", "tt.* 词汇表与方言黏合层"),
    "ch20": ("ir-dialects", "原理篇：布局即函数"),
    "ch21": ("ir-dialects", "Distributed 布局：Blocked/Slice/MMA"),
    "ch22": ("ir-dialects", "Shared 编码与 swizzle"),
    "ch23": ("ir-dialects", "原理篇：LinearLayout 统一布局"),
    "ch24": ("ir-dialects", "ttg./ttng. 算子：convert_layout/异步拷贝"),
    "ch25": ("analysis-transforms", "AxisInfo 与 Coalesce"),
    "ch26": ("analysis-transforms", "共享内存分配与屏障 Membar"),
    "ch27": ("analysis-transforms", "原理篇：Tensor Core 与 MMA 布局"),
    "ch28": ("analysis-transforms", "AccelerateMatmul 与布局最优化"),
    "ch29": ("analysis-transforms", "原理篇：软件流水线与模调度"),
    "ch30": ("analysis-transforms", "流水线落地：建模与展开"),
    "ch31": ("analysis-transforms", "Prefetch/Warp Specialization"),
    "ch32": ("conversion-lowering", "五级台阶第一跳：TTIR→TTGIR"),
    "ch33": ("conversion-lowering", "ConvertLayoutOp 三条搬运路径"),
    "ch34": ("conversion-lowering", "共享内存降级与访存向量化"),
    "ch35": ("conversion-lowering", "矩阵乘指令选择/LLVM→PTX 出口"),
    "ch36": ("backends-hw", "CUDABackend：注入五段 stages"),
    "ch37": ("backends-hw", "PTX→cubin→发射：ptxas/launcher"),
    "ch38": ("backends-hw", "AMD HIP 后端：第二种实现对照"),
    "ch39": ("tooling-ecosystem", "proton 剖析钩子与 do_bench"),
    "ch40": ("tooling-ecosystem", "AOT compile/link 与 SASS 反汇编"),
    "ch41": ("tooling-ecosystem", "triton-opt 家族与 tutorials 阶梯"),
    "ch42": ("tooling-ecosystem", "原理篇：FlashAttention 在线 softmax"),
    "ch43": ("tooling-ecosystem", "收官实战：fused-attention 端到端"),
}

# ── 横条（面包屑）版式常量 ────────────────────────────────────────────────
# 9 个 chip 排成单行，宽度按各自 label 文本自适应；chip 间留 GAP 放 → 箭头。
# 画布高度固定为常量（窄长条），callout 挂在高亮 chip 下方、不撑高画布。
MARGIN_X = 34            # 左右外边距
CHIP_TOP = 96            # chip 顶边 y
CHIP_H = 46              # chip 高
CHIP_PAD = 20            # chip 内水平留白（单侧）
CHIP_MIN_W = 84          # chip 最小宽（短标签兜底）
GAP = 40                 # 相邻 chip 间距（容纳箭头）
LABEL_FS = 15            # chip 标签字号
CALLOUT_GAP = 20         # 高亮 chip 底 → callout 顶 的竖向间距
CALLOUT_H = 34           # 「本章深入」小标高
CALLOUT_FS = 12.5        # 小标字号
CANVAS_H = 240           # 画布高（固定；配合 ~1500 宽得 ≈6:1 窄长条）


def text_width(s, fs):
    """估算文本像素宽（CJK≈fs*0.98，拉丁字母数字≈fs*0.56，其余≈fs*0.4）。"""
    return fs * sum(
        (0.98 if ('⺀' <= c <= '鿿' or '＀' <= c <= '￯') else
         0.56 if (c.isascii() and c.isalnum()) else 0.4)
        for c in s
    )


def build(highlight: str) -> str:
    spine_keys = [k for k, _, _ in STAGES]
    sub_label = None
    hl_key = highlight
    if highlight in ALIASES:
        hl_key, sub_label = ALIASES[highlight]
    elif highlight and highlight not in spine_keys:
        chs = [k for k in ALIASES if k.startswith("ch")]
        ps = [k for k in ALIASES if k.startswith("P") and len(k) == 2]
        raise SystemExit(
            f"未知 --highlight {highlight!r}。\n"
            f"  Part 键: {', '.join(spine_keys)}\n"
            f"  Part 别名: {', '.join(ps)}\n"
            f"  章号键: {', '.join(chs)}\n"
            "（Part 章用 Part 键/别名；按章发车用章号键 chNN；meta 总览用 ''。"
            " 子系统键与 Part 键同名，无需单列。）"
        )

    # 每个 chip 宽度按 label 自适应；顺次从左到右布局，累计出画布宽度。
    widths = [max(CHIP_MIN_W, int(text_width(label, LABEL_FS)) + 2 * CHIP_PAD)
              for _, label, _ in STAGES]
    xpos, x = [], MARGIN_X
    for wi in widths:
        xpos.append(x)
        x += wi + GAP
    w = x - GAP + MARGIN_X
    h = CANVAS_H
    cy_mid = CHIP_TOP + CHIP_H // 2

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
              'markerWidth="7" markerHeight="5" orient="auto">'
              '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="{w // 2}" y="38" text-anchor="middle" font-size="20" '
              f'font-weight="bold" fill="#0f172a">Triton 降级阶梯 · 全书地图（9 Part 书脊）</text>')
    subtitle = ("你在这里：高亮处为本章所在 Part（← 一门 DSL 一路降级成 PTX →）" if hl_key
                else "全书总览：一门 DSL 如何一路降级成 PTX（后续各章逐 Part 放大）")
    L.append(f'<text x="{w // 2}" y="64" text-anchor="middle" font-size="13" '
              f'fill="#64748b">{esc(subtitle)}</text>')

    # 链接箭头：前一 chip 右边 → 后一 chip 左边（同一行 y，体现单向降级链）。
    for i in range(len(STAGES) - 1):
        x1 = xpos[i] + widths[i]
        x2 = xpos[i + 1]
        L.append(f'<line x1="{x1 + 6}" y1="{cy_mid}" x2="{x2 - 4}" y2="{cy_mid}" '
                  f'stroke="#94a3b8" stroke-width="2" marker-end="url(#a)"/>')

    # 9 个 chip
    for i, (key, label, _sub) in enumerate(STAGES):
        x0, wi = xpos[i], widths[i]
        on = (key == hl_key)
        fill = "#2563eb" if on else "#f1f5f9"
        stroke = "#1d4ed8" if on else "#cbd5e1"
        tcol = "white" if on else "#475569"
        sw = 2.5 if on else 1.5
        fw = "bold" if on else "normal"
        L.append(f'<rect x="{x0}" y="{CHIP_TOP}" width="{wi}" height="{CHIP_H}" '
                  f'rx="13" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{x0 + wi // 2}" y="{cy_mid + 5}" text-anchor="middle" '
                  f'font-size="{LABEL_FS}" font-weight="{fw}" fill="{tcol}">{esc(label)}</text>')

    # 高亮 chip 下方挂细小「本章深入：<标签>」小标（仅逐章别名带 sub_label 时）。
    if sub_label and hl_key in spine_keys:
        hi = spine_keys.index(hl_key)
        hx, hw = xpos[hi], widths[hi]
        anchor = hx + hw // 2
        text = "本章深入 · " + sub_label
        cw = max(hw, int(text_width(text, CALLOUT_FS)) + 30)
        cx = hx + hw // 2 - cw // 2
        cx = max(8, min(cx, w - cw - 8))
        cy = CHIP_TOP + CHIP_H + CALLOUT_GAP
        # 连接线：chip 底中点 → callout 顶（x 收敛进 callout 宽度内，避免斜出框外）。
        lx = max(cx + 16, min(anchor, cx + cw - 16))
        L.append(f'<line x1="{anchor}" y1="{CHIP_TOP + CHIP_H}" x2="{lx}" y2="{cy}" '
                  f'stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="4 3"/>')
        L.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{CALLOUT_H}" '
                  f'rx="{CALLOUT_H // 2}" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.8"/>')
        L.append(f'<text x="{cx + cw // 2}" y="{cy + CALLOUT_H // 2 + 4}" '
                  f'text-anchor="middle" font-size="{CALLOUT_FS}" font-weight="bold" '
                  f'fill="#6d28d9">{esc(text)}</text>')

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlight", default="")
    ap.add_argument("--out", default="roadmap.svg")
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build(a.highlight))
    print("wrote", a.out)
