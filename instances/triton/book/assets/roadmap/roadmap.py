#!/usr/bin/env python3
"""Book roadmap master — renders the Triton 书脊（9 Part 降级阶梯）as an SVG,
highlighting the current chapter's Part. Reused as each chapter's Roadmap ("你在这里").

本书主线命题：一门 GPU DSL（@triton.jit 里的一行 Python）一路编译成 PTX/cubin，
每一层（追踪 → TTIR → TTGIR+布局 → 优化 pass → LLVM → PTX）都是一次带理由的降级。
9 个 Part 就是这条降级阶梯的 9 级台阶——横向 9 格宽度超出画布预算，故按「之」字形
（boustrophedon）折成 3 行 × 3 列：每行内从左到右逐级降一格（视觉上的台阶），
到行尾折下一行继续降，整体读来仍是自顶向下、从入口 DSL 走到工具生态的单向阶梯。

Usage:
  python3 roadmap.py --highlight orientation   --out roadmap.svg   # Part 键: 高亮一个 Part
  python3 roadmap.py --highlight P5             --out roadmap.svg   # 等价 Part 别名 P1..P9
  python3 roadmap.py --highlight ch23           --out roadmap.svg   # 章号键: 同上 + "本章深入" 框
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
# ⚠️ 键名即语义：每个键对应它所高亮的那个 Part 框，且与 outline.json 里每章的
# subsystem 字段原样同名（本书里 subsystem 与 Part 一一对应，无跨 Part 情形）。
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

# 3 行 × 3 列的之字形（boustrophedon）网格：每行 3 个 Part 从左到右排布，
# 行内每列下沉 DY_STEP（台阶感），行末折到下一行、继续下沉——整体单向降级。
COLS = 3
BW, BH = 280, 88
GAP_X = 40
DY_STEP = 18          # 行内逐列下沉量（台阶视觉）
ROW_GAP = 100          # 行间额外间距（须容纳最深列下方的"本章深入"回环，见下方推导）
X0, Y0 = 40, 100


def esc_cjk_width(s):
    """估算标签像素宽（CJK≈1em，拉丁字母数字≈0.58em）用于 callout 自适应宽度。"""
    return 13.5 * sum(
        (1.0 if ('⺀' <= c <= '鿿' or '＀' <= c <= '￯') else
         0.58 if (c.isascii() and c.isalnum()) else 0.5)
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

    rows = [STAGES[i:i + COLS] for i in range(0, len(STAGES), COLS)]
    row_height = BH + (COLS - 1) * DY_STEP   # 一行的纵向footprint（含台阶下沉）

    w = X0 * 2 + COLS * BW + (COLS - 1) * GAP_X
    h = Y0 + len(rows) * row_height + (len(rows) - 1) * ROW_GAP + 100  # 尾部余量放 callout

    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
    L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
              'markerWidth="7" markerHeight="5" orient="auto">'
              '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
    L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
    L.append(f'<text x="{w // 2}" y="40" text-anchor="middle" font-size="20" '
              f'font-weight="bold" fill="#0f172a">Triton 降级阶梯 · 全书地图（9 Part 书脊）</text>')
    subtitle = ("你在这里（高亮处为本章所在 Part；台阶从入口 DSL 一路降到工具生态）" if hl_key
                else "全书总览：一门 DSL 如何一路编译成 PTX（后续各章逐 Part 放大）")
    L.append(f'<text x="{w // 2}" y="66" text-anchor="middle" font-size="13" '
              f'fill="#64748b">{esc(subtitle)}</text>')

    # 记录每个 Part 框的几何，供画箭头/callout 复用
    geo = {}
    for r, row in enumerate(rows):
        row_top = Y0 + r * (row_height + ROW_GAP)
        for c, (key, label, sub) in enumerate(row):
            x = X0 + c * (BW + GAP_X)
            y = row_top + c * DY_STEP
            geo[key] = (x, y)

    # Part 框
    for key, label, sub in STAGES:
        x, y = geo[key]
        on = (key == hl_key)
        fill = "#2563eb" if on else "#f1f5f9"
        stroke = "#1d4ed8" if on else "#cbd5e1"
        tcol = "white" if on else "#0f172a"
        scol = "#dbeafe" if on else "#64748b"
        sw = 3 if on else 2
        L.append(f'<rect x="{x}" y="{y}" width="{BW}" height="{BH}" rx="10" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        L.append(f'<text x="{x + BW // 2}" y="{y + 34}" text-anchor="middle" '
                  f'font-size="15" font-weight="bold" fill="{tcol}">{esc(label)}</text>')
        L.append(f'<text x="{x + BW // 2}" y="{y + 58}" text-anchor="middle" '
                  f'font-size="10.5" fill="{scol}">{esc(sub)}</text>')

    # 箭头：行内从左框右边中点 → 右框左边中点（自然带出台阶下沉的斜线）；
    # 行末：从本行最后一框底边中点 → 下一行首框顶边中点（折行下沉，延续单向降级）。
    for r, row in enumerate(rows):
        for c in range(len(row) - 1):
            k1, k2 = row[c][0], row[c + 1][0]
            x1, y1 = geo[k1]
            x2, y2 = geo[k2]
            L.append(f'<line x1="{x1 + BW}" y1="{y1 + BH // 2}" x2="{x2 - 3}" '
                      f'y2="{y2 + BH // 2}" stroke="#64748b" stroke-width="2" '
                      f'marker-end="url(#a)"/>')
        if r < len(rows) - 1:
            k_last = row[-1][0]
            k_next = rows[r + 1][0][0]
            xl, yl = geo[k_last]
            xn, yn = geo[k_next]
            L.append(f'<line x1="{xl + BW // 2}" y1="{yl + BH}" '
                      f'x2="{xn + BW // 2}" y2="{yn - 3}" stroke="#64748b" '
                      f'stroke-width="2" stroke-dasharray="5 3" marker-end="url(#a)"/>')

    # off-Part: 在高亮 Part 框下方画「本章深入：<子系统>」标注框 + 连线
    if sub_label and hl_key in geo:
        hx, hy = geo[hl_key]
        cy = hy + BH + 38
        ch = 54
        lbl_w = esc_cjk_width(sub_label)
        cw = max(BW + 16, int(lbl_w) + 32)
        cx = hx + BW // 2 - cw // 2
        cx = max(8, min(cx, w - cw - 8))
        L.append(f'<line x1="{hx + BW // 2}" y1="{hy + BH}" '
                  f'x2="{hx + BW // 2}" y2="{cy}" stroke="#7c3aed" '
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
