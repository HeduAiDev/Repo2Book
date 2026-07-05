"""生成 ch17 四张配图：
  1) inheritance-depth-by-component —— 310P 按组件挑继承深度（三层/特例/两层直继承）
  2) slot-mapping-triton-vs-numpy   —— 基类 Triton slot_mapping vs 310 CPU NumPy 路径
  3) horizontal-cuts                —— is_310p() 一个布尔分流，辐射四个横切域
  4) kv-cache-constraints           —— 310P KV cache 分配的三处受限（拒绝/对齐/格式）
全部坐标由 Python 计算，无手填魔数。"""
import os
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


FONT = 'font-family="sans-serif"'


def box(x, y, w, h, fill, stroke, rx=10, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, anchor="middle", fill="#1e293b", weight="normal", mono=False):
    fam = 'font-family="monospace"' if mono else FONT
    return (f'<text x="{x}" y="{y}" {fam} font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')


def vline(x, y1, y2, stroke, sw=2, marker="ar", dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" marker-end="url(#{marker})"{d}/>')


def defs_arrow(mid, color):
    return (f'<marker id="{mid}" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" '
            f'markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="{color}"/></marker>')


# ----------------------------------------------------------------------------
# 图 1：按组件挑继承深度
# ----------------------------------------------------------------------------
def diagram_inheritance(path):
    W, H = 1200, 760
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>' + defs_arrow("ar", "#475569") + defs_arrow("im", "#7c3aed") + '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 40, "310P 不一刀切：按组件挑继承深度", 24, weight="bold"))
    L.append(text(W / 2, 68, "实线=继承，紫虚线=import 复用（非继承）；箭头是否经过中层，区分三种深度", 14, fill="#64748b"))

    colw, gap = 360, 30
    x0 = (W - 3 * colw - 2 * gap) / 2
    cols = [x0, x0 + colw + gap, x0 + 2 * (colw + gap)]
    top_y, mid_y, bot_y = 116, 286, 456
    bh = 78

    def node(x, y, title, sub, fill, stroke, tcol):
        nonlocal L
        L.append(box(x, y, colw, bh, fill, stroke, sw=2.2))
        L.append(text(x + colw / 2, y + 30, title, 14.5, weight="bold", fill=tcol, mono=True))
        L.append(text(x + colw / 2, y + 54, sub, 11.5, fill="#475569"))

    # ---- 组 1：三层（vLLM → 昇腾主栈 → 310）----
    cx = cols[0]
    L.append(text(cx + colw / 2, top_y - 18, "① 主执行体 · 三层", 15, weight="bold", fill="#1d4ed8"))
    node(cx, top_y, "GPUModelRunner / InputBatch", "vLLM 基座", "#eff6ff", "#3b82f6", "#1d4ed8")
    node(cx, mid_y, "NPUModelRunner / NPUInputBatch", "昇腾主栈 · 继承+猴补（第 14 章）", "#eff6ff", "#3b82f6", "#1d4ed8")
    node(cx, bot_y, "NPUModelRunner310 / …310", "再继承一层", "#dbeafe", "#2563eb", "#1d4ed8")
    L.append(vline(cx + colw / 2, top_y + bh, mid_y, "#475569", 2.4))
    L.append(vline(cx + colw / 2, mid_y + bh, bot_y, "#475569", 2.4))

    # ---- 组 2：BlockTable 特例 ----
    cx = cols[1]
    L.append(text(cx + colw / 2, top_y - 18, "② BlockTable · 特例", 15, weight="bold", fill="#7c3aed"))
    node(cx, top_y, "_compute_slot_mapping_kernel", "vLLM 只贡献 Triton kernel", "#faf5ff", "#a855f7", "#7c3aed")
    node(cx, mid_y, "class BlockTable:", "昇腾独立类 · 不继承 vLLM BlockTable", "#faf5ff", "#a855f7", "#7c3aed")
    node(cx, bot_y, "BlockTable(AscendBlockTable)", "建在昇腾独立类之上", "#f3e8ff", "#9333ea", "#7c3aed")
    # 虚线 import：top → mid（非继承）
    L.append(vline(cx + colw / 2, top_y + bh, mid_y, "#7c3aed", 2.2, marker="im", dash="6,4"))
    L.append(text(cx + colw - 6, (top_y + bh + mid_y) / 2 + 4, "import 复用", 11, anchor="end", fill="#7c3aed"))
    L.append(vline(cx + colw / 2, mid_y + bh, bot_y, "#475569", 2.4))

    # ---- 组 3：两层直继承（跳过昇腾中间层）----
    cx = cols[2]
    L.append(text(cx + colw / 2, top_y - 18, "③ 与昇腾无关 · 两层", 15, weight="bold", fill="#059669"))
    node(cx, top_y, "KVBlockZeroer / ShardedStateLoader", "vLLM 基类", "#ecfdf5", "#10b981", "#047857")
    L.append(box(cx, mid_y, colw, bh, "#f8fafc", "#cbd5e1", sw=2, dash="5,4"))
    L.append(text(cx + colw / 2, mid_y + 34, "（无昇腾中间层）", 13, fill="#94a3b8"))
    L.append(text(cx + colw / 2, mid_y + 56, "直接继承 vLLM 基类", 11.5, fill="#94a3b8"))
    node(cx, bot_y, "AscendKVBlockZeroer310 / …310", "两层直继承", "#d1fae5", "#059669", "#047857")
    # 实线继承 top → bot，绕过中层（从 top 底直连 bot 顶，穿过虚框间隙在两侧）
    L.append(vline(cx + colw / 2, top_y + bh, bot_y, "#475569", 2.4))

    # ---- 底部：310 这层改了什么 ----
    note_y = 566
    L.append(box(40, note_y, W - 80, 158, "#fffbeb", "#f59e0b", rx=12, sw=2))
    L.append(text(W / 2, note_y + 28, "310 这层各自改了什么（差异集中在「无 Triton / 无 MLA / 受限 dtype·格式」）", 15, weight="bold", fill="#b45309"))
    items = [
        "NPUModelRunner310：slot_mapping 与 decode 元数据 Triton → CPU NumPy；KV 钉死 FRACTAL_NZ；换 310 sampler",
        "NPUInputBatch310：唯一改动——__init__ 末尾把 self.block_table 换成 _310p 的 MultiGroupBlockTable",
        "BlockTable(310)：覆写 compute_slot_mapping，Triton kernel → NumPy 在 CPU 算 slot_mapping",
        "AscendKVBlockZeroer310：去 Triton，收集 (k,v) 张量列表 + 逐块切片 .zero_()",
        "ShardedStateLoader310：永远单 part（去 max_size 分片）+ 额外产出 parameters_type_map.json",
    ]
    for i, s in enumerate(items):
        L.append(text(70, note_y + 56 + i * 21, "• " + s, 12.5, anchor="start", fill="#78350f"))

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


# ----------------------------------------------------------------------------
# 图 2：slot_mapping Triton vs NumPy
# ----------------------------------------------------------------------------
def diagram_slot_mapping(path):
    W, H = 1160, 740
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>' + defs_arrow("ab", "#2563eb") + defs_arrow("ag", "#059669") + '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 40, "同一个 slot 公式，两条执行路径", 24, weight="bold"))
    L.append(text(W / 2, 68, "slot = block_number × block_size + (position mod block_size)", 14, fill="#64748b", mono=True))

    colw = 500
    lx, rx = 40, W - 40 - colw
    top = 100
    bh, gap = 70, 44

    def flow(x, title, sub, accent, accent_bg, marker, steps, hint):
        nonlocal L
        L.append(box(x, top, colw, 56, accent_bg, accent, rx=10, sw=2.5))
        L.append(text(x + colw / 2, top + 24, title, 16, weight="bold", fill=accent))
        L.append(text(x + colw / 2, top + 44, sub, 12, fill="#475569"))
        y = top + 56 + gap
        for i, (label, code) in enumerate(steps):
            L.append(box(x, y, colw, bh, "#ffffff", accent, rx=8, sw=1.6))
            L.append(text(x + 22, y + 27, label, 13, anchor="start", weight="bold", fill=accent))
            L.append(text(x + 22, y + 50, code, 11.5, anchor="start", fill="#334155", mono=True))
            if i < len(steps) - 1:
                L.append(vline(x + colw / 2, y + bh, y + bh + gap, accent, 2.2, marker=marker))
            y += bh + gap
        L.append(box(x, y + 6, colw, 44, accent_bg, accent, rx=8, sw=2))
        L.append(text(x + colw / 2, y + 33, hint, 12.5, weight="bold", fill=accent))
        return y + 50

    steps_l = [
        ("输入在 NPU 上", "positions / query_start_loc  (device tensor)"),
        ("Triton kernel 启动", "_compute_slot_mapping_kernel[(num_reqs+1,)](…)"),
        ("直接写 device", "→ self.slot_mapping.gpu"),
    ]
    flow(lx, "基座路径（910B / 主栈）", "vllm_ascend/worker/block_table.py", "#2563eb", "#dbeafe", "ab",
         steps_l, "kernel 读 block_table.gpu → 天然形成 NPU 流序")

    steps_r = [
        ("输入强制在 CPU", "_to_numpy：device 张量直接 raise"),
        ("NumPy 在 host 算", "np.add(block_numbers*block_size, offsets, out=…)"),
        ("算完再上传", "slot_mapping.np → copy_to_gpu(num_tokens)"),
    ]
    flow(rx, "310P 路径（无 Triton）", "vllm_ascend/_310p/block_table.py", "#059669", "#d1fae5", "ag",
         steps_r, "丢了 kernel 的隐式流序 → 需手动补回")

    # 底部贯通注释
    ny = H - 78
    L.append(box(40, ny, W - 80, 58, "#fffbeb", "#f59e0b", rx=10, sw=2))
    L.append(text(W / 2, ny + 24, "减法引出的补偿：_update_states 在 condense（move_row 改写 block_table.np）的步骤手动 synchronize", 13.5, weight="bold", fill="#b45309"))
    L.append(text(W / 2, ny + 46, "torch.npu.current_stream().synchronize()  —— 仅在布局变更步补回基座靠 Triton kernel 隐式得到的流序", 12, fill="#92400e", mono=True))

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


# ----------------------------------------------------------------------------
# 图 3：is_310p() 横切四域
# ----------------------------------------------------------------------------
def diagram_horizontal_cuts(path):
    W, H = 1220, 580
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>' + defs_arrow("ar", "#475569") + '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 40, "一个布尔分流，散落四个横切域", 24, weight="bold"))
    L.append(text(W / 2, 68, "差异尽量收进 _310p/，外部文件只留一行 is_310p() 判断", 14, fill="#64748b"))

    # 中心节点（顶部居中）
    ccx, ccy = W / 2, 130
    cw, ch = 380, 74
    L.append(box(ccx - cw / 2, ccy - ch / 2, cw, ch, "#ede9fe", "#7c3aed", rx=14, sw=3))
    L.append(text(ccx, ccy - 4, "is_310p()", 22, weight="bold", fill="#6d28d9", mono=True))
    L.append(text(ccx, ccy + 22, "SOC \"Ascend310P3\" → _310P 枚举（构建期烧进）", 11.5, fill="#5b21b6"))

    cards = [
        ("platform", "vllm_ascend/platform.py", "#2563eb", "#dbeafe", [
            "worker_cls = NPUWorker310",
            "（整条 310 栈入口）",
            "不开 custom_ops=['all']",
        ]),
        ("attention", "get_attn_backend_cls", "#0891b2", "#cffafe", [
            "backend_map_310 →",
            "AscendAttentionBackend310",
            "MLA / SFA 注释掉 = 不支持",
        ]),
        ("distributed", "patch_distributed.py", "#059669", "#d1fae5", [
            "broadcast：all_gather 取 src",
            "int64 all_reduce：",
            "all_gather + 本地 sum/max",
        ]),
        ("ops", "REGISTERED_ASCEND_OPS", "#db2777", "#fce7f3", [
            "rotary_embedding / sampler",
            "换 310 同形版",
            "运行期按 is_310p() 选实现",
        ]),
    ]
    n = len(cards)
    bw, bh = 270, 178
    margin = 28
    gap = (W - 2 * margin - n * bw) / (n - 1)
    card_top = 300
    for i, (name, mod, accent, accent_bg, lines) in enumerate(cards):
        x = margin + i * (bw + gap)
        y = card_top
        L.append(box(x, y, bw, bh, accent_bg, accent, rx=12, sw=2.4))
        L.append(text(x + bw / 2, y + 30, name, 17, weight="bold", fill=accent))
        L.append(text(x + bw / 2, y + 52, mod, 11, fill="#475569", mono=True))
        L.append(f'<line x1="{x+18}" y1="{y+64}" x2="{x+bw-18}" y2="{y+64}" stroke="{accent}" stroke-width="1" opacity="0.4"/>')
        for j, s in enumerate(lines):
            L.append(text(x + 18, y + 90 + j * 24, s, 12, anchor="start", fill="#334155"))
        # 从中心底边向各卡片顶边中点连线
        tx = x + bw / 2
        L.append(f'<line x1="{ccx}" y1="{ccy + ch/2}" x2="{tx}" y2="{y}" stroke="#475569" '
                 f'stroke-width="2" marker-end="url(#ar)"/>')

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


# ----------------------------------------------------------------------------
# 图 4：KV cache 三处受限
# ----------------------------------------------------------------------------
def diagram_kv_constraints(path):
    W, H = 1060, 720
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    L.append('<defs>' + defs_arrow("ar", "#475569") + '</defs>')
    L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    L.append(text(W / 2, 40, "受限硬件的 KV cache：三处约束写进分配流程", 23, weight="bold"))
    L.append(text(W / 2, 66, "initialize_kv_cache_tensors → _allocate_kv_cache_tensors → AscendKVBlockZeroer310", 13, fill="#64748b", mono=True))

    cx = W / 2
    bw, bh = 620, 64
    x = cx - bw / 2
    steps = [
        ("kv_cache_spec", "进入 initialize_kv_cache_tensors", "#475569", "#f1f5f9", None),
        ("① 拒绝不支持的特性", "KV-transfer / Deepseek-Sparse / MLA → raise ValueError", "#dc2626", "#fee2e2", "格式·能力"),
        ("② 筛 kernel block_size", "support_size × head_size ≤ 128×128（_ATTENTION_BLOCK_SIZE_LIMIT）", "#d97706", "#fef3c7", "对齐"),
        ("③ 专用格式分配", "torch_npu.empty_with_format(acl_format=FRACTAL_NZ)", "#2563eb", "#dbeafe", "格式"),
        ("得到 (k_cache, v_cache)", "K / V 分开分配（非单 Tensor）", "#0891b2", "#cffafe", None),
        ("KV 清零去 Triton", "init_meta 收集 (k,v) 列表 → zero_block_ids 切片 kv[start:end].zero_()", "#059669", "#d1fae5", None),
    ]
    top = 104
    gap = 36
    y = top
    for i, (title, sub, accent, bg, tag) in enumerate(steps):
        L.append(box(x, y, bw, bh, bg, accent, rx=10, sw=2.2))
        L.append(text(x + 24, y + 27, title, 14.5, anchor="start", weight="bold", fill=accent))
        L.append(text(x + 24, y + 49, sub, 11.5, anchor="start", fill="#334155", mono=True))
        if tag:
            tagw = 92
            L.append(box(x + bw - tagw - 12, y + 14, tagw, 36, "#ffffff", accent, rx=8, sw=1.6))
            L.append(text(x + bw - tagw / 2 - 12, y + 37, tag, 12, weight="bold", fill=accent))
        if i < len(steps) - 1:
            L.append(vline(cx, y + bh, y + bh + gap, "#475569", 2.2))
        y += bh + gap

    L.append('</svg>')
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    d = os.path.dirname(os.path.abspath(__file__))
    diagram_inheritance(os.path.join(d, "inheritance-depth-by-component.svg"))
    diagram_slot_mapping(os.path.join(d, "slot-mapping-triton-vs-numpy.svg"))
    diagram_horizontal_cuts(os.path.join(d, "310p-horizontal-cuts.svg"))
    diagram_kv_constraints(os.path.join(d, "kv-cache-310-constraints.svg"))
    print("done")
