#!/usr/bin/env python3
"""Generate ch18 diagrams: three-key routing table, contract対账 map, resolution flow."""
import os, subprocess
import xml.sax.saxutils as xs

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return xs.escape(s)


def box(x, y, w, h, fill, stroke, rx=6, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=13, fill="#1e293b", anchor="middle", weight="normal", mono=False, italic=False):
    fam = "monospace" if mono else "sans-serif"
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}"{st}>{esc(s)}</text>')


def arrow(x1, y1, x2, y2, color="#64748b", sw=2, dash=None, marker="a"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def defs():
    return ('<defs>'
            '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
            '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
            '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
            '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
            '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
            '<path d="M0,0 L10,3 L0,6 Z" fill="#0d9488"/></marker>'
            '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
            '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
            '</defs>')


def save(name, svg):
    sp = os.path.join(HERE, name + ".svg")
    pp = os.path.join(HERE, name + ".png")
    with open(sp, "w") as f:
        f.write(svg)
    assert subprocess.run(["xmllint", "--noout", sp]).returncode == 0, name
    subprocess.run(["rsvg-convert", "-z", "2", sp, "-o", pp], check=True)
    print("wrote", pp)


# ── Diagram 1: three-key routing table ───────────────────────────────
def routing_table():
    w, h = 1040, 600
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', defs(),
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w/2, 38, "三元 key 路由表：(use_mla, use_sparse, use_compress) → 4 个昇腾后端", 19, "#0f172a", weight="bold"))
    L.append(text(w/2, 62, "NPUPlatform.get_attn_backend_cls 查 backend_map，返回点分类路径字符串", 13, "#475569"))

    # key column headers
    kx = [150, 290, 430]          # column centers for three key dims
    head_y = 100
    L.append(text(kx[0], head_y, "use_mla", 13.5, "#1d4ed8", weight="bold", mono=True))
    L.append(text(kx[1], head_y, "use_sparse", 13.5, "#1d4ed8", weight="bold", mono=True))
    L.append(text(kx[2], head_y, "use_compress", 13.5, "#1d4ed8", weight="bold", mono=True))

    rows = [
        ((True, False, False), "AscendMLABackend", "MLA · 多头潜在注意力", "#1e40af", "#dbeafe", "#3b82f6"),
        ((False, False, False), "AscendAttentionBackend", "标准 MHA（本章样板）", "#6d28d9", "#ede9fe", "#8b5cf6"),
        ((True, True, False), "AscendSFABackend", "稀疏 SFA", "#0f766e", "#ccfbf1", "#14b8a6"),
        ((True, False, True), "AscendDSABackend", "DSA（需 use_compress 字段）", "#9a3412", "#ffedd5", "#fb923c"),
    ]
    row_y0 = 126
    rh = 64
    cellw = 78
    for i, (key, cls, label, tcol, fill, stroke) in enumerate(rows):
        cy = row_y0 + i * rh
        # key cells
        for j, v in enumerate(key):
            vfill = "#dcfce7" if v else "#fee2e2"
            vstroke = "#22c55e" if v else "#ef4444"
            vtxt = "True" if v else "False"
            L.append(box(kx[j] - cellw/2, cy, cellw, 40, vfill, vstroke, rx=6, sw=1.5))
            L.append(text(kx[j], cy + 25, vtxt, 13, "#166534" if v else "#991b1b", weight="bold", mono=True))
        # arrow
        L.append(arrow(kx[2] + cellw/2 + 6, cy + 20, 600, cy + 20, color=stroke, sw=2.2))
        # backend box
        L.append(box(610, cy - 2, 380, 44, fill, stroke, rx=8, sw=2))
        L.append(text(800, cy + 16, cls, 14, tcol, weight="bold", mono=True))
        L.append(text(800, cy + 33, label, 11.5, tcol))

    # dim hint under each header column bracket
    L.append(text(290, row_y0 + 4 * rh + 14, "绿=True　红=False", 11.5, "#64748b"))

    # side branches
    sb_y = row_y0 + 4 * rh + 40
    L.append(box(150, sb_y, 360, 56, "#f8fafc", "#cbd5e1", rx=8, sw=1.5, dash="5,4"))
    L.append(text(330, sb_y + 22, "旁支①  FA3：selected_backend==FLASH_ATTN", 12, "#475569", weight="bold", mono=True))
    L.append(text(330, sb_y + 41, "且 _validate_fa3_backend 通过 → AscendFABackend", 11.5, "#64748b"))
    L.append(box(540, sb_y, 360, 56, "#f8fafc", "#cbd5e1", rx=8, sw=1.5, dash="5,4"))
    L.append(text(720, sb_y + 22, "旁支②  310p：is_310p() 为真", 12, "#475569", weight="bold", mono=True))
    L.append(text(720, sb_y + 41, "→ 走 backend_map_310（推理卡专表）", 11.5, "#64748b"))

    L.append(f'</svg>')
    save("routing-table", "\n".join(L))


# ── Diagram 2: contract対账 map ──────────────────────────────────────
def contract_map():
    w, h = 1080, 660
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', defs(),
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w/2, 36, "契约对账：vLLM AttentionBackend  ↔  AscendAttentionBackend", 19, "#0f172a", weight="bold"))
    L.append(text(w/2, 60, "哪些是 v1 强制契约、哪些是昇腾额外保留", 13, "#475569"))

    lx, lw = 50, 420
    rx, rw = 610, 420
    L.append(text(lx + lw/2, 92, "vLLM 侧契约（vllm/v1/attention/backend.py）", 14, "#1d4ed8", weight="bold"))
    L.append(text(rx + rw/2, 92, "昇腾落点（vllm_ascend/.../attention_v1.py）", 14, "#6d28d9", weight="bold"))

    # group A label
    L.append(box(lx, 108, lw, 24, "#dbeafe", "#3b82f6", rx=5, sw=1.5))
    L.append(text(lx + lw/2, 125, "组 A · 4 个 @abstractmethod（必须实现）", 12.5, "#1e3a8a", weight="bold"))

    methods = [
        ("get_name()", "→ 伪装返回 \"FLASH_ATTN\"  ⚠ HACK", True),
        ("get_impl_cls()", "→ 按 enable_cp() 运行期二选一", False),
        ("get_builder_cls()", "→ 按 enable_cp() 运行期二选一", False),
        ("get_kv_cache_shape()", "→ (2, num_blocks, block_size, …)", False),
    ]
    my = 140
    mh = 50
    for i, (m,落点, hack) in enumerate(methods):
        cy = my + i * (mh + 8)
        # left
        L.append(box(lx, cy, lw, mh, "#eff6ff", "#93c5fd", rx=7, sw=1.5))
        L.append(text(lx + 16, cy + 30, m, 14, "#1e3a8a", weight="bold", mono=True, anchor="start"))
        # right
        rfill = "#fef2f2" if hack else "#f5f3ff"
        rstroke = "#dc2626" if hack else "#c4b5fd"
        L.append(box(rx, cy, rw, mh, rfill, rstroke, rx=7, sw=2 if hack else 1.5))
        tcol = "#b91c1c" if hack else "#5b21b6"
        L.append(text(rx + 16, cy + 30, 落点, 12.5, tcol, weight="bold" if hack else "normal", anchor="start"))
        # connector
        col = "#dc2626" if hack else "#8b5cf6"
        mk = "r" if hack else "b"
        L.append(arrow(lx + lw + 4, cy + mh/2, rx - 4, cy + mh/2, color=col, sw=2, marker=mk))

    # group B label
    gby = my + 4 * (mh + 8) + 6
    L.append(box(lx, gby, lw, 24, "#fef9c3", "#eab308", rx=5, sw=1.5))
    L.append(text(lx + lw/2, gby + 17, "组 B · 基类自带默认实现（可覆写，非 abstract）", 12.5, "#854d0e", weight="bold"))
    cy = gby + 32
    L.append(box(lx, cy, lw, mh, "#fefce8", "#fde047", rx=7, sw=1.5))
    L.append(text(lx + 16, cy + 22, "get_supported_kernel_block_sizes()", 13, "#854d0e", weight="bold", mono=True, anchor="start"))
    L.append(text(lx + 16, cy + 40, "默认 [MultipleOf(1)]", 11.5, "#a16207", anchor="start"))
    L.append(box(rx, cy, rw, mh, "#fefce8", "#fde047", rx=7, sw=1.5))
    L.append(text(rx + 16, cy + 30, "→ 覆写返回 [128]", 13, "#854d0e", weight="bold", anchor="start"))
    L.append(arrow(lx + lw + 4, cy + mh/2, rx - 4, cy + mh/2, color="#eab308", sw=2, marker="a"))

    # group C: ascend-only, no v1 contract
    gcy = cy + mh + 18
    L.append(box(lx, gcy + 14, lw, mh + 16, "#f1f5f9", "#94a3b8", rx=7, sw=1.5, dash="5,4"))
    L.append(text(lx + lw/2, gcy + 38, "（v1 基类无此契约点）", 13, "#64748b", weight="bold"))
    L.append(text(lx + lw/2, gcy + 58, "基座 vllm/v1/attention/ 全目录无 def、无调用", 11, "#94a3b8"))
    L.append(box(rx, gcy, rw, mh + 44, "#f1f5f9", "#475569", rx=7, sw=2))
    L.append(text(rx + rw/2, gcy + 22, "组 C · 昇腾额外保留（v0 遗留接口）", 12.5, "#334155", weight="bold"))
    L.append(text(rx + 16, gcy + 46, "swap_blocks() / copy_blocks()", 13.5, "#1e293b", weight="bold", mono=True, anchor="start"))
    L.append(text(rx + 16, gcy + 66, "按 (2, …) 布局做块级索引搬运", 11.5, "#475569", anchor="start"))
    # no arrow from left — annotate the gap
    L.append(text((lx + lw + rx) / 2, gcy + 40, "✗", 22, "#94a3b8", weight="bold"))

    L.append(f'</svg>')
    save("contract-map", "\n".join(L))


# ── Diagram 3: resolution flow ───────────────────────────────────────
def resolution_flow():
    w, h = 1080, 540
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', defs(),
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w/2, 36, "OOT 后端被接进来的两条线：占位声明 vs 真正解析路径", 19, "#0f172a", weight="bold"))

    # main resolution chain (vertical)
    cx = 300
    bw, bh = 460, 52
    steps = [
        ("selector.get_attn_backend(…)", "vLLM 组装 AttentionSelectorConfig（含 use_mla/use_sparse）", "#eff6ff", "#3b82f6", "#1e3a8a"),
        ("_cached_get_attn_backend(…)  @cache", "调 current_platform.get_attn_backend_cls(…)", "#eff6ff", "#3b82f6", "#1e3a8a"),
        ("NPUPlatform.get_attn_backend_cls(…)", "三元 key 查 backend_map → 返回点分类路径字符串", "#ede9fe", "#8b5cf6", "#5b21b6"),
        ("resolve_obj_by_qualname(\"vllm_ascend.…\")", "importlib 延迟解析字符串 → 后端类对象", "#eff6ff", "#3b82f6", "#1e3a8a"),
        ("AscendAttentionBackend  类", "vLLM 拿到后端类，后续调 get_name/get_impl_cls…", "#ccfbf1", "#14b8a6", "#0f766e"),
    ]
    y0 = 80
    gap = 86
    for i, (title, sub, fill, stroke, tcol) in enumerate(steps):
        cy = y0 + i * gap
        L.append(box(cx - bw/2, cy, bw, bh, fill, stroke, rx=8, sw=2))
        L.append(text(cx, cy + 22, title, 13.5, tcol, weight="bold", mono=True))
        L.append(text(cx, cy + 40, sub, 11.5, tcol))
        if i < len(steps) - 1:
            L.append(arrow(cx, cy + bh, cx, cy + gap, color=stroke, sw=2.4))

    L.append(text(cx, y0 + 5 * gap - 8, "主解析路径（运行期，按需触发）", 12.5, "#475569", weight="bold"))

    # side: register_backend placeholder (dashed, parallel)
    sx = 800
    sw_ = 240
    L.append(box(sx, 95, sw_, 320, "#fafafa", "#a8a29e", rx=10, sw=1.8, dash="6,5"))
    L.append(text(sx + sw_/2, 122, "导入期 · 占位声明", 13.5, "#57534e", weight="bold"))
    L.append(box(sx + 20, 145, sw_ - 40, 72, "#f5f5f4", "#78716c", rx=7, sw=1.5))
    L.append(text(sx + sw_/2, 170, "@register_backend(", 12, "#44403c", weight="bold", mono=True))
    L.append(text(sx + sw_/2, 188, "  CUSTOM, \"ASCEND\")", 12, "#44403c", weight="bold", mono=True))
    L.append(text(sx + sw_/2, 206, "装饰 AscendAttentionBackend", 10.5, "#57534e"))
    L.append(arrow(sx + sw_/2, 217, sx + sw_/2, 248, color="#a8a29e", sw=2, dash="5,4"))
    L.append(box(sx + 20, 250, sw_ - 40, 64, "#f5f5f4", "#78716c", rx=7, sw=1.5))
    L.append(text(sx + sw_/2, 274, "_ATTN_OVERRIDES[CUSTOM]", 11, "#44403c", weight="bold", mono=True))
    L.append(text(sx + sw_/2, 292, "= \"ASCEND\"", 11, "#44403c", weight="bold", mono=True))
    L.append(text(sx + sw_/2, 308, "占住 CUSTOM 槽，装饰器 no-op", 10, "#57534e"))
    L.append(box(sx + 20, 332, sw_ - 40, 66, "#fffbeb", "#d97706", rx=7, sw=1.5))
    L.append(text(sx + sw_/2, 354, "声明「CUSTOM 由昇腾接管」", 10.5, "#92400e", weight="bold"))
    L.append(text(sx + sw_/2, 372, "但本版不靠它解析后端——", 10.5, "#92400e"))
    L.append(text(sx + sw_/2, 388, "真正解析走左侧点分路径", 10.5, "#92400e"))

    L.append(f'</svg>')
    save("resolution-flow", "\n".join(L))


if __name__ == "__main__":
    routing_table()
    contract_map()
    resolution_flow()
