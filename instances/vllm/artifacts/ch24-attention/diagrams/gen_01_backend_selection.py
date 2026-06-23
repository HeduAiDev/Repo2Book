#!/usr/bin/env python3
"""01-backend-abstraction-and-selection: 抽象层 + 注册表 + 选择三段。"""
import xml.sax.saxutils as xs

OUT = "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch24-attention/diagrams/01-backend-abstraction-and-selection.svg"


def esc(s):
    return xs.escape(str(s))


W, H = 1100, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, lines, fs=13, tcol="#0f172a", weight="normal"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * (fs + 3) / 2 + fs * 0.35
    for i, t in enumerate(lines):
        L.append(f'<text x="{x + w/2}" y="{cy + i*(fs+3)}" font-family="sans-serif" '
                 f'font-size="{fs}" font-weight="{weight}" text-anchor="middle" fill="{tcol}">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, col="#475569", marker="a"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
             f'stroke-width="1.8" marker-end="url(#{marker})"/>')


def label(x, y, t, fs=12, col="#475569", weight="normal", anchor="middle"):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{fs}" '
             f'font-weight="{weight}" text-anchor="{anchor}" fill="{col}">{esc(t)}</text>')


label(W/2, 30, "AttentionBackend 抽象 → 注册表懒加载 → 按配置选后端", 18, "#0f172a", "bold")

# ---- 段① 抽象层（左） ----
L.append('<rect x="20" y="56" width="340" height="540" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>')
label(190, 78, "① 抽象层：AttentionBackend", 14, "#1d4ed8", "bold")

box(40, 96, 300, 130, "#dbeafe", "#3b82f6", [
    "六个抽象 staticmethod（后端身份证）",
    "get_name / get_impl_cls",
    "get_builder_cls",
    "get_kv_cache_shape",
    "get_kv_cache_stride_order",
    "validate_configuration（能力聚合）",
], fs=12, weight="normal")

box(40, 246, 300, 60, "#dcfce7", "#22c55e",
    ["AttentionMetadataBuilder.build()", "Common → 后端专属 metadata"], fs=12)
box(40, 326, 300, 80, "#fef3c7", "#f59e0b",
    ["AttentionImpl", "forward（算+读 paged KV）", "do_kv_cache_update（写 paged KV）"], fs=12)
box(40, 426, 300, 60, "#ede9fe", "#8b5cf6",
    ["FlashAttentionBackend 等具体后端", "= 把六个方法落地实现"], fs=12, weight="bold")
arrow(190, 226, 190, 246)
arrow(190, 306, 190, 326)
arrow(190, 406, 190, 426)
label(190, 520, "选后端 / 问 KV shape / 问优先级", 11, "#64748b")
label(190, 538, "都发生在『还没实例化对象』阶段", 11, "#64748b")
label(190, 556, "→ 用 staticmethod 仅凭类就能查", 11, "#1d4ed8")

# ---- 段② 注册表（中） ----
L.append('<rect x="380" y="56" width="320" height="540" rx="10" fill="#fef2f2" stroke="#fecaca"/>')
label(540, 78, "② 注册表：AttentionBackendEnum", 14, "#b91c1c", "bold")

box(400, 96, 280, 120, "#fee2e2", "#ef4444", [
    "枚举值 = 默认类路径字符串：",
    "FLASH_ATTN = \"...flash_attn.",
    "  FlashAttentionBackend\"",
    "FLASHINFER = \"...\"",
    "TRITON_ATTN = \"...\"",
    "CUSTOM = None（待 register）",
], fs=11.5)
box(400, 236, 280, 70, "#fecaca", "#dc2626",
    ["get_class()", "= _ATTN_OVERRIDES 查覆盖表", "→ 回退枚举值"], fs=12, weight="bold")
box(400, 336, 280, 56, "#fff7ed", "#f97316",
    ["resolve_obj_by_qualname", "用到才真正 import（懒加载）"], fs=12)
box(400, 420, 280, 56, "#fff7ed", "#f97316",
    ["register_backend(...)", "运行时覆盖 / 注册第三方 CUSTOM"], fs=12)
arrow(540, 216, 540, 236)
arrow(540, 306, 540, 336)
label(540, 510, "重依赖（FlashInfer/FlashMLA）", 11, "#64748b")
label(540, 528, "若 import 全部后端会拖崩缺失依赖", 11, "#64748b")
label(540, 546, "→ 用到哪个才 import 哪个", 11, "#b91c1c")

# ---- 段③ 选择（右） ----
L.append('<rect x="720" y="56" width="360" height="540" rx="10" fill="#f0fdf4" stroke="#bbf7d0"/>')
label(900, 78, "③ 选择：get_attn_backend", 14, "#15803d", "bold")

box(740, 96, 320, 56, "#dcfce7", "#22c55e",
    ["get_attn_backend(head_size, dtype,", "kv_cache_dtype, use_mla, ...)"], fs=12, weight="bold")
box(740, 168, 320, 50, "#dcfce7", "#22c55e",
    ["AttentionSelectorConfig（可哈希键）", "→ @cache 同配置只解一次"], fs=12)
box(740, 234, 320, 56, "#bbf7d0", "#16a34a",
    ["platform.get_attn_backend_cls", "（按 compute capability 决策）"], fs=12, weight="bold")
box(740, 306, 320, 78, "#dcfce7", "#22c55e",
    ["显式 --attention-backend ?",
     "  是 → 只校验它，不合法直接报错",
     "  否 → 按优先级列表逐个",
     "       validate_configuration 过滤"], fs=11.5)
box(740, 400, 320, 50, "#bbf7d0", "#16a34a",
    ["取 invalid_reasons==[] 中优先级最高者", "→ 返回类路径字符串"], fs=11.5, weight="bold")
box(740, 466, 320, 50, "#fef3c7", "#f59e0b",
    ["若后端要求特定 KV layout（HND）", "→ set_kv_cache_layout 全局生效"], fs=11.5)
for y1, y2 in [(152, 168), (218, 234), (290, 306), (384, 400), (450, 466)]:
    arrow(900, y1, 900, y2, col="#15803d", marker="ag")

# 段间连线
arrow(360, 156, 400, 156)
label(380, 146, "实现", 10, "#64748b")
arrow(700, 264, 740, 262, col="#15803d", marker="ag")
label(720, 252, "查表", 10, "#64748b")

label(900, 548, "head_size / dtype / compute capability", 11, "#64748b")
label(900, 566, "→ 决定 FlashAttention / FlashInfer / Triton", 11, "#15803d")

L.append('</svg>')
open(OUT, "w").write('\n'.join(L))
print("ok", OUT)
