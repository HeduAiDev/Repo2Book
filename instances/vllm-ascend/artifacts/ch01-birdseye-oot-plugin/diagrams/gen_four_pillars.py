#!/usr/bin/env python3
"""OOT 四要素总览：从安装期声明到运行期分发的一条链 + 两段式 patch 侧支。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 940
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7.5" markerHeight="5.5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7.5" markerHeight="5.5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '<marker id="ap" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7.5" markerHeight="5.5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#9333ea"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="42" font-family="sans-serif" font-size="25" font-weight="bold" fill="#0f172a" text-anchor="middle">OOT 插件四要素：装上就被发现 · 一个平台类接管分发 · 改不动处打补丁 · 每个扩展点登记昇腾实现</text>')

# 两列表头
L.append(f'<text x="360" y="86" font-family="sans-serif" font-size="17" font-weight="bold" fill="#b45309" text-anchor="middle">vllm-ascend 侧（登记实现）</text>')
L.append(f'<text x="960" y="86" font-family="sans-serif" font-size="17" font-weight="bold" fill="#2563eb" text-anchor="middle">vLLM 侧（预留扩展点 · 反向钩入）</text>')
L.append(f'<line x1="660" y1="100" x2="660" y2="{H-30}" stroke="#e2e8f0" stroke-width="1.5" stroke-dasharray="4 5"/>')

AL, AR = 130, 590   # ascend 列左右
VL, VR = 730, 1190  # vLLM 列左右


def box(x, y, w, h, title, sub, col, bg, mono=True):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{bg}" stroke="{col}" stroke-width="2"/>')
    tf = "monospace" if mono else "sans-serif"
    L.append(f'<text x="{x+w/2}" y="{y+27}" font-family="{tf}" font-size="15" font-weight="bold" fill="{col}" text-anchor="middle">{esc(title)}</text>')
    if sub:
        L.append(f'<text x="{x+w/2}" y="{y+49}" font-family="sans-serif" font-size="12.5" fill="#475569" text-anchor="middle">{esc(sub)}</text>')
    return (x, y, w, h)


def arrow(x1, y1, x2, y2, col="#2563eb", mk="a", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2.4"{d} marker-end="url(#{mk})"/>')


def label(x, y, t, col="#64748b", anchor="middle", size=12):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" fill="{col}" text-anchor="{anchor}">{esc(t)}</text>')


# ---- 支柱一 band ----
L.append(f'<rect x="40" y="108" width="{W-80}" height="150" rx="12" fill="#fffbeb" stroke="#fde68a" stroke-width="1.5"/>')
L.append(f'<text x="58" y="130" font-family="sans-serif" font-size="14" font-weight="bold" fill="#b45309">支柱一 · 安装期挂入（entry points）</text>')
a1 = box(AL, 150, AR - AL, 70, "setup.py  entry_points", "vllm.platform_plugins / vllm.general_plugins", "#b45309", "#fef3c7")
v1 = box(VL, 150, VR - VL, 70, "plugins.load_plugins_by_group", "importlib.metadata.entry_points(group=…)", "#2563eb", "#eff6ff")
arrow(a1[0] + a1[2], 185, v1[0], 185)
label(660, 176, "pip install", "#b45309")
label(660, 202, "按组名反查", "#2563eb")

# ---- 支柱二 band ----
L.append(f'<rect x="40" y="278" width="{W-80}" height="330" rx="12" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.5"/>')
L.append(f'<text x="58" y="300" font-family="sans-serif" font-size="14" font-weight="bold" fill="#1d4ed8">支柱二 · 运行期分发（一个 NPUPlatform 接管所有「问平台要组件」）</text>')

a2 = box(AL, 318, AR - AL, 66, "register()", "return \"vllm_ascend.platform.NPUPlatform\"", "#b45309", "#fef3c7")
v2 = box(VL, 318, VR - VL, 66, "resolve_current_platform_cls_qualname", "probe 激活 · OOT 优先于 builtin", "#2563eb", "#dbeafe")
arrow(a2[0] + a2[2], 351, v2[0], 351)
label(660, 344, "只返回字符串", "#b45309")
label(660, 366, "选中 OOT", "#2563eb")

v3 = box(VL, 408, VR - VL, 62, "current_platform（模块级 __getattr__）", "首次访问才 resolve + 实例化（懒加载）", "#2563eb", "#dbeafe")
arrow((v2[0] + v2[2] / 2), v2[1] + v2[3], (v3[0] + v3[2] / 2), v3[1])

a3 = box(AL, 486, AR - AL, 104, "class NPUPlatform(Platform)", "", "#b45309", "#fef3c7")
label(AL + (AR - AL) / 2, 538, "_enum = PlatformEnum.OOT", "#475569", size=12.5)
label(AL + (AR - AL) / 2, 560, "get_attn_backend_cls / get_device_communicator_cls …", "#475569", size=11.5)
label(AL + (AR - AL) / 2, 578, "每个钩子返回一个昇腾类的 qualname", "#94a3b8", size=11)
# v3 → NPUPlatform 实例化
arrow(v3[0], v3[1] + v3[3] / 2, a3[0] + a3[2], a3[1] + 20, "#2563eb", "a")
label(660, 452, "实例化", "#2563eb")

# ---- 支柱三 band ----
L.append(f'<rect x="40" y="628" width="{W-80}" height="220" rx="12" fill="#faf5ff" stroke="#e9d5ff" stroke-width="1.5"/>')
L.append(f'<text x="58" y="650" font-family="sans-serif" font-size="14" font-weight="bold" fill="#7e22ce">支柱三 · 两段式 monkey-patch（给没有工厂钩子的地方）</text>')

a4 = box(AL, 668, 300, 60, "pre_register_and_update()", "平台被选定后首个回调", "#7e22ce", "#f3e8ff")
a5 = box(AL, 764, 300, 60, "每个 worker.__init__", "worker 启动时", "#7e22ce", "#f3e8ff")
p1 = box(478, 668, 336, 60, "adapt_patch(is_global_patch=True)", "import patch.platform（副作用）", "#7e22ce", "#faf5ff")
p2 = box(478, 764, 336, 60, "adapt_patch(is_global_patch=False)", "import patch.worker（副作用）", "#7e22ce", "#faf5ff")
arrow(a4[0] + a4[2], 698, p1[0], 698, "#9333ea", "ap")
arrow(a5[0] + a5[2], 794, p2[0], 794, "#9333ea", "ap")
box(858, 668, VR - 858, 60, "engine-core / scheduler 层符号", "platform 段：约 25 条改写", "#7e22ce", "#f3e8ff", mono=False)
box(858, 764, VR - 858, 60, "worker / 模型 / 算子层符号", "worker 段：约 22 条改写", "#7e22ce", "#f3e8ff", mono=False)
arrow(p1[0] + p1[2], 698, 858, 698, "#9333ea", "ap")
arrow(p2[0] + p2[2], 794, 858, 794, "#9333ea", "ap")

# 支柱二 → 支柱三 连线：NPUPlatform 触发 platform 段
arrow(a3[0] + 60, a3[1] + a3[3], a4[0] + 60, a4[1], "#9333ea", "ap")
label(AL + 150, 620, "① platform 段", "#7e22ce", "start", 11.5)

# 底注：第四要素
L.append(f'<rect x="40" y="866" width="{W-80}" height="52" rx="10" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1.5"/>')
L.append(f'<text x="{W/2}" y="898" font-family="sans-serif" font-size="14.5" font-weight="bold" fill="#15803d" text-anchor="middle">要素四 · 往每个扩展点登记昇腾实现：get_*_cls 返回的每个类名，就是后面各章 zoom-in 的昇腾实现（注意力 / 通信 / 编译 / worker …）</text>')

L.append('</svg>')
open("four_pillars.svg", "w").write("\n".join(L))
print("wrote four_pillars.svg")
