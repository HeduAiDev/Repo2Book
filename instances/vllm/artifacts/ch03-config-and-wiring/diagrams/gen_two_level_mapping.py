#!/usr/bin/env python3
"""ch03 figure: two-level mapping — flat EngineArgs -> structured VllmConfig -> impl classes."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
         'markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# Column headers
cols = [
    (40, "扁平参数", "EngineArgs（CLI 风格）", "#0e7490"),
    (440, "结构化配置", "VllmConfig（聚合体）", "#7c3aed"),
    (860, "实现选择", "具体类（推迟实例化）", "#b45309"),
]
for x, title, sub, color in cols:
    L.append(f'<text x="{x}" y="40" font-size="20" font-weight="bold" fill="{color}">{esc(title)}</text>')
    L.append(f'<text x="{x}" y="62" font-size="13" fill="#64748b">{esc(sub)}</text>')

# Left column: flat args (single source of truth note)
left = [
    "model = ModelConfig.model",
    "tensor_parallel_size = ...",
    "optimization_level = O2",
    "async_scheduling = None",
    "enforce_eager = False",
]
lx, ly0, lw, lh, lgap = 40, 80, 360, 38, 12
for i, txt in enumerate(left):
    y = ly0 + i * (lh + lgap)
    L.append(f'<rect x="{lx}" y="{y}" width="{lw}" height="{lh}" rx="6" '
             f'fill="#ecfeff" stroke="#0e7490" stroke-width="1.4"/>')
    L.append(f'<text x="{lx + 12}" y="{y + 25}" font-size="14" '
             f'fill="#0f172a" font-family="monospace">{esc(txt)}</text>')
left_bottom = ly0 + len(left) * (lh + lgap)
L.append(f'<text x="{lx}" y="{left_bottom + 14}" font-size="11" fill="#64748b">'
         f'每个默认值借自子 Config 同名属性</text>')
L.append(f'<text x="{lx}" y="{left_bottom + 30}" font-size="11" fill="#64748b">'
         f'（单一真相源，CLI 与子配置不漂移）</text>')

# Middle column: VllmConfig box containing sub-configs
mx, my, mw, mh = 440, 80, 360, 440
L.append(f'<rect x="{mx}" y="{my}" width="{mw}" height="{mh}" rx="10" '
         f'fill="#faf5ff" stroke="#7c3aed" stroke-width="2"/>')
L.append(f'<text x="{mx + mw // 2}" y="{my + 24}" text-anchor="middle" '
         f'font-size="15" font-weight="bold" fill="#7c3aed">VllmConfig</text>')
subs = ["ModelConfig", "CacheConfig", "ParallelConfig",
        "SchedulerConfig", "CompilationConfig", "KernelConfig"]
sx, sy0, sw, sh, sgap = mx + 24, my + 44, mw - 48, 34, 10
for i, txt in enumerate(subs):
    y = sy0 + i * (sh + sgap)
    L.append(f'<rect x="{sx}" y="{y}" width="{sw}" height="{sh}" rx="5" '
             f'fill="white" stroke="#a78bfa" stroke-width="1.3"/>')
    L.append(f'<text x="{sx + 10}" y="{y + 22}" font-size="13" '
             f'fill="#0f172a" font-family="monospace">{esc(txt)}</text>')
postinit_y = sy0 + len(subs) * (sh + sgap)
L.append(f'<rect x="{sx}" y="{postinit_y}" width="{sw}" height="{sh + 8}" rx="5" '
         f'fill="#fef9c3" stroke="#ca8a04" stroke-width="1.3"/>')
L.append(f'<text x="{sx + 10}" y="{postinit_y + 18}" font-size="12.5" '
         f'fill="#854d0e">__post_init__：交叉校验 + 推导</text>')
L.append(f'<text x="{sx + 10}" y="{postinit_y + 33}" font-size="11" '
         f'fill="#854d0e">async 决策 / O0-O3 应用 / cudagraph 落定</text>')

# Right column: impl classes, three factory groups
rx, rw, rh = 860, 280, 34
groups = [
    ("Executor.get_class", ["UniProcExecutor", "MultiprocExecutor", "RayDistributedExecutor"], 104, "#b45309"),
    ("get_scheduler_cls", ["Scheduler", "AsyncScheduler"], 300, "#b45309"),
    ("make_client", ["InprocClient", "SyncMPClient", "AsyncMPClient"], 446, "#b45309"),
]
right_anchor_ys = []
for gname, items, gy, color in groups:
    L.append(f'<text x="{rx}" y="{gy - 6}" font-size="13" font-weight="bold" '
             f'fill="{color}" font-family="monospace">{esc(gname)}</text>')
    right_anchor_ys.append(gy + 6)
    for i, it in enumerate(items):
        y = gy + i * (rh + 8)
        bold = (i == 0)
        fillc = "#fff7ed" if bold else "white"
        L.append(f'<rect x="{rx}" y="{y}" width="{rw}" height="{rh}" rx="5" '
                 f'fill="{fillc}" stroke="#fb923c" stroke-width="1.3"/>')
        L.append(f'<text x="{rx + 10}" y="{y + 22}" font-size="13" '
                 f'fill="#0f172a" font-family="monospace">{esc(it)}</text>')

# Arrow: left -> middle (create_engine_config)
L.append(f'<line x1="{lx + lw}" y1="260" x2="{mx}" y2="260" '
         f'stroke="#475569" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(lx + lw + mx) // 2}" y="248" text-anchor="middle" '
         f'font-size="12.5" fill="#334155">create_engine_config()</text>')
L.append(f'<text x="{(lx + lw + mx) // 2}" y="278" text-anchor="middle" '
         f'font-size="11" fill="#64748b">第一级映射</text>')

# Arrows: middle -> right (three factories)
for ay in right_anchor_ys:
    L.append(f'<line x1="{mx + mw}" y1="{ay + 20}" x2="{rx}" y2="{ay + 10}" '
             f'stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<text x="{(mx + mw + rx) // 2}" y="60" text-anchor="middle" '
         f'font-size="12.5" fill="#334155">三个工厂（第二级映射）</text>')

# Footer note
L.append(f'<text x="40" y="{H - 18}" font-size="12" fill="#64748b">'
         f'两级映射：参数空间 → 配置空间 → 实现空间。下游模块只需传一个 VllmConfig 即拿到全部上下文。</text>')
L.append('</svg>')

with open("two-level-mapping.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote two-level-mapping.svg")
