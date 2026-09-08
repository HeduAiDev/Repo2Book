#!/usr/bin/env python3
"""ch23 机制图 5 · 新旧两代模型布局（figure_spec ch23-fig-two-layouts，模板 before-after）

放大自本章 L2 章图拍片 ① registry 解析（站 1-2）× south 新布局格 · L0：GPU 执行臂 × 模型层框。

claim: 同一张 registry 表里两代布局并存——DeepseekV32 走扁平相对名（'deepseek_v2' →
前台拼 vllm.model_executor.models. 前缀）、DeepseekV4 走 vllm. 开头全限定路径
（vllm.models.deepseek_v4，_resolve_module_name 原样放行），新布局 __init__.py 按
current_platform 三分发 nvidia/amd/xpu 子包。

数字全部取自 figure_spec.numbers（两代布局并存 / 三分支平台分发 / 同款全限定条目与
三类出口 / import 只在 load_model_cls 一刻）。坐标由常量/循环计算；文本全 esc()。
行号基线 vLLM v0.27.1（registry.py / vllm/models/deepseek_v4/__init__.py）。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'book' / 'cartography'))
import l0_common as lc  # noqa: E402

lc.reset()

W, H = 1600, 896
MX = 56
BXR = 1544

# ---------------- 标题区 ----------------
lc.text(MX, 34, '同一张 registry 表，两代布局并存：扁平相对名 vs vllm. 开头全限定路径',
        15.5, lc.C_TXT, 'start', True, maxw=1000, tag='title')
lc.text(MX, 58, '老商业街（model_executor/models/ 扁平目录）什么平台都卖、柜台越堆越长；新开发区（vllm/models/<name>/ 独栋）一层一个平台专柜，大门口听口音（current_platform）分流——两条街的门牌系统都认',
        10, lc.C_MUTE, 'start', maxw=1090, tag='subtitle')
_ch = '放大自 L2 拍片 ① registry 解析（站 1-2）× south 新布局 · L0：GPU 执行臂 × 模型层框'
_cw = lc.chip_w(_ch)
lc.rect(BXR - _cw, 12, _cw, 20, '#ffffff', lc.C_MUTE, rx=9, sw=1.1, dash=True)
lc.text(BXR - _cw / 2, 26.5, _ch, 9.5, lc.C_GPU_S, 'middle', True, maxw=_cw - 4, tag='chip')

# ---------------- 顶部：同一张表的两行 ----------------
CBX, CBY, CBW, CBH = MX + 24, 92, BXR - MX - 48, 92
lc.rect(CBX, CBY, CBW, CBH, '#f8fafc', lc.C_MUTE, rx=8, sw=1.4)
lc.text(CBX + 16, CBY + 20, '_TEXT_GENERATION_MODELS 同一张表·相邻两行（vllm/model_executor/models/registry.py:L92-L95）',
        9.5, lc.C_TXT, 'start', True, maxw=760, tag='cb:t')
lc.text(CBX + 16, CBY + 44, '"DeepseekV32ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),',
        9, '#334155', 'start', tag='cb:l1')
lc.text(CBX + 760, CBY + 44, '← 扁平相对名（L94）', 8.5, lc.C_MUTE, 'start', tag='cb:t1')
lc.text(CBX + 16, CBY + 64, '"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", "DeepseekV4ForCausalLM"),',
        9, '#334155', 'start', tag='cb:l2')
lc.text(CBX + 760, CBY + 64, '← vllm. 开头全限定路径（L95）', 8.5, lc.C_MUTE, 'start',
        tag='cb:t2')
lc.text(CBX + 16, CBY + 82, '几百个条目全是字符串对——查表本身一行源码都不 import', 8.2,
        lc.C_MUTE, 'start', tag='cb:n')

# 表 → 判定盒
lc.seg(CBX + CBW / 2, CBY + CBH, CBX + CBW / 2, 214, lc.C_MUTE, 2.0, 'std')
lc.text(CBX + CBW / 2 + 10, 206, '条目逐个过 _resolve_module_name', 8.5, lc.C_MUTE, 'start',
        tag='a0')

# ---------------- 中部：_resolve_module_name 判定 ----------------
RVX, RVY, RVW, RVH = 500, 214, 600, 128
lc.rect(RVX, RVY, RVW, RVH, '#ffffff', lc.C_GPU_S, rx=9, sw=1.8)
lc.text(RVX + 18, RVY + 24, '_resolve_module_name(mod_relname)', 10.5, lc.C_GPU_S, 'start',
        True, maxw=RVW - 36, tag='rv:t')
lc.text(RVX + 18, RVY + 48, 'if mod_relname.startswith("vllm."):', 9, '#334155', 'start',
        tag='rv:l1')
lc.text(RVX + 18, RVY + 66, '    return mod_relname', 9, '#334155', 'start', tag='rv:l2')
lc.text(RVX + 210, RVY + 66, '# 原样放行 → 右（新街）', 8.2, lc.C_GPU_S, 'start', tag='rv:c2')
lc.text(RVX + 18, RVY + 84, 'return f"vllm.model_executor.models.{mod_relname}"', 9,
        '#334155', 'start', tag='rv:l3')
lc.text(RVX + 18, RVY + 102, '# ↑ 不以 vllm. 开头 → 拼前缀 → 左（老街）', 8.2, lc.C_MUTE,
        'start', tag='rv:c3')
lc.text(RVX + RVW - 14, RVY + RVH - 10, 'registry.py:L1439-L1445', 8, lc.C_FAINT, 'end',
        tag='rv:ft')

# 判定 → 两街
PY0 = 384
lc.seg(RVX + 120, RVY + RVH, 420, PY0, lc.C_MUTE, 2.0, 'std')
lc.text(400, 366, '拼前缀 → 老街', 8.5, lc.C_MUTE, 'end', True, tag='aL')
lc.seg(RVX + RVW - 120, RVY + RVH, 1180, PY0, lc.C_GPU_S, 2.0, 'std')
lc.text(1200, 366, '原样放行 → 新街', 8.5, lc.C_GPU_S, 'start', True, tag='aR')

# ---------------- 左面板：旧扁平布局 ----------------
LPX, LPY, LPW, LPH = 80, PY0, 680, 360
lc.rect(LPX, LPY, LPW, LPH, '#ffffff', lc.C_MUTE, rx=9, sw=1.6)
lc.text(LPX + 18, LPY + 24, '旧扁平布局：一条街全平台都卖', 11.5, lc.C_TXT, 'start', True,
        maxw=LPW - 36, tag='lp:t')
# 目录 → 文件
fy = LPY + 42
lc.rect(LPX + 18, fy, LPW - 36, 30, '#f8fafc', lc.C_MUTE, rx=5, sw=1.2)
lc.text(LPX + 30, fy + 20, 'vllm/model_executor/models/　（扁平一层目录）', 9, '#334155',
        'start', tag='lp:dir')
lc.seg(LPX + 60, fy + 30, LPX + 60, fy + 46, lc.C_MUTE, 1.4, 'std')
fby = fy + 46
lc.rect(LPX + 18, fby, LPW - 36, 108, '#ffffff', '#64748b', rx=5, sw=1.4)
lc.text(LPX + 30, fby + 20, 'deepseek_v2.py —— 一个文件', 9.5, lc.C_TXT, 'start', True,
        maxw=LPW - 60, tag='lp:file')
for i, ln in enumerate(['· DSV2 / DSV3 / DSV3.2 全在这一件里',
                        '· nvidia / amd / xpu 的平台分支挤在同一文件',
                        '· 老街照常营业：几百个条目大多还在这条街（L92-L94 一带）',
                        '· 旗舰深度绑定平台 kernel（FlashMLA / MegaMoE）时没法读——迁移动机']):
    lc.text(LPX + 30, fby + 40 + i * 17, ln, 8.3, '#334155', 'start', maxw=LPW - 60,
            tag='lp:fl' + str(i))
ry = fby + 108 + 22
lc.rect(LPX + 18, ry, LPW - 36, 52, '#f8fafc', lc.C_MUTE, rx=5, sw=1.2, dash=True)
lc.text(LPX + 30, ry + 20, '前台翻登记册自动补街名前缀：', 8.5, lc.C_TXT, 'start', True,
        maxw=LPW - 60, tag='lp:r1')
lc.text(LPX + 30, ry + 38, "→ vllm.model_executor.models.deepseek_v2", 8.8, '#334155',
        'start', tag='lp:r2')
lc.text(LPX + 18, LPY + LPH - 12, '条目写相对名（' + "'deepseek_v2'" + '）——新旧归一在判定盒完成',
        8, lc.C_FAINT, 'start', maxw=LPW - 36, tag='lp:ft')

# ---------------- 右面板：新硬件隔离布局 ----------------
RPX, RPY, RPW, RPH = 840, PY0, 680, 360
lc.rect(RPX, RPY, RPW, RPH, lc.C_GPU_F, lc.C_GPU_S, rx=9, sw=1.8)
lc.text(RPX + 18, RPY + 24, '新硬件隔离布局：旗舰独栋，一层一个平台专柜', 11.5, lc.C_GPU_S,
        'start', True, maxw=RPW - 36, tag='rp:t')
fy = RPY + 42
lc.rect(RPX + 18, fy, RPW - 36, 30, '#ffffff', lc.C_GPU_S, rx=5, sw=1.2)
lc.text(RPX + 30, fy + 20, 'vllm/models/deepseek_v4/　（独栋目录树）', 9, '#334155', 'start',
        tag='rp:dir')
lc.seg(RPX + 60, fy + 30, RPX + 60, fy + 46, lc.C_GPU_S, 1.4, 'std')
gy = fy + 46
lc.rect(RPX + 18, gy, RPW - 36, 34, '#ffffff', lc.C_GPU_S, rx=5, sw=1.5)
lc.text(RPX + 30, gy + 22, '__init__.py 大门口：听口音分流（current_platform）', 9.2,
        lc.C_GPU_S, 'start', True, maxw=RPW - 60, tag='rp:gate')
# 三分支
by = gy + 46
BR = [('is_rocm()', 'amd/'), ('is_xpu()', 'xpu/'), ('else', 'nvidia/')]
bw = (RPW - 36 - 24) / 3
for i, (cond, sub) in enumerate(BR):
    bx = RPX + 18 + i * (bw + 12)
    lc.seg(RPX + 60, by - 12, bx + bw / 2, by - 12 if i == 1 else by - 12, lc.C_GPU_S, 1.0)
    lc.rect(bx, by, bw, 44, '#ffffff', lc.C_GPU_S, rx=5, sw=1.3)
    lc.text(bx + bw / 2, by + 19, cond, 8.8, '#334155', 'middle', True, tag='rp:b' + str(i))
    lc.text(bx + bw / 2, by + 35, '→ ' + sub + ' 专柜', 8.2, lc.C_GPU_S, 'middle', tag='rp:s' + str(i))
# 各专柜导出三类
oy = by + 60
lc.text(RPX + 30, oy + 4, '各专柜再导出三类（__init__.py L17-L32 三分支）：', 8.5, lc.C_TXT,
        'start', True, maxw=RPW - 60, tag='rp:o')
for i, ln in enumerate(['model.py → DeepseekV4ForCausalLM（主模型）',
                        'mtp.py → DeepSeekV4MTP（多 token 预测头）',
                        'dspark.py → DSparkDeepseekV4ForCausalLM（draft 模型）']):
    lc.text(RPX + 44, oy + 22 + i * 16, ln, 8.3, '#334155', 'start', maxw=RPW - 80,
            tag='rp:ol' + str(i))
ry2 = oy + 74
lc.rect(RPX + 18, ry2, RPW - 36, 44, '#ffffff', lc.C_GPU_S, rx=5, sw=1.2, dash=True)
lc.text(RPX + 30, ry2 + 18, '条目写全地址：' + "'vllm.models.deepseek_v4'" + '——vllm. 开头原样放行',
        8.5, lc.C_GPU_S, 'start', True, maxw=RPW - 60, tag='rp:r1')
lc.text(RPX + 30, ry2 + 34, 'docstring 原话：hardware-isolated … picks the right one for the current platform',
        7.8, lc.C_MUTE, 'start', maxw=RPW - 60, tag='rp:r2')
lc.text(RPX + 18, RPY + RPH - 12, '同款全限定条目：DSparkDraftModel（L617）· DeepSeekV4MTPModel（L643）',
        8, lc.C_FAINT, 'start', maxw=RPW - 36, tag='rp:ft')

# ---------------- 底部注 ----------------
NY = PY0 + LPH + 24
lc.rect(MX + 24, NY, BXR - MX - 48, 58, '#ffffff', lc.C_MUTE, rx=7, sw=1.1, dash=True)
lc.text(MX + 42, NY + 20, '惰性 import：登记册全是字符串对，只在 load_model_cls 三行（L1017-L1019）importlib.import_module + getattr——说「现在要进房」那一刻才出发，一张表只出动一个导游',
        8.8, lc.C_TXT, 'start', maxw=BXR - MX - 84, tag='nb:1')
lc.text(MX + 42, NY + 40, "两条街并存 = 迁移进行时（'[1/N]'）——读者要能分辨条目走哪条街；DSV4 接入清单六件套归 ch28 capstone（预告）",
        8.8, lc.C_MUTE, 'start', maxw=BXR - MX - 84, tag='nb:2')

# ---------------- 页脚 ----------------
lc.text(MX, NY + 92, '读图：顶部同一张表的两行分别走左右两条解析路——左拼前缀进老街扁平文件；右原样放行进新街独栋，门口按 current_platform 三分发平台专柜。查表零 import，import 推迟到实例化那一刻。',
        9, lc.C_MUTE, 'start', maxw=BXR - MX, tag='ft:1')
lc.text(MX, NY + 110, '行号基线 vLLM v0.27.1（vllm/model_executor/models/registry.py · vllm/models/deepseek_v4/__init__.py）',
        8, lc.C_FAINT, 'start', maxw=BXR - MX, tag='ft:2')

# ---------------- 装配输出 ----------------
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
       f'<rect width="{W}" height="{H}" fill="white"/>', lc.DEFS]
svg += [s for _, s in lc.ELEMS]
svg.append('</svg>')
out = HERE / 'ch23-fig-two-layouts.svg'
out.write_bytes('\n'.join(svg).encode('utf-8'))
print(f'wrote {out}  ({W}x{H}, {len(lc.ELEMS)} elems)')
if lc.WARN:
    print('--- OVERFLOW WARNINGS ---')
    for w_ in lc.WARN:
        print(' ', w_)
    sys.exit(1)
