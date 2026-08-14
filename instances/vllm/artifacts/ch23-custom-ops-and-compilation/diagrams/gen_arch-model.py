#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，真正的渲染逻辑在 scripts/arch_model_figure.py。

为什么是薄封装:这张图**全书每章一张、同一套画法**,数据全部来自
book/cartography/arch-model.json(← outline-final.json 的章-子系统归属 + 各章 dossier
的 code_spine)。逻辑放共享渲染器,单章只声明「我是哪一章」,避免 153 份复制品各自漂移。

改图请改 scripts/arch_model_figure.py;改数据请改 scripts/arch_model.py 后重新 build。
"""
import subprocess
import sys
from pathlib import Path

CHAPTER = 'ch23'
INSTANCE = 'vllm'

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'

subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)], check=True)
# PNG 转换：优先真实 librsvg（rsvg-convert，2x 高清）；本机无 librsvg 时按渲染器
# 既定的 cairosvg shim 兜底（renderer 注释已声明此路径，字体栈显式走 Microsoft YaHei
# 保证中文不豆腐）。cairosvg 无 -z 参数，scale=2 等价于 rsvg-convert -z 2。
import shutil
rsvg = shutil.which('rsvg-convert')
if rsvg is not None:
    conv = subprocess.run([rsvg, '-z', '2', str(svg), '-o', str(png)],
                          capture_output=True, text=True)
else:
    conv = None
if conv is None or conv.returncode != 0:
    import cairosvg
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2.0)
print(f'✓ {svg.name} / {png.name}')
