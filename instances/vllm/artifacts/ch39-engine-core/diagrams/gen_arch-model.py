#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，渲染逻辑在 scripts/arch_model_figure.py。

为什么是薄封装:这张图**全书每章一张、同一套画法**,数据全部来自
book/cartography/arch-model.json。逻辑放共享渲染器,单章只声明「我是哪一章」,避免复制品漂移。

改图请改 scripts/arch_model_figure.py;改数据请改 scripts/arch_model.py 后重新 build。
"""
import shutil
import subprocess
import sys
from pathlib import Path

CHAPTER = 'ch39'
INSTANCE = 'vllm'

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'

subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)], check=True)

# rsvg-convert 优先，不可用时回退 cairosvg（Windows 环境常见：
# WindowsApps 存根的 python3 挡在 PATH 前面时 rsvg-convert.bat 无法执行）
ok = False
if shutil.which('rsvg-convert'):
    try:
        subprocess.run(['rsvg-convert', '-z', '2', str(svg), '-o', str(png)],
                       check=True, capture_output=True)
        ok = True
    except (subprocess.CalledProcessError, OSError):
        ok = False
if not ok:
    import cairosvg
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)

print(f'✓ {svg.name} / {png.name}')
