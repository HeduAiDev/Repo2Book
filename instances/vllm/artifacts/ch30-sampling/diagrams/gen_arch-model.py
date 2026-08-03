#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，真正的渲染逻辑在 scripts/arch_model_figure.py。

为什么是薄封装:这张图**全书每章一张、同一套画法**,数据全部来自
book/cartography/arch-model.json(← outline-final.json 的章-子系统归属 + 各章 dossier
的 code_spine)。逻辑放共享渲染器,单章只声明「我是哪一章」,避免 153 份复制品各自漂移。

改图请改 scripts/arch_model_figure.py;改数据请改 scripts/arch_model.py 后重新 build。
"""
import os
import subprocess
import sys
from pathlib import Path

CHAPTER = 'ch30'
INSTANCE = 'vllm'

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'

# 渲染器在收尾会 print '✓ …'——Windows 控制台默认 GBK 编码会抛 UnicodeEncodeError，
# 子进程环境显式置 UTF-8（否则一键重生成在 Windows 上必挂）。
env = dict(os.environ, PYTHONIOENCODING='utf-8')

# Step 1: 渲染 SVG（数据来自 arch_model.json，由 arch_model_figure.py 渲染）
subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)],
               check=True, env=env)

# Step 2: SVG → PNG（cairosvg, scale=2；本环境 rsvg-convert 是 Python wrapper 不兼容 -z）
import cairosvg  # noqa: E402
cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
print(f'OK {svg.name} / {png.name}')
