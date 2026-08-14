#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，渲染逻辑在 scripts/arch_model_figure.py。

为什么是薄封装:这张图**全书每章一张、同一套画法**,数据全部来自
book/cartography/arch-model.json(← outline-final.json 的章-子系统归属 + 各章 dossier
的 code_spine)。逻辑放共享渲染器,单章只声明「我是哪一章」,避免 153 份复制品各自漂移。

改图请改 scripts/arch_model_figure.py;改数据请改 scripts/arch_model.py 后重新 build。

渲染器里 print('✓ …') 在 GBK 控制台下会 UnicodeEncodeError → 给子进程显式
PYTHONIOENCODING=utf-8;rsvg-convert 在部分 Windows 环境是 shebang 损坏的
cairosvg 包装(exit 49),捕获 CalledProcessError 降级 cairosvg(同 ch35 的做法)。
"""
import os
import subprocess
import sys
from pathlib import Path

CHAPTER = 'ch03'
INSTANCE = 'vllm'

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'

env = dict(os.environ, PYTHONIOENCODING='utf-8')
subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)],
               check=True, env=env)
try:
    subprocess.run(['rsvg-convert', '-z', '2', str(svg), '-o', str(png)], check=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    import cairosvg
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
print(f'[OK] {svg.name} / {png.name}')
