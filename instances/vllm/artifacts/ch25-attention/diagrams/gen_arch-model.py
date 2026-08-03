#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，真正的渲染逻辑在 scripts/arch_model_figure.py。

为什么是薄封装:这张图**全书每章一张、同一套画法**,数据全部来自
book/cartography/arch-model.json(← outline-final.json 的章-子系统归属 + 各章 dossier
的 code_spine)。逻辑放共享渲染器,单章只声明「我是哪一章」,避免 153 份复制品各自漂移。

改图请改 scripts/arch_model_figure.py;改数据请改 scripts/arch_model.py 后重新 build。
"""
import subprocess
import sys
import shutil
from pathlib import Path

CHAPTER = 'ch25'
INSTANCE = 'vllm'

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'

subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)], check=True)

# rsvg-convert 优先（保留中文渲染精度）；找不到或运行失败时降级为 cairosvg。
# 实测本机 rsvg-convert 是 .BAT shim（CreateProcess 直接炸、本体也崩），
# 故 fallback 必须同时覆盖「找不到」与「运行失败」两种情形。
def _rsvg(svg_path, png_path):
    if not shutil.which('rsvg-convert'):
        return False
    try:
        r = subprocess.run(['rsvg-convert', '-z', '2', str(svg_path), '-o', str(png_path)])
        return r.returncode == 0
    except OSError:      # .BAT shim 无法被 CreateProcess 直接执行
        return False

if not _rsvg(svg, png):
    import cairosvg
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2)
print(f'√ {svg.name} / {png.name}')
