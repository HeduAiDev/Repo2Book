#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，渲染逻辑在 scripts/arch_model_figure.py。"""
import subprocess, sys
from pathlib import Path

CHAPTER = 'ch03'
INSTANCE = 'vllm'

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'
subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)], check=True)
subprocess.run(['rsvg-convert', '-z', '2', str(svg), '-o', str(png)], check=True)
print(f'✓ {svg.name} / {png.name}')
