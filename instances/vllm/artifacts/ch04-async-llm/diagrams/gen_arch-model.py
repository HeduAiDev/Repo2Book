#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，真正的渲染逻辑在 scripts/arch_model_figure.py。

为什么是薄封装:这张图**全书每章一张、同一套画法**,数据全部来自
book/cartography/arch-model.json(← outline-final.json 的章-子系统归属 + 各章 dossier
的 code_spine)。逻辑放共享渲染器,单章只声明「我是哪一章」,避免 153 份复制品各自漂移。

改图请改 scripts/arch_model_figure.py;改数据请改 scripts/arch_model.py 后重新 build。

rsvg 解析(2026-08-03 本机加固):本机没有真 librsvg,`rsvg-convert` 是 Miniconda/Scripts
下的 cairosvg shim(python 脚本 + .bat);裸调 `rsvg-convert -z 2` 会因 shim 的简易参数
解析把 `-z` 当成输入文件而失败,且其 shebang 的 python3 是 WindowsApps 空壳(exit 49)。
故用 shutil.which 解析(命中 .bat),svg 放第一位、不带 -z(scale 1,与全书已提交
PNG 同口径;真 librsvg 环境下同样成立)。

GBK 控制台加固(2026-08-04):本机终端默认 GBK,渲染器收尾 `print('✓ …')` 会抛
UnicodeEncodeError(子进程 exit 1,PNG 因此不生成);给子进程注入 PYTHONIOENCODING=utf-8,
父进程自身 stdout 也 reconfigure 为 utf-8(replace 容错)。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

CHAPTER = 'ch04'
INSTANCE = 'vllm'

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'

subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)],
               check=True, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
rsvg = shutil.which('rsvg-convert')
if not rsvg:
    raise SystemExit('rsvg-convert 未找到(本机以 cairosvg shim 顶替)')
subprocess.run([rsvg, str(svg), '-o', str(png)], check=True)
print(f'✓ {svg.name} / {png.name}')
