#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，真正的渲染逻辑在 scripts/arch_model_figure.py。

为什么是薄封装:这张图**全书每章一张、同一套画法**,数据全部来自
book/cartography/arch-model.json(← outline-final.json 的章-子系统归属 + 各章 dossier
的 code_spine)。逻辑放共享渲染器,单章只声明「我是哪一章」,避免 153 份复制品各自漂移。

改图请改 scripts/arch_model_figure.py;改数据请改 scripts/arch_model.py 后重新 build。

渲染器说明:优先 rsvg-convert(librsvg/Pango 逐字 CJK 回退);本机 conda 的
rsvg-convert 因 DLL 故障对任何 SVG 都退 49(2026-08-02 起),自动回退 node sharp。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

CHAPTER = 'ch18'
INSTANCE = 'vllm'

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'

env = dict(os.environ, PYTHONIOENCODING='utf-8')
subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)],
               check=True, env=env)

rsvg_ok = False
if shutil.which('rsvg-convert'):
    try:
        r = subprocess.run(['rsvg-convert', '-z', '2', str(svg), '-o', str(png)],
                           capture_output=True, text=True)
        rsvg_ok = (r.returncode == 0 and png.exists())
        if not rsvg_ok:
            print(f'rsvg-convert 失败(rc={r.returncode}),回退 node sharp')
    except OSError as e:
        print(f'rsvg-convert 不可执行({e}),回退 node sharp')
if rsvg_ok:
    print(f'✓ {svg.name} / {png.name} (rsvg-convert)')
    sys.exit(0)

node = shutil.which('node')
if not node:
    print('✗ 无可用渲染器:rsvg-convert 与 node 均不可用')
    sys.exit(1)
subprocess.run([node, '-e',
                "const s=require('sharp');"
                "s(process.argv[1],{density:144}).png().toFile(process.argv[2])"
                ".then(i=>console.log('✓ sharp',i.width+'x'+i.height))"
                ".catch(e=>{console.error(e.message);process.exit(1)})",
                str(svg), str(png)], check=True)
print(f'✓ {svg.name} / {png.name} (sharp)')
