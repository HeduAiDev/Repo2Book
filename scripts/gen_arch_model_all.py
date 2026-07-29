#!/usr/bin/env python3
"""为一本书的每一章生成渐进式架构模型图 + 每章的 gen 封装脚本。

  python3 scripts/gen_arch_model_all.py --instance vllm
"""
import argparse, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance

GEN_TPL = '''#!/usr/bin/env python3
"""本章的「渐进式架构模型」图 —— 薄封装，渲染逻辑在 scripts/arch_model_figure.py。"""
import subprocess, sys
from pathlib import Path

CHAPTER = {cid!r}
INSTANCE = {inst!r}

here = Path(__file__).resolve().parent
repo = here.parents[4]
svg = here / 'arch-model.svg'
png = here / 'arch-model.png'
subprocess.run([sys.executable, str(repo / 'scripts' / 'arch_model_figure.py'),
                '--chapter', CHAPTER, '--instance', INSTANCE, '--out', str(svg)], check=True)
subprocess.run(['rsvg-convert', '-z', '2', str(svg), '-o', str(png)], check=True)
print(f'✓ {{svg.name}} / {{png.name}}')
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', required=True)
    a = ap.parse_args()
    ad = Path(instance.artifacts_dir(a.instance))
    ok = bad = 0
    for ch in sorted(p for p in ad.iterdir() if p.is_dir() and p.name.startswith('ch')):
        cid = ch.name.split('-')[0]
        diag = ch / 'diagrams'
        diag.mkdir(exist_ok=True)
        (diag / 'gen_arch-model.py').write_text(
            GEN_TPL.format(cid=cid, inst=a.instance), encoding='utf-8')
        r = subprocess.run([sys.executable, str(diag / 'gen_arch-model.py')],
                           capture_output=True, text=True)
        if r.returncode == 0 and (diag / 'arch-model.png').exists():
            ok += 1
        else:
            bad += 1
            print(f'  ✗ {ch.name}: {r.stderr.strip()[:160]}')
    print(f'✓ {ok} 章生成成功, ✗ {bad} 章失败')

if __name__ == '__main__':
    main()
