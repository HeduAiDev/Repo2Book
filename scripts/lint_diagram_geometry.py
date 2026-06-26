#!/usr/bin/env python3
"""图元几何检查(lint_diagrams 查不出的): 文字越界/相撞/压框 + 箭头端点不连通。
解析 SVG 的 <text>/<rect>/<line>/<path>, 估算几何, 报四类「明显」问题:
  overflow   : 文字盒越出画布(viewBox)边界;
  text-text  : 两段文字盒大面积相撞;
  text-rect  : 文字大幅插进一个非容纳它的大框内部(标签压/侵入别的框);
  arrow-loose: 带箭头(marker)的连线, 某端点悬空——附近(容差内)既无框边也无另一线端点。

文字宽度估算: CJK/全角≈1.0em, 数字/拉丁≈0.58em, 空格≈0.3em, 其余≈0.5em。启发式, 容差留宽只报明显。

用法: python3 scripts/lint_diagram_geometry.py --all | <a.svg> ...
退出码 1 = 有问题。
"""
import re
import sys
import glob
import math
import xml.etree.ElementTree as ET

NS = '{http://www.w3.org/2000/svg}'
ARROW_TOL = 6.0   # 箭头端点到框边/他端点的容差
EDGE_TOL = 6.0


def char_w(c):
    o = ord(c)
    if o == 0x20:
        return 0.30
    if 0x2E80 <= o <= 0x9FFF or 0xFF00 <= o <= 0xFFEF or 0x3000 <= o <= 0x303F:
        return 1.0
    if c.isascii() and c.isalnum():
        return 0.58
    return 0.5


def text_w(s, size):
    return size * sum(char_w(c) for c in s)


def parse_vb(root):
    vb = root.get('viewBox')
    if vb:
        p = [float(x) for x in vb.replace(',', ' ').split()]
        return p[0], p[1], p[2], p[3]
    return 0.0, 0.0, float(root.get('width', 0) or 0), float(root.get('height', 0) or 0)


def path_endpoints(d):
    # 取 path 的首末坐标点(粗略): 抓所有数字对
    nums = re.findall(r'-?\d+\.?\d*', d)
    pts = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums) - 1, 2)]
    return (pts[0], pts[-1]) if len(pts) >= 2 else None


def collect(root):
    texts, rects, arrows = [], [], []
    for el in root.iter():
        tag = el.tag.replace(NS, '')
        if tag == 'text':
            try:
                x = float(el.get('x', 0)); y = float(el.get('y', 0))
            except ValueError:
                continue
            size = float(el.get('font-size', 14) or 14)
            anchor = el.get('text-anchor', 'start')
            s = ''.join(el.itertext()).strip()
            if not s:
                continue
            w = text_w(s, size)
            x0 = x - w / 2 if anchor == 'middle' else (x - w if anchor == 'end' else x)
            texts.append({'s': s, 'x0': x0, 'x1': x0 + w, 'yt': y - 0.78 * size, 'yb': y + 0.20 * size})
        elif tag == 'rect':
            try:
                x = float(el.get('x', 0)); y = float(el.get('y', 0))
                w = float(el.get('width', 0)); h = float(el.get('height', 0))
            except ValueError:
                continue
            if w > 0 and h > 0:
                rects.append({'x0': x, 'y0': y, 'x1': x + w, 'y1': y + h, 'w': w, 'h': h})
        elif tag == 'line':
            try:
                p = ((float(el.get('x1', 0)), float(el.get('y1', 0))),
                     (float(el.get('x2', 0)), float(el.get('y2', 0))))
            except ValueError:
                continue
            arrows.append({'pts': p, 'marker': bool(el.get('marker-end') or el.get('marker-start'))})
        elif tag == 'path':
            ep = path_endpoints(el.get('d', ''))
            if ep:
                arrows.append({'pts': ep, 'marker': bool(el.get('marker-end') or el.get('marker-start'))})
    return texts, rects, arrows


def near_rect_edge(pt, rects, tol):
    x, y = pt
    for r in rects:
        on_v = (abs(x - r['x0']) <= tol or abs(x - r['x1']) <= tol) and (r['y0'] - tol <= y <= r['y1'] + tol)
        on_h = (abs(y - r['y0']) <= tol or abs(y - r['y1']) <= tol) and (r['x0'] - tol <= x <= r['x1'] + tol)
        inside = (r['x0'] - tol <= x <= r['x1'] + tol and r['y0'] - tol <= y <= r['y1'] + tol)
        if on_v or on_h or inside:
            return True
    return False


def near_point(pt, others, tol):
    for o in others:
        if math.hypot(pt[0] - o[0], pt[1] - o[1]) <= tol:
            return True
    return False


def check(path):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        return [f'SVG 解析失败: {e}']
    vx, vy, vw, vh = parse_vb(root)
    texts, rects, arrows = collect(root)
    issues = []
    PAD = 2.0
    # (1) overflow — 文字盒越出画布
    for t in texts:
        if t['x1'] > vx + vw + PAD or t['x0'] < vx - PAD or t['yb'] > vy + vh + PAD or t['yt'] < vy - PAD:
            over = round(max(t['x1'] - (vx + vw), vx - t['x0'], t['yb'] - (vy + vh), vy - t['yt']))
            if over > 4:
                issues.append(f"overflow: 「{t['s'][:24]}」越出画布约 {over}px")
    # (1b) rect-overflow — 框越出画布(被裁切)。文字居中时字不越界但框边被切, 故单独查。
    for r in rects:
        if r['x1'] > vx + vw + PAD or r['x0'] < vx - PAD or r['y1'] > vy + vh + PAD or r['y0'] < vy - PAD:
            over = round(max(r['x1'] - (vx + vw), vx - r['x0'], r['y1'] - (vy + vh), vy - r['y0']))
            if over > 4:
                issues.append(f"rect-overflow: 一个 {round(r['w'])}×{round(r['h'])} 框越出画布约 {over}px(被裁切)")
    # (2) text-text
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            ox = min(a['x1'], b['x1']) - max(a['x0'], b['x0'])
            oy = min(a['yb'], b['yb']) - max(a['yt'], b['yt'])
            if ox > 0 and oy > 0:
                wmin = min(a['x1'] - a['x0'], b['x1'] - b['x0'])
                hmin = min(a['yb'] - a['yt'], b['yb'] - b['yt'])
                if wmin > 0 and hmin > 0 and ox > 0.45 * wmin and oy > 0.45 * hmin:
                    issues.append(f"text-text 相撞: 「{a['s'][:14]}」×「{b['s'][:14]}」")
    # (3) text-rect: 文字大幅插进一个大框(非自身容纳框)内部
    for t in texts:
        tw = t['x1'] - t['x0']; cx = (t['x0'] + t['x1']) / 2; cy = (t['yt'] + t['yb']) / 2
        for r in rects:
            if r['w'] < 55 or r['h'] < 26:   # 跳过图例小色块
                continue
            if r['x0'] <= cx <= r['x1'] and r['y0'] <= cy <= r['y1']:
                continue  # 中心在框内=本框标签, 不算
            ox = min(t['x1'], r['x1']) - max(t['x0'], r['x0'])
            oy = min(t['yb'], r['y1']) - max(t['yt'], r['y0'])
            if ox > 0.30 * tw and oy > 0.4 * (t['yb'] - t['yt']):  # 横向插进>30%字宽 且 纵向明显
                issues.append(f"text-rect 侵入: 「{t['s'][:18]}」插进别的框内")
                break
    # (3b) tag-on-title: 同一个框内两段文字的盒子相交=角标压住框内正文。
    # 合法的多行标签是竖向堆叠、互不相交; 角标(贴右上角)会与居中标题重叠 → 抓它。
    for r in rects:
        if r['w'] < 55 or r['h'] < 26:
            continue
        inside = [t for t in texts
                  if r['x0'] <= (t['x0'] + t['x1']) / 2 <= r['x1']
                  and r['y0'] <= (t['yt'] + t['yb']) / 2 <= r['y1']]
        flagged = False
        for i in range(len(inside)):
            for j in range(i + 1, len(inside)):
                a, b = inside[i], inside[j]
                ox = min(a['x1'], b['x1']) - max(a['x0'], b['x0'])
                oy = min(a['yb'], b['yb']) - max(a['yt'], b['yt'])
                if ox > 3 and oy > 3:   # 框内两段文字真正二维相交(留 3px 容差)
                    issues.append(f"tag-on-title: 框内「{a['s'][:12]}」与「{b['s'][:12]}」相压")
                    flagged = True
                    break
            if flagged:
                break
    # (4) arrow-loose: 带 marker 的连线端点悬空
    all_ep = [p for a in arrows for p in a['pts']]
    for a in arrows:
        if not a['marker']:
            continue
        (p1, p2) = a['pts']
        if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < 6:
            continue
        for end, label in ((p1, '起点'), (p2, '末端')):
            others = [q for q in all_ep if q is not end]
            if not near_rect_edge(end, rects, ARROW_TOL) and not near_point(end, others, ARROW_TOL):
                issues.append(f"arrow-loose: 一条箭头{label}({round(end[0])},{round(end[1])})悬空, 未接到任何框边/端点")
    seen = set(); out = []
    for it in issues:
        if it not in seen:
            seen.add(it); out.append(it)
    return out


def main():
    args = sys.argv[1:]
    files = sorted(glob.glob('instances/vllm/artifacts/ch*/diagrams/*.svg')) \
        if (args == ['--all'] or not args) else args
    total = 0; perch = {}
    for f in files:
        iss = check(f)
        if iss:
            ch = f.split('/')[-3] if '/artifacts/' in f else f
            perch.setdefault(ch, []).append((f.split('/')[-1], iss)); total += len(iss)
    for ch in sorted(perch):
        print(f'❌ {ch}:')
        for name, iss in perch[ch]:
            for it in iss:
                print(f'    {name}: {it}')
    print('\n✓ 无明显几何问题' if total == 0 else f'\n共 {total} 处, 涉及 {len(perch)} 章')
    return 0 if total == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
