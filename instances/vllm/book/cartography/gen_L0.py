#!/usr/bin/env python3
"""L0 全书唯一权威架构图 —— 薄壳（v3 图系 Phase 2）

绘制主体在 l0_common.build_l0()（与 fable 执笔版逐行同源，坐标零改动）；
本文件只做装配：svg 头 + 白底 + defs + 元素流。gen_L1.py 复用同一 build_l0()
做「L0 minimap 高亮框 + Part 区域裁切放大 + Part 标题带」——同源强制（FIGURE-SYSTEM.md §0）。

对标用户手绘参照（画布9.png）的信息密度：
  每个组件框 = 类名 + 2~4 个真实方法/契约行 + 规范源码路径；
  每条数据箭头带消息名/方法名；进程边界显式（ROUTER/DEALER、PUSH/PULL、帧序）。
覆盖全系统：API 进程双泳道 → ZMQ 边界 → EngineCore 进程
  （五拍循环 + 调度·显存账本 + GPU 执行臂 + 采样与出口）。
所有内容（类名/方法名/数字/契约）取自 ARCHITECTURE.md 与 deepread/*.json（pin v0.27.1）。
"""
import sys
from pathlib import Path

from l0_common import DEFS, W, build_l0


def main():
    elems, geo, warn = build_l0()
    H = geo['H']
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         DEFS]
    L += [s for _, s in elems]
    L.append('</svg>')
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'L0-architecture.svg'
    out.write_bytes('\n'.join(L).encode('utf-8'))   # LF 一律（CLAUDE.md 坑 #8）
    print(f'OK {out} (viewBox 0 0 {W} {H})')
    if warn:
        print(f'--- {len(warn)} OVERFLOW WARNINGS ---')
        for w in warn:
            print(w)
    else:
        print('no overflow warnings')


if __name__ == '__main__':
    main()
