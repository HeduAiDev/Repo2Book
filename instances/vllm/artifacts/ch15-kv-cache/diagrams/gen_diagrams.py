#!/usr/bin/env python3
"""Generate ch15 paged-KV-cache diagrams.

paging-overview / free-queue-lru / block-hash-chain / prefix-hit-alloc.
PNG rendered via rsvg-convert (CJK fallback). Do NOT use ImageMagick convert.
"""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

DEFS = (
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="ared" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
    '<marker id="agrn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
    '<marker id="ablu" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1d4ed8"/></marker>'
    '</defs>'
)

def box(x, y, w, h, fill, stroke, rx=8, sw=1.5):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def txt(x, y, s, size=13, anchor="middle", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    return (f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')

def line(x1, y1, x2, y2, color="#475569", marker="a", dash=None, width=1.6):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = f' marker-end="url(#{marker})"' if marker else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}"{d}{m}/>')

def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
            + DEFS + f'<rect width="{w}" height="{h}" fill="white"/>' + body + '</svg>')


# ============ 1: paging-overview ============
def paging_overview():
    w, h = 880, 470
    b = []
    b.append(txt(w/2, 30, "分页：逻辑 token 序列 → block table → 物理块池", 16, weight="bold"))

    # left: two requests' logical token sequences sliced by block_size
    b.append(txt(150, 64, "请求 A 的 token（block_size=4 切块）", 11.5, fill="#475569"))
    b.append(txt(150, 230, "请求 B 的 token（前 8 个与 A 相同）", 11.5, fill="#475569"))
    def slice_row(x0, y, labels, fills):
        out = []
        for i, (lab, f) in enumerate(zip(labels, fills)):
            out.append(box(x0 + i*66, y, 60, 34, f, "#64748b", rx=5))
            out.append(txt(x0 + i*66 + 30, y+22, lab, 11, mono=True))
        return out
    b += slice_row(40, 78, ["t0..3", "t4..7", "t8..11"], ["#dbeafe", "#dbeafe", "#bfdbfe"])
    b += slice_row(40, 244, ["t0..3", "t4..7", "t8..9"], ["#dbeafe", "#dbeafe", "#fde68a"])

    # middle: block tables (logical->physical)
    b.append(txt(450, 64, "block table（逻辑块 → 物理 block_id）", 11.5, fill="#475569"))
    def btable(x, y, rows):
        out = [box(x, y, 150, 18*len(rows)+8, "#f8fafc", "#94a3b8", rx=6)]
        for i, (lg, ph) in enumerate(rows):
            out.append(txt(x+12, y+20+i*18, f"逻辑块{lg}", 10.5, anchor="start", mono=True))
            out.append(txt(x+140, y+20+i*18, f"→ blk {ph}", 10.5, anchor="end", mono=True, fill="#1d4ed8"))
        return out
    b += btable(375, 86, [(0, 3), (1, 4), (2, 5)])
    b += btable(375, 252, [(0, 3), (1, 4), (2, 6)])

    # right: physical block pool grid
    b.append(txt(745, 64, "BlockPool 物理块网格", 11.5, fill="#475569"))
    pool_x, pool_y = 660, 86
    ids = [0, 1, 2, 3, 4, 5, 6, 7]
    fills = {3: "#bfdbfe", 4: "#bfdbfe", 5: "#dbeafe", 6: "#fde68a", 0: "#e2e8f0"}
    cell_y = {}
    for i, bid in enumerate(ids):
        r, c = divmod(i, 2)
        x = pool_x + c*86
        y = pool_y + r*42
        f = fills.get(bid, "#f1f5f9")
        b.append(box(x, y, 78, 34, f, "#64748b", rx=5))
        lab = f"blk {bid}" + (" (null)" if bid == 0 else "")
        b.append(txt(x+39, y+21, lab, 10.5, mono=True))
        cell_y[bid] = (x, y)
    # arrows from btables to shared physical blocks 3,4 (both requests)
    bx, by = cell_y[3]
    b.append(line(525, 110, bx, by+10, color="#1d4ed8", marker="ablu", width=1.4))
    b.append(line(525, 276, bx, by+24, color="#15803d", marker="agrn", width=1.4))
    b.append(txt((525+bx)/2, by-6, "同物理块被两请求前缀共享", 10, fill="#15803d", weight="bold"))

    b.append(txt(w/2, h-22, "CacheConfig.block_size 决定切块/哈希/分配/命中的最小粒度（这里 = 4）",
                 11.5, fill="#b91c1c"))
    return svg(w, h, "".join(b)), "paging-overview"


# ============ 2: free-queue-lru ============
def free_queue_lru():
    w, h = 900, 540
    b = []
    b.append(txt(w/2, 30, "FreeKVCacheBlockQueue：哨兵双向链表 + LRU 驱逐序", 16, weight="bold"))

    def node(x, y, label, fill, stroke):
        return box(x, y, 70, 36, fill, stroke, rx=6) + txt(x+35, y+23, label, 11.5, mono=True)

    def chain(y, items, title, note=""):
        out = [txt(60, y-12, title, 12, anchor="start", weight="bold", fill="#334155")]
        x = 60
        prev_cx = None
        for lab, fill, stroke in items:
            out.append(node(x, y, lab, fill, stroke))
            cx = x + 35
            if prev_cx is not None:
                out.append(line(prev_cx+35, y+18, cx-37, y+18, color="#94a3b8", marker=None, width=1.3))
                out.append(line(cx-37, y+24, prev_cx+35, y+24, color="#cbd5e1", marker=None, width=1.0))
            prev_cx = cx
            x += 96
        if note:
            out.append(txt(60, y+58, note, 11, anchor="start", fill="#475569"))
        return out

    HEAD = ("head", "#f1f5f9", "#94a3b8")
    TAIL = ("tail", "#f1f5f9", "#94a3b8")
    def blk(n, fill="#dbeafe"):
        return (f"blk{n}", fill, "#2563eb")

    # state 1: initial order by block_id
    b += chain(80, [HEAD, blk(1), blk(2), blk(3), blk(4), blk(5), TAIL],
               "① 初始：按 block_id 升序",
               "队首=最久未用（LRU 先驱逐）　队尾=最近释放　head/tail 哨兵永不弹出")
    # state 2: popleft takes front
    b += chain(210, [HEAD, blk(2, "#fde68a"), blk(3), blk(4), blk(5), TAIL],
               "② popleft()：取队首 blk1（已被分配走）",
               "下一个 LRU 候选变成 blk2；num_free_blocks 减 1")
    # state 3: free_blocks appends reversed request blocks to tail
    b += chain(340, [HEAD, blk(2, "#fde68a"), blk(3), blk(5), blk(7, "#bbf7d0"), blk(6, "#bbf7d0"), TAIL],
               "③ free_blocks(reversed)：请求块逆序入队尾",
               "尾块 blk7 排在 blk6 前 → 前缀更长的尾块更早被驱逐（同请求内 LRU 细分）")
    # state 4: touch removes middle in O(1)
    b += chain(470, [HEAD, blk(2, "#fde68a"), blk(5), blk(7, "#bbf7d0"), blk(6, "#bbf7d0"), TAIL],
               "④ touch(blk3)：命中复用 → remove() 从链中间 O(1) 摘除",
               "blk3 被新请求前缀命中，立刻移出驱逐候选——deque 做不到中间 O(1) 删除")

    return svg(w, h, "".join(b)), "free-queue-lru"


# ============ 3: block-hash-chain ============
def block_hash_chain():
    w, h = 900, 420
    b = []
    b.append(txt(w/2, 30, "链式块哈希：前缀完全一致才得到相同 hash", 16, weight="bold"))

    def hbox(x, y, title, body1, body2, fill):
        out = [box(x, y, 220, 78, fill, "#0369a1", rx=8)]
        out.append(txt(x+110, y+22, title, 12.5, weight="bold", mono=True))
        out.append(txt(x+110, y+44, body1, 10.5, mono=True, fill="#334155"))
        out.append(txt(x+110, y+62, body2, 10.5, mono=True, fill="#334155"))
        return out

    y = 90
    b += hbox(40, y, "block0_hash", "= H(NONE_HASH,", "  tok[0:4], extra0)", "#e0f2fe")
    b += hbox(340, y, "block1_hash", "= H(block0_hash,", "  tok[4:8], extra1)", "#e0f2fe")
    b += hbox(640, y, "block2_hash", "= H(block1_hash,", "  tok[8:12], extra2)", "#e0f2fe")
    b.append(line(260, y+39, 338, y+39, color="#1d4ed8", marker="ablu", width=2.0))
    b.append(line(560, y+39, 638, y+39, color="#1d4ed8", marker="ablu", width=2.0))
    b.append(txt(300, y+30, "parent", 10, fill="#1d4ed8"))
    b.append(txt(600, y+30, "parent", 10, fill="#1d4ed8"))

    # extra keys sources
    ey = 240
    b.append(txt(w/2, ey, "extra_keys 来源（generate_block_hash_extra_keys）", 12.5, weight="bold", fill="#334155"))
    def src(x, lab, sub):
        out = [box(x, ey+18, 200, 56, "#fef9c3", "#a16207", rx=8)]
        out.append(txt(x+100, ey+40, lab, 11.5, mono=True))
        out.append(txt(x+100, ey+60, sub, 10, fill="#475569"))
        return out
    b += src(60, "mm_hash + 块内偏移", "多模态特征隔离")
    b += src(350, "lora_name", "适配器命名空间隔离")
    b += src(640, "cache_salt（仅首块）", "显式租户命名空间隔离")

    b.append(txt(w/2, h-22,
                 "相同 token 但 parent / extra 不同 → 不同 hash，杜绝跨语义错误命中别人的 KV",
                 11.5, fill="#b91c1c"))
    return svg(w, h, "".join(b)), "block-hash-chain"


# ============ 4: prefix-hit-alloc ============
def prefix_hit_alloc():
    w, h = 880, 560
    b = []
    b.append(txt(w/2, 30, "allocate_slots 三段式 + 前缀命中复用时序", 16, weight="bold"))

    def stage(y, n, title, fill, lines):
        out = [box(70, y, 740, 30 + 18*len(lines), fill, "#475569", rx=8)]
        out.append(txt(96, y+24, f"段{n}　{title}", 13, anchor="start", weight="bold"))
        for i, ln in enumerate(lines):
            out.append(txt(120, y+44+i*18, ln, 11, anchor="start", mono=True, fill="#334155"))
        return out, 30 + 18*len(lines)

    y = 60
    s, dh = stage(y, "前", "get_computed_blocks：沿 block_hashes 查 cached_block_hash_to_block", "#eef2ff",
                  ["命中 k 块即停（链一断即停）→ num_new_computed_tokens = k·block_size",
                   "max_cache_hit_length = num_tokens − 1（至少重算最后一个 token 取 logits）"])
    b += s; y += dh + 18
    b.append(line(440, y-18, 440, y, marker="a"))

    s, dh = stage(y, "一", "容量检查", "#f1f5f9",
                  ["num_blocks_to_allocate（含可驱逐命中块计数）> 空闲块数 → 返回 None"])
    b += s; y += dh + 18
    b.append(line(440, y-18, 440, y, marker="a"))

    s, dh = stage(y, "二", "allocate_new_computed_blocks → touch(命中块)", "#ecfdf5",
                  ["ref_cnt += 1；若命中块 ref_cnt 原为 0（在 free queue）→ remove() 救回",
                   "挂进 req_to_blocks，标记 num_cached_block，不重新分配物理块"])
    b += s; y += dh + 18
    b.append(line(440, y-18, 440, y, marker="agrn", color="#15803d"))

    s, dh = stage(y, "三", "allocate_new_blocks → get_new_blocks(剩余 token 的块数)", "#f1f5f9",
                  ["popleft_n 取块；每块 _maybe_evict_cached_block（有旧 hash 则摘除并 reset）",
                   "再 cache_blocks → cache_full_blocks 给新满块写 hash 并 insert 进 map"])
    b += s; y += dh + 18

    b.append(box(70, y, 740, 46, "#fef2f2", "#b91c1c", rx=8))
    b.append(txt(90, y+19, "被抢占请求重排回来：从 0 重 prefill，但其前缀块若未被取走复用、hash 仍在 map →",
                 11, anchor="start", fill="#7f1d1d"))
    b.append(txt(90, y+38, "get_computed_blocks 直接命中、touch 救回，重算只剩命中长度之外的部分。",
                 11, anchor="start", fill="#7f1d1d", weight="bold"))
    return svg(w, h, "".join(b)), "prefix-hit-alloc"


if __name__ == "__main__":
    import subprocess
    from pathlib import Path
    here = Path(__file__).parent
    for fn in (paging_overview, free_queue_lru, block_hash_chain, prefix_hit_alloc):
        content, name = fn()
        svg_path = here / f"{name}.svg"
        svg_path.write_text(content, encoding="utf-8")
        subprocess.run(["rsvg-convert", "-z", "2", str(svg_path), "-o",
                        str(here / f"{name}.png")], check=True)
        print("wrote", name)
