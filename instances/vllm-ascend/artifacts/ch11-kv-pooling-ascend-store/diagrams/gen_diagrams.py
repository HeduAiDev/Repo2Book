#!/usr/bin/env python3
"""Generate ch11 diagrams: two-end decoupling, pool-vs-pd contrast, key/value addressing."""
import os, subprocess
import xml.sax.saxutils as xs

HERE = os.path.dirname(os.path.abspath(__file__))


def esc(s):
    return xs.escape(s)


def box(x, y, w, h, fill, stroke, rx=6, sw=2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, s, size=13, fill="#1e293b", anchor="middle", weight="normal", mono=False, italic=False):
    fam = "monospace" if mono else "sans-serif"
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{fam}" '
            f'font-size="{size}" fill="{fill}" font-weight="{weight}"{st}>{esc(s)}</text>')


def arrow(x1, y1, x2, y2, color="#64748b", sw=2, dash=None, marker="a"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}"{d} marker-end="url(#{marker})"/>')


def defs():
    return ('<defs>'
            '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
            '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
            '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
            '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
            '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
            '<path d="M0,0 L10,3 L0,6 Z" fill="#0d9488"/></marker>'
            '</defs>')


def save(name, svg):
    sp = os.path.join(HERE, name + ".svg")
    pp = os.path.join(HERE, name + ".png")
    with open(sp, "w") as f:
        f.write(svg)
    assert subprocess.run(["xmllint", "--noout", sp]).returncode == 0, name
    subprocess.run(["rsvg-convert", "-z", "2", sp, "-o", pp], check=True)
    print("wrote", pp)


# ── Diagram 1: two-end decoupling ────────────────────────────────────
def two_end():
    w, h = 980, 560
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', defs(),
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w/2, 36, "两端解耦：调度进程「决定搬什么」，worker 进程后台线程「异步搬」", 18, "#0f172a", weight="bold"))

    # lanes
    sch_x, sch_w = 40, 380
    wrk_x, wrk_w = 560, 380
    lane_y, lane_h = 70, 360
    L.append(box(sch_x, lane_y, sch_w, lane_h, "#eff6ff", "#bfdbfe", rx=10, sw=2))
    L.append(box(wrk_x, lane_y, wrk_w, lane_h, "#f5f3ff", "#ddd6fe", rx=10, sw=2))
    L.append(text(sch_x + sch_w/2, lane_y + 24, "Scheduler 进程 · KVPoolScheduler", 15, "#1d4ed8", weight="bold"))
    L.append(text(sch_x + sch_w/2, lane_y + 42, "（无 KV、无后端连接）", 11, "#3b82f6"))
    L.append(text(wrk_x + wrk_w/2, lane_y + 24, "Worker 进程 · KVPoolWorker", 15, "#6d28d9", weight="bold"))
    L.append(text(wrk_x + wrk_w/2, lane_y + 42, "（持有 m_store 后端 + 收/发后台线程）", 11, "#7c3aed"))

    # scheduler boxes
    s1y = lane_y + 64
    L.append(box(sch_x + 30, s1y, 320, 46, "#dbeafe", "#3b82f6"))
    L.append(text(sch_x + 190, s1y + 19, "get_num_new_matched_tokens", 12.5, "#1e3a8a", weight="bold", mono=True))
    L.append(text(sch_x + 190, s1y + 36, "问池：本请求多少前缀已在池中", 11, "#1e40af"))

    s2y = s1y + 110
    L.append(box(sch_x + 30, s2y, 320, 44, "#dbeafe", "#3b82f6"))
    L.append(text(sch_x + 190, s2y + 18, "建 LoadSpec → build_connector_meta", 12, "#1e3a8a", weight="bold", mono=True))
    L.append(text(sch_x + 190, s2y + 35, "每拍打包 load/save 请求 = 一个节拍", 11, "#1e40af"))

    # worker boxes
    w1y = lane_y + 64
    L.append(box(wrk_x + 30, w1y, 320, 44, "#ede9fe", "#7c3aed"))
    L.append(text(wrk_x + 190, w1y + 18, "LookupKeyServer → lookup_scheduler", 12, "#5b21b6", weight="bold", mono=True))
    L.append(text(wrk_x + 190, w1y + 35, "m_store.exists(keys) 查池命中前缀", 11, "#6d28d9"))

    w2y = w1y + 92
    L.append(box(wrk_x + 30, w2y, 320, 44, "#ede9fe", "#7c3aed"))
    L.append(text(wrk_x + 190, w2y + 18, "start_load_kv / wait_for_save", 12, "#5b21b6", weight="bold", mono=True))
    L.append(text(wrk_x + 190, w2y + 35, "add_request 入队即返回（主循环不阻塞）", 10.5, "#6d28d9"))

    w3y = w2y + 70
    L.append(box(wrk_x + 30, w3y, 320, 58, "#fef3c7", "#d97706"))
    L.append(text(wrk_x + 190, w3y + 20, "request_queue（解耦点）", 12.5, "#92400e", weight="bold", mono=True))
    L.append(text(wrk_x + 190, w3y + 38, "收/发线程 while True: get → _handle_request", 10.5, "#b45309", mono=True))
    L.append(text(wrk_x + 190, w3y + 52, "join() = 背压屏障，等 put 落地才放行", 10.5, "#b45309"))

    # external pool
    pool_y = lane_y + lane_h + 30
    L.append(box(260, pool_y, 460, 56, "#ecfdf5", "#10b981", rx=10, sw=2.5))
    L.append(text(490, pool_y + 23, "External Pool · Backend 契约（MooncakeBackend）", 14, "#047857", weight="bold"))
    L.append(text(490, pool_y + 42, "exists / put / get → 内容寻址的共享 KV 池", 11.5, "#059669"))

    # arrows: scheduler s1 -> worker w1 (zmq lookup)
    L.append(arrow(sch_x + 350, s1y + 22, wrk_x + 28, w1y + 22, "#7c3aed", 2.2, marker="b"))
    L.append(text((sch_x + 350 + wrk_x + 28)/2, s1y + 12, "② zmq REQ→REP lookup", 11, "#7c3aed", weight="bold"))
    # worker w1 -> scheduler s2 (hit count back) -- curved feel via straight
    L.append(arrow(wrk_x + 28, w1y + 40, sch_x + 350, s2y + 14, "#7c3aed", 2, dash="5,4", marker="b"))
    L.append(text((sch_x + 350 + wrk_x + 28)/2 + 10, s2y - 6, "④ 回命中长度", 11, "#7c3aed"))
    # scheduler s2 -> worker w2 (metadata下发)
    L.append(arrow(sch_x + 350, s2y + 36, wrk_x + 28, w2y + 22, "#1d4ed8", 2.4, marker="a"))
    L.append(text((sch_x + 350 + wrk_x + 28)/2, s2y + 64, "⑤ SchedulerOutput 携 metadata", 11, "#1d4ed8", weight="bold"))
    # worker w2 -> queue
    L.append(arrow(wrk_x + 190, w2y + 44, wrk_x + 190, w3y, "#d97706", 2.2))
    L.append(text(wrk_x + 250, w2y + 60, "⑥ 入队", 10.5, "#b45309"))
    # queue -> pool
    L.append(arrow(wrk_x + 190, w3y + 58, 600, pool_y, "#0d9488", 2.4, marker="g"))
    L.append(text(wrk_x + 120, w3y + 78, "⑦ 后台线程 put / get", 11, "#0d9488", weight="bold"))
    # worker w1 -> pool (exists during lookup)
    L.append(arrow(wrk_x + 120, w1y + 44, 430, pool_y, "#0d9488", 1.8, dash="4,4", marker="g"))
    L.append(text(wrk_x + 10, pool_y - 16, "③ exists", 10.5, "#0d9488"))
    # step ① on scheduler
    L.append(text(sch_x + 60, s1y - 8, "① vLLM 调度调用", 11, "#1d4ed8"))
    # step ⑧ get_finished
    L.append(arrow(660, pool_y + 56, wrk_x + 300, w3y + 58, "#0d9488", 1.6, dash="3,4", marker="g"))
    L.append(text(720, pool_y + 78, "⑧ get_finished 回收 block", 11, "#0d9488"))

    L.append('</svg>')
    return '\n'.join(L)


# ── Diagram 2: pool vs PD contrast ───────────────────────────────────
def contrast():
    w, h = 980, 500
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', defs(),
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w/2, 34, "两种省重算：P2P 直传（ch10） vs 经外存储池化复用（ch11）", 18, "#0f172a", weight="bold"))

    # left panel
    lx, lw = 30, 440
    L.append(box(lx, 60, lw, 300, "#fef2f2", "#fecaca", rx=12, sw=2))
    L.append(text(lx + lw/2, 86, "ch10 · PD 分离 = 节点间 P2P 直传", 15, "#b91c1c", weight="bold"))
    L.append(box(lx + 50, 130, 150, 60, "#fee2e2", "#ef4444"))
    L.append(text(lx + 125, 156, "Prefill 实例", 13, "#991b1b", weight="bold"))
    L.append(text(lx + 125, 175, "算出 prompt 的 KV", 10.5, "#b91c1c"))
    L.append(box(lx + 240, 130, 150, 60, "#fee2e2", "#ef4444"))
    L.append(text(lx + 315, 156, "Decode 实例", 13, "#991b1b", weight="bold"))
    L.append(text(lx + 315, 175, "接着往下吐 token", 10.5, "#b91c1c"))
    L.append(arrow(lx + 200, 160, lx + 238, 160, "#dc2626", 3))
    L.append(text(lx + lw/2, 215, "KV 点对点直发，一次性", 12, "#dc2626", weight="bold"))
    L.append(text(lx + lw/2, 250, "拓扑：点对点（prefill → decode）", 11.5, "#7f1d1d"))
    L.append(text(lx + lw/2, 272, "节拍：随传随用，传完即弃", 11.5, "#7f1d1d"))
    L.append(text(lx + lw/2, 294, "复用面：单次（这一对实例）", 11.5, "#7f1d1d"))
    L.append(text(lx + lw/2, 330, "寻址：拓扑路由（谁发给谁）", 11.5, "#7f1d1d"))

    # right panel
    rx, rw = 510, 440
    L.append(box(rx, 60, rw, 300, "#eff6ff", "#bfdbfe", rx=12, sw=2))
    L.append(text(rx + rw/2, 86, "ch11 · KV 池化 = 经外存储中转复用", 15, "#1d4ed8", weight="bold"))
    # pool center
    L.append(box(rx + 150, 150, 140, 64, "#dbeafe", "#2563eb", rx=10))
    L.append(text(rx + 220, 176, "共享 KV 池", 13, "#1e3a8a", weight="bold"))
    L.append(text(rx + 220, 196, "by chunk_hash", 10.5, "#1d4ed8", mono=True))
    # producers
    L.append(box(rx + 20, 118, 100, 40, "#eff6ff", "#3b82f6"))
    L.append(text(rx + 70, 142, "Req A", 12, "#1e40af", weight="bold"))
    L.append(box(rx + 20, 200, 100, 40, "#eff6ff", "#3b82f6"))
    L.append(text(rx + 70, 224, "Req B", 12, "#1e40af", weight="bold"))
    L.append(box(rx + 320, 118, 100, 40, "#eff6ff", "#3b82f6"))
    L.append(text(rx + 370, 142, "Req C", 12, "#1e40af", weight="bold"))
    L.append(box(rx + 320, 200, 100, 40, "#eff6ff", "#3b82f6"))
    L.append(text(rx + 370, 224, "另一实例", 11, "#1e40af", weight="bold"))
    L.append(arrow(rx + 120, 138, rx + 148, 168, "#2563eb", 2.2))
    L.append(arrow(rx + 120, 220, rx + 148, 196, "#2563eb", 2.2))
    L.append(text(rx + 130, 250, "put（存）", 10.5, "#1d4ed8"))
    L.append(arrow(rx + 290, 178, rx + 318, 142, "#0d9488", 2.2, marker="g"))
    L.append(arrow(rx + 290, 192, rx + 318, 222, "#0d9488", 2.2, marker="g"))
    L.append(text(rx + 300, 252, "get（按 key 捞）", 10.5, "#0d9488"))
    L.append(text(rx + rw/2, 294, "拓扑：星型经池 · 节拍：存后异步复用", 11.5, "#1e3a5f"))
    L.append(text(rx + rw/2, 316, "复用面：跨请求 / 跨实例 · 寻址：内容哈希", 11.5, "#1e3a5f"))
    L.append(text(rx + rw/2, 338, "存前 lookup 去重 → 同前缀只存一份", 11.5, "#1e3a5f"))

    # bottom note
    L.append(box(30, 392, 920, 78, "#f8fafc", "#cbd5e1", rx=10))
    L.append(text(w/2, 416, "同样是「不重算已知前缀」，但一个靠拓扑直传、一个靠内容寻址的共享存储。", 13, "#334155", weight="bold"))
    L.append(text(w/2, 440, "直传省一次跨节点的搬运；池化把 KV 攒成可被任何后来请求命中的共享资产——", 12, "#475569"))
    L.append(text(w/2, 460, "key 由 prompt 前缀内容哈希决定，所以相同前缀跨请求、跨实例都能复用。", 12, "#475569"))

    L.append('</svg>')
    return '\n'.join(L)


# ── Diagram 3: key & value addressing ────────────────────────────────
def addressing():
    w, h = 940, 440
    L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">', defs(),
         f'<rect width="{w}" height="{h}" fill="white"/>']
    L.append(text(w/2, 34, "「搬哪个 chunk」与「在显存哪一段」分两路算，再汇入后端契约", 18, "#0f172a", weight="bold"))

    # top path: key
    L.append(box(40, 80, 180, 56, "#eff6ff", "#3b82f6"))
    L.append(text(130, 104, "token 序列", 13, "#1e40af", weight="bold"))
    L.append(text(130, 122, "+ prefix block_hashes", 10.5, "#1d4ed8", mono=True))
    L.append(box(320, 80, 200, 56, "#dbeafe", "#2563eb"))
    L.append(text(420, 102, "process_tokens", 13, "#1e3a8a", weight="bold", mono=True))
    L.append(text(420, 121, "按 chunk 切，逐块生成", 10.5, "#1d4ed8"))
    L.append(box(620, 74, 280, 68, "#e0e7ff", "#4f46e5"))
    L.append(text(760, 96, "PoolKey.to_string()", 13, "#3730a3", weight="bold", mono=True))
    L.append(text(760, 114, "model@pcp@dcp@tp@pp@group", 10, "#4338ca", mono=True))
    L.append(text(760, 130, "@cache_role@cache_family@chunk_hash", 9.5, "#4338ca", mono=True))
    L.append(arrow(220, 108, 318, 108, "#2563eb", 2.4))
    L.append(arrow(520, 108, 618, 108, "#2563eb", 2.4))
    L.append(text(130, 70, "「搬哪个」= 名字", 12, "#1d4ed8", weight="bold"))

    # bottom path: value addr
    L.append(box(40, 230, 180, 56, "#f0fdf4", "#16a34a"))
    L.append(text(130, 254, "block_id", 13, "#15803d", weight="bold", mono=True))
    L.append(text(130, 272, "（该 chunk 占的物理块）", 10, "#16a34a"))
    L.append(box(320, 226, 200, 64, "#dcfce7", "#16a34a"))
    L.append(text(420, 248, "prepare_value", 13, "#14532d", weight="bold", mono=True))
    L.append(text(420, 266, "base_addr + block_id×stride", 9.5, "#15803d", mono=True))
    L.append(text(420, 282, "size = block_len/bs ×(end−start)", 9.5, "#15803d", mono=True))
    L.append(box(620, 232, 280, 56, "#d1fae5", "#059669"))
    L.append(text(760, 256, "(addr, size) 列表", 13, "#065f46", weight="bold", mono=True))
    L.append(text(760, 274, "每层 K/V 一段显存区间", 10.5, "#047857"))
    L.append(arrow(220, 258, 318, 258, "#16a34a", 2.4, marker="g"))
    L.append(arrow(520, 258, 618, 258, "#16a34a", 2.4, marker="g"))
    L.append(text(130, 220, "「在哪」= 显存地址", 12, "#15803d", weight="bold"))

    # converge to backend
    L.append(box(330, 350, 280, 60, "#fef3c7", "#d97706", rx=10, sw=2.5))
    L.append(text(470, 376, "Backend.put(keys, addrs, sizes)", 13, "#92400e", weight="bold", mono=True))
    L.append(text(470, 396, "/ get(...) · exists(keys)", 11.5, "#b45309", mono=True))
    L.append(arrow(760, 142, 540, 348, "#4f46e5", 2.2, marker="b"))
    L.append(arrow(760, 288, 540, 352, "#059669", 2.2, marker="g"))
    L.append(text(700, 230, "keys", 11, "#4f46e5", weight="bold"))
    L.append(text(700, 320, "addrs/sizes", 11, "#059669", weight="bold"))

    # dedup note
    L.append(box(40, 350, 250, 60, "#fef2f2", "#ef4444", rx=10))
    L.append(text(165, 374, "put 前先 exists(keys) 去重", 11.5, "#b91c1c", weight="bold"))
    L.append(text(165, 393, "池里已有的 chunk 跳过 → 只存一份", 10.5, "#dc2626"))
    L.append(arrow(290, 380, 328, 380, "#ef4444", 2))

    L.append('</svg>')
    return '\n'.join(L)


if __name__ == "__main__":
    save("two-end-decoupling", two_end())
    save("pool-vs-pd-contrast", contrast())
    save("key-and-value-addressing", addressing())
