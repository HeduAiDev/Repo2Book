#!/usr/bin/env python3
"""ch30-nixl-read-flow: NIXL RDMA READ 单边读时序图。两泳道 D-worker / P-worker。"""
import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 1060, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>')
L.append('<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>')
L.append('<marker id="arBig" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>')
L.append('<marker id="arG" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>')
L.append('</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
L.append(f'<text x="{w/2}" y="32" text-anchor="middle" font-size="21" font-weight="bold" fill="#0f172a">NIXL：D 单边 RDMA READ 直接从 P 显存拉 KV</text>')

# two lifelines
dx = 280   # D-worker (decode) lifeline
px = 800   # P-worker (prefill) lifeline
top = 70
bot = 600
for x, name, col in [(dx, "D-worker (decode)", "#dcfce7"), (px, "P-worker (prefill)", "#dbeafe")]:
    L.append(f'<rect x="{x-130}" y="{top}" width="260" height="40" rx="8" fill="{col}" stroke="#475569" stroke-width="1.5"/>')
    L.append(f'<text x="{x}" y="{top+26}" text-anchor="middle" font-size="16" font-weight="bold" fill="#0f172a">{esc(name)}</text>')
    L.append(f'<line x1="{x}" y1="{top+40}" x2="{x}" y2="{bot}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6,5"/>')

def event(x, y, text, fill="#f1f5f9", stroke="#64748b", tcol="#0f172a", bw=210, bh=38):
    L.append(f'<rect x="{x-bw/2}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    lines=text.split("\n")
    yy=y+bh/2-(len(lines)-1)*8+4
    for i,ln in enumerate(lines):
        L.append(f'<text x="{x}" y="{yy+i*15}" text-anchor="middle" font-size="12.5" fill="{tcol}">{esc(ln)}</text>')

def msg(x1, y, x2, text, marker="url(#ar)", col="#475569", dash=None, fs=12):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{col}" stroke-width="1.8"{d} marker-end="{marker}"/>')
    midx=(x1+x2)/2
    L.append(f'<text x="{midx}" y="{y-7}" text-anchor="middle" font-size="{fs}" fill="{col}">{esc(text)}</text>')

y = 135
event(dx, y, "start_load_kv：首遇 P，\n_remote_agents 里没有它", "#dcfce7", "#22c55e");
y += 60
msg(dx+12, y, px-12, "后台握手：交换内存注册元数据 (agent metadata)", dash="5,4")
y += 18
msg(px-12, y+18, dx+12, "握手回包；本请求挂起，待 _ready_requests 取出补发", dash="5,4")
y += 70
event(dx, y, "make_prepped_xfer('READ') + transfer\nhandle 存入 _recving_transfers", "#fee2e2", "#ef4444", tcol="#b91c1c")
y += 58
# big READ arrow P显存 -> D paged buffer
L.append(f'<line x1="{px-12}" y1="{y}" x2="{dx+12}" y2="{y}" stroke="#b91c1c" stroke-width="4.5" marker-end="url(#arBig)"/>')
L.append(f'<text x="{(px+dx)/2}" y="{y-10}" text-anchor="middle" font-size="13.5" font-weight="bold" fill="#b91c1c">RDMA 单边 READ：P 显存 → D paged buffer</text>')
L.append(f'<text x="{(px+dx)/2}" y="{y+18}" text-anchor="middle" font-size="12" fill="#b91c1c">（P 端 CPU/GPU 完全不参与传输）</text>')
y += 56
event(dx, y, "后续 step 轮询 check_xfer_state(handle)\nPROC=留着 / DONE=收完成", "#dcfce7", "#22c55e")
y += 58
msg(dx+12, y, px-12, "读完 send_notif 通知 P", marker="url(#arG)", col="#16a34a")
y += 50
event(dx-120, y, "get_finished →\nfinished_recving", "#dcfce7", "#22c55e", bw=200, bh=40)
event(px+0, y, "get_finished 收通知 →\nfinished_sending，释放被读 block", "#dbeafe", "#3b82f6", bw=250, bh=40)
y += 64
# timeout note
L.append(f'<rect x="{px-150}" y="{y}" width="300" height="40" rx="6" fill="#fef9c3" stroke="#eab308" stroke-width="1.3"/>')
L.append(f'<text x="{px}" y="{y+18}" text-anchor="middle" font-size="12" fill="#854d0e">超时兜底 (reqs_to_send 过期)：</text>')
L.append(f'<text x="{px}" y="{y+33}" text-anchor="middle" font-size="12" fill="#854d0e">防 D 失联把 P 的 block 永久占住</text>')

L.append('</svg>')
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch30-pd-disaggregation/diagrams/ch30-nixl-read-flow.svg","w").write('\n'.join(L))
print("ok")
