#!/usr/bin/env python3
"""l0_common —— v3 图系共享模块（FIGURE-SYSTEM.md §0 硬规则 2/3 的落点）

L0/L1/L2 三层共享：布局常量、配色常量、字体栈、绘制原语。
改配色/术语只许改这里的常量，全层联动。

本模块持有：
  - 画布/配色/字体常量（唯一真相源）；
  - 绘制原语（text/rect/seg/parrow/…）：每次发射同时记录元素 bbox
    （供 gen_L1.py 做 minimap 框内/框外亮暗分区与裁切判定的逐元素分区）；
  - build_l0()：L0 全部绘制主体（自 fable 执笔的 gen_L0.py 逐行迁移，未改任何坐标），
    返回 (ELEMS, GEO, WARN)。ELEMS = [(bbox, svg_str), …]；GEO = 关键布局坐标字典
    （L1 的 Part→区域映射从这里取坐标，L0 改版自动联动，杜绝坐标漂移）。
"""
import xml.sax.saxutils as xs

FONT = 'Microsoft YaHei, SimHei, Noto Sans CJK SC, PingFang SC, sans-serif'
C_TXT, C_MUTE, C_FAINT = '#0f172a', '#475569', '#94a3b8'
C_API_S, C_API_F = '#2563eb', '#eff6ff'      # API 进程 = 蓝
C_ZMQ_S, C_ZMQ_F = '#7c3aed', '#f5f3ff'      # ZMQ 边界 = 紫
C_ENG_S, C_ENG_F = '#ea580c', '#fff7ed'      # EngineCore 进程 = 橙
C_GPU_S, C_GPU_F = '#16a34a', '#f0fdf4'      # GPU 执行臂 = 绿
C_KV_S, C_KV_F = '#0891b2', '#ecfeff'        # 显存账本 = 青
C_SAM_S, C_SAM_F = '#be185d', '#fdf2f8'      # 采样出口 = 品红
C_ABORT = '#dc2626'                          # abort = 红虚线

W = 2200

DEFS = ('<defs>'
        '<marker id="std" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4.2" orient="auto">'
        f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_MUTE}"/></marker>'
        '<marker id="dn" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
        f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_API_S}"/></marker>'
        '<marker id="up" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
        f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_ENG_S}"/></marker>'
        '<marker id="ab" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
        f'<path d="M0,0 L10,3 L0,6 Z" fill="{C_ABORT}"/></marker>'
        '</defs>')

# ---------- 发射状态：元素流（带 bbox）+ 越界警告 ----------
ELEMS = []   # [(bbox(x0,y0,x1,y1) | None, svg_str)]
WARN = []


def reset():
    ELEMS.clear()
    WARN.clear()


def esc(s):
    return xs.escape(str(s))


def tw(s, fs, bold=False):
    """宽度估算：CJK=fs，ASCII=0.58*fs，粗体*1.07"""
    n = sum(1 for c in str(s) if ord(c) > 0x2E80)
    return (n * fs + (len(str(s)) - n) * fs * 0.58) * (1.07 if bold else 1.0)


def fit(s, fs, maxw, tag, bold=False):
    if tw(s, fs, bold) > maxw:
        WARN.append(f'OVERFLOW [{tag}] need {tw(s, fs, bold):.0f} > {maxw:.0f}: {s[:40]}')
    return fs


def text_svg(x, y, s, fs, fill, anchor, bold):
    b = ' font-weight="bold"' if bold else ''
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{fs}" '
            f'fill="{fill}" text-anchor="{anchor}"{b}>{esc(s)}</text>')


def rect_svg(x, y, w, h, fill, stroke, rx, sw, dash):
    d = ' stroke-dasharray="6,4"' if dash else ''
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, fs=11, fill=C_TXT, anchor='middle', bold=False, maxw=None, tag=''):
    if maxw:
        fit(s, fs, maxw, tag or s[:16], bold)
    w = tw(s, fs, bold)
    x0 = x - w / 2 if anchor == 'middle' else (x - w if anchor == 'end' else x)
    ELEMS.append(((x0 - 2, y - 0.85 * fs - 1.5, x0 + w + 2, y + 0.25 * fs + 1.5),
                  text_svg(x, y, s, fs, fill, anchor, bold)))


def rect(x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=False):
    ELEMS.append(((x - 2, y - 2, x + w + 2, y + h + 2),
                  rect_svg(x, y, w, h, fill, stroke, rx, sw, dash)))


def seg(x1, y1, x2, y2, color, sw=1.8, marker=None, dash=False):
    m = f' marker-end="url(#{marker})"' if marker else ''
    d = ' stroke-dasharray="6,4"' if dash else ''
    s = (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
         f'stroke="{color}" stroke-width="{sw}"{d}{m}/>')
    ELEMS.append(((min(x1, x2) - 8, min(y1, y2) - 8, max(x1, x2) + 8, max(y1, y2) + 8), s))


def parrow(pts, color, sw=1.8, marker='std', dash=False):
    """折线箭头：pts = [(x,y),...]"""
    d = ' stroke-dasharray="6,4"' if dash else ''
    path = ' '.join(f'{"M" if i == 0 else "L"}{p[0]:.1f},{p[1]:.1f}' for i, p in enumerate(pts))
    s = (f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{sw}"{d} '
         f'marker-end="url(#{marker})"/>')
    xs_ = [p[0] for p in pts]
    ys_ = [p[1] for p in pts]
    ELEMS.append(((min(xs_) - 8, min(ys_) - 8, max(xs_) + 8, max(ys_) + 8), s))


def alabel(x, y, s, fs=9, fill=C_MUTE, anchor='start'):
    text(x, y, s, fs, fill, anchor, maxw=9999, tag='arrow:' + s[:14])


def comp(x, y, w, title, lines, file, stroke, fill='#ffffff', badge='', lh=17, tfs=12, lfs=9.5):
    """组件框：类名(粗) + 方法行若干 + 底部源码路径。返回 (y, h)。"""
    n = len(lines)
    h = 34 + n * lh + (20 if file else 6)
    rect(x, y, w, h, fill, stroke, rx=7, sw=1.4)
    tx = x + 14
    text(tx, y + 22, title, tfs, C_TXT, 'start', True, maxw=w - 28 - (46 if badge else 0), tag=title)
    if badge:
        bw = 16 + 11 * len(badge)
        rect(x + w - bw - 8, y + 6, bw, 20, '#ffedd5', C_ENG_S, rx=9, sw=1.1)
        text(x + w - bw / 2 - 8, y + 20, badge, 9.5, C_ENG_S, 'middle', True)
    for i, ln in enumerate(lines):
        text(tx, y + 40 + i * lh, ln, lfs, '#334155', 'start', maxw=w - 26, tag=title + ':' + ln[:12])
    if file:
        text(tx, y + h - 9, file, 9, C_FAINT, 'start', maxw=w - 26, tag='file:' + title)
    return y, h


def queue_glyph(x, y, w, name, sub):
    h = 96
    rect(x, y, w, h, '#ffffff', C_MUTE, rx=7, sw=1.3)
    text(x + w / 2, y + 18, name, 10, C_TXT, 'middle', True, maxw=w - 10, tag=name)
    bw, gap = 7, 5
    bx = x + (w - (3 * bw + 2 * gap)) / 2
    for i in range(3):
        rect(bx + i * (bw + gap), y + 32, bw, 26, '#cbd5e1', C_MUTE, rx=1.5, sw=1.0)
    text(x + w / 2, y + 78, sub, 9, C_MUTE, 'middle', maxw=w - 8, tag=name + ':sub')
    return h


def build_l0():
    """L0 全部绘制主体（与 fable 执笔版逐行同源）。返回 (ELEMS, GEO, WARN)。"""
    reset()

    MX, CW = 36, W - 72            # 外边距 / 内容宽
    BXR = MX + CW                  # 内容右缘 2164

    # ---------- 标题 ----------
    text(MX, 38, 'vLLM v1 全书架构图（L0 · 唯一权威图）：一个请求的一生', 20, C_TXT, 'start', True)
    text(MX, 64, '源码 pin vLLM v0.27.1 · 三段式进程解耦 · 引擎逐拍循环「调度 → 执行 → 采样 → 回收」 · 显存共享账本',
         10.5, C_MUTE, 'start')

    # ---------- 用户行 ----------
    UY, UH, UW, UGAP = 92, 56, 250, 150
    UX0 = (W - (3 * UW + 2 * UGAP)) / 2          # 575
    user_cx = [UX0 + UW / 2 + i * (UW + UGAP) for i in range(3)]   # 700/1100/1500
    for i, cx in enumerate(user_cx):
        rect(cx - UW / 2, UY, UW, UH, '#ffffff', C_MUTE, rx=10, sw=1.5)
        text(cx, UY + 24, f'user{i + 1}', 12, C_TXT, 'middle', True)
        text(cx, UY + 43, 'HTTP 客户端', 9, C_MUTE)

    # ========== API 进程 ==========
    AY = 192
    LANE_W = 600
    LCX, RCX = 560, 1770                       # 左泳道(上行) / 右泳道(下行) 中心
    DIVX = (LCX + RCX) // 2                    # 泳道虚线分隔 1165
    ENT_Y, ENT_H = AY + 40, 58                 # 入口条
    LY0 = AY + 116                             # 泳道首框 y = 308
    PITCH = 168                                # 泳道框间距（框高取 comp() 实际返回，不用常量）
    AH = 596

    rect(MX, AY, CW, AH, C_API_F, C_API_S, rx=12, sw=2.4)
    text(MX + 16, AY + 24, 'API 进程（frontend · 零 GPU）', 13.5, C_API_S, 'start', True)
    text(BXR - 16, AY + 24, 'tokenize / detokenize / HTTP / SSE 的 CPU 活全在此进程', 9.5, C_MUTE, 'end')

    # 入口条（两个使用面）
    rect(MX + 20, ENT_Y, CW - 40, ENT_H, '#ffffff', C_API_S, rx=8, sw=1.5)
    text(MX + 36, ENT_Y + 22, 'OpenAI server / LLM（离线） / AsyncLLM（在线）—— 两个使用面，同一套三件套',
         12, C_TXT, 'start', True, maxw=CW - 500, tag='entry')
    text(BXR - 36, ENT_Y + 22, 'vllm/entrypoints · vllm/v1/engine/async_llm.py', 9, C_FAINT, 'end')
    text(MX + 36, ENT_Y + 43, '· generate(prompt, sampling_params) 逐条 yield RequestOutput　·　add_request 双登记：先本进程建表，后跨进程发引擎',
         9.5, '#334155', 'start', maxw=CW - 110, tag='entry:sub')

    # 用户 ↔ 入口条 箭头（下蓝上橙）
    for i, cx in enumerate(user_cx):
        seg(cx - 14, UY + UH, cx - 14, ENT_Y, C_API_S, 1.8, 'dn')
        seg(cx + 14, ENT_Y, cx + 14, UY + UH, C_ENG_S, 1.8, 'up')
        alabel(cx + 21, UY + UH + 18, 'sse', 9, C_MUTE)
    alabel(user_cx[1] - 20, UY + UH + 18, 'HTTP 请求', 9, C_API_S, 'end')

    # 泳道分隔虚线
    seg(DIVX, LY0 - 6, DIVX, AY + AH - 20, C_FAINT, 1.2, dash=True)

    # ---- 右泳道（下行：tokenize → 组包） ----
    rx = RCX - LANE_W / 2
    _, r1h = comp(rx, LY0, LANE_W, 'Renderer.render',
                  ['· render_chat / render_cmpl 两个入口',
                   '· _tokenize_prompt: tokenizer.encode → TokensPrompt',
                   '· tokenize 在前端进程，文本不过 IPC',
                   '· 原始 prompt 的阻塞 tokenize 跑线程池，不下事件循环'],
                  'vllm/renderers/base.py', C_API_S)
    _, r2h = comp(rx, LY0 + PITCH, LANE_W, 'InputProcessor',
                  ['· process_inputs: 校验 + params.clone() + 整理 mm_features',
                   '· assign_request_id 双轨：外部 id + 内部「id-8位hex」',
                   '· 组装 EngineCoreRequest（只有 token ids）'],
                  'vllm/v1/engine/input_processor.py', C_API_S)
    _, r3h = comp(rx, LY0 + 2 * PITCH, LANE_W, 'EngineCoreRequest（msgspec Struct）',
                  ['· prompt_token_ids + sampling_params + mm_features',
                   '· array_like 按位置编码 · omit_defaults 省字节',
                   '· 大 tensor 走零拷贝独立帧 / OOB 共享内存旁路'],
                  'vllm/v1/engine/__init__.py', C_API_S)
    rbox_h = [r1h, r2h, r3h]
    # 下行箭头
    seg(RCX, ENT_Y + ENT_H, RCX, LY0, C_API_S, 2.0, 'dn')
    alabel(RCX + 7, ENT_Y + ENT_H + 13, 'add_request(prompt, sampling_params)', 9, C_API_S)
    for k, lab in [(0, 'TokensPrompt（只有 token ids）'), (1, 'process_inputs 产出')]:
        seg(RCX, LY0 + k * PITCH + rbox_h[k], RCX, LY0 + (k + 1) * PITCH, C_API_S, 2.0, 'dn')
        alabel(RCX + 7, LY0 + k * PITCH + rbox_h[k] + 26, lab, 9, C_API_S)

    # ---- 左泳道（上行：拆包 → 拼字 → 流式） ----
    lx = LCX - LANE_W / 2
    _, l1h = comp(lx, LY0, LANE_W, 'RequestOutputCollector ×3（每请求一个）',
                  ['· 单槽 + Event + 生产侧合并',
                   '· 刻意不用 asyncio.Queue（慢消费者防堆积）',
                   '· 三态契约：DELTA / CUMULATIVE / FINAL_ONLY'],
                  'vllm/v1/engine/output_processor.py', C_API_S)
    _, l2h = comp(lx, LY0 + PITCH, LANE_W, 'OutputProcessor',
                  ['· process_outputs: 按 request_id demux 到各 Collector',
                   '· 增量 detokenize + logprobs，组装 RequestOutput',
                   '· 内部 id 反查外部 id · stop-string 判定在这里'],
                  'vllm/v1/engine/output_processor.py', C_API_S)
    _, l3h = comp(lx, LY0 + 2 * PITCH, LANE_W, 'output_handler（单任务）',
                  ['· await get_output_async() 整批取回',
                   '· 按 chunk_size=128 切片逐片 process_outputs',
                   '· 片间 await asyncio.sleep(0) 让事件循环喘气'],
                  'vllm/v1/engine/async_llm.py', C_API_S)
    lbox_h = [l1h, l2h, l3h]
    # 上行箭头（朝上）
    seg(LCX, LY0, LCX, ENT_Y + ENT_H, C_ENG_S, 2.0, 'up')
    alabel(LCX + 7, ENT_Y + ENT_H + 13, 'yield RequestOutput → SSE', 9, C_ENG_S)
    for k, lab in [(0, 'RequestOutput · 按 request_id 扇出'), (1, 'EngineCoreOutputs 分片喂入')]:
        seg(LCX, LY0 + (k + 1) * PITCH, LCX, LY0 + k * PITCH + lbox_h[k], C_ENG_S, 2.0, 'up')
        alabel(LCX + 7, LY0 + k * PITCH + lbox_h[k] + 26, lab, 9, C_ENG_S)

    # 双登记虚线（入口条 → OutputProcessor，本进程建表先于跨进程）
    parrow([(1010, ENT_Y + ENT_H), (1010, LY0 + PITCH + lbox_h[1] / 2), (lx + LANE_W, LY0 + PITCH + lbox_h[1] / 2)],
           C_API_S, 1.5, 'std', dash=True)
    alabel(1017, LY0 + PITCH + 30, '① 本进程建表 OutputProcessor.request_states', 9, C_API_S)
    alabel(1017, LY0 + PITCH + 44, '（先于跨进程发送，保证回程到达时表已存在）', 9, C_MUTE)

    # ========== ZMQ 边界带（AsyncMPClient） ==========
    ZY = AY + AH + 16                            # 804
    ZH = 168
    rect(MX, ZY, CW, ZH, C_ZMQ_F, C_ZMQ_S, rx=12, sw=2.2)
    # 左侧边界标签
    text(MX + 16, ZY + 28, '进程边界', 13, C_ZMQ_S, 'start', True)
    text(MX + 16, ZY + 50, 'ZMQ + msgpack', 10.5, C_ZMQ_S, 'start', True)
    text(MX + 16, ZY + 72, '序列化：array_like · 零拷贝帧', 9, C_MUTE, 'start', maxw=200, tag='zmq:l3')
    text(MX + 16, ZY + 90, '大 tensor → OOB 共享内存旁路', 9, C_MUTE, 'start', maxw=200, tag='zmq:l4')
    text(MX + 16, ZY + 112, '无反压 HWM=0 · fire-and-forget', 9, C_MUTE, 'start', maxw=200, tag='zmq:l5')
    text(MX + 16, ZY + 132, '· DP 部署另起 DPCoordinator（控制面）', 9, C_MUTE, 'start', maxw=200, tag='zmq:l6')
    text(MX + 16, ZY + 148, '  → 详见分布式章', 9, C_MUTE, 'start', maxw=200, tag='zmq:l7')

    BAND_W = 600
    _, zrh = comp(RCX - BAND_W / 2, ZY + 30, BAND_W, 'AsyncMPClient.add_request_async',
                  ['· ROUTER(bind) → 引擎 DEALER(connect, identity=rank)',
                   '· 帧序 (identity, type_byte, *payload) · 首帧寻址',
                   '· send_multipart(copy=False) 零拷贝直传'],
                  'vllm/v1/engine/core_client.py', C_ZMQ_S)
    _, zlh = comp(LCX - BAND_W / 2, ZY + 30, BAND_W, 'AsyncMPClient.get_output_async',
                  ['· PULL(bind) 收全部引擎输出 ← 引擎 PUSH(connect)',
                   '· HWM=0 无反压 · 大内存机 0.5GB 内核缓冲',
                   '· 多前端时引擎按 client_index 选 PUSH socket'],
                  'vllm/v1/engine/core_client.py', C_ZMQ_S)

    # 右：下行穿越（EngineCoreRequest → add_request_async）
    seg(RCX, LY0 + 2 * PITCH + rbox_h[2], RCX, ZY + 30, C_API_S, 2.2, 'dn')
    alabel(RCX + 7, ZY + 22, 'encoder.encode → 多帧', 9, C_API_S)
    # 左：上行穿越（get_output_async → output_handler）
    seg(LCX, ZY + 30, LCX, LY0 + 2 * PITCH + lbox_h[2], C_ENG_S, 2.2, 'up')
    alabel(LCX + 7, ZY + 22, 'await 取回整批', 9, C_ENG_S)

    # ========== EngineCore 进程 ==========
    EY = ZY + ZH + 16                            # 988
    ROW1_Y, ROW1_H = EY + 68, 96                 # 队列/socket 框
    LOOP_X, LOOP_Y, LOOP_W, LOOP_H = 806, EY + 40, 700, 150
    PUSH_X, PUSH_W = 470, 180
    OQ_X, OQ_W = 670, 120
    IQ_X, IQ_W = 1522, 130
    DLR_X, DLR_W = 1668, 260
    EH = 792

    rect(MX, EY, CW, EH, C_ENG_F, C_ENG_S, rx=12, sw=2.4)
    text(MX + 16, EY + 24, 'EngineCore 进程（busy loop · 只做调度 + 执行）', 13.5, C_ENG_S, 'start', True)
    text(BXR - 16, EY + 24, 'GPU 上下文只在此进程与 worker · vllm/v1/engine/core.py', 9.5, C_MUTE, 'end')

    # 左侧注解：为什么 IO 线程 + 队列
    rect(MX + 20, ROW1_Y, 380, ROW1_H, 'none', C_FAINT, rx=8, sw=1.1, dash=True)
    text(MX + 32, ROW1_Y + 18, '为什么队列 + 专属 IO 线程 ×2', 9.5, C_MUTE, 'start', True)
    text(MX + 32, ROW1_Y + 38, '· ZMQ IO 线程释放 GIL，与 GPU 前向重叠', 9, C_MUTE, 'start', maxw=360, tag='io:l1')
    text(MX + 32, ROW1_Y + 56, '· 序列化 / 反序列化也不占 busy loop', 9, C_MUTE, 'start', maxw=360, tag='io:l2')
    text(MX + 32, ROW1_Y + 74, '· 空闲时阻塞在 input_queue.get，不空转', 9, C_MUTE, 'start', maxw=360, tag='io:l3')

    # 右侧注解：启动握手
    rect(DLR_X + DLR_W + 16, ROW1_Y, BXR - 20 - (DLR_X + DLR_W + 16), ROW1_H, 'none', C_FAINT, rx=8, sw=1.1, dash=True)
    hx = DLR_X + DLR_W + 28
    text(hx, ROW1_Y + 18, '启动握手（同一 socket）', 9.5, C_MUTE, 'start', True)
    text(hx, ROW1_Y + 38, '· DEALER 先发言 HELLO → READY', 9, C_MUTE, 'start', maxw=180, tag='hs:l1')
    text(hx, ROW1_Y + 56, '· EngineCoreReadyResponse 带回', 9, C_MUTE, 'start', maxw=180, tag='hs:l2')
    text(hx, ROW1_Y + 74, '  max_model_len / num_gpu_blocks', 9, C_MUTE, 'start', maxw=180, tag='hs:l3')

    # DEALER 收端
    rect(DLR_X, ROW1_Y, DLR_W, ROW1_H, '#ffffff', C_MUTE, rx=8, sw=1.4)
    text(DLR_X + 12, ROW1_Y + 20, 'DEALER(connect) 收端', 10.5, C_TXT, 'start', True, maxw=DLR_W - 24, tag='dealer')
    text(DLR_X + 12, ROW1_Y + 42, '· identity = engine_index（2 字节小端）', 9, '#334155', 'start', maxw=DLR_W - 22, tag='dealer:l1')
    text(DLR_X + 12, ROW1_Y + 60, '· 输入 IO 线程：decode → Request', 9, '#334155', 'start', maxw=DLR_W - 22, tag='dealer:l2')
    text(DLR_X + 12, ROW1_Y + 78, '· preprocess_add_request 不占忙循环', 9, '#334155', 'start', maxw=DLR_W - 22, tag='dealer:l3')

    queue_glyph(IQ_X, ROW1_Y, IQ_W, 'input_queue', 'queue.Queue · 保序')
    queue_glyph(OQ_X, ROW1_Y, OQ_W, 'output_queue', 'queue.Queue')

    # PUSH 发端
    rect(PUSH_X, ROW1_Y, PUSH_W, ROW1_H, '#ffffff', C_MUTE, rx=8, sw=1.4)
    text(PUSH_X + 12, ROW1_Y + 20, 'PUSH(connect) 发端', 10.5, C_TXT, 'start', True, maxw=PUSH_W - 24, tag='push')
    text(PUSH_X + 12, ROW1_Y + 42, '· 输出 IO 线程', 9, '#334155', 'start')
    text(PUSH_X + 12, ROW1_Y + 60, '· 按 client_index 选 socket', 9, '#334155', 'start', maxw=PUSH_W - 22, tag='push:l2')
    text(PUSH_X + 12, ROW1_Y + 78, '· encode_into 复用 bytearray', 9, '#334155', 'start', maxw=PUSH_W - 22, tag='push:l3')

    # ---- 穿越进程边界的箭头（在引擎带与 socket 框之后画：全程可见，端到端贴框边） ----
    # 右：add_request_async 框底 → DEALER 框顶（请求进引擎）
    seg(RCX, ZY + 30 + zrh, RCX, ROW1_Y, C_API_S, 2.4, 'dn')
    alabel(RCX + 8, ZY + ZH + 26, "EngineCoreRequest · type=ADD b'\\x00' · client_index 随请求过线", 9.5, C_API_S)
    # 左：PUSH 框顶 → get_output_async 框底（输出回 API 进程）
    seg(LCX, ROW1_Y, LCX, ZY + 30 + zlh, C_ENG_S, 2.4, 'up')
    alabel(LCX + 8, ZY + ZH + 26, 'EngineCoreOutputs · 每步整批聚合 1 条 · 引擎按 client_index 选 PUSH 回发', 9.5, C_ENG_S)
    # ABORT 红虚线（下行，两泳道之间：API 带底边 → EngineCore 带顶边）
    ABX = (LCX + RCX) // 2                     # 1165
    seg(ABX, AY + AH, ABX, EY, C_ABORT, 1.6, 'ab', dash=True)
    alabel(ABX + 8, ZY + 40, "ABORT b'\\x01'", 9.5, C_ABORT, 'start')
    alabel(ABX + 8, ZY + 56, '断连 / stop-string → 反向 abort', 9, C_MUTE)
    alabel(ABX + 8, ZY + 72, '只带内部 id · 与 ADD 同一条 socket', 9, C_MUTE)
    alabel(ABX + 8, ZY + 88, '引擎侧双投递 input_queue + aborts_queue', 9, C_MUTE)

    # 行内小箭头：DEALER→iq→loop→oq→PUSH（语义链统一标注在循环框下方）
    seg(DLR_X - 2, ROW1_Y + 48, IQ_X + IQ_W + 2, ROW1_Y + 48, C_MUTE, 1.6, 'std')
    seg(IQ_X - 2, ROW1_Y + 48, LOOP_X + LOOP_W + 2, ROW1_Y + 48, C_MUTE, 1.6, 'std')
    seg(LOOP_X - 2, ROW1_Y + 48, OQ_X + OQ_W + 2, ROW1_Y + 48, C_MUTE, 1.6, 'std')
    seg(OQ_X - 2, ROW1_Y + 48, PUSH_X + PUSH_W + 2, ROW1_Y + 48, C_MUTE, 1.6, 'std')
    text(LOOP_X + LOOP_W / 2, LOOP_Y + LOOP_H + 24,
         'IO 线程：recv_multipart → decode → put ｜ 忙循环取请求 ｜ put_nowait → encode_into → send_multipart（按 client_index 选 socket）',
         9, C_MUTE, 'middle', maxw=LOOP_W + 400, tag='io-chain')

    # ---- step 循环框 ----
    rect(LOOP_X, LOOP_Y, LOOP_W, LOOP_H, '#ffffff', C_ENG_S, rx=8, sw=1.8)
    text(LOOP_X + 12, LOOP_Y + 20, 'EngineCoreProc.run_busy_loop → EngineCore.step() 逐拍循环',
         11.5, C_ENG_S, 'start', True, maxw=LOOP_W - 140, tag='loop')
    text(LOOP_X + LOOP_W - 12, LOOP_Y + 20, 'vllm/v1/engine/core.py', 9, C_FAINT, 'end')

    CHIP_W, CHIP_H, CHIP_GAP = 131, 72, 6
    chip_cx = []
    chips = [
        ('① schedule()', 'RUNNING 先于 WAITING', '只认 token 数'),
        ('② execute_model', 'non_block=True', '→ Future 立即返回'),
        ('③ get_grammar_bitmask', 'CPU 活藏进 GPU', '前向的窗口期'),
        ('④ sample_tokens', 'future.result() 后', '先位掩码后采样'),
        ('⑤ update_from_output', '状态推进 + free 块', '组装回程消息'),
    ]
    for i, (t, s1, s2) in enumerate(chips):
        cx0 = LOOP_X + 10 + i * (CHIP_W + CHIP_GAP)
        chip_cx.append(cx0 + CHIP_W / 2)
        rect(cx0, LOOP_Y + 34, CHIP_W, CHIP_H, '#fff7ed', '#fdba74', rx=6, sw=1.2)
        text(cx0 + CHIP_W / 2, LOOP_Y + 51, t, 9, '#9a3412', 'middle', True, maxw=CHIP_W - 8, tag='chip:' + t)
        text(cx0 + CHIP_W / 2, LOOP_Y + 69, s1, 9, '#334155', 'middle', maxw=CHIP_W - 8, tag='chip:' + s1)
        text(cx0 + CHIP_W / 2, LOOP_Y + 87, s2, 9, C_MUTE, 'middle', maxw=CHIP_W - 8, tag='chip:' + s2)
        if i < 4:
            seg(cx0 + CHIP_W, LOOP_Y + 70, cx0 + CHIP_W + CHIP_GAP, LOOP_Y + 70, C_ENG_S, 1.5, 'std')
    # 回环：⑤ → ① 下一拍
    fb_y = LOOP_Y + 34 + CHIP_H + 14
    parrow([(chip_cx[4], LOOP_Y + 34 + CHIP_H), (chip_cx[4], fb_y), (chip_cx[0], fb_y),
            (chip_cx[0], LOOP_Y + 34 + CHIP_H)], C_MUTE, 1.4, 'std')
    text((chip_cx[0] + chip_cx[4]) / 2, fb_y + 14, '下一拍 · v0.27.1 起默认开启异步调度（本循环即重叠版）', 9, C_MUTE, 'middle')

    # ========== 三列：调度·显存 / GPU 执行臂 / 采样出口 ==========
    CY0 = EY + 232
    COL_W, COL_GAP = 616, 90
    AX = 116
    BX = AX + COL_W + COL_GAP                  # 822
    CX = BX + COL_W + COL_GAP                  # 1528
    ACX = AX + COL_W / 2                     # 424
    BCX = BX + COL_W / 2                     # 1130
    CCX = CX + COL_W / 2                     # 1836
    GAP_AB = AX + COL_W                      # 732
    GAP_BC = BX + COL_W                      # 1438

    def col_header(x, s, note, color):
        text(x + 2, CY0 + 14, s, 12.5, color, 'start', True)
        text(x + COL_W - 2, CY0 + 14, note, 9, C_MUTE, 'end', maxw=COL_W - 200, tag='colh:' + s[:6])

    col_header(AX, '调度 · 显存账本', '第二原则：一切调度先对账', C_KV_S)
    col_header(BX, 'GPU 执行臂', '第一原则：GPU 不空转', C_GPU_S)
    col_header(CX, '采样与出口', 'GPU 全程，logits 不落 CPU', C_SAM_S)

    # ---- A 列（青） ----
    _, a1h = comp(AX, CY0 + 30, COL_W, 'Scheduler',
                  ['· schedule()：RUNNING 先于 WAITING · 抢占拍不收新',
                   '· 只认 token 数，无 prefill / decode 相位',
                   '· update_from_output：状态推进 + 组装 EngineCoreOutputs',
                   '· RequestStatus 单 IntEnum · >PREEMPTED 即 finished'],
                  'vllm/v1/core/sched/scheduler.py', C_KV_S, badge='①⑤')
    A1Y = CY0 + 30
    A2Y = CY0 + 30 + a1h + 16
    _, a2h = comp(AX, A2Y, COL_W, 'KVCacheManager',
                  ['· get_computed_blocks → 前缀命中（返回 token 计数）',
                   '· allocate_slots → None = 触发抢占的唯一信号',
                   '· 分配三重预算 free−reserved−watermark（水位抑制抢占抖动）',
                   '· free 逆序归还 = LRU 隐藏不变量'],
                  'vllm/v1/core/kv_cache_manager.py', C_KV_S)
    A3Y = A2Y + a2h + 16
    _, a3h = comp(AX, A3Y, COL_W, 'BlockPool + 前缀缓存',
                  ['· KVCacheBlock 固定块池 · ref_cnt 共享前缀',
                   '· 链式哈希（非 radix 树）· 满块 + 块内 CoW 部分命中（partial prefix cache）',
                   '· 抢占 = recompute-only：不清哈希，回 waiting 队头',
                   '· block_id 是调度器 ↔ worker 的唯一共享键'],
                  'vllm/v1/core/block_pool.py', C_KV_S)
    # A 列内部箭头（对账双向 + 分配/归还）
    seg(ACX - 6, CY0 + 30 + a1h, ACX - 6, A2Y, C_KV_S, 1.5, 'std')
    seg(ACX + 6, A2Y, ACX + 6, CY0 + 30 + a1h, C_KV_S, 1.5, 'std')
    alabel(ACX + 12, A2Y - 4, '每拍对账', 9, C_KV_S)
    seg(ACX, A2Y + a2h, ACX, A3Y, C_KV_S, 1.5, 'std')
    alabel(ACX + 7, A3Y - 4, 'touch / free 块', 9, C_KV_S)

    # ---- B 列（绿） ----
    _, b1h = comp(BX, CY0 + 30, COL_W, 'Executor → Worker',
                  ['· Executor：进程拓扑编排（Uni / MultiProc）',
                   '· Worker：设备生命周期 · 显存 profile 一次定池',
                   '· execute_model(scheduler_output, non_block=True)'],
                  'vllm/v1/executor', C_GPU_S, badge='②')
    B1Y = CY0 + 30
    B2Y = CY0 + 30 + b1h + 16
    _, b2h = comp(BX, B2Y, COL_W, 'GPUModelRunner',
                  ['· execute_model → 暂存 ExecuteModelState，返回 None',
                   '· InputBatch 持久批次 · 差量调和（不重发全量）',
                   '· CpuGpuBuffer 预分配固定地址（CUDA Graph 前提）',
                   '· sample_tokens(grammar_output)：两段式第二幕'],
                  'vllm/v1/worker/gpu_model_runner.py', C_GPU_S, badge='②④')
    B3Y = B2Y + b2h + 16
    _, b3h = comp(BX, B3Y, COL_W, '模型层 forward + 编译',
                  ['· DecoderLayer 拼装 · Attention = 插座（MLA / GQA 变体）',
                   '· piecewise torch.compile：注意力处切图',
                   '· CUDA Graph 按形状查表回放',
                   '· slot_mapping 由 Triton kernel 在 GPU 上算',
                   '· 新布局 vllm/models/<name>/：硬件隔离（旗舰架构）'],
                  'vllm/model_executor/models · vllm/compilation', C_GPU_S)
    seg(BCX, CY0 + 30 + b1h, BCX, B2Y, C_GPU_S, 1.6, 'std')
    alabel(BCX + 7, B2Y - 4, 'execute_model 穿三层', 9, C_GPU_S)
    seg(BCX, B2Y + b2h, BCX, B3Y, C_GPU_S, 1.6, 'std')
    alabel(BCX + 7, B3Y - 4, 'set_forward_context → model.forward', 9, C_GPU_S)

    # ---- C 列（品红） ----
    _, c1h = comp(CX, CY0 + 30, COL_W, 'compute_logits',
                  ['· 只在采样位物化：hidden_states[logits_indices]',
                   '· lm_head GEMM + TP gather · 裁词表 padding'],
                  'vllm/model_executor/layers/logits_processor.py', C_SAM_S)
    C1Y = CY0 + 30
    C2Y = CY0 + 30 + c1h + 16
    _, c2h = comp(CX, C2Y, COL_W, 'Sampler.forward · 9 步管线',
                  ['· fp32 → allowed / bad_words → min_tokens / logit_bias',
                   '· 惩罚 → greedy 快路径 / 温度 / min_p / top-k / top-p',
                   '· Gumbel 式 random_sample：全程 GPU 无同步',
                   '· top-k logprobs 用变换前的 raw logits'],
                  'vllm/v1/sample/sampler.py', C_SAM_S, badge='④')
    C3Y = C2Y + c2h + 16
    _, c3h = comp(CX, C3Y, COL_W, '结构化输出位掩码',
                  ['· xgrammar FSM → fill_next_token_bitmask',
                   '· apply_token_bitmask_inplace：非法位 logits = -inf'],
                  'vllm/v1/structured_output', C_SAM_S, badge='③④')
    C4Y = C3Y + c3h + 16
    _, c4h = comp(CX, C4Y, COL_W, 'spec decode（投机解码）',
                  ['· drafter.propose → 排进下一步 → target 一次验证',
                   '· RejectionSampler 拒绝采样，保分布不变',
                   '· 启用时整体替换 Sampler 管线'],
                  'vllm/v1/sample/rejection_sampler.py', C_SAM_S)
    seg(CCX, CY0 + 30 + c1h, CCX, C2Y, C_SAM_S, 1.6, 'std')
    alabel(CCX + 7, C2Y - 4, 'logits [采样位, vocab]', 9, C_SAM_S)
    seg(CCX, C3Y, CCX, C2Y + c2h, C_SAM_S, 1.6, 'std')
    alabel(CCX + 7, C3Y - 4, 'bitmask H2D → -inf', 9, C_SAM_S)

    # ---- 跨列箭头 ----
    # SchedulerOutput：A1 → B1（差量协议）
    so_y = CY0 + 30 + 52
    parrow([(GAP_AB, so_y), (BX, so_y)], C_ENG_S, 2.2, 'std')
    text(GAP_AB + COL_GAP / 2, so_y - 26, 'SchedulerOutput', 9, C_ENG_S, 'middle', True)
    text(GAP_AB + COL_GAP / 2, so_y - 12, '差量：新全量 / 老 diff', 9, C_MUTE, 'middle', maxw=COL_GAP + 30, tag='so:sub')
    # block_ids：A3 → B2（折线经 A/B 间隙）
    bt_x = GAP_AB + COL_GAP / 2
    parrow([(GAP_AB, A3Y + 40), (bt_x, A3Y + 40), (bt_x, B2Y + 60), (BX, B2Y + 60)], C_KV_S, 1.8, 'std')
    text(bt_x, (A3Y + 40 + B2Y + 60) / 2 - 8, 'block_ids', 9, C_KV_S, 'middle')
    text(bt_x, (A3Y + 40 + B2Y + 60) / 2 + 6, '→ 块表', 9, C_KV_S, 'middle')
    # hidden_states：B2 → C1（折线经 B/C 间隙；切片细节写在 C1 框内）
    hs_x = GAP_BC + 20
    hs_y = B2Y + 80
    parrow([(GAP_BC, hs_y), (hs_x, hs_y), (hs_x, CY0 + 30 + c1h / 2), (CX, CY0 + 30 + c1h / 2)],
           C_GPU_S, 1.8, 'std')
    text((hs_x + CX) / 2, CY0 + 30 + c1h / 2 - 8, 'hidden_states', 9, C_GPU_S, 'middle')

    # ---- 回程总线：Sampler → update_from_output → output_queue → PUSH ----
    BUS_Y = CY0 + 520
    tap_x = GAP_BC + 52
    tap_y = C2Y + c2h / 2
    parrow([(CX, tap_y), (tap_x, tap_y), (tap_x, BUS_Y), (76, BUS_Y),
            (76, ROW1_Y + 120), (560, ROW1_Y + 120), (560, ROW1_Y + ROW1_H)], C_ENG_S, 2.4, 'up')
    text((tap_x + 76) / 2, BUS_Y - 10,
         '回程：sampled_token_ids（D2H）→ ⑤ update_from_output 推进状态机 · 组装 EngineCoreOutputs → output_queue → PUSH 送出',
         9.5, C_ENG_S, 'middle', maxw=tap_x - 100, tag='bus')

    # ========== 页脚：读图 + 原则 + 图例 ==========
    FY = EY + EH + 24
    text(MX, FY, '读图：请求自顶部 users 进 API 进程（蓝）下行——tokenize → 双轨 id → EngineCoreRequest；穿 ZMQ 边界（紫）进引擎（橙）；'
                 '引擎逐拍 ①→⑤ 循环；新 token 沿橙线上行回 API——detokenize → SSE 流出。',
         10, C_MUTE, 'start', maxw=CW, tag='ft:read')
    text(MX, FY + 20, '第一原则：GPU 是最贵的员工，一切 CPU 活不让它等（tokenize / detokenize 挪出进程 · bitmask 藏进 GPU 窗口 · 固定地址 + CUDA Graph · 异步调度）；'
                      '第二原则：显存是共享账本，一切调度先对账（token 预算 + 块池）。',
         10, C_MUTE, 'start', maxw=CW, tag='ft:principle')
    # 图例行
    ly = FY + 44
    swatches = [(C_API_S, 'API 进程'), (C_ZMQ_S, 'ZMQ 边界'), (C_ENG_S, 'EngineCore 进程'),
                (C_GPU_S, 'GPU 执行臂'), (C_KV_S, '显存账本'), (C_SAM_S, '采样出口')]
    lx0 = MX
    for color, name in swatches:
        rect(lx0, ly - 9, 16, 11, '#ffffff', color, rx=3, sw=1.6)
        text(lx0 + 21, ly + 1, name, 9.5, C_TXT, 'start')
        lx0 += 21 + tw(name, 9.5) + 22
    seg(lx0 + 4, ly - 3, lx0 + 34, ly - 3, C_API_S, 2.0, 'dn')
    text(lx0 + 40, ly + 1, '请求下行', 9.5, C_TXT, 'start')
    lx0 += 40 + tw('请求下行', 9.5) + 18
    seg(lx0 + 4, ly - 3, lx0 + 34, ly - 3, C_ENG_S, 2.0, 'up')
    text(lx0 + 40, ly + 1, '输出上行', 9.5, C_TXT, 'start')
    lx0 += 40 + tw('输出上行', 9.5) + 18
    seg(lx0 + 4, ly - 3, lx0 + 34, ly - 3, C_ABORT, 1.6, 'ab', dash=True)
    text(lx0 + 40, ly + 1, 'abort', 9.5, C_TXT, 'start')
    lx0 += 40 + tw('abort', 9.5) + 18
    rect(lx0 + 4, ly - 11, 22, 15, '#ffedd5', C_ENG_S, rx=7, sw=1.1)
    text(lx0 + 15, ly + 1, '①', 9, C_ENG_S, 'middle', True)
    text(lx0 + 32, ly + 1, '= EngineCore.step() 第几拍', 9.5, C_TXT, 'start')
    lx0 += 32 + tw('= EngineCore.step() 第几拍', 9.5) + 18
    text(lx0, ly + 1, '框内灰字 = 规范源码路径', 9.5, C_MUTE, 'start')

    H = FY + 76
    GEO = dict(W=W, H=H, MX=MX, CW=CW, BXR=BXR, UY=UY, UH=UH, UW=UW, UX0=UX0,
               AY=AY, AH=AH, ENT_Y=ENT_Y, ENT_H=ENT_H, LY0=LY0, PITCH=PITCH,
               LCX=LCX, RCX=RCX, DIVX=DIVX, ZY=ZY, ZH=ZH, EY=EY, EH=EH,
               ROW1_Y=ROW1_Y, ROW1_H=ROW1_H,
               LOOP_X=LOOP_X, LOOP_Y=LOOP_Y, LOOP_W=LOOP_W, LOOP_H=LOOP_H,
               PUSH_X=PUSH_X, PUSH_W=PUSH_W, OQ_X=OQ_X, OQ_W=OQ_W,
               IQ_X=IQ_X, IQ_W=IQ_W, DLR_X=DLR_X, DLR_W=DLR_W, ABX=ABX,
               CY0=CY0, COL_W=COL_W, COL_GAP=COL_GAP,
               AX=AX, BX=BX, CX=CX, ACX=ACX, BCX=BCX, CCX=CCX,
               GAP_AB=GAP_AB, GAP_BC=GAP_BC, BUS_Y=BUS_Y, FY=FY,
               A1Y=A1Y, a1h=a1h, A2Y=A2Y, a2h=a2h, A3Y=A3Y, a3h=a3h,
               B1Y=B1Y, b1h=b1h, B2Y=B2Y, b2h=b2h, B3Y=B3Y, b3h=b3h,
               C1Y=C1Y, c1h=c1h, C2Y=C2Y, c2h=c2h, C3Y=C3Y, c3h=c3h, C4Y=C4Y, c4h=c4h)
    return ELEMS, GEO, WARN
