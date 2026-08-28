# v3 图系规范：三层缩放（唯一图源，放大镜关系）

> Phase 2 的宪法。配套：`gen_L0.py`（L0）、`gen_L1.py`（L1×8）、`gen_L2.py`（L2×40，随章滚动）。
> 用户裁决史：v2 的病根一 =「同一架构三套画法、非粒度缩放关系」——本规范根治之。

## 0. 硬规则

1. **一张图原则**：全书只有一种架构画法 = L0 的画法。任何新图必须能回答「它是 L0 哪一块的放大」，答不出就不许存在。
2. **同源强制**：L0/L1/L2 共享布局常量模块（`l0_common.py`）、配色常量（C_API_S 蓝/C_ZMQ_S 紫/C_ENG_S 橙/C_GPU_S 绿/C_KV_S 青/C_TXT/C_MUTE）、字体栈、图例样式。改配色/术语只许改常量，全层联动。**角色色即身份**（exp-2026-08-28：ch18-fig-diff-protocol 把 worker 框画成橙色、与 ch12 的绿色 worker 泳道分叉）：worker/GPU 执行臂在任何图里恒用 C_GPU_S 绿、EngineCore 恒用 C_ENG_S 橙——角色色不得按图内美观改用他色，角色变了颜色没变 = 图系分叉。
3. **几何同源**：L1 = L0 的 viewBox 裁切放大（物理保证）；L2 = L1 对应块的展开（组件框位置从 L1 坐标推导，方法级细节在框内展开）。
4. **文字与门禁**：图上文字全走 `fit()`（防越界）；箭头端点必须落框边（strict 几何门禁 `lint_diagram_geometry.py`——cartography 路径自动 strict）。
5. **Part 标注**：L0 上每块标「第几 Part 打开」（F9 伏笔：逐章点亮）。

## 1. 三层的职责

| 层 | 数量 | 画什么 | 何时出现 |
|----|:---:|--------|---------|
| L0 | 1 | 全书骨架（一个请求的一生，方法名级密度） | ch1 首次给出；每章开篇 zoom 引用；ch40 点亮版收官 |
| L1 | 8 | Part 区域放大 + hook 标题带 + 本 Part 章目录 + 左侧 L0 minimap 高亮框（2026-08-15 用户裁决：IDE 概览图模式取代区域外淡出） | 每 Part 首章开篇 |
| L2 | 40 | 章图：本章块的组件 + **方法级展开** + 本章站号 | 每章开篇（正文前） |

## 2. L2 章图契约（渲染器 gen_L2.py + L2-spec 数据契约，随章 dossier 滚动产出）

**数据源（与 v2 的关键区别）**：
- 组件/方法清单吃 **deepread 卡 + 本章 dossier**（curated，深读过）——**不走 v2 的 arch-model.json 自动抽取**（v2 归属 bug 修了三轮的教训：自动抽取的「多数票/兜底」必然出错）。
- 站号吃本章 dossier 的 code_spine（每站 file:line）。
- L0 缩放路径吃 pedagogy-plan.json 的 `l0_zoom` 字段。

**L2-spec/1 数据契约（定型 2026-08-15，样板 = `l2-specs/ch9.json`）**——gen_L2.py 的唯一输入，
Phase 3 起每章 dossier 顺产一份，渲染器零改代码出图：

```
chapter / part / title / hook / l0_zoom / depends_on / reading     # 单值字段（hook·l0_zoom·depends_on 直拷 pedagogy-plan）
l0_region     {anchors: [名]} 或 {rects: [[x0,y0,x1,y1],…]}        # minimap 高亮区域
              # 锚名词表（GEO 派生，L0 改版自动联动）：
              #   full / api_band / zmq_band / engine_band / loop_box
              #   / kv_column / gpu_column / sample_column
frame         {title, file}          # 本章舞台外框（进程/系统边界，橙系进程框风格）
center        {name, title, where}   # 核心机制区外框（可选；其 zone=center 组件按序成拍片+回环）
components    [{name, role, zone, kind, file, methods[], stations[], note[]}]
              # zone ∈ north|center|south   north=请求进出条（左→右）· center=主角拍片 · south=支撑/why 注
              # role ∈ engine|gpu|kv|api|zmq|sample|io|plain|beat —— 只映射 l0_common 配色常量（同源强制，spec 不带色值）
              # kind ∈ comp|queue|note
              # note = 挂在该组件旁的 why 小注（虚线框）
flows         [{from, to, label, up, dash, color_role}]   # from/to 填组件名 / frame / center.name；同行横向、跨行纵向自动锚边
loop          {label}                # center 末拍片 → 首拍片 回环（可选）
stations      [{n, where, what}]     # 本章站号账本（左下「站号轨道」逐行渲染）
```

渲染器内置校验（渲染前置闸）：zone/role/kind 合法、flows 引用可解析、**站号徽标 ⊆ 账本且账本每站有挂点**。

**版式**（画布 2200 = L0/L1 家族宽；detail 可比 L0 局部更宽——站号与方法名需要空间）：
- 顶部窄条（96px）：`ch{N} · {标题}` + hook 一句 + 右上「L2 · L0 位置：{l0_zoom}」。
- 左栏：minimap（L0 全图 ×0.2，高亮框=本章 L0 区域，框外 opacity 0.45 退后——沿 L1 IDE 概览图模式）
  + 其下「站号轨道」（第 N 站徽标 + where · what 逐行）。
- 右区 detail **不裁切 L0**（L0 无方法级/站号级密度）——用 l0_common 原语/配色**新画**：
  frame 外框 → north 行 → center 拍片行（框内方法签名+file:line+站号徽标，拍间箭头、标签下沉说明行、回环）
  → south 行；flows 全部端点贴框边（strict 几何门禁）。
- 底部：读图一行 + 前置依赖章（pedagogy-plan depends_on）+ 本章埋/收伏笔（pedagogy-plan foreshadows 自动带出）。
- primer 章（4 张）：无站号（原理章无源码走线），主体改为推导链图（顿悟图头图风格沿 v2 ch21 样板）——
  此类 spec 走独立 kind，Phase 3 到 primer 章时再定。

**验收链**：渲染 → Read PNG 亲眼看自查（六项：越界/相撞/压框/箭头悬空/同源/断言溯源）→ strict 几何门禁
（cartography 路径自动 strict；minimap/detail 组各带 data-minimap/data-detail ctx 标记，沿 gen_L1 模式）
→ 独立盲审（插画者≠审图者）→ manifest 登记。

## 3. 正文插图规则（writer 契约用）

1. **开篇即图**：每章 `## 你在这里` 段放 L2 章图；Part 首章先放 L1 再 L2。
2. 图注三要素：(a) 这块在 L0 的位置（认得感）；(b) 本章打开什么；(c) 站号=请求流经顺序、正文按讲解需要编排。
3. 正文机制图（讲解算法的独立图）：允许存在，但**架构性内容必须回指 L0/L1/L2**，不许另立架构画法。
4. 禁止：出现第二种架构图风格；图上杜撰类名/方法名/站号。

## 4. 与 v2 资产的边界

- v2 的 arch-model.*（39 张）与 chapter-map.*：**不迁移**。v2 封版不动。
- v2 的 svg-diagram skill 与 illustrator 契约：继续有效（机制图用它画）。
- fable 画图/读图专用（CLAUDE.md #7）；opus 写渲染器代码；strict 几何门禁兜底。
