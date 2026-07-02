---
name: svg-diagram
description: >
  Generate technical diagrams as valid SVG+PNG using Python scripts. Use this
  skill whenever you need to create diagrams for technical documentation,
  book chapters, or educational content — especially for dense many-to-many
  graphs (tiling patterns, state evolution tables, architecture overviews)
  where Mermaid auto-layout fails and Excalidraw manual coordinates cause
  misalignment. Triggers on: "create a diagram", "draw this", "visualize
  the tiling", "make a figure", "需要画图", "画一个图", or any request to
  illustrate a technical concept. DO NOT use Excalidraw or Mermaid for
  diagrams with >3 connected elements — use this skill instead.
---

# SVG Diagram Generator v2 — 画一张"好图"的完整方案

一张好图 = **论点 + 数据 + 版式 + 双重验收**。本 skill 定义从 figure-spec 到验收的全流程。
渲染管线不变:Python 生成 SVG → xmllint 校验 → rsvg-convert 转 PNG。

## Step 0:先有 figure-spec,再动笔(无 spec 不绘图)

绘图前必须有(或先写出)这张图的 spec:

```json
{
  "figure_id": "fig-m1-preempt",
  "claim": "一句话:这张图让读者看懂什么(写不成一句话 → 拆成两张图)",
  "template": "state-table|swimlane|layout|tensor-flow|before-after|state-machine|flow|tiling",
  "numbers": [{"value": "512", "provenance": "traces/m1.json 或 vllm/...:L123"}],
  "elements": ["图中每个视觉组及其含义"],
  "caption_draft": "图注草稿——给结论,不描述画面"
}
```

## 设计规则(绘图时逐条对照)

1. **一图一论点**:整张图为 claim 服务;与 claim 无关的元素删掉。
2. **元素预算**:每个视觉组 ≤7 个元素;超了就分组加留白,或拆图。
3. **颜色即语义**:颜色只编码状态/类别,不做装饰;>2 种语义色必须画图例。
4. **数字皆有出处**:图中每个数字来自 spec.numbers(trace 或源码常量)。**禁止即兴加"示意数字"**。
5. **阅读顺序显式**:符合左上→右下,否则用 ①②③ 编号标出看图顺序。
6. **图注给结论**:图注是 claim 的读者版(「队列长 3→2→1:LIFO 每轮恰弹出一个」),不写「本图展示了…的结构」。

## 模板库(按 spec.template 选,参考 references/ 对应示例改)

| template | 用途 | 参考 |
|---|---|---|
| state-table | 状态逐轮演化/数值追踪 | references/example-softmax-trace.py |
| swimlane | 跨组件/跨进程时序协议 | references/example-swimlane.py |
| layout | 内存/块表/KV 页/张量布局 | references/example-layout.py |
| before-after | 优化前后双态对比 | references/example-before-after.py |
| state-machine | 状态机/生命周期流转 | references/example-state-machine.py |
| tensor-flow | 张量形状流(shape 沿箭头标注) | 用 flow 骨架,每条边标 shape |
| tiling | 分块/many-to-many 连接 | references/example-fa-tiling.py |
| flow | 简单线性流程(<5 节点可用 Mermaid 替代) | SKILL 模板 C |

## 生成规则(Python 脚本,CRITICAL)

1. **全部坐标由循环/常量计算,零手写魔数**。
2. **所有文本过 `xml.sax.saxutils.escape()`**;绝不预转义(`&lt;` 会被二次转义)。
3. 箭头端点从元素边缘计算(source.right → target.left),`marker-end` 在 `<defs>` 定义一次。
4. 多行文本用多个 `<text>` + y 偏移(SVG 无 `<br/>`);`font-weight="bold"` 用属性不用 CSS。
5. viewBox 为文字留边;`text-anchor="end"` 的 x ≥ 50。
6. 中文:`font-family="sans-serif"`,**不要**强制 CJK 字体;rsvg-convert 自动逐字回退。

SVG 骨架:

```python
import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 700, 400
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
         'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
# ... 元素全部循环生成 ...
L.append('</svg>')
```

## 渲染与三重验收(强制顺序,缺一 = 图未完成)

```bash
python3 gen_<figure_id>.py                       # 生成 <figure_id>.svg
xmllint --noout <figure_id>.svg                  # 1a. XML 语法
python3 scripts/validate_svg.py <figure_id>.svg  # 1b. 语义(双转义/裁剪/缺箭头)
rsvg-convert -z 2 <figure_id>.svg -o <figure_id>.png   # 勿用 ImageMagick convert(丢中文/错位)
```

2. **视觉自查(必须做,没看过渲染结果的图 = 未完成)**:用 Read 工具打开 **PNG**(不是 SVG),
   亲眼看,逐项如实判定:
   - `claim_readable_10s`:不看正文,10 秒内能从图上得到 claim 吗?
   - `numbers_match_spec`:图上每个数字逐个与 spec.numbers 对(多字/少字/错字都算 false)。
   - `no_overlap`:无文字相撞/压框/越界。
   - `arrows_attached`:每条箭头两端都贴着元素边缘,无悬空。
   - `cjk_rendered`:中文无豆腐块/缺字。
   - `reading_order_clear`:第一眼知道从哪看起。
   任一 false → 改脚本 → 重渲 → **重新 Read PNG** 再判。全 true 才算过,结果写进
   `diagrams/figure-manifest.json` 对应条目的 `selfcheck`。**凭想象填表 = 造假。**

3. **盲审(由流程/另一 agent 执行)**:只看 PNG + figure-spec(不看生成代码),复述图的论点、
   逐个核数字。verdict 写进 manifest 的 `blind_review`。

## Common Pitfalls(保留)

1. 文本里写 RAW `<-`,交给 esc() 转义;绝不手写 `&lt;`。
2. `text-anchor="end"` 且 x < 文本宽 → 左侧裁剪;行标签 x ≥ 50。
3. 箭头端点悬空 → 一律从元素坐标计算。
4. SVG 无 `<br/>`;用多 `<text>`。
5. `font-weight:bold`(CSS 语法)无效 → 用 `font-weight="bold"` 属性。
6. `color:` 是 CSS,SVG 属性用 `fill=`。
7. 根元素必须有 `xmlns="http://www.w3.org/2000/svg"`。
