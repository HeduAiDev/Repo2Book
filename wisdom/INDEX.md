# Wisdom Index — Universal repo2book Patterns

跨实例通用经验（任何 repo2book 书都适用）。门槛：2+ 实例出现才"提升"为 wisdom。
> v2 现实：本体系已从"从零简化重写"转为**源码解读型**（直接解读真实源码、单章工作流编排）。下表已按 v2 + v0.21.0 重基 + 通用化 + vllm-ascend 试点的经验更新。深层条目见同目录 writing/architecture/debugging/testing.md。

| # | Category | Pattern | Roles | Confirmed In |
|---|----------|---------|-------|-------------|
| W01 | debugging | F.linear weight shape is [out, in] | impl, tester | vllm, pytorch |
| W02 | testing | Preemption tests: both requests must fit initially | tester | vllm |
| W03 | writing | 数值追踪 2+ 轮、列全中间量（口头结论不够，摆成 mini 表）| writer, reviewer | vllm, vllm-ascend |
| W04 | architecture | 反压闸门防级联错误（阶段二元闸，失败即停）| all | vllm |
| W05 | debugging | 加速器代码 host 跑不动→只验可读控制流，行为以源码为准（容器版本可能有行号差）| tester, impl | vllm, vllm-ascend |
| W06 | writing | 公式 lint：\text→\mathrm、$$ 独占行、inline 仅简单式（auto-REJECT）| writer, reviewer | vllm |
| W07 | writing | 引行号前**先读实现**；引用走规范源码路径，绝不 instances/*/source/ | writer | vllm, vllm-ascend |
| W08 | architecture | 横向通信：Reviewer↔Writer 绕过 Lead 的有界 revise 回环 | reviewer, writer | vllm |
| W09 | writing | 大白话不书面语：句 15–25 字、超 40 必拆、禁"综上所述/不难看出" | writer | vllm, vllm-ascend |
| W10 | testing | ref_cnt = -1 = "not ready" in cache pools | tester, impl | vllm |
| W11 | debugging | SVG 文字裁切：text-anchor="end" 且 x<30 会被切 | writer | vllm |
| W12 | architecture | 单章工作流：dossier(真相源)→减法实现+test→真源码叙事→多维评审→归档；6 闸门 + 逃生舱 | all | vllm, vllm-ascend |
| W13 | writing | 内嵌真源码**逐字保留**（含签名/类型标注，别改一行内容）；非相邻方法拼块要加 `# … 省略 …` | writer, reviewer | vllm, vllm-ascend |
| W14 | writing | 主线术语首现处给一次性中文注解（如 qualname/OOT），后文沿用英文 | writer | vllm, vllm-ascend |
| W15 | writing | 图几何：文字不越界/不相撞/不压框/不裁切、箭头接框边；一图一核心对比（lint_diagram_geometry）| writer | vllm, vllm-ascend |
| W16 | writing | 章内锚点须 GitHub-slug 可解析（lint_anchors）；中文标点全角（lint_punct）| writer, reviewer | vllm |
| W17 | architecture | 逃生舱：任一阶段发现路线错→返回 BLOCKED 早停升级 Lead；dossier 对抗性自核拦事实错（实战拦下 ch31/ch01 错误）| all | vllm |
| W18 | architecture | 防假通过：评审 agent 全失败**不静默 APPROVE**；archive 注入完整 review 对象 + 崩溃重试 | reviewer, all | vllm |
| W19 | architecture | 源码版本升级：difflib 行级对齐**确定性重映射**行号（remap_lines），内容真改处才定点重抽片段 | all | vllm |
| W20 | debugging | **git push 须前台**（后台 shell SSH 鉴权失败）；只在用户要求时提交/推送 | all | vllm |
| W21 | writing | 姊妹篇/衍生仓：主线讲衍生仓、每章钉一个对位基座章、对照基座源码（基座已在另一实例可直引）| writer, architecture | vllm-ascend |
| W22 | architecture | 实例无关化：脚本经 `instance.py` 认活动实例；`new_instance` **继承**通用约定（voice-guide/wisdom）而非退回白板 | all | vllm-ascend |
| W23 | architecture | cartography **强制子系统覆盖交叉核对**：列源码每个顶层子系统，逐一确认被某章 key_source_paths 覆盖或显式点名，否则漏章（playbook RUNBOOK §0.6 / schema book_outline.json v2）| team-lead, analyst | vllm-ascend |
| W24 | architecture | cartography 易低估漏掉的子系统：PD 分离(proxy 调度 + KV 亲和/命中路由)、KV 池化/外存储、KV 卸载(host/CPU 分层)、芯片分代变体(如 310P 整套子类化)、网络加载——别压成一章 | team-lead | vllm-ascend |

## Category Quick Reference

- **debugging**: 静默失败 / 错值 / 迷惑错误的根因模式
- **testing**: 测试设计、边界目录、容器/加速器怪癖
- **writing**: 叙事结构、公式、图、声线、术语、伏笔呈现规则
- **architecture**: 工作流设计、agent 通信、闸门、逃生舱、版本升级、实例无关化

## Role Quick Reference

| Role | Query Priority |
|------|---------------|
| implementer | debugging > architecture > testing > writing |
| tester | testing > debugging > architecture > writing |
| writer | writing > architecture > debugging > testing |
| reviewer | writing > architecture > testing > debugging |
| team-lead | architecture > all others equally |
