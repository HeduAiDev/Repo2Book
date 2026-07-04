# SDD 任务简报 lint-exp-N1 — book-retro / book-gap-audit：args 若为字符串须先 JSON.parse 再判，防 CFG 静默回退

> 来源：Lead 复盘发现的新问题（不在 retro-2026-07-05.json / wisdom-candidates.json 候选清单内，
> Lead 直接批准，落点为 `.claude/workflows/book-retro.js` 与 `.claude/workflows/book-gap-audit.js`
> 的 CFG 解析逻辑）。本简报为落地前的 SDD 任务定义，**不改代码**——workflow 编排逻辑属于 Curator
> 权限外的文件，需 Lead 另行走 TDD 小任务落地。

## 背景 / 现象

`book-retro.js` 与 `book-gap-audit.js`（以及 `chapter-pipeline.js`/`chapter-retrofit.js` 同款
写法）都用如下模式解析入参：

```js
const CFG = { instance: 'vllm-ascend', chapters: null, date: 'undated', repo_root: '...' }
const A = (typeof args !== 'undefined' && args && args.instance) ? args : CFG
```

已知本环境"Workflow 的 args 注入不可靠"（`chapter-pipeline.js` 顶部注释明确记录了这一点）。若
host 把 `args` 以 **JSON 字符串**（而非已解析对象）形式注入，`args.instance` 对字符串取属性会
返回 `undefined`，三元表达式判定为假，于是**静默**回退到脚本内 `CFG` 默认值——调用方以为自己
传的参数生效了，实际跑的是硬编码默认配置（如默认 `instance: 'vllm-ascend'`），且没有任何报错
或日志提示这次回退，问题只能靠事后对照产物目录才能发现。这与"args 注入不可靠"是两个独立问题
叠加：其一是 host 注入本身不稳定（已知、有旁注），其二是**即便注入到了，字符串形态也会被当前
判断逻辑误判为"没传"**——本简报只处理后者。

## 规则描述（拟）

在 `book-retro.js`、`book-gap-audit.js` 的 CFG 解析处，在三元判断前插入一步归一化：

```js
let _args = args
if (typeof _args === 'string') {
  try { _args = JSON.parse(_args) } catch (e) { _args = null }
}
const A = (_args && _args.instance) ? _args : CFG
if (_args && !_args.instance) {
  log('⚠️ args 已收到但缺 instance 字段或解析失败，回退到脚本内 CFG 默认值:' + JSON.stringify(CFG))
}
```

要点：
1. **先判断 `typeof args === 'string'`，尝试 `JSON.parse`**，解析失败则视为无效（保留原有回退
   行为，不让解析异常中断流水线）。
2. **回退到 CFG 时必须显式 `log()` 一条警告**，而不是静默——防止"以为传参生效、实际跑了默认
   配置"这类事故重演（book-retro/book-gap-audit 影响的是复盘/审计报告的实例范围，误判范围会
   导致报告文不对题却不自知）。
3. 同款模式也存在于 `chapter-pipeline.js`/`chapter-retrofit.js`（用 `chapter_id` 作判据字段），
   建议一并修（Lead 决定是否本次一起做，或作为后续跟进）。

## blocking / warn 定级建议

- 这不是叙事/dossier 质量闸门，而是**编排层防御性修复**，无"blocking/warn"意义上的 verdict；
  建议按"必须做"而非"可选"对待——因为当前是**完全静默**的回退，一旦命中就没有任何信号能让
  Lead 发现参数没生效。
- 若额外产出静态检查（见下），该检查本身对 `.claude/workflows/*.js` 建议判 **warn**（提示性
  代码审查项，不阻断任何 pipeline 运行）。

## 测试用例草案

workflow 脚本体依赖 host 注入的 `args`/`agent`/`log` 等全局量，不便直接 pytest 化；建议拆成
两部分：

1. **提取纯函数单测**（推荐）：把 CFG 归一化逻辑抽成 `.claude/workflows/lib/resolve-cfg.js`
   导出的纯函数 `resolveCfg(args, CFG, requiredKey)`，用 Node 内置 `node --test` 写单测：

```js
// .claude/workflows/lib/resolve-cfg.test.js
const { test } = require('node:test')
const assert = require('node:assert')
const { resolveCfg } = require('./resolve-cfg')

test('object args with required key passes through', () => {
  const cfg = { instance: 'default' }
  const r = resolveCfg({ instance: 'vllm' }, cfg, 'instance')
  assert.strictEqual(r.instance, 'vllm')
})

test('JSON-string args is parsed before key check', () => {
  const cfg = { instance: 'default' }
  const r = resolveCfg(JSON.stringify({ instance: 'vllm-ascend' }), cfg, 'instance')
  assert.strictEqual(r.instance, 'vllm-ascend')
})

test('unparseable string falls back to CFG without throwing', () => {
  const cfg = { instance: 'default' }
  const r = resolveCfg('{not json', cfg, 'instance')
  assert.strictEqual(r.instance, 'default')
})

test('object args missing required key falls back to CFG', () => {
  const cfg = { instance: 'default' }
  const r = resolveCfg({ chapters: [] }, cfg, 'instance')
  assert.strictEqual(r.instance, 'default')
})
```

2. **可选：新增静态防回归检查** `scripts/lint_workflow_cfg.py`（Python，与其余 `scripts/lint_*.py`
   风格一致），扫 `.claude/workflows/*.js` 找 `args && args.<key>) ? args : CFG` 模式且其上文
   若干行内**没有** `typeof args === 'string'`/`JSON.parse` 字样，判该文件 warn，防止未来新增
   workflow 脚本重犯同一坑：

```python
# scripts/tests/test_lint_workflow_cfg.py（若采纳该静态检查）
def test_flags_workflow_without_string_guard(tmp_path):
    ...
def test_passes_workflow_with_json_parse_guard(tmp_path):
    ...
```

## 提醒

请 Lead 走 TDD 小任务落地（workflow 编排逻辑不在 Curator 权限内）。建议范围：先修
`book-retro.js`/`book-gap-audit.js` 两处（本次批准范围），`chapter-pipeline.js`/
`chapter-retrofit.js` 的同款模式作为可选一并处理或后续单独立项；静态防回归检查视精力决定是否
本次一并做。
