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

test('undefined args falls back to CFG without logging', () => {
  const cfg = { instance: 'default' }
  let logged = null
  const r = resolveCfg(undefined, cfg, 'instance', (msg) => { logged = msg })
  assert.strictEqual(r.instance, 'default')
  assert.strictEqual(logged, null, 'args 本就没传，不应算"收到但回退"，不应告警')
})

test('object args missing required key logs a warning naming the fallback', () => {
  const cfg = { instance: 'default' }
  let logged = null
  const r = resolveCfg({ chapters: [] }, cfg, 'instance', (msg) => { logged = msg })
  assert.strictEqual(r.instance, 'default')
  assert.ok(logged && logged.includes('instance'), '回退到 CFG 时必须显式 log 一条警告，点名缺失字段')
})

test('unparseable JSON-string args logs a warning (parse failure is a received-but-invalid case)', () => {
  const cfg = { instance: 'default' }
  let logged = null
  const r = resolveCfg('{not json', cfg, 'instance', (msg) => { logged = msg })
  assert.strictEqual(r.instance, 'default')
  assert.ok(logged, '字符串 args 解析失败也应视为"收到但回退"，须告警而非静默')
})
