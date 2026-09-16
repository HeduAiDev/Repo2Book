export const meta = {
  name: 'one-fig',
  description: '单图任务：illustrator 按 dossier supplement 画一张图（model 经 args.model 指定，默认 claude-haiku-4-5——2026-09-17 网关换代 fable/deepseek-flash 双下线）。',
  phases: [{ title: 'Fig' }],
}
// args: { prompt: string }（完整任务提示词）+ CFG 兜底
const CFG = { prompt: '' }
const A = (typeof args !== 'undefined' && args && args.prompt) ? args : CFG
if (!A.prompt) { return { error: 'no prompt' } }
const r = await agent(
  A.prompt,
  { schema: { type: 'object', additionalProperties: false, required: ['status', 'note'], properties: { status: { type: 'string', enum: ['OK', 'BLOCKED'] }, note: { type: 'string' }, blocker_reason: { type: 'string' } } },
    label: 'fig:' + (A.label || 'one'), phase: 'Fig', agentType: 'illustrator', model: A.model || 'claude-haiku-4-5-20251001' }
)
return r
