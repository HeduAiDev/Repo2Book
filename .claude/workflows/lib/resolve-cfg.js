// resolve-cfg.js — CFG 归一化纯函数（lint-exp-N1）
//
// 背景：本环境已知"Workflow 的 args 注入不可靠"（见 chapter-pipeline.js 顶部注释）。
// 若 host 把 args 以 JSON 字符串（而非已解析对象）形式注入，`args.instance` 对字符串
// 取属性会返回 undefined，旧写法 `(args && args.instance) ? args : CFG` 会静默回退到
// 脚本内 CFG 默认值——调用方以为参数生效了，实际跑的是硬编码默认配置，且没有任何报错。
//
// resolveCfg 处理两件事：
//   1. 若 args 是字符串，先尝试 JSON.parse 再判断（防止字符串形态被误判为"没传"）。
//   2. 回退到 CFG 时，若 args 确实"收到过"（非 undefined/null），显式调用 log() 告警，
//      而不是静默——防止"以为传参生效、实际跑了默认配置"的事故重演。
//      注意：args 本就未传（undefined/null）是正常调用路径，不告警；
//      args 传了但缺必需字段、或字符串解析失败，都算"收到但回退"，需要告警。

function resolveCfg(args, CFG, requiredKey, log) {
  const warn = typeof log === 'function' ? log : () => {}

  const wasProvided = args !== undefined && args !== null
  let _args = args
  let parseFailed = false

  if (typeof _args === 'string') {
    try {
      _args = JSON.parse(_args)
    } catch (e) {
      _args = null
      parseFailed = true
    }
  }

  const hasKey = _args && typeof _args === 'object' && _args[requiredKey]
  if (hasKey) {
    return _args
  }

  if (wasProvided) {
    const reason = parseFailed
      ? 'JSON.parse 失败'
      : `缺 ${requiredKey} 字段`
    warn(
      `⚠️ args 已收到但${reason}，回退到脚本内 CFG 默认值: ${JSON.stringify(CFG)}`
    )
  }

  return CFG
}

module.exports = { resolveCfg }
