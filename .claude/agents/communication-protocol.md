# Agent 通信协议

每个 agent 遵守三件事：**出站通知、入站检查、超时升级**。

## 1. 出站通知

完成工作后，把状态写入 `/tmp/book-factory/{chapter}/{role}-status.json`：

```json
{"agent": "writer", "chapter": "05-xxx", "status": "done", "output": "instances/vllm/artifacts/05-xxx/narrative/chapter.md", "time": "2026-05-05T10:00:00Z"}
```

状态值: `working` | `done` | `blocked` | `waiting_for_{other_agent}`

## 2. 入站检查

开始工作前，检查上游 agent 的状态文件是否存在且有效：

```
如果我是 writer，我应该等 implementer 完成：
  implementer-status.json 存在 && status == "done" → 继续
  implementer-status.json 存在 && status == "blocked" → 检查 blocked_reason，解决或升级
  implementer-status.json 不存在 || 超过 30 分钟未更新 → 升级到 Team Lead
```

## 3. 超时升级

以下情况立即报告 Team Lead（在状态文件的 `blocked_reason` 中写清楚）：

- 等待上游 agent 超过 30 分钟无响应
- 收到下游 agent 的请求但无法处理
- 自身任务超过预期时间 2 倍仍未完成
- 发现上游 agent 的输出有问题（文件损坏、内容不完整等）

报告中必须包含：**谁出了问题、什么时候发现的、你等了多久、建议 Team Lead 怎么做**。

## 状态文件位置

```
/tmp/book-factory/{chapter}/
  implementer-status.json
  tester-status.json
  writer-status.json
  reviewer-status.json
  researcher-status.json
  archivist-status.json
```
