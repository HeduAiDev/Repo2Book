# 高级引擎运维: 弹性 EP 不停机扩缩状态机(ElasticEPScalingState scale_up/scale_down 各阶段、core.py reinitialize_distributed 不停机重建分布式组、_eep_scale_up_before_kv_init 在 KV init 前扩、专家迁移/DP 维度重分配; 与 ch21 DP wave + ch07 EngineCore 关系) + Responses API 多轮(OpenAIServingResponses 跨轮会话: context 保存前轮 output 喂下轮、harmony 共享 list、有状态会话)。全书 code 章收官

- **Type**: delivery
- **Chapter**: 33
- **Date**: 2026-06-24
- **Timestamp**: 2026-06-24T17:38:21Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: ch33, engine-core, elastic-ep, scale-up, scale-down, reinitialize-distributed, responses-api, multiturn, harmony, part-viii, APPROVED

## What happened

ch33 reviewer 判定 APPROVED, 全书 code 章收官。两条主线: (1)弹性 EP 不停机扩缩——ElasticEPScalingState 四角色状态机(ScaleUp{Existing,New}/ScaleDown{Remaining,Removing}), busy-loop 每轮 progress() 只推一步(§33.4), reinitialize_distributed 按 new_dp_size 判 scale_up/down、SHUTDOWN_CURRENT_RANK 判 removing、转非阻塞(§33.5); existing 引擎 9 步、new 引擎 4 步把扩入提前到 KV cache init 之前(_eep_scale_up_before_kv_init, 用同步显存额度而非 determine_available_memory)(§33.8); _switch_and_prepare 以 all_reduce(MAX) 收敛 [engines_running,current_wave,step_counter] 三元组对接 ch21 DP wave(§33.6); 两阶段 TCP-store barrier 化解通知偏序、最多多一次 forward step(§33.7); scale_down 余留减/被裁退、removing 引擎 raise SystemExit(§33.9)。(2)Responses API 有状态多轮——construct_input_messages 非 harmony 路径经 response_store 的 prev_response_output 回灌上轮 output(§33.11); harmony 路径 HarmonyContext._messages 与 msg_store 共享同一 list, append_output extend 自动留存(§33.12); response_store/msg_store 存取(§33.10)。36 测试用例(host 纯单元, 不 import vllm, 替身驱动状态机+多轮拼接)全绿, 4/4 linter 全过。reviewer verdict=APPROVED, 6 条 issue 全 non-blocking+negotiable: §33.13 用非 harmony 两轮测试当共享 list 活证据名实不符(建议补澄清: 该测试注入 HarmonyContext 走 harmony 机制, 非 harmony 经 prev_response_output 回灌)、'36 个断言'应改'36 个测试用例'、run_busy_loop_once 杜撰方法名应改 run_busy_loop、§33.4'三个收尾细节'实只两条 bullet、EP 线缺 all_reduce(MAX) 三元组数值追踪(建议补 [1,7,142] 逐分量 MAX)、§33.7 两阶段 barrier 缺跨 rank/step 时间线追踪。bible 登记 4 个精简版接口。

## Why it matters

全书代码之旅终点(Part VIII 收官)。承接 ch07 EngineCore busy-loop + ch21 DP wave 三元组——弹性 EP 把'不停机'的难点(busy-loop 里塞不下阻塞重建)用状态机+两阶段 barrier 化解。Responses 多轮承接 ch10 logprobs/输出装配与 ch32 OpenAIServing 基类。foreshadow due ch33 为空: 非任何伏笔 plant/payoff(对 ch07/ch21/ch32 引用均以'此前已细讲'呼应, 非 bible 强制项)。

## What to remember

ch33 reviewer 判定 APPROVED, 全书 code 章收官。两条主线: (1)弹性 EP 不停机扩缩——ElasticEPScalingState 四角色状态机(ScaleUp{Existing,New}/ScaleDown{Remaining,Removing}), busy-loop 每轮 progress() 只推一步(§33.4), reinitialize_distri...
