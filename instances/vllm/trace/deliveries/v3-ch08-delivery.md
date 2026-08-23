# v3 ch08《输出的另一个维度：logprobs》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第八章、Part II 第五章）
- **Chapter**: v3 ch08 · Part II 分而治之：进程边界与消息 · kind=code（L0 缩放：上行泳道 logprobs 支路）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-22 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，14 条 issue（1 blocking figure-integration + 13 non-blocking/negotiable），全文见 `artifacts-v3/ch08-logprobs/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（90 用例 host 全绿 ~3.4s：真 Rust byte-fallback tokenizer 0.22.2/真 msgpack 过线/真 torch CPU，无平台分支不进容器）；write↔review 3 轮；L2/图 1 轮；盲审 1 轮零失败（5 图全 PASS）。注：figure-manifest 里 L2 的「修后复验·第 2 轮」是 blocking 修复后的 revise 步盲审再验证（只看修后 PNG+issue 清单），非 ledger 新正式轮次。
- **归档时抽查（issue 兑现状态）**：blocking 1 条（正文三处把 L0 品红「采样与出口」列说成「绿色 GPU 执行臂的采样列」，figure-integration auto-REJECT 维）**已兑现**——现稿 L17「橙色 EngineCore 带里的采样出口列（品红那列，…sampler.py 的 ④ sample_tokens 框就画在这列）」+「①-④ 在引擎侧的 GPU 执行臂与采样出口列」、L96/L305「采样出口列」三处字级全改；L2 图同步复渲（图题/顶部图例/「①-④ 对应 L0 位置」面板/底注五处口径，复用 ch1 已立列名「采样出口列」、色彩词用「品红」）后盲审第 2 轮复验 PASS（盲审判词：五处口径与 suggested_fix 近逐字吻合、全图 0 处「绿色采样列」残留）。13 条 non-blocking **抽查均未在稿**（站 5-7 标题「卡车」未改「班车」L303、hook 预支未定义的 k+1 L7、m10 量化列 44001 与「同一算术口径」按语自相矛盾 L975、站 4 prefer 分支行号 L132-L135 应 L133-L136 L292、站 14 非流式 L886-892 应 L885/L889-L895 L1339、站 6 省略标注 L1931-L1938 应含 L1939 L424、m6 唯一入口五行分派未内嵌 L549、m13 缺小节级直觉句 L1113、rank「OpenAI 无对应物」论断缺就地指证 L270、工序号↔站号两套编号无对照、站 6 num_logprobs 三名并存缺括注、SamplingMetadata 首现未解释、钳底 −9999.0「JSON 装不下极小值」理由漏洞 L1441）——以 APPROVED 归档；全部为字级/半句级，writer 定点小修一轮可全消，清单留 review-report.json。
- **bible 登记（v3 侧车）**：glossary-v3 +13（logprob/top_logprobs/prompt_logprobs/logprob_token_ids 点名册/FlatLogprobs/LogprobsProcessor/logprobs_mode 四态/rank token 名次/cumulative_logprob/bytes 字段/ChatCompletionLogProb/SamplingMetadata/copy stream·D2H）；concepts-v3 +13（raw logprobs 语义/bytes 字段/k+1 列形状/批均一张量团体餐/log 域记账/U+FFFD 修正两轴/非增量解码/FlatLogprobs 对象账/prompt logprobs 支路/rank 链 dict 键去重/支路零耦合/搭主泳道班车/两刀截断区分）；interfaces-v3 +26 条（Logprob+FlatLogprobs/create_*+append_logprobs_for_next_position/LogprobsProcessor 整类八方法/convert_ids_list_to_tokens/Sampler.forward+compute_logprobs+gather_logprobs/batched_count_greater_than/greedy 快路径/gather_specific_token_logprobs/SamplingMetadata/InputBatch·CachedRequestState logprobs 域/LogprobsTensors·Lists 四接口/AsyncGPUModelRunnerOutput/_get_prompt_logprobs_dict/Scheduler.update_from_output logprobs 行/线载体+msgpack 钩子链/process_outputs 骨架/RequestState logprobs 段/SamplingParams 四参数/LogprobsMode/ChatCompletionLogProb 三类/to_sampling_params logprobs 段/_get_top_logprobs+_create_chat_logprobs/_get_decoded_token 等）；figures.json 追加 5 张（L2-ch8=l2-logprobs-lane + m1-raw-snapshot/m2-gather-triple/m8-ufffd-repair/m10-flat-vs-nested，book:v3）。
- **伏笔对账**：本章无应埋、无应收（pedagogy-plan foreshadows 十组的 planted/paid 集合均不含 8，与 run-ledger foreshadow_due:[] 一致），foreshadow-v3.json 零改动。正文结尾「下一章把循环框整个放大——EngineCore 逐拍循环」是 ch01 已埋 F1（paid=ch09）的自然过渡非新伏笔；两处「留给后续 RL/scoring 专题」为书外范围注记不入账。
- **图登记门禁**：`python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0（5/5 登记核过）。

## Why it matters

Part II logprobs 支路章：把 ch7 开头留的念想（EngineCoreOutput.new_logprobs 字段每次路过都没打开）展开成整条支路的机制账——「惩罚不扭曲模型意见」的 raw 留底语义（NOTE(woosuk) 与 V0 分道、v0.27 logprobs_mode 四态把 V0 语义降级成显式开关、RL ratio e^1.8≈6 倍的生态压力）、gather 三件套缩 [num_tok,k+1]（被采样恒列 0 的物理起点）、批均一张量团体餐与两刀截断、同车 D2H/过线（logprobs 没有专车、msgpack 钩子给 ndarray/tensor 发原生类型护照）、非增量解码与 U+FFFD 修正两轴（横向候选各修各的/纵向前文共用、上下文 4 上界来自 RFC 3629）、FlatLogprobs GC 对象账 O(L×k)→O(1)、prompt 支路「补考」（两方法契约/分块批改末块交付/首位 None/与前缀缓存互斥）、出口 token/logprob/bytes 三件套（bytes 是唯一不受修正干扰的字节真相）。兑现 ch7 impl-notes 登记的回填契约（CompletionOutput.logprobs/cumulative_logprob 与 RequestState.logprobs_processor 传参恢复逐字真码，未重考古）；ch9（EngineCore 循环）可按「前章已立」直接引用采样出口列与两泳道同车不同步的账；后续 Part VII 采样章（随机路径/惩罚公式）与 ch38（OpenAI 门面）有本章接口面可接。interfaces-v3 累计 5 章 92 条。

## What to remember

1. **【writer 定点小修清单待用】** 13 条 negotiable 全部未兑现即归档（APPROVED 不阻断合规，ch02/ch07 有先例）——其中 5 条是可核性硬伤类（两处行号偏移 L292/L1339、省略标注 L424、m10 表后散文数字 44001 与素材 42001 打架、钳底理由 L1441 会被读者用 case 2 输入当场证伪一半），建议下轮 writer 小修优先吃掉；「卡车→班车」与「hook 去 k+1」两条是隐喻/阶梯一致性，也值得修。
2. **【m6 五行分派是现成素材】** dossier embed_excerpts 已备 update_from_output L348-L352 逐字段片段、全书未内嵌——若后续做 retrofit-deep-revise 或 ch9 讲 process_outputs 第 3 步时，可直接取用（正文一句话点「new_logprobs 走生成路、new_prompt_logprobs_tensors 走 prompt 路」）。
3. **【跨章契约兑现核验】** ch7 impl-notes 登记的 ch8 回填点已按登记项兑现（本章 impl-notes「RequestState logprobs 段 L224-225/L366-371/L404-420 逐字」+ CompletionOutput 两字段实参），无需后续动作——记录于此供 ch9+ 引用时免复查。
4. **【工具卫生延续】** active_instance=vllm 期间 lint_figures_registered 显式章目录即可（本次未带 REPO2BOOK_INSTANCE 也 exit 0）；lint_dossier/lint_fidelity 对 artifacts-v3 的静默跳过缺陷（ch04-ch07 已记）仍未修，本章由 impl-notes 收工审计（148 引用区间机械核对、12 处偏移修正、must_keep 50 全核在）兜底。`python3` WindowsApps 坏桩（exit 49 零输出）本章第三次复发（gen_L2 复跑 + 本归档脚本均踩），INSTANCE.md L31/L44 已有记载，无需新增条目。
