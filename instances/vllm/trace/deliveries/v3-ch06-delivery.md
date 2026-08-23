# v3 ch06《下行：从文本到 token》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第六章、Part II 第三章）
- **Chapter**: v3 ch06 · Part II 分而治之：进程边界与消息 · kind=code（L0 缩放：蓝色 API 进程带下行泳道）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-17 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，18 条 issue（1 blocking cognitive-ladder + 17 non-blocking），全文见 `artifacts-v3/ch06-downlink-text-to-token/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（57 用例 host 全绿）；write↔review 3 轮；L2/图 3 轮；盲审 3 轮（轮 1 FAIL——pool-offload 三条泳道标签被框顶边拦腰切字 [rect-rect 盲区]；轮 2 FAIL 两处——pool-offload「带图的路」标签被泳道虚线删除线穿过 + mm-flatten 首根括弧标签压「展平后」节标题 [轮 2 复盲还推翻了轮 1 对 mm-flatten 的误判 PASS，教训=盲审独立性价值]；轮 3 零失败）。
- **归档时抽查（issue 兑现状态）**：blocking 1 条（L38「Renderer 正式成为三件套第一件」与 ch1 钦定定义「第一件=Renderer+InputProcessor 两块」跨章矛盾）已按最简改法兑现（现稿「三件套第一件『Renderer+InputProcessor』里可复用的那一半」）。其余 15 条内容类 non-blocking **逐条核实在稿**：7 实参省略标注（两处）、arrival_time 方法名↔行号同序、m4 姊妹后门 async_llm.py:L337-L342 补句、m7 八类拒绝路径运行级绿证据、render-pipeline 图注「右侧 completion 面」、双池节借用池长句拆两句、「回程地址章」→「回程路由章」对齐 ch4 原词、mm_kwargs↔MultiModalFeatureSpec 特征载荷搭桥、decoder_inputs 绑回 processed_inputs、mm 缩写首现括注、异步调度回收竞态 gloss、增量解码要原文指路第 7 章、DictPrompt 联合别名引定义两行、all_stop 括注、图注裸「第 2 章」补链接。
- **bible 登记（v3 侧车）**：glossary-v3 +17（chat 模板/EngineInput 家族/DictPrompt/TypedDict/fast tokenizer/arrival_time/双池/借用池 TokenizerPool/mm 特征 MultiModalFeatureSpec/PlaceholderRange/占位符-特征拼接/P0-P1 两级 mm 缓存/LoRA/VLM/代理键与自然键/生日界/max_model_len）；concepts-v3 +16（tokenize 在前端文本不过线/渲染四步流水/快路径分流/救响应性不救吞吐/双池分工/借用池/Rust 批路径放 GIL/TypedDict 判别联合/七道关口/克隆补全不变式/占位换座/展平排序不变式/两级缓存省 IPC/identifier 双轨/双轨 id 深挖/诞生即 tokenized）；interfaces-v3 +21 条（BaseRenderer 双池装配/四步流水四方法/_process_multimodal/make_async/InputPreprocessor 兜底/process_inputs 组装主干/校验链三方法/assign_request_id/argsort_mm_positions+PlaceholderRange+MultiModalFeatureSpec/SamplingParams 三补全方法/EngineCoreRequest/add_request 分流主干/_add_request 双登记等）；figures.json 追加 4 张（L2-ch6=l2 章图 / ch06-fig-render-pipeline=m1 / ch06-fig-pool-offload=m2 / ch06-fig-mm-flatten=m8，book:v3）。
- **伏笔对账**：本章无应埋、无应收（pedagogy-plan foreshadows 的 planted/paid 集合均不含 6，与 run-ledger foreshadow_due:[] 一致），foreshadow-v3.json 零改动。正文对 F3 client_index 仅作 ch4 回顾（站 9 字段表「回程路由章」）非收款——收款仍按计划在 ch34。注意：pedagogy-plan ch6 条目 notes 里「伏笔埋：EngineCoreRequest 字段 → ch5 帧序」是定稿前的陈旧草稿注（ch5 在 ch6 之前，时序不成立；正文实际以回指方式接 ch5 字节账），正式 foreshadows 数组为准、无需登记。
- **图登记门禁**：`REPO2BOOK_INSTANCE=vllm PYTHONIOENCODING=utf-8 python scripts/lint_figures_registered.py <章目录>` 显式传参 exit 0（active_instance=triton-ascend，无参模式照不到 vllm）。

## Why it matters

Part II 下行章：把 ch1「文本不过线、token 过线」的一句结论展开成整条泳道的机制账——为什么切词必须在过线前做完且不许跑在事件循环上（#11963 序列化账 + #49608 停顿 6.14s→0.99s 但总耗时仅 −8% 的「救响应性不救吞吐」）、双池与借用池两层并发安全（#38418 键序 / #36557 Already borrowed）、七道关口错误前移、params 克隆隔离不变式、mm 占位-换座范式与展平排序不变式、两级 mm 缓存命中省 IPC、双轨 id 的代理键思维与生日界算账。此后 ch07（上行 detokenize/扇出）、ch08（logprobs）、后续 mm 相关章可按「前章已立」直接引用；interfaces-v3 自 ch04 开张后登记下行泳道接口 21 条。

## What to remember

1. **【Lead 决策项·工具卫生第三次】** review issue 5：lint_diagrams.py 在 Windows GBK 控制台打印 ✓/❌ 即 UnicodeEncodeError 假失败（0 实质问题却 exit 1）——与 ch04/ch05 记录的 GBK 家族问题同根（ch04/ch05 是 lint 输出侧，本章波及按 exit code 判定的门禁）。建议 lint_* 脚本入口统一 `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`，或 RUNBOOK/CLAUDE.md linter 命令统一前置 PYTHONIOENCODING=utf-8；INSTANCE.md「用 python + PYTHONIOENCODING=utf-8」的一行提示仍缺。
2. **【Lead 决策项·跨章升级】** review issue 11（非本章缺陷）：ch01 chapter.md:L122 对 L0 图泳道方位文说右、图在左（下行 Renderer/InputProcessor 实画左列、上行 Collector/OutputProcessor 右列），是 exp-2026-08-16 用户抓「ch1 左右写反」后的残留；建议修文对齐图并全文 grep ch01-ch05 的「左边/右边」复核——L0 是全书唯一权威地图，不修会持续污染后续各章方位叙述。
3. **【linter 缺陷仍未修（ch04/ch05 记录的延续）】** lint_dossier/lint_fidelity 只认 `artifacts` 目录名、对 `artifacts-v3` 静默跳过锚点核验——本章 impl-notes 机械复核 49 个核心锚点零失配兜底，但该缺陷对后续 v3 章仍是系统性风险。
4. **【盲审纪律实证】** mm-flatten 轮 1 的误判 PASS 被轮 2 独立复盲推翻（像素级核验标签墨迹压标题墨迹带）——再次印证 CLAUDE.md「盲审必须独立」：自审/前轮结论天然带确认偏误，多轮独立复盲是必要成本非冗余。
