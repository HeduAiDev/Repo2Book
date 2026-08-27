# v3 ch11《抢占与请求的一生》交付归档（APPROVED）

- **Type**: delivery（v3 Archive 站，chapter-pipeline-v3 第十一章、Part III 第三章——出序生产：ch12 已于同日先归档，本章补齐后 **Part III 四章（ch09-12）全部收官**）
- **Chapter**: v3 ch11 · Part III 引擎的心跳：调度循环 · kind=code（L0 缩放：调度账本+状态机）
- **Pin**: vLLM v0.27.1（6e448d0ea）；行号基线即此版
- **Date**: 2026-08-27 · **Agents**: pipeline 各站（analyst→researcher→implementer→tester→explainer→illustrator→writer→reviewer）+ archivist（本记录）
- **Verdict**: APPROVED，16 条 issue（0 blocking；15 negotiable + 1 条逐机制勾选表存档非缺陷，其中 5 条 reader-comprehension 维），全文见 `artifacts-v3/ch11-preemption-request-lifecycle/reviews/review-report.json`

## What happened

- **回环**（`reviews/run-ledger.json`）：impl↔test 1 轮（42 项纯单元 host 全绿 0.06s、不 import vllm——双法验证：host 未装 vllm 且套件照过；tester 另在容器内对**真 vLLM 调度器**做 61/61 行为交叉核验、from-scratch 驱动用后即删）；write↔review 2 轮；L2/图 2 轮；盲审 1 轮零失败（9 图全 PASS——L2-ch11 复盲审含「四笔虚线小注」计数修正 + 8 张机制图独立盲审）。
- **归档时抽查（issue 兑现状态）**：16 条全部 negotiable，抽查 8 处标记（L55 IntEnum 引文插入语、L167「唯一触发信号」与 L310 reset_prefix_cache 第二抢占源无互指、L572「出生就带」超证、L518 hash_i code span 体例、L192「戏眼」、L225 priority 数值方向约定、L700 m9 缺「账面 49 token」口径、站 11 无 ch10 回指）均**未在稿**——以 APPROVED 归档（ch02/ch07/ch08/ch09/ch10/ch12 同先例），writer 定点小修清单留 review-report.json。
- **bible 登记（v3 侧车）**：glossary-v3 +11（num_stale_output_tokens / drop_stale_output / num_in_flight_tokens / 锁步冲销 / 水位（watermark） / 队头阻塞 / swap（换出） / 管线深度 / 背压 / resumable（流式输入会话） / priority 数值约定——state.json 前注点名的前四条主场术语全部落账、ch12 未抢注如约）；concepts-v3 +17（单 IntEnum 状态机、水位三限定准入、recompute-only 取舍全史、stale 平行账、free 不清哈希、抢占环有限终止与自我放弃、守卫 why、被抢集三个化身、resumed 整表替换登记、check_stop 五连判、finish_reason 先抓再办+resumable 假终点、finish_requests 两遍法幂等、终点逆序归还+除名、热循环四动作面、逐 token 先入账后判停、remove_all 快路径、spec 拒绝回扣）；interfaces-v3 +25 条（抢占环/_preempt_request 六件事/守卫/双队列遍历/_try_promote/前缀重命中/恢复准入/回流落位/update_from_output 主线/_update_request_with_output 逐字/_handle_stopped_request 坍缩/finish_requests/_free 族/_update_after_schedule/KVCacheManager 水位两处+free 哈希不清+get_computed_blocks/get_request_block_hasher/RequestStatus+计数器群/check_stop 族/SchedulerConfig 两旋钮/SchedulerOutput+ModelRunnerOutput/EngineCoreOutput 四字段/interface 契约+PauseState/FCFSRequestQueue 十操作）；figures.json 追加 9 张（L2-ch11 + m1/m2/m3/m4/m6/m7/m8/m10 八张机制图，book:v3）。
- **伏笔对账**：本章应埋 F2「抢占恢复撞前缀缓存」——正文实埋（L2 图注 L11「埋在第 ⑥ 拍片」+ 站 5 L291「哈希留表=站 9 的全部伏笔」+ 站 9 正戏 L468-L534 + m7 实测 + prefix-rehit 图 + L2 图第 9 站注记），planted done:true 已登记 foreshadow-v3.json；应收无（pedagogy-plan F1-F10 的 paid 集合均不含 11，与 run-ledger foreshadow_due 一致）。**回修债一条**（review issue 7）：正文 L11「第 15 章」为裸章号——ch15 成稿后升级为规范跨章链接，已记入 F2 evidence。
- **图登记门禁**：`python scripts/lint_figures_registered.py <章目录>` 显式传参 + REPO2BOOK_INSTANCE=vllm（active_instance=triton-ascend，无参模式照不到 vllm）exit 0（9/9 登记核过）。

## Why it matters

Part III 第三章、出序补齐即收官：把 ch10 留下的两扇门（allocate_slots None 的抢占内环、守卫的 why）与 ch9 第 ⑤ 拍盖着省略号的「请求生命周期内景」三张欠条一次兑现。一章走完请求的一生：一枚 IntEnum 记生死（>PREEMPTED 一次比较、枚举顺序=隐式 API）→ 池子见底那一拍的抢占环（FCFS 抢队尾最年轻者、六件事带回与首调度同构初态、recompute-only 是 v1 首提交起的唯一路径——swap 连配置项一起被删）→ stale 在途输出平行账（赋值不累加/锁步冲销/排空前推迟恢复/drop-mode/同步自中和——异步调度的账单，为 ch12 立桩）→ 恢复（free 不清哈希→前缀重命中 65 只补 1——F2 埋给 ch15；水位三限定准入 #44594 治 decode-heavy 超收抖动；resumed 整表替换）→ 收尾（热循环四动作面/check_stop 五连判/finish_reason 先抓再办+resumable 假终点/终点逆序归还+账本除名/finish_requests 两遍法幂等——ch9 abort 双投递的引擎侧前提）。与 ch10 合起来第 ① 拍黑盒从外到里全部打开；Part III 至此完整，调度器故事只剩异步形态（ch12 已交）。

## What to remember

1. **【writer 定点小修清单待用】** 16 条 negotiable 全部未兑现即归档（APPROVED 不阻断合规，先例一致）。最优先四条：①L55 标「官方文档原话」的引文含插入语 "(and always)" 与链接页面不一致——「原话」是逐字承诺，改一行；②L167「allocate_slots 返回 None 是抢占的唯一触发信号」与 L310 自述的 reset_prefix_cache（同一 _preempt_request 的第二调用源）绝对论断打架——补「调度路径上」两字或一句指针；③L225 PRIORITY「取最大=优先级最低」缺数值方向约定（越小越优先、类 Unix nice）——glossary-v3 已记约定（priority 数值约定，首现章 ch11），正文补半句括注即可；④m9 场景（L700）缺「prefill 已产出首 token、账面 49」口径——站 9 刚教过 cap 复算，认真读者会得出与表矛盾的 2 块（m7 L518 有现成句式可复用）。其余：§7.3 节号、出生就带→早期就带、hash_i 提为 $`…`$、戏眼→看点、背压人话版、流水线并行括注、站 11 回指 ch10、m4 P5「补 3 送 1」搭桥、被抢集三个化身一句话串联、幂性→幂等性。
2. **【F2 回收债已记账】** ch15 成稿后两件事：正文 L11 裸章号链接化（../../ch15-<slug>/narrative/chapter.md）+ foreshadow-v3.json F2 paid 翻 true——均记在 F2 evidence 里，ch15 归档时 archivist 对账。
3. **【state.json 前注兑现】** 前一 archivist 在 v3.status 预告的「ch11 主场术语 num_stale_output_tokens/drop_stale_output/num_in_flight_tokens/锁步冲销待归档时登记，ch12 未抢注」——本次四条全部落账（另加 7 条），ch12 归档时的克制得到回报。
4. **【python3 坏桩持续】** host `python3` 仍是 WindowsApps 坏桩（exit 49，test-report 有双法记载）；本章全程 Miniconda `python`。台账第二次确认，后续站照此办理。
