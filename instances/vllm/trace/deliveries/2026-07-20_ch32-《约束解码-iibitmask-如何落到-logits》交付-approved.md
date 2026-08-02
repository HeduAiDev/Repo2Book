# 《约束解码 II:bitmask 如何落到 logits》交付-APPROVED

- **Type**: delivery
- **Chapter**: 32
- **Date**: 2026-07-20
- **Timestamp**: 2026-07-20T21:51:49Z
- **Agents involved**: archivist, analyst, implementer, tester, explainer, illustrator, writer, reviewer, team-lead
- **User present**: True
- **Tags**: delivery, ch32, part-vi, structured-output, constrained-decoding, bitmask, triton-kernel, default-path, spec-decode, reasoning-gate, copy-stream, dossier-escape, citation-range

## What happened

ch32-structured-output(kind=deep, Part VI 双章下篇, source_pin v0.21.0 ad7125a4)交付并归档,verdict=APPROVED(评审 9 项全 non-blocking,已由 writer/illustrator 逐条修完)。十二节走完掩码的下半程——从调度器筛人到 GPU 上一次带谓词的写回:
①§32.1 门控用同一个谓词写两遍——has_structured_output_requests 在 scheduler.py:L942-951 置位、L1224-1246 收集,判据逐字相同,故「掩码行数 = 本步产出 logits 的结构化请求数」是构造性成立而非碰巧;prefill 中段的请求本步不采样,一行不占。
②§32.2 **行序做成数据不做成约定**:GrammarOutput 把 structured_output_request_ids 随掩码同传,不变式是「第 k 行属于 ids[k]」,**不是**「掩码序 = batch 序」;worker 用 req_id→batch 下标字典 + cu_num_logits 区间重建映射。这类错位不报错,只静默生成坏 JSON。缓冲按最坏情况 max_num_seqs×(1+num_spec) 行 × ceil(|V|/32) 列开一次跨步复用,故「不受约束」必须显式写整行 -1,靠留空会吃到上一步残留位。
③§32.3 装配前先过草稿安检:validate_tokens 过滤 + -1 补齐把长度夹逼回调度时定下的数,必须排在掩码之前;异步调度延后采样(pending_structured_output_tokens / AsyncScheduler._update_after_schedule / DraftTokensHandler)。
④§32.4 并行填充三道门——构造期 128<max_num_seqs、运行期批量>128、且无投机;因一步内请求数不超过 max_num_seqs,**max_num_seqs ≤ 128 的部署里并行分支是结构性死代码**;第三道门排除投机是正确性要求(投机行共享同一语法对象的推进顺序)。
⑤§32.5 投机把一行掩码变成 1+k 行:逐位置试探性 accept_tokens 推进、末尾一次 rollback 使本步净位移为零;被语法作废的位子由 num_invalid_spec_tokens 从接受率分母扣掉。
⑥§32.6 两道独立推理门:should_fill_bitmask 管填不填、should_advance 管推不推进;侦测到思考结束的那一步只置 reasoning_ended、**明确不推进**,约束从下一步生效(否则会拿结束标记本身喂语法);enable_in_reasoning 打开则第一步就受约束。
⑦§32.7 掩码搭前向的车:engine core 先 execute_model(non_block=True) 发车、再算掩码(纯 CPU),落点卡在 compute_logits 之后、采样器之前——两头各有硬理由。
⑧**【全章路线前提】§32.7 岔口**:VLLM_USE_V2_MODEL_RUNNER 默认 False(envs.py:251 / 取值 lambda :1711-1713),gpu_worker.py:L316-330 据此二选一,**默认部署构造 GPUModelRunnerV1**,掩码落点 gpu_model_runner.py:L4243-4247 → structured_output/utils.py:L44 apply_grammar_bitmask → **xgrammar 自带的 xgr.apply_token_bitmask_inplace,没有 vLLM 自写 Triton kernel**。
⑨§32.8 默认路径:把紧凑掩码摊平成与 logits 同形的 sorted_bitmask、空位填全允许,再整张交库函数——花 CPU 与内存换调用界面极简。
⑩§32.9-32.11 V2 路径(opt-in,书里定位为**演进方向而非主线**):StructuredOutputsWorker 预分配缓冲 + 独立 copy_stream 两次异步 H2D、两次方向相反的 wait_stream;紧凑行按 cu_num_logits 展开到 logits 行(启动前 assert num_masks==len(mapping));_apply_grammar_bitmask_kernel 位解包成位矩阵 + 一次带谓词 tl.store(-inf),谓词兼任词表尾越界保护。
⑪§32.12 为什么必须 -inf:写 -inf 使非法 token 概率精确为 0,与温度/top-k/top-p 任意组合正交;数值表证 C=-20 时非法总概率 0.999929 且叠加 top_k=4 后约束彻底失效而不报错,C=-10000 「碰巧」有效但把合法 logit 压到 -10001 即回到 0.996143——**有效性依赖数据不依赖设计**。
产出:1144 行正文,20 段 dossier 引文 + 正文引文行号区间全绿;精简版 15 文件(structured_outputs/buffer_utils/spec_decode_utils/utils/structured_output_manager/output/input_batch/async_scheduler/scheduler/engine_core/model_runner 等)38 测试全过、零 skip——**本机有真 GPU(RTX PRO 6000 Blackwell, CUDA 13.1, torch 2.11.0+cu130, triton 3.6.0),GPU 标记测试真的在设备上跑了真实 @triton.jit kernel**(全书罕见);8 张机制图 + chapter-map 共 9 图 blind_review 全 PASS(round 1 干净);全 10 项 linter green。
**dossier-verify 逃生舱经过**:自核确认 20 段引文逐字 + 行号全部准确(ch31 行号教训未复发),但抓出 9 项实质问题,**最关键一项改变了全章路线**——Lead 发车 focus 把 V2 kernel 当默认行为讲错了,订正为「默认路径为主线、V2 为演进方向」;另修 delete 误删 step_with_batch_queue 致 must_keep 落空、data_flow 缺延后采样因果链、max_rollback 跨后端错误概括(LMFE 直接 raise 拒绝投机 / guidance 无此参数用 rollback_lag)、m13 与 ch31 重复降 cross_ref、3 处缺锚、2 处锚起于注释或空行、图预算 9→8。
run-ledger:impl_test_rounds=1、write_review_rounds=1、blind_rounds=1、map_rounds=1、escalated=dossier-verify。bible.py due ch32 空(无应埋/应回收伏笔;全书 26 条伏笔早已 100% resolved),与上篇 ch31 的回滚上界/validate_tokens 语义只回指不重讲。
**诚实边界**:kernel 耗时/占比未实测;18.6 KiB/行、4.6 MiB/批、grid 第二维 19 等均按源码常量(|V|=152064、BLOCK_SIZE=8192)推算并已注明;并行线程上界 max(1,min(cpu_count//2,8)) 依部署机 CPU 数。
**工具验证**:本章是 citation_range 检查(exp-2026-07-20-01)首次在新章上生效的验证,全绿。

## Why it matters

ch31 讲「语法怎么变成能逐 token 判合法的对象」,ch32 是它唯一的兑现处:没有这一章,约束解码只是一个编译好却从未落到 logits 上的对象。本章把三条最容易讲错、且错了不报错的事实钉死为可引用基座:(1)**行序不变式是数据不是约定**——掩码序与 batch 序被允许不同,按行号硬对齐会静默互换两个请求的约束;(2)**缓冲跨步复用**使「不受约束」必须显式写整行 -1,留空即读到上一步残留位;(3)**-inf 不是「一个很小的负数」的近似**,它是「约束不改采样」这句话的物理兑现——用有限常数时约束会在特定数据下彻底失效且不报错。
更重要的是**路线诚实性**:本章最花哨的高潮(自写 Triton kernel 上的一次位运算)在 pin 上是 opt-in 且默认关闭。若按发车 focus 的原始表述写下去,整章会给读者一个「vLLM 默认用自写 kernel 应用语法掩码」的错误世界模型——这类错误比行号错严重得多,因为它污染的是读者的部署直觉。dossier 对抗性自核在写作前拦下,Lead 订正路线为「默认路径为主线、V2 为演进方向」,评审终审逐条确认该前提在导言/§32.7/§32.8/§32.11 末尾反复出现(env 开关点名 8 处、「默认」28 处)。这为后续所有涉及 opt-in 新路径的章节立了一条可复用的写作规范。

## What to remember

ch32-structured-output(deep, Part VI 下篇):约束解码后半场=一张紧凑掩码怎么落到 logits。六处命门——(1)门控同一谓词写两遍(scheduler.py:L942-951 置位 / L1224-1246 收集),掩码行数=本步产出 logits 的结构化请求数是构造性成立;(2)**行序不变式是「第 k 行属于 ids[k]」,不是「掩码序=batch 序」**,GrammarOutput 把 req_id 列表随掩码同传,worker 用字典+cu_num_logits 重建映射,错位不报错只生成坏 JSON;(3)缓冲按 max_num_seqs×(1+num_spec) 行跨步复用,「不受约束」必须显式写整行 -1;(4)并行填充三道门,max_num_seqs ≤ 128 时是结构性死代码,第三道门排除投机是正确性要求;(5)投机 1+k 行=逐位置 accept 推进 + 末尾一次 rollback 净位移归零,num_invalid_spec_tokens 从接受率分母扣;(6)两道推理门语义不同(should_fill_bitmask 填不填 / should_advance 推不推进),思考结束那一步只置标志不推进。
**最关键的路线事实(后续章勿讲反)**:VLLM_USE_V2_MODEL_RUNNER 默认 False → 默认部署是 GPUModelRunnerV1 → structured_output/utils.py:apply_grammar_bitmask → **xgrammar 库函数 xgr.apply_token_bitmask_inplace**;vLLM 自写的 _apply_grammar_bitmask_kernel(gpu/structured_outputs.py:L85-115)**只在显式开 V2 时才走到**,书里定位为演进方向。
-inf 的论证(§32.12)是「约束不改采样」的最终兑现:有限常数 C=-20 时非法总概率 0.999929、叠 top_k=4 后约束彻底失效不报错;C=-10000 碰巧有效但合法 logit=-10001 时回到 0.996143——有效性依赖数据不依赖设计。
38 测试全过零 skip(本机真 GPU,Triton kernel 真跑)、9 图 blind PASS、全 linter green、APPROVED。经验候选:opt-in/默认关闭的新路径若被当作章节高潮,必须在高潮之前把 env 门与默认值交代清楚(本章由 dossier 自核抓出,可考虑固化成 dossier 契约的一项自核清单)。
