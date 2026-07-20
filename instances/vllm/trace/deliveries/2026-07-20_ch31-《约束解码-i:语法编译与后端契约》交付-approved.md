# 《约束解码 I:语法编译与后端契约》交付-APPROVED

- **Type**: delivery
- **Chapter**: 31
- **Date**: 2026-07-20
- **Timestamp**: 2026-07-20T16:54:52Z
- **Agents involved**: archivist, analyst, implementer, tester, explainer, illustrator, writer, reviewer, team-lead
- **User present**: True
- **Tags**: delivery, ch31, part-vi, structured-output, constrained-decoding, xgrammar, guidance, outlines, lm-format-enforcer, bitmask, async-compile, contract-abc, dossier-escape, env-incident

## What happened

ch31-structured-output(kind=deep, Part VI 双章上篇, source_pin v0.21.0 ad7125a4)交付并归档,verdict=APPROVED(评审初判 REVISE 的 3 处行号 blocking 已由 writer 修完并经 Lead 对 pin 复核)。九节主线:①§31.1 约束不改采样算法,只在采样前加一层位掩码;②§31.2 入口六形态(json/json_object/regex/choice/grammar/structural_tag)+ StructuredOutputsParams.__post_init__ 双向互斥校验 + get_structured_output_key 归一成 (枚举, 字符串);③§31.3 后端在进引擎之前(前端校验期)就选好,auto 是两级降级阶梯(xgrammar→outlines/guidance,lm-format-enforcer 从不在这条路上),全引擎单后端;④§31.2 尾+§31.6 **校验期原地改写请求**——validate_xgrammar_grammar 用 choice_as_grammar 把 choice 转成 EBNF 后置 so_params.choice=None、so_params.grammar=...,**故 xgrammar compile_grammar 只有五分支、无 CHOICE**;⑤§31.4 命门:grammar_init 把编译扔线程池,门控真正落点是 StructuredOutputRequest.grammar property(读到 Future 就 100us poll、就绪则原地替换),调度侧 _is_blocked_waiting_status / _try_promote_blocked_waiting_request 负责晋级(阻塞态枚举值 2→就绪 1),external_launcher 退回同步编译;⑥§31.5 六方法请求级契约(accept_tokens/validate_tokens/rollback/fill_bitmask/is_terminated/reset)+ 三方法引擎级契约(compile_grammar/allocate_token_bitmask/destroy),accept_tokens 真实调用点在调度器 update_from_output(scheduler.py:L1359-L1372);⑦§31.7 四后端同契约四种活法,is_terminated 四种算法;⑧**编译复用各后端各自为政**——structured_output_key **不是**跨请求缓存键、vLLM 不做同 schema 去重(xgrammar 库内 GrammarCompiler cache_enabled / outlines 自建 / guidance 无 / LMFE 仅 tokenizer);⑨§31.8 位掩码一行 18.3 KB(18752 B) vs logits 585.9 KB(600000 B),恰好 1/32,便宜到能进热路径;§31.9 出口交棒 get_grammar_bitmask → 下一章 ch32。产出:精简版 9 文件(backend_types/backend_xgrammar/backend_guidance/so_request/structured_output_manager/request/sampling_params/scheduler/utils)73 测试全过(host 纯 Python,第三方库 try/except 顶层导入 + 测试内 Fake 替身 monkeypatch);8 张机制图 + chapter-map 共 9 图,blind_review 全 PASS;全 linter green(含 lint_chapter_map --require);21 机制勾选表零缺口。**诚实边界**:host 无 CUDA/无 vLLM 安装,真机取证须进容器(scripts/vllm_docker.sh);xgrammar 两行数字为替身复刻、guidance 两行为精简版真实计数,正文已就地披露。**dossier-verify 逃生舱经过**:对抗性自核抓出五处命门错误并连带订正 Lead 发车 focus——xgrammar 无 CHOICE 分支(真相是校验期就地改写,补机制 m20)、delete 清单误删 validate_xgrammar_grammar(唯一改写点)、位掩码 4.7KB→18.3KB、is_grammar_ready/reset 零调用者应如实标注、structured_output_key 非编译缓存键;Lead 逐条修正后 skip_dossier 复跑,五处命门在终审复核中无一回退。**环境事件**:后台隔离守卫三次拦下子 agent(writer/illustrator/reviewer)写共享 checkout,三者均正确拒绝用 Bash 绕过(判定为守卫实现缺口而非许可);用户批准 .claude/settings.json 设 worktree.bgIsolation=none 后解除,但已缓存的 write-agent 失败结果导致 resume 秒回,故 Lead 改为直接驱动剩余工位(writer/reviewer/illustrator 独立派工,chapter-map 成品从 job tmp 落盘并 append manifest)。run-ledger:impl_test_rounds=1、write_review_rounds=1、blind_rounds=0、map_rounds=1、escalated=dossier-verify + write。bible.py due ch31 为空(无应埋/应回收伏笔)。

## Why it matters

约束解码是 vLLM 里少数横跨前端校验、引擎管理、线程池、调度器状态机与采样热路径的功能,本章把前半场(一段语法怎么变成能逐 token 判合法的对象)钉死成可引用基座:六方法/三方法契约签名(ch32 及后续投机解码章直接复用勿改名)、阻塞门的真正落点(grammar property 而非零调用的 is_grammar_ready)、位掩码 1/32 的代价账,以及三条最容易讲错的事实——xgrammar 五分支不是六分支、structured_output_key 不是跨请求缓存键、vLLM 不做同 schema 去重。这三条恰是发车 focus 的原始表述里说反的地方,由 dossier 对抗性自核在写作前拦下,避免整章建在错误前提上。

## What to remember

ch31-structured-output(deep, Part VI 上篇):约束解码前半场=语法怎么编译成逐 token 判合法的对象。五处命门——(1)校验期原地改写请求(choice→EBNF)导致 xgrammar compile_grammar 只有五分支无 CHOICE;(2)异步编译的阻塞门真正落点是 StructuredOutputRequest.grammar property,调度侧 _is_blocked_waiting_status/_try_promote_blocked_waiting_request 晋级(状态 2→1);(3)六方法请求级契约 + 三方法引擎级契约,accept_tokens 真实调用点 scheduler.py:L1359-L1372,reset/is_grammar_ready 零调用者(如实标注);(4)编译复用各后端各自为政,structured_output_key 非跨请求缓存键、vLLM 不做同 schema 去重;(5)位掩码一行 18.3KB vs logits 585.9KB=1/32。73 测试全过、9 图 blind PASS、全 linter green、APPROVED。经验候选:lint_fidelity 只验引文在文件中出现、不验 '# path:La-Lb' 区间是否真对应(本章 3 处行号 blocking 全靠人肉复核抓出)。
