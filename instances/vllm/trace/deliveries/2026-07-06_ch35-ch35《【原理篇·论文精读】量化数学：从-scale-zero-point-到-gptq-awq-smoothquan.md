# ch35《【原理篇·论文精读】量化数学：从 scale/zero-point 到 GPTQ/AWQ/SmoothQuant》交付-APPROVED

- **Type**: delivery
- **Chapter**: 35
- **Date**: 2026-07-06
- **Timestamp**: 2026-07-06T01:03:17Z
- **Agents involved**: archivist, writer, reviewer, illustrator
- **User present**: False
- **Tags**: primer, quantization, gptq, awq, smoothquant, fp8, paper-grounding

## What happened

ch35-primer-quantization（原理章，GPTQ arXiv:2210.17323 + AWQ arXiv:2306.00978 + SmoothQuant arXiv:2211.10438 论文精读）四段式：动机(W8A8/FP8 为何省又为何险) -> 推导(均匀量化 scale/zero-point/per-channel；GPTQ 二阶 Hessian 补偿+lazy batch+Cholesky；AWQ 激活感知缩放 s=s_X^alpha 网格搜索；SmoothQuant 迁移因子 s_j 数学等价搬迁) -> 数值推演(4 个参考实现文件 uniform_quant.py/gptq.py/awq.py/smoothquant.py 忠实复现论文算法，7 张图全部盲审 PASS，8 个 mechanism 逐条数值表与 trace 逐位一致) -> 落地(vllm quant_config/quant_method.apply 调用面 + FP8 e8m0 装载面，回指 ch22 模型定义章与 ch25 DeepSeek-V4 章)。reviewer verdict=APPROVED，12 条 issue 全 non-blocking/negotiable：3 条论文引用锚合并不完备（SmoothQuant Fig.3/Table1 跨小节合并成 §2；AWQ Eq.4-5 单标 Eq.5；OCP Microscaling §5 游离在三篇论文范围外且不可核实）；1 条 m3-gptq invariant 论证偏薄弱（终止性推理未落地，只给了 blocksize 不变性旁证）；1 条 m6 源码层为间接引用未重复行号；6 条 reader-comprehension 可读性建议（AWQ s_X 定义/Hessian 逆的投影解释/175B 模型锚点/OPT-6.7B 锚点/OCP 标准介绍/B=128 与 blocksize 前置澄清）。run-ledger：impl_test_rounds=1、write_review_rounds=3、blind_rounds=1(0 failures)、无升级。bible.py due ch35（REPO2BOOK_INSTANCE=vllm 下正确定位）为空，无应埋/应回收伏笔——早先 dossier 记录的 f15（拒绝采样保分布定理）经核实为跨实例(vllm-ascend)误路由的残留记录，已在本次交付确认与本章无关。Book Bible 登记 4 条精简参考实现接口签名（uniform_quant/gptq/awq/smoothquant 四模块）+ figures.json 新增 7 条 mechanism->figure 映射（m1~m6,m8；m7 为 supporting 协议类无需图）。

## Why it matters

巩固 primer 系列（ch34 flash-attention/ch35 quantization/ch36 EAGLE）论文精读方法论的第二个交付实例：素材先行(explainer.json+traces) + lint_paper_grounding 引用锚门禁 + 三篇奠基论文横向对照(GPTQ二阶补偿 vs AWQ激活感知缩放 vs SmoothQuant难度迁移)在同一套小矩阵数值上互相印证。为后续任何涉及权重/激活量化、FP8 装载、模型精简版量化调用面的章节提供可链接的理论基座。

## What to remember

ch35-primer-quantization：GPTQ/AWQ/SmoothQuant 三法论文精读+vllm quant_config/apply/FP8 e8m0 落地面，APPROVED，12 non-blocking issues（3 条引用锚精度、1 条 invariant 论证薄弱、6 条可读性、1 条 OCP MX 规范游离论文包外待核实、1 条源码间接引用）。4 接口+7 图注册进 Bible。bible.py due 需带 REPO2BOOK_INSTANCE=vllm 才能定位到 vllm 实例的 bible（否则误读 vllm-ascend 的伏笔表，本次已澄清 f15 误路由不影响本章）。
