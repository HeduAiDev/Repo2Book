# 论文包补充 — chunked prefill 的调度谱系（§九 落地节）

> 内部参考，正文仅引公式 + 出处。本文件是 §九「chunked prefill——连合并都不需要的拆分」
> 引用的三篇论文的**要点摘录 + 出处原样**（非全文；三篇均只以要点入文，正文不内嵌其算法）。
> 核实日期 2026-07-11：三篇 arXiv 号 / 标题 / 作者 / 核心主张均经 WebSearch 逐一核对，见下。

---

## 1. Sarathi（chunked prefill 的提出者）

- **arXiv**: arXiv:2308.16369
- **标题（原样）**: *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*
- **作者（原样）**: Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee
- **提交**: 2023-08-31
- **source_url**: https://arxiv.org/abs/2308.16369

### 要点
- LLM 推理分两相：**prefill**（处理输入 prompt）与 **decode**（自回归逐 token 生成）。
  prefill 在小 batch 就能吃满 GPU 算力；decode 一次只生成 1 token/请求，算力利用率极低。
- **chunked-prefills**：把一个 prefill 请求切成**等大小的 chunk**（"splits a prefill request
  into equal sized chunks"）。
- **decode-maximal batching**：用**单个 prefill chunk** + 其余 slot 塞满 decode 组成一批
  （"constructs a batch using a single prefill chunk and populates the remaining slots with decodes"）。
- 收益：prefill chunk 吃满算力，decode 请求**捎带**（piggyback）执行，相比 decode-only 批
  "cost up to an order of magnitude less"。

### 本章用法
§九「为什么要主动去拆：调度动机」引其 chunked-prefills + piggyback 动机；回指 ch13 §13.2
溯源节（该节已把 chunked prefill 归到 Sarathi 论文根）。

---

## 2. Sarathi-Serve（stall-free 调度 + token 预算）

- **arXiv**: arXiv:2403.02310
- **标题（原样）**: *Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*
- **发表**: OSDI'24（USENIX；arXiv v3 2024-06-17）
- **source_url**: https://arxiv.org/abs/2403.02310 ；USENIX PDF: https://www.usenix.org/system/files/osdi24-agrawal.pdf

### 要点
- **chunked-prefills**：把 prefill 请求切成**近等大小的 chunk**（"splits a prefill request into
  near equal sized chunks"），据此构造 **stall-free**（无停顿）调度：**加入新请求而不暂停在途
  decode**（"adds new requests in a batch without pausing ongoing decodes"）。
- **token budget（token 预算）**：每次调度先按用户指定 SLO 算出**一批最多可执行的 token 数**
  （"calculates the budget of maximum number of tokens that can be executed in a batch based on
  user specified SLO"）。
- **调度顺序**：每个调度迭代——**先**打包所有在途 decode，**再**纳入任一部分完成的 prefill，
  **只有**在途请求全部安置后**才**接纳新请求（"first packs all the running decodes ... then
  includes any partially completed prefill, and only after all the running requests have been
  accommodated, admits new requests"）。
- 效果：stall-free 调度让大 batch 提吞吐的同时，把 batching 对延迟的影响降到最低。

### 本章用法
§九引其 stall-free + token budget，说明"长 prefill 被自动切成刚好填进预算余量的 chunk、
永不打断在途 decode"；点明这是 ch13"token 为中心、不分相"数轴的论文根之一。

---

## 3. DeepSpeed-FastGen / Dynamic SplitFuse（平行发明）

- **arXiv**: arXiv:2401.08671
- **标题（原样）**: *DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and
  DeepSpeed-Inference*
- **作者（部分，原样）**: Connor Holmes, Masahiro Tanaka, 等（DeepSpeed 团队）
- **source_url**: https://arxiv.org/abs/2401.08671

### 要点
- **Dynamic SplitFuse**：一种"动态 prompt 与 generation 的分解-统一"策略（"a novel prompt and
  generation composition strategy" / "dynamic prompt and generation decomposition and
  unification"）——把长 prompt 拆成小块、与 generation（decode）token 融进同一批算，提升连续
  批处理与系统吞吐。
- **性能（原样）**：相对当时 SOTA（含 vLLM）"up to 2.3x higher effective throughput, 2x lower
  latency on average, and up to 3.7x lower (token-level) tail latency"。
- 系统由 DeepSpeed-MII + DeepSpeed-Inference 组合实现。

### 本章用法
§九作为**平行发明对照**：与 Sarathi 系几乎同期、独立提出"拆长 prompt 并与 decode 融批"的
同一主意；两条线殊途同归，底层同一地基（因果注意力逐行独立，拆 query 轴不损精度）。

---

## 与本章 §九 论证的关系（一句话）

三篇都在**调度层**给出"为什么要拆 prefill"的动机；本章 §九 补的是**注意力层**的"为什么拆了
零代价"——因果注意力逐行独立（第 i 行只依赖位置 ≤ i 的 KV，是绝对位置的纯函数）+ KV 写入逐
token 幂等（`reshape_and_cache_flash` 一 token 一槽）⇒ 沿 query 轴拆块后逐块输出直接拼接，
与一次性整段计算逐字节相同（explainer trace `chunked-prefill-row-independence`，实测 max|差|=0.0），
连 §六 的 LSE（⊕）合并都不需要，`flash_attn.py` 因此对 chunked prefill 零特判。
