# impl-notes — ch36 primer《EAGLE:特征级自回归与树验证》

本章 `kind: primer`——不是目标代码仓的减法精简版，而是**论文忠实的小型参考实现**
（PyTorch/NumPy，纯 CPU，小参数即可全量跑完，用于配合 explainer 产出可示教的数值轨迹）。
每个 `def`/`class` 用 `# PAPER: §x Eq.y` 锚定论文出处（替代普通章节的 `# SOURCE:`）；
另有少量注释里的 vLLM 行号引用（如 `llm_base_proposer.py:L664-669`）——那些不是
`# SOURCE:` 减法锚，只是说明"参考实现的哪个控制流对应 vLLM 落地的哪一段"，帮读者对照。

论文包：`arXiv:2401.15077`（EAGLE，`paper.md`）+ `arXiv:2406.16858`（EAGLE-2，
`paper-eagle2.md`），均在 `instances/vllm/book/papers/ch36-primer-eagle/`。

## 文件清单

- `feature_autoregression.py` —— EAGLE §2（token 层自回归的 T→E→F→p→t 记号）、
  §3.1/Fig.6（Autoregression Head：FC(2h→h) 融合 + 单层 decoder）、§3.1/Fig.3-5
  （"超前一步 token"的 shift-and-splice 构造）、§3.2（训练目标 Smooth-L1 + 交叉熵，
  `w_cls=0.1`）。`ToyTargetLLM` 是"目标 LLM"的极简占位（真实 backbone 内部机制不是
  EAGLE 的研究对象，只有它的 T→E→F→p→t 接口是）。
- `chain_drafting.py` —— EAGLE §3.1 drafting phase 的**链式特例**（对应 dossier
  paper_origin_note：vLLM v1 的 eagle proposer 默认路径就是这个特例，不是论文的动态树）。
  `propose_chain` 复现 vLLM `llm_base_proposer.py:L392-L592` 的"第一遍前向 + 逐步回喂"
  控制流；`greedy_sample` 对应 `_greedy_sample`（L386-L390）。
- `speculative_sampling.py` —— EAGLE §2（接受准则 `min(1,p/p̂)` + 残差分布
  `norm(max(0,p-p̂))`）+ Appendix A.2 Algorithm 1（多轮投机采样，验证树而非链所需的
  递归)。**不重新证明保分布定理**——那是 ch28 §28.5 的工作，本章链接引用。
- `draft_tree.py` —— EAGLE §3.1/Appendix A.1（静态 k 叉树）+ §3.3（树验证=多轮投机
  采样的递归应用）+ EAGLE-2 §4.1（value `V_i=∏置信度`、top-k 扩展）+ §4.2（top-m 重排、
  树注意力掩码）+ §3.2（良好校准：置信度≈接受率，动态树可行性依据）。

## 1:1 Paper Map（参考实现符号 ↔ 论文出处 ↔ 说明）

| 参考实现符号 | 论文出处 | 对应关系 / 取舍 |
|---|---|---|
| `feature_autoregression.ToyTargetLLM` | arXiv:2401.15077 §2（T→E→F→p→t 记号） | 极简占位"目标 LLM"：真实 embedding+GRU backbone+LM Head，backbone 内部结构不是论文研究对象，只有 forward_prefix/next_token_distribution 这两个接口的语义（特征序列、softmax 出 token）要忠实 |
| `feature_autoregression.AutoregressionHead` | arXiv:2401.15077 §3.1 Fig.6 | fc(2h→h) 融合 [token_embed⊕feature] + 一层 decoder-like 变换（Linear+SiLU+残差），逐字对应论文"FC 层降维 + decoder 层"描述；不含张量并行/量化/权重命名重映射（那些是 `vllm/model_executor/models/llama_eagle.py` 的工程细节，非算法本身） |
| `feature_autoregression.build_shifted_token_input` | arXiv:2401.15077 §1/§3.1 Fig.3-5 | 单条请求的"左移一位+末位塞采样 token"；vLLM `set_inputs_first_pass`（`llm_base_proposer.py:L664-669`）的批量/多请求边界处理是工程细节，不属于论文机制，未纳入 |
| `feature_autoregression.regression_loss` / `classification_loss` / `combined_loss` | arXiv:2401.15077 §3.2（`L_reg`/`L_cls`/`L=L_reg+w_cls·L_cls`） | 训练期损失，vLLM 推理仓不含训练代码；本章按 dossier 要求作"原理背景 + 数值推演"，不在推理调用面里使用 |
| `chain_drafting.greedy_sample` | arXiv:2401.15077 §3.1（`p_4=LM_Head(f_3)`, `t_4~p_4`，取 argmax） | 对应 vLLM `_greedy_sample`（`llm_base_proposer.py:L386-L390`）；额外返回 confidence（`c_j`）供 EAGLE-2 的 value 计算复用 |
| `chain_drafting.propose_chain` | arXiv:2401.15077 §3.1 drafting phase，vLLM 链式特例 | 复现 `propose()`（`llm_base_proposer.py:L392-L592`）控制流：第一遍对 shifted-token+prefix-feature 融合出第 1 个草稿 token，随后 `num_speculative_tokens-1` 次把"上一步 token 的 embedding + 上一步产出的 feature"回喂；省略 position/slot_mapping/cudagraph 缓冲等工程细节 |
| `speculative_sampling.accept_reject` / `residual_distribution` | arXiv:2401.15077 §2 | 接受准则与残差分布的逐字实现；证明见 ch28 §28.5，本章不重复 |
| `speculative_sampling.multi_round_speculative_sampling` | arXiv:2401.15077 Appendix A.2 Algorithm 1 | 递归：逐个候选 token 用当前（可能已被前面拒绝调整过的）分布试接受，全部拒绝则从最终残差分布重采；`draft_tree.verify_tree` 在每个树节点调用它 |
| `draft_tree.TreeNode` / `build_root` / `compute_value` | arXiv:2406.16858 §4.1 Eq.(`V_i=∏p_j≈∏c_j`) | value = 从根到节点路径上置信度之积；根节点置信度固定 1.0（已被接受的起点） |
| `draft_tree.expand_node` / `build_static_tree` | arXiv:2401.15077 §3.1/Appendix A.1 | 每节点展开 top-`branching_k` 个候选（按草稿模型自己的 softmax 概率排序）；`build_static_tree` 是 EAGLE-1 的静态满 k 叉树（每层每节点都展开） |
| `draft_tree.expansion_phase` | arXiv:2406.16858 §4.1 Expansion Phase | 只展开当前层 value 最高的 top-k 节点，对照 `build_static_tree` 的"全展开" |
| `draft_tree.reranking_phase` / `_assert_connected` | arXiv:2406.16858 §4.2 Reranking Phase | 全树按 `(-value, depth)` 排序取 top-m，"值相同优先浅层"保证连通性（父节点 value ≥ 子节点 value，浅层优先 tiebreak 确保被选中的节点其父节点必也被选中） |
| `draft_tree.build_tree_attention_mask` | arXiv:2406.16858 §4.2（树注意力掩码：每 token 只看祖先） | mask[i,j]=True 当且仅当 j 是 i 的祖先或 i 本身 |
| `draft_tree.verify_tree` | arXiv:2401.15077 §3.3（树验证=SpecInfer 式递归多轮投机采样） | 深度优先：每个节点把子节点当作候选集合，调用 `multi_round_speculative_sampling`；接受路径匹配某个草稿子节点则继续下探，否则停在该处（重采样/bonus token，无子树可下探） |
| `draft_tree.calibration_curve` | arXiv:2406.16858 §3.2（良好校准：confidence≈acceptance rate，Fig.6） | 分桶统计 (confidence, accepted) 对，产出与论文 Fig.6 同风格的校准曲线数据，支撑"动态树可用置信度近似接受率"这一设计依据 |

## 取舍说明（不发明论文没有的机制）

- **不实现真实 LLaMA backbone / 张量并行 / 量化 / cudagraph**：`ToyTargetLLM` 用
  `nn.Embedding` + 单个 `GRUCell` 占位"目标 LLM"，只保证 T→E→F→p→t 接口语义忠实；
  论文和 vLLM 落地都不关心"目标 LLM 内部具体怎么算特征"，只关心草稿模型怎么用这个
  特征——这正是 dossier code_spine 圈定的边界。
- **vLLM v1 默认路径是链式，不是树**：`chain_drafting.py` 只实现链式特例（dossier
  paper_origin_note 明确指出 `dummy_run` 里仍有 `FIXME: when using tree-based
  specdec` 注释，树式投机解码尚未在这条路径落地）；`draft_tree.py` 实现的是论文原理
  （EAGLE-1 静态树 + EAGLE-2 动态树），两者在参考实现里是两个独立模块，不混为一谈。
- **不重新证明投机采样保分布定理**：`speculative_sampling.py` 只实现 §2 的接受准则/
  残差分布和 Appendix A.2 的递归控制流，完整的分布保持证明（含 worked example + Monte
  Carlo 验证）留给 ch28 §28.5，本章 narrative 直接链接引用。
- **草稿头训练（`regression_loss`/`classification_loss`）不接入任何 optimizer /
  训练循环**：§3.2 的训练目标是纯背景知识（vLLM 推理仓不含训练代码），只用来支持
  dossier 要求的"数值推演"（手算一步 loss），不构成可运行的训练流程。
- **`_one_hot_like`（draft_tree.py）不是论文机制**：树节点只存了单个 top-1 置信度标量
  （`confidence`），不存完整 vocab 分布；`verify_tree` 需要给 `multi_round_speculative_
  sampling` 一个完整分布形状的 `p_hat`，`_one_hot_like` 只是把这一个标量摊成一个合法
  分布（该 token 处放 confidence，其余均匀）以便调用签名对得上——这是参考实现的工程
  胶水，不是论文/vLLM 的算法内容，已在函数 docstring 里明确标注。

## 测试

`tests/test_feature_autoregression.py`（10 例）、`tests/test_chain_drafting.py`（7 例）、
`tests/test_speculative_sampling.py`（11 例）、`tests/test_draft_tree.py`（16 例）——TDD：
先按 dossier/论文记录的行为写断言，再核对/微调实现。`conftest.py` 把 `implementation/`
加入 `sys.path`，host 直接跑（纯 CPU torch/numpy，无需进容器）：

```
cd instances/vllm/artifacts/ch36-primer-eagle
python3 -m pytest tests/ -q
```

44 passed（无 xfail/skip）。
