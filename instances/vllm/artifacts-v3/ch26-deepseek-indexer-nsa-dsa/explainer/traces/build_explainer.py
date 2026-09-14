# ch26 explainer.json 组装器 —— 表格 rows 直接取自 m0X.json 的 table_rows_echo
#（由各 run_*.py 运行产出），保证 lint_explainer 的数字溯源逐字成立。
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


m01, m02, m03, m04 = load("m01.json"), load("m02.json"), load(
    "m03.json"), load("m04.json")
m05, m06, m09, m10 = load("m05.json"), load("m06.json"), load(
    "m09.json"), load("m10.json")
m11, m12, m13 = load("m11.json"), load("m12.json"), load("m13.json")

L0 = "L0 模型层 indexer 框（gpu_column；l2-specs/ch26.json l0_region）"

doc = {
  "chapter_id": "ch26-deepseek-indexer-nsa-dsa",
  "chapter_title": "DeepSeek 索引器 NSA→DSA",
  "pin": "vLLM v0.27.1 (6e448d0ea)",
  "kind": "code",
  "trace_environment":
    "11 个 trace 全部 host 运行（trace_source=run），驱动脚本与原始输出在 "
    "explainer/traces/（run_m01/02/03/04/05/06/09/10/11/12/13.py + 同名 json + "
    "trace_common.py 公共装配件，复用 tests/ 的真实注入件与 builder 真路径）。"
    "host：Windows CPython 3.11、torch CPU float32、无 CUDA/vLLM 安装——精简版"
    "（implementation/，测试 36 passed）+ HOST SEAM 镜像（impl-notes §Seam B1："
    "DeepGEMM 打分核族 / top_k_per_row 族 / FlashMLA 稀疏核 / 融合量化核的"
    "精确数学镜像，等价性由 36 例测试数值闭环）。取证环境与 pin 的差异"
    "（writer 须就近挑明，exp-0718-1）：① trace 数值是『kernel 数学』的 host "
    "实证、非 GPU kernel 位级产物（CUDA 核 clean_logits 语义以 0 承载『窗外语"
    "义不参与选择』）；② 全部数值档 float32（生产 bf16 部分）——V4 压缩核在 "
    "amax 前做 bf16 round-trip（quant_input 位），镜像已同型；③ m06 三核分派："
    "host is_cuda=False → 实跑 per_row 兜底核；cooperative/persistent 的选择"
    "条件按 sm90 布尔求值列账（dispatch_conditions_sm90）——生产 sm90 上 "
    "topk=2048 且 rows≤32 会真选 cooperative；④ MINI 心算档把头数/上下文缩到 "
    "心算规模（H=2 vs 实尺 64、L≤70 vs 实尺 163840）但保 head_dim=128 / "
    "quant_block=128 / 132B 布局实尺几何；DSV3.2 实尺只做真实例化的形状/字节"
    "账（m01 dsv32_shapes、m09），不跑前向；⑤ V4 具体 compress_ratios 逐层 "
    "Pattern 随 checkpoint 发布（vLLM 无默认值）——m10 用 [1,4,128] 三实例"
    "演示分型机制，不杜撰 61 层排布。",
  "figure_policy":
    "本章 L0 缩放位置 = 模型层 indexer 框（pedagogy-plan l0_zoom；L0 区域 "
    "gpu_column，见 cartography/l2-specs/ch26.json）。全部 figure-spec 均可回答"
    "「它是 L0 哪一块的放大」：m01 = ② 装配·小头+IndexCache（站 3）；m02 = "
    "IndexCache 池 + 站 4；m03 = topk_indices_buffer + 站 1/6/11/12 的写读协议；"
    "m05 = ⑤ 打分+选块 prefill 臂（站 9）；m10 = V4 三类层 compress_ratios 表"
    "（站 13 装配面）；m11 = V4 压缩索引 K^IComp（站 13）；m13 = ⑥ 消费·一核"
    "双源（站 14）。开篇图走 L2 章图（cartography/l2-specs/ch26.json，gen_L2 "
    "渲染），本文件只产正文机制图 spec——按 FIGURE-SYSTEM.md §3：架构性内容"
    "回指 L0/L1/L2、不另立架构画法；配色走 l0_common 角色色（GPU 侧恒绿、KV "
    "恒青），图面不出现内部路径。",
  "mechanisms": []
}

M = doc["mechanisms"]

# ═══════════════ m01 DSA 打分器（worked example + figure） ═══════════════
M.append({
  "mechanism_id": "ch26-m01",
  "intuition":
    "考古队进墓先派一支便宜的调查犬队：64 只小狗（每只鼻子 128 维）逐个闻过"
    "所有出土陶片，各报一个气味强度；ReLU 把『没闻到』归零，队长按每只狗的"
    "可靠度加权（权重可以是负的——不可靠的狗反而扣分）汇总成每片一个分数。"
    "只有总分最高的 2048 片才请专家（主 MLA 128 头×576 维）上手细看。犬队"
    "自己不挖墓（无反向）、吃便宜粮（FP8）、头数还只有专家队一半——这就是"
    "『打分便宜到能扫全部历史』。",
  "worked_example": {
    "params": {
      "档位": "手算档 H=2/D=4 + 真路径档 H=2/D=128/T=5 + DSV3.2 实尺形状账",
      "手算": "q_h1=[1,0,2,-1], q_h2=[0,3,1,1], 4 个 k, w=[1.5,-0.5]",
      "真路径": "make_small_hf_config（index_n_heads=2、head_dim=128、"
               "indexer 专属 interleave RoPE）",
      "实尺": "DSV3.2 config（index_n_heads=64、hidden 7168、q_lora 1536、"
              "index_topk 2048、max_model_len 163840）"},
    "trace_source": "run",
    "trace_ref": "traces/m01.json",
    "table": {
      "columns": ["阶段", "动作", "实测数值", "判定/说明"],
      "rows": m01["table_rows_echo"]}
  },
  "invariant": {
    "claim":
      "scale 折叠不改变打分值：把 q 除以 ue8m0 scale（2 的幂）存成 fp8、同时把 "
      "该 scale 连同 softmax_scale、n_head_scale 乘进 weights，I_{t,s} 的数学值"
      "不变——这正是 deepseek_v2.py:L817『标量归一化搬出核』合法性的根据。",
    "argument":
      "设 s>0（ue8m0 scale 是 2 的幂、恒正）。ReLU(s·x)=s·ReLU(x) 对 s>0 成立"
      "（正部线性、负部仍为 0），故 Σ_j (w_j·s)·ReLU((q_j/s)·k) = "
      "Σ_j w_j·ReLU(q_j·k)——逐项相等，不求和也相等。softmax_scale 与 "
      "n_head_scale 是额外正标量、同样可提进 w_j。trace 实证：折叠后 weights "
      "与独立参考 w·q_scale·softmax_scale·n_head_scale 最大偏差 0.000000，"
      "q_fp8 与参考量化逐位相等=true。"
  },
  "quantified":
    "手算例 4 个 key×2 头=8 个点积 3 步出 top-2；真路径一拍 T=5：q_fp8 "
    "[5,2,128] float8、weights 折进 3 个标量。实尺（m01 dsv32_shapes）：wq_b "
    "[8192,1536]（64 头×128 维从 1536 维潜向量上投、复制不切 TP）、"
    "wk_weights_proj [192,7168] 一枪出 128 维 key + 64 个逐头权重、k_cache "
    "132B/条、workspace 6553600 条。对照打分与主 MLA 的每对 MAC 账归 m09"
    "（64×128=8192 vs 128×576=73728，11.11%）。",
  "figure_specs": [{
    "figure_id": "ch26-fig-indexer-head",
    "template": "layout",
    "l0_zoom": L0 + "装配幕 ② 小头+IndexCache（L2 站 3）的类型展开。",
    "claim":
      "indexer 是一套与主注意力完全独立的打分头：头表全来自 config.index_*"
      "（64 头×128 维，与主 128 头无关）、wq_b 从 1536 维潜向量上投且复制不切 "
      "TP、wk_weights_proj 一枪 GEMM 出 128 维 key 与 64 个逐头权重、key 经 "
      "LayerNorm(eps=1e-6) 与专属 interleave RoPE 后以 132B 量化条目进 "
      "IndexCache——主 MLA 一个头都没动用。",
    "numbers": [
      {"value": "wq_b [8192, 1536]——q_lora_rank 1536 上投 64×128，"
                "ReplicatedLinear『no tensor parallel, just replicated』",
       "provenance": "traces/m01.json dsv32_shapes.wq_b_weight_shape"},
      {"value": "wk_weights_proj [192, 7168]，output_sizes [128, 64]——一枪 "
                "GEMM 出 key + 逐头权重两片",
       "provenance": "traces/m01.json dsv32_shapes"},
      {"value": "indexer 64 头 vs 主注意力 128 头——两套头表互不相干"
                "（字面证据=字段前缀 index_*）",
       "provenance": "traces/m01.json independent_head_evidence + "
                     "deepseek_v2.py:L655-L660"},
      {"value": "k_norm eps=1e-6；softmax_scale=0.0884（128^-0.5）；"
                "n_head_scale=0.125（64^-0.5）——三标量折进 weights",
       "provenance": "traces/m01.json dsv32_shapes / forward_run"},
      {"value": "k_cache 132B/条；workspace 6553600 条（163840×40）",
       "provenance": "traces/m01.json dsv32_shapes.k_cache_head_dim 与 "
                     "max_total_seq_len"},
      {"value": "FP8 wk + BF16 weights_proj 融合加载（两段缓冲、到齐反量化"
                "bf16）——单独训练的 checkpoint 形态痕迹",
       "provenance": "traces/m01.json fp8_wk_load + deepseek_v2.py:L822-L871"}
    ],
    "caption_draft":
      "『独立小头』的解剖现场：左边主 MLA（128 头×576 维潜向量，ch25 的主角）"
      "与右边 indexer 小头（64 头×128 维）并排——头表、权重、归一化、RoPE "
      "（interleave 与主 RoPE 相反）、缓存（132B vs 576 元素）五件事全部独立。"
      "数据流三条：q_c（1536 维，ch25 站 9 的低秩瓶颈共享）经 wq_b 上投成 "
      "64×128 的 q；hidden 经 wk_weights_proj 一枪出 k 与逐头权重 w；k 过 "
      "k_norm+专属 RoPE 后量化进 IndexCache。weights 上标三个被折进来的标量"
      "（q_scale·softmax_scale·n_head_scale）——打分核里只剩点积+ReLU+加权和。",
    "illustrator_notes":
      "layout 模板左右对照：左侧主 MLA 框（GPU 绿）、右侧 indexer 小头框（GPU "
      "绿但视觉上更小的框+『64 头 FP8 无反向』徽标）；中间共享输入 q_c/hidden "
      "两条箭头。权重几何用 [out,in] 标注。IndexCache 条目画 132B 字节条（青色 "
      "KV 角色）。数字全取 numbers.provenance；不出现内部路径；配色 l0_common。"
  }]
})

# ═══════════════ m02 IndexCache（figure-only） ═══════════════
M.append({
  "mechanism_id": "ch26-m02",
  "figure_specs": [{
    "figure_id": "ch26-fig-index-cache-ledger",
    "template": "layout",
    "l0_zoom": L0 + "IndexCache 池（north 组件）+ 站 4 spec 自报的展开。",
    "claim":
      "IndexCache 每条 132 字节 = 128B fp8 值 + 4B fp32 scale（ue8m0 幂次），"
      "spec 自报 num_kv_heads=1 的 MLAAttentionSpec——『只有一根向量、无 K+V "
      "之分』，与主 KV cache（bf16 576 元素 = 1152B/token）分开分配、分开分组"
      "的第二本账：缓存的是打分原料（量化索引键），不是分数。",
    "numbers": [
      {"value": "132 = 128 + 4：前 128B fp8 值 + 尾 4B fp32 scale；量化+缓存"
                "插入一步融合（indexer_k_quant_and_cache）",
       "provenance": "traces/m02.json layout_roundtrip（回环 dequant 偏差与 "
                     "ue8m0 参考一致）"},
      {"value": "spec：MLAAttentionSpec(num_kv_heads=1, head_size=132, "
                "dtype=uint8)；后端 V3.2 块 64 / V4 块 256；stride_order 恒等"
                "=不支持跨层布局",
       "provenance": "traces/m02.json spec_self_report"},
      {"value": "workspace 魔数账：40×163840 = 6553600 条 ×132B = 865075200 B "
                "= 825 MB——(576×2//132)×5=40 对齐 flashmla_sparse 的 "
                "5×max_model_len workspace",
       "provenance": "traces/m02.json workspace_account + indexer.py:L442-L452 "
                     "注释原文"},
      {"value": "第二本账字节对比：132B vs 主 KV bf16 1152B/token/layer = "
                "11.46%；满长 163840 单层 21626880 B vs 188743680 B",
       "provenance": "traces/m02.json second_ledger_bytes"},
      {"value": "『每 token 只算一次』：历史 token 的索引键跨拍只读——decode "
                "打分直接对它 paged 读（m06）",
       "provenance": "dossier ch26-m02 note + 站 10（sparse_attn_indexer.py:"
                     "L530-L688）"}
    ],
    "caption_draft":
      "两本账的对照：主 KV cache 存 576 维潜向量（bf16 1152B/token/layer，"
      "ch25 的产物），IndexCache 存 indexer 的量化索引键（132B/token/layer）"
      "——同是『每 token 一条』却是两本独立账本：spec 不同（uint8/132B/单向量 "
      "vs bf16/576 元素）、分开分配、分开分组。条目字节条图解 128B 值+4B "
      "scale 布局；右下角 workspace 账：魔数 40 的推导 "
      "（(576×2//132)×5）让 indexer workspace 恰好塞进 flashmla_sparse 的 "
      "workspace 预算（825 MB @ 163k）。",
    "illustrator_notes":
      "layout 模板上下（或左右）两本账对照：主 KV 条（青色，宽）vs IndexCache "
      "条（青色，窄，标注 132B=128+4 字节放大图）；spec 卡片式标注 "
      "num_kv_heads=1/head_size=132/uint8/块 64(V3.2)·256(V4)。右下角 workspace "
      "算式卡。数字全取 numbers.provenance；配色 l0_common（KV 恒青）。"
  }]
})

# ═══════════════ m03 topk_indices_buffer（figure-only） ═══════════════
M.append({
  "mechanism_id": "ch26-m03",
  "figure_specs": [{
    "figure_id": "ch26-fig-buffer-protocol",
    "template": "swimlane",
    "l0_zoom": L0 + "topk_indices_buffer（north 组件）——站 1 分配、站 6 接线、"
                       "站 11 落账、站 12 消费的写读协议。",
    "claim":
      "一块 [max_num_batched_tokens, index_topk] int32 裸 buffer 是 indexer 与"
      "稀疏 MLA 的唯一接口：每层的 indexer 先于 mla_attn 被调用、纯副作用写"
      "本拍 query token 的行（返回值无人接收），稀疏后端取前 num_actual_toks "
      "行读——无所有权封装的共享让 skip 层『不写只读旧值』的跨层复用成为"
      "可能。",
    "numbers": [
      {"value": "形状 [512, 2048] int32 = 4194304 B（4 MiB）；例 8192 token 档 "
                "= 67108864 B（64 MiB）",
       "provenance": "traces/m03.json allocation"},
      {"value": "两层共用同一对象（layer0/layer1 的 topk_indices_buffer is "
                "buf = true）；非 v32 模型 buffer=None",
       "provenance": "traces/m03.json allocation"},
      {"value": "-1 哨兵两例：decode 因果自界 rowEnd=1 时行 [0, -1]（越界高分 "
                "9 不得入选）；空上下文 prefill 整行 -1（op 预清）",
       "provenance": "traces/m03.json sentinel_case_decode_bound / "
                     "sentinel_case_empty_context"},
      {"value": "行=本拍 query token：同一行拍 1 写 [7, 1]、拍 2 整块重写 "
                "[2, 5]——历史行旧值不残留",
       "provenance": "traces/m03.json row_rewrite_across_steps"},
      {"value": "接线位：if indexer and is_sparse and not skip_topk: "
                "indexer(...)——返回值无人接收（mutates_args 声明写者身份）",
       "provenance": "vllm/model_executor/layers/mla.py:L205-L206（dossier "
                     "embed_excerpts[5]）"}
    ],
    "caption_draft":
      "共享 buffer 的泳道协议：泳道=层（L0..L60 + skip 层 + 消费后端），时间轴="
      "拍。每拍每层 indexer 在 mla_attn 之前写 buffer 的本拍 token 行（-1 预清"
      "→top-k 覆写），同层稀疏 MLA 随即读走；skip 层的 indexer 泳道画虚线"
      "（不写、直接读别层写的同一行）。右侧放大一行：因果窗不足时尾部 -1 哨兵"
      "（[0,-1] 例）、跨拍整块重写（[7,1]→[2,5] 例）——『每轮只对新增 token "
      "算 index』在 buffer 上的字面形态。",
    "illustrator_notes":
      "swimlane 模板：3-4 条生命线（layer i 的 indexer / layer i 的稀疏 MLA / "
      "skip 层 / V3.2 消费后端）+ 中央 buffer 竖条（GPU 绿）。消息=水平箭头"
      "（写=实线向 buffer、读=实线从 buffer、skip 层读=虚线）。时间轴两拍，"
      "第二拍行重写高亮。数字全取 numbers.provenance；UML 时序图法（消息水平"
      "直线、禁折线）。"
  }]
})

# ═══════════════ m04 双预算切块（worked example） ═══════════════
M.append({
  "mechanism_id": "ch26-m04",
  "intuition":
    "搬家车装货：车斗挂两张限重牌——车长（workspace N：历史条目总数）和货重"
    "（logits 预算 M·N·4 字节）。司机挨家收货，下一家会使任一限超标就把车"
    "封条发车、剩下下一车；一家货自己就超重时（单请求超预算），按楼层拆"
    "（query 维 M 子切）——O(L²) 打分矩阵的峰值显存就是被这两张限重牌压住的。",
  "worked_example": {
    "params": {
      "心算档": "seq=[300,200,250], qry=[100,60,80], workspace=500, "
                "logits 预算=120000 B（30000 元素）",
      "N 约束档": "seq=[100,100,100], qry=[10,10,10], workspace=250（预算放开）",
      "M 子切档": "seq=[1000], qry=[40], workspace=100, 预算=10000 B（2500 元素）",
      "DSV3.2 实尺": "16384 query × 163840 历史，预算 512 MiB"},
    "trace_source": "run",
    "trace_ref": "traces/m04.json",
    "table": {
      "columns": ["轮次（贪心累积）", "动作", "M·N 账", "判定"],
      "rows": m04["table_rows_echo"]}
  },
  "invariant": {
    "claim":
      "切块结果是不重叠不遗漏的覆盖，且每块同时满足两条预算：N ≤ workspace 与 "
      "M·N·4 ≤ max_logits_bytes。",
    "argument":
      "覆盖性：外层 while 每轮 end 严格递增（请求要么被贪心收下、要么单请求"
      "强制收下 end+=1），有限请求必耗尽；块内再按 max_q=预算元素//chunk_n 对 "
      "[0, chunk_m) 连续切片——相邻 slice 首尾相接。预算性：贪心只在 "
      "new_n ≤ workspace 且 new_m*new_n ≤ max_logits_elems 时才收（else "
      "break）；单请求兜底后 max_q 向下取整保证 q·chunk_n ≤ 预算元素。单调量"
      "是 end：每轮至少 +1、至多 n 轮必停。"
  },
  "quantified":
    "心算档 3 请求 3 块（每块 M·N ≤ 30000）；M 子切档 40 行 → 20 片（每片 2 "
    "行）；DSV3.2 实尺：不切块的 logits 10737418240 B（10240 MiB）→ max_q=819 "
    "切成 21 片、片峰 536739840 B（511.99 MiB ≤ 512 MiB 预算）——峰值从 10240 "
    "MiB 压到 512 MiB 边界（m04.json dsv32_full_prefill）。workspace 侧 "
    "6553600 条（825 MB）由魔数 40 与 flashmla_sparse workspace 对齐（m02）。",
})

# ═══════════════ m05 prefill 打分核链（worked example + figure） ═══════════════
M.append({
  "mechanism_id": "ch26-m05",
  "intuition":
    "阅卷前的收卷：答卷分散在各考场储物柜（分页 IndexCache 的物理块）里，"
    "先把每个学生的卷子按块表收拢到一张长桌（gather 进连续 workspace），再"
    "逐题打分（fp8_fp4_mqa_logits——I 就是每份答卷的得分），最后每题只把前 "
    "k 名写上榜（top_k_per_row_prefill 按因果边界选块写 buffer 行）。块表"
    "换算让物理块 2/3 的碎片在长桌上回到学生序（请求序）连续。",
  "worked_example": {
    "params": {
      "几何": "2 请求：seq=[70,6], qry=[6,4]（req0 历史 64、req1 历史 2）、"
              "topk=6、block_size=64、物理块从 2 起（物理位≠逻辑位）",
      "打分头": "H=2、head_dim=128（实尺 64）、q FP8（scale 已折 weights）",
      "因果": "cu_seqlen_ks/ke 按请求基址+start_pos+1+offset"},
    "trace_source": "run",
    "trace_ref": "traces/m05.json",
    "table": {
      "columns": ["行", "query（请求/位置）", "因果窗 [ks, ke)",
                  "buffer 行（相对请求基址）", "top 分值（前 3）"],
      "rows": m05["table_rows_echo"]}
  },
  "invariant": {
    "claim":
      "任何行的选中索引严格落在其因果窗 [ks, ke) 内（未来 token 绝不入选），"
      "且窗不足 topk 时尾部恰为 k−candidates 个 -1 哨兵。",
    "argument":
      "选择核只读 logits[r, ks[r]:ke[r]] 切片——窗外的分数根本不进比较集"
      "（结构性排除，非过滤）；窗内是有限全序排序（降序、tie 取小 index），"
      "恰好选出 min(k, ke-ks) 个。哨兵侧：buffer 行先被 op 预清 -1，选择核"
      "只覆写前 min(k, 窗长) 个位置——尾部哨兵无人覆写、天然保留。trace "
      "实证：全部 10 行与独立暴力参考逐行相等，req1 首行窗长 3 → 尾部 3 个 "
      "-1。"
  },
  "quantified":
    "本例一拍 M=10、N=76：logits 矩阵 10×76×4=3040 B；实尺账归 m04/m09"
    "（16384×163840×4 B 不切 10240 MiB、双预算切块片峰 512 MiB）。gather "
    "侧：76 条×132B=10032 B workspace（实尺 6553600 条=825 MB 上限）。块表"
    "换算例：req0 逻辑 pos 63→slot 191（物理块 2）、pos 64→slot 192（物理"
    "块 3）——跨块边界的间接寻址（F7 在稀疏路径的再现）。",
  "figure_specs": [{
    "figure_id": "ch26-fig-prefill-scoring",
    "template": "tensor-flow",
    "l0_zoom": L0 + "⑤ 打分+选块的 prefill 臂（L2 站 9）。",
    "claim":
      "prefill 打分三段流水：cp_gather 把分页 IndexCache 按块表收成请求序"
      "连续 workspace（物理块 2/3 的碎片回到逻辑序）→ fp8_fp4_mqa_logits 对 "
      "(q_fp8, weights)×(k_quant, k_scale) 算 I_{t,:}=Σ_j w_j·ReLU(q_j·k_s)"
      "（ReLU 与逐头加权在核内，O(L²) 本尊）→ top_k_per_row_prefill 按 "
      "cu_seqlen_ks/ke 因果边界选 top-k 写 buffer 行（不足 -1 哨兵）。",
    "numbers": [
      {"value": "块表 [[2,3],[4,0]]：req0 逻辑 pos 63→slot 191、pos 64→192、"
                "pos 65→193（跨块边界）；req1 pos 0→256",
       "provenance": "traces/m05.json paged_gather_slots"},
      {"value": "因果边界：cu_seqlen_ks=[0×6, 70×4]，cu_seqlen_ke=[65,66,67,"
                "68,69,70, 73,74,75,76]——req0 首行窗 [0,65)、req1 首行窗 "
                "[70,73)",
       "provenance": "traces/m05.json chunk_metadata"},
      {"value": "workspace 76 条×132B（请求序铺放）；logits [10, 76] fp32",
       "provenance": "traces/m05.json chunk_metadata.local_total_seq_lens 与 "
                     "logits_matrix_rounded3"},
      {"value": "选块对账：全部 10 行与暴力参考相等=true；req1 首行窗长 3 → "
                "选中 [1,2,0,-1,-1,-1]（3 个 -1 哨兵）",
       "provenance": "traces/m05.json per_row_account（r6）"},
      {"value": "打分核消费位：q_scale 已折 weights 的契约（核内只剩点积+"
                "ReLU+加权和）",
       "provenance": "sparse_attn_indexer.py:L500-L507 + traces/m01.json "
                     "forward_run.fold_formula"}
    ],
    "caption_draft":
      "prefill 打分的数据流三段：左侧分页 IndexCache（物理块 2/3/4 的碎片格）"
      "经块表箭头 gather 成中间的请求序 workspace 长条（76×132B，标注跨块"
      "边界例 pos63→191/pos64→192）；中段打分核把 (q_fp8, weights) 与 "
      "(k_quant, k_scale) 变成 logits 矩阵 [10,76]（行=query token、列=历史"
      "token，逐头点积→ReLU→加权在核内）；右段 top_k_per_row_prefill 按每行"
      "的 [ks,ke) 因果窗（图上画两条窗框）选 top-6 写 buffer 行——req1 首行"
      "窗只 3 条、尾部 3 个 -1。与 m06 对照：prefill 先收卷再打分，decode "
      "不收卷直接翻柜。",
    "illustrator_notes":
      "tensor-flow 模板三段横排：分页格（青 KV 色，物理块号错开强调碎片）→ "
      "workspace 长条（连续、请求序分色）→ logits 矩阵（热力示意但数值用 "
      "top3 标注）→ buffer 行条（GPU 绿，-1 哨兵灰显）。因果窗画两个半透明"
      "框（req0/req1 各一）。数字全取 numbers.provenance；不出现内部路径。"
  }]
})

# ═══════════════ m06 decode paged 打分（worked example） ═══════════════
M.append({
  "mechanism_id": "ch26-m06",
  "intuition":
    "decode 每步只给『刚写的这一两笔』找对象：不收卷（免 gather），拿着问题"
    "按储物柜编号表（block_table）逐柜直接翻看打分——query 是本拍 token"
    "（含投机解码的 spec 窗口），key 是全历史缓存。选完 top-k 有三把剪刀"
    "（cooperative/persistent/per_row），剪出来的一模一样，只是批形不同各"
    "有所长。",
  "worked_example": {
    "params": {
      "几何": "B=2 请求、next_n=2（spec 窗口）、L=40、topk=6、block_size=16、"
              "块表 [[5,6,7],[9,10,11]]（3 块/请求）",
      "2D seq_lens": "[b,j] = L - next_n + j + 1 = [[39,40],[39,40]]",
      "三核对账": "[3,64] logits、seq_lens [64,40,7]、k=8"},
    "trace_source": "run",
    "trace_ref": "traces/m06.json",
    "table": {
      "columns": ["行", "query（请求/spec 位）", "rowEnd（因果界）",
                  "buffer 行", "top3 分值"],
      "rows": m06["table_rows_echo"]}
  },
  "invariant": {
    "claim":
      "spec 窗口的因果性由 rowEnd = seq_len − next_n + j + 1 逐行保证：spec 位 "
      "j 只看到它之前的 L−next_n+j+1 个 token，最后一个 spec 位恰好看全 L；"
      "三核分派输出逐位一致。",
    "argument":
      "j 从 0 到 next_n−1 时 rowEnd 严格 +1 递增（单调量），j=next_n−1 时 "
      "rowEnd=L——第 j 位 query 对应的第 j 个 token 之前恰有这么多历史。三核"
      "一致性：per_row 是插入排序（降序、tie 取小 index），cooperative/"
      "persistent 的 HOST SEAM 镜像实现同一序（csrc sampler.cu 同构 tie-"
      "break）——trace 对账 per_row==cooperative==persistent 逐位相等=true。"
      "host 无 CUDA 实跑 per_row 兜底核（环境差异已在 trace_environment 声明）。"
  },
  "quantified":
    "本例每拍 4 行×40 历史=160 个 (q,key) 对——只有新增 token 的行，历史 40 条"
    "全从 IndexCache 直读（对比 m05 prefill 同场景要收 76 条卷）。实尺 decode "
    "（L=131072、topk=2048）：每 token 打分扫全历史 131072 条、选 2048——"
    "indexer 自身仍 O(L²) 的 decode 形态（m09：64×128×131072=1073741824 MAC）。"
    "分派账：topk=2048、rows≤32、sm90 → cooperative；rows=33 → persistent；"
    "topk∉{512,1024,2048} → per_row 兜底。",
})

# ═══════════════ m09 复杂度诚实账（worked example） ═══════════════
M.append({
  "mechanism_id": "ch26-m09",
  "intuition":
    "机场安检的诚实账：所有人都要过金属探测门（indexer 扫全部历史——便宜："
    "探测器少一半、低精度、不用开箱），只有响门的才开箱细查（top-k 条目真算 "
    "MLA）。排队总人数一个没少（O(L²) 没消灭），但开箱（贵检查）从全部人降到 "
    "k 个人——省的是贵的那头。",
  "worked_example": {
    "params": {
      " decode 步 QK 侧 MAC 账": "L=131072（128k）、k=2048；主 MLA 128 头×576 "
                                  "维（吸收态 MQA）、indexer 64 头×128 维",
      "实尺": "L=163840（DSV3.2 max_model_len）→ 80×",
      "第二本账": "IndexCache 132B/token/layer × 61 层"},
    "trace_source": "run",
    "trace_ref": "traces/m09.json",
    "table": {
      "columns": ["账项", "构成", "量", "判定/对照"],
      "rows": m09["table_rows_echo"]}
  },
  "invariant": {
    "claim":
      "总打分量恒等式：total_sparse = indexer + sparse_main，其中 indexer 与 "
      "dense 主注意力的每对 (query, key) MAC 之比是结构常数 8192/73728 = "
      "1/9——与上下文长度无关。",
    "argument":
      "三个量都是整数乘积：dense=128·576·L、indexer=64·128·L、sparse_main="
      "128·576·k。比例 64·128/(128·576)=1/9 由头数×维数一次性约掉（与 L、k "
      "无关）；主注意力侧节省 L/k 在 k 整除 L 时是精确整数（131072=64×2048、"
      "163840=80×2048）。基例 L=k 时节省 1×（全选），归纳步：L 每增 k，dense "
      "多 128·576·k 而 sparse_main 不变——节省比严格 +1。"
  },
  "quantified":
    "128k 一步一层的 QK MAC：dense 9663676416 → 稀疏后 1073741824（indexer）+"
    "150994944（稀疏主）=1224735776，总算量降至 1/7.89；主注意力侧自身省 64×"
    "（128k）/80×（163840）。显存面：prefill 打分矩阵 16384×163840×4 B=10240 "
    "MiB 被双预算切块压到 512 MiB 片峰（m04）；IndexCache 全模型 61 层满长 "
    "1.22 GiB——主 KV bf16 同规模 10.72 GiB 的 11.46%。FP8/FP4 再给 indexer "
    "2-4× 吞吐常数因子（未经 GPU 计时，比例是结构性的）。",
})

# ═══════════════ m10 V4 三类层（figure-only） ═══════════════
M.append({
  "mechanism_id": "ch26-m10",
  "figure_specs": [{
    "figure_id": "ch26-fig-v4-layer-types",
    "template": "tiling",
    "l0_zoom": L0 + "V4 三类层 compress_ratios 逐层表（L2 south 组件，站 13 "
                       "装配面）。",
    "claim":
      "compress_ratios 逐层表把 V4 的每一层定型为三类之一：SWAonly(1) 只滑窗、"
      "C4A(4) 带 indexer 做 4 压 1 的块级 top-k、C128A(128) 压缩后 candidates"
      "(1280)≤topk(2048) 在 metadata 期直算全选——只有 C4A 建 indexer，MTP 层"
      "恒回 1。",
    "numbers": [
      {"value": "三实例 ratios [1,4,128] → indexer 建/不建 [False, True, "
                "False]；类型映射 1→swaonly、4→c4a、128→c128a",
       "provenance": "traces/m10.json layer_types"},
      {"value": "『Only C4A uses sparse attention and hence has indexer』"
                "（attention.py:L277-L278 注释原话）",
       "provenance": "vllm/models/deepseek_v4/attention.py:L276-L297（dossier "
                     "embed_excerpts[11]）"},
      {"value": "C128A 全选账：163840//128=1280 ≤ 2048 → metadata 期直算；"
                "C4A：163840//4=40960 > 2048 → 真打分；边界上下文 4×2048=8192",
       "provenance": "traces/m10.json c128a_full_select"},
      {"value": "MTP 护栏：layer_id 5 ≥ num_hidden_layers 3 → compress_ratio=1"
                "（逐字 L207-L213：MTP 层不在表内）",
       "provenance": "traces/m10.json layer_types.mtp_compress_ratio"},
      {"value": "SWA 与 C4A 压缩块共享物理张量页：SWA 块宽 64 与 C4A KV 块 "
                "[256//4, head_dim] 同页宽；三类层各有 FlashMLASchedMeta 不共享 "
                "planner",
       "provenance": "traces/m10.json layer_types.swa_note + flashmla.py:"
                     "L210-L226"}
    ],
    "caption_draft":
      "V4 的层是一条逐层定型的带子：compress_ratios 表（checkpoint 发布）把层"
      "染成三色——SWAonly（只滑窗）、C4A（带 indexer，4 token 压 1 块后 top-k）、"
      "C128A（128 压 1，压缩后 1280 ≤ 2048 直接全选、不建 indexer）。只有 "
      "C4A 格画 indexer 小图标；MTP 层钉在带尾恒为 1。三类层各有独立的 "
      "FlashMLASchedMeta（planner 不共享）——消费侧一核双源见 m13 图。",
    "illustrator_notes":
      "tiling 模板：一条层带（61 格示意但只标注类型分界，不杜撰具体排布——"
      "Pattern 随 checkpoint）；三色图例（swaonly/c4a/c128a，KV 青系深浅）；"
      "c4a 格内嵌 4→1 压缩小图标；右侧 MTP 尾格。数字全取 numbers.provenance；"
      "注记『具体逐层 Pattern 随 checkpoint 发布』。"
  }]
})

# ═══════════════ m11 V4 压缩索引（worked example + figure） ═══════════════
M.append({
  "mechanism_id": "ch26-m11",
  "intuition":
    "图书馆装订：每 4 本薄册按各册的重要性（softmax 门控）配页装订成一卷，"
    "索引卡只记卷号——且装订时还参考前一架的 4 本（overlap 窗宽 8：前半经"
    "头0、后半经头1 两套配页标准）。找书先在卷目录（压缩坐标系）里挑，不用"
    "翻每一本；k_cache 的插入也归装订车间（compressor）管，索引台"
    "（indexer_op）只管打分。",
  "worked_example": {
    "params": {
      "压缩窗": "ratio=4、overlap=True（coff=2）、窗宽 8；边界 token pos 7"
                "（(7+1)%4==0）触发",
      "窗内值": "头0 列 a_p=one-hot(维 p)（p=0..3）、头1 列 b_p=one-hot(维 "
                "p+8)（p=4..7）；门控分数 [1,2,3,4,1,1,2,2]",
      "坐标系": "builder 压缩坐标：seq_lens [16,8]→[4,2]、slot 15→3、7→17"},
    "trace_source": "run",
    "trace_ref": "traces/m11.json",
    "table": {
      "columns": ["窗位 t", "pos", "读哪头（状态切片）", "门控 score",
                  "softmax 权重"],
      "rows": m11["table_rows_echo"]}
  },
  "invariant": {
    "claim":
      "每个 token 恰属一个压缩块（块号 = pos//ratio），且只有边界 token "
      "（(pos+1)%ratio==0）触发一次压缩写；窗内 softmax 权重和恒为 1——压缩"
      "向量是窗内 kv 的凸组合。",
    "argument":
      "块划分：整数除法 pos//4 把位置集划分成互不相交的 4 元组块（余数 "
      "0..3 各归其块），边界条件 (pos+1)%4==0 在每块内恰对最大 pos 成立一次"
      "——每块恰写一次。凸组合：softmax 按窗维归一（trace 实测权重和 "
      "1.0000），故 |compressed| ≤ max|kv_win|——RMSNorm 前范数有界、量化 "
      "scale 必有限（ue8m0 幂次存在）。"
  },
  "quantified":
    "窗 8 项的门控权重 [0.0259, 0.0704, 0.1913, 0.5200, 0.0259, 0.0259, "
    "0.0704, 0.0704]——score=4 的 token 独占 52%；压缩向量（one-hot 例）"
    "RMSNorm rrms=19.878185 后 FP8 scale=0.03125（2^-5）写 slot 7（132B/条）。"
    "坐标系收益：序列长压到 1/4（[16,8]→[4,2]；实尺 163840→40960 个候选块）"
    "——top-k 在块级选（m10：40960>2048 才真打分；≤8192 全选，m12）。",
  "figure_specs": [{
    "figure_id": "ch26-fig-compressor-window",
    "template": "state-table",
    "l0_zoom": L0 + "V4 压缩索引 K^IComp（L2 south 组件，站 13 的 compressor 臂）。",
    "claim":
      "DeepseekCompressor 在 (pos+1)%4==0 的边界 token 上把 8-token 窗（前半"
      "读状态头0、后半读头1——overlap）softmax 门控压成 1 个块：Σ kv·score → "
      "RMSNorm → GPT-J RoPE（位置=压缩位 (pos//4)*4）→ FP8 132B 一步融合写进"
      "压缩坐标系的 IndexCache—— indexer 活在 //4 坐标系里（max_model_len//4、"
      "seq_lens//4、slot=(pos+1)%4==0）。",
    "numbers": [
      {"value": "窗 8 项门控权重 [0.0259, 0.0704, 0.1913, 0.5200, 0.0259, "
                "0.0259, 0.0704, 0.0704]（score [1,2,3,4,1,1,2,2] 的 softmax，"
                "和 1.0000）",
       "provenance": "traces/m11.json compress_window.window_items"},
      {"value": "头切换：t_i<4 读头0（pos 0-3）、t_i≥4 读头1（pos 4-7）——"
                "overlap 窗比 ratio 宽一倍",
       "provenance": "traces/m11.json compress_window.head_switch_note + "
                     "compressor.py:L68-L84（HOST SEAM 逐式）"},
      {"value": "RMSNorm rrms=19.878185（eps=1e-6、weight=1）；RoPE 只打末 64 维、"
                "压缩位 4=(7//4)*4；FP8 scale=0.03125 写 slot 7",
       "provenance": "traces/m11.json compress_window"},
      {"value": "压缩坐标：seq_lens [16,8]→[4,2]；slot 15→3、7→17"
                "（storage_block=16=block//4 分页）；16 token→4 块：top-k 在 "
                "[0..3] 选",
       "provenance": "traces/m11.json compressed_coordinates"},
      {"value": "skip_k_cache_insert=True：K^IComp 插入归 compressor"
                "（indexer_op 不插）；省打分不省建缓存（m12）",
       "provenance": "traces/m10.json layer_types.c4a_skip_k_cache_insert + "
                     "attention.py:L820"}
    ],
    "caption_draft":
      "压缩窗状态表：8 行窗位（pos 0-7）各标读哪个头切片、门控分数与 softmax "
      "权重（0.52 的最大权重行高亮），底部一行输出链——Σ kv·门控 → RMSNorm"
      "（rrms 19.88）→ RoPE（压缩位 4，只打末 64 维）→ FP8（scale 0.03125）"
      "写 slot 7。右侧小坐标轴：token 位置 0..15 → 压缩块 0..3（(pos+1)//4），"
      "indexer 的 top-k 从 16 个 token 位搬进 4 个块位——『活在 //4 坐标系』"
      "的字面图解。",
    "illustrator_notes":
      "state-table 模板：主表 8 行×5 列（窗位/pos/头切片/分数/权重，权重列画"
      "比例条）；表下输出链横条（compress→norm→rope→fp8 四步，末步接 132B "
      "字节条）。右缘小坐标换算轴（token→block）。数字全取 numbers.provenance；"
      "KV 青色/GPU 绿。"
  }]
})

# ═══════════════ m12 短上下文全选（worked example） ═══════════════
M.append({
  "mechanism_id": "ch26-m12",
  "intuition":
    "候选人比席位还少时不用投票：全体当选，直接把名单填成 0..n-1——但学籍"
    "（IndexCache 里的压缩键）照建，下一届（后续拍）要用。数学上『n≤k 的 "
    "top-k = 全选』与分数无关，所以打分可以整个跳过、一行 _fill 直填。",
  "worked_example": {
    "params": {
      "行账": "positions [7,15,20,31,32]、TOP_K=8、COMPRESS_RATIO=4",
      "边界": "pos 31：candidates=(31+1)//4=8 恰等于 topk（≤ 含等号）",
      "批判定": "max_seq_len//4 ≤ topk_tokens 才整批走快路径"},
    "trace_source": "run",
    "trace_ref": "traces/m12.json",
    "table": {
      "columns": ["pos", "candidates=(pos+1)//4", "选中", "哨兵"],
      "rows": m12["table_rows_echo"]}
  },
  "invariant": {
    "claim":
      "candidates ≤ topk 时全选即唯一最优：无论分数如何，n 个元素选 k≥n 个的 "
      "top-k 恒为全集——跳过打分不损失任何信息；且快路径不跳过建缓存。",
    "argument":
      "直接论证：top-k 的输出是输入多重集的确定性函数；当候选数 n≤k 时，"
      "任何降序取前 k 的过程都取尽全部 n 个（排序是全序、无元素可被跳过），"
      "与键值无关——故 fill(0..n-1) 与『真打分再选』逐位相等。边界 n=k 仍全"
      "选（≤ 含等号，trace pos31 行无 -1）。缓存侧：快路径分支先调 "
      "compressor 再 fill（attention.py:L843-L845 注释原话『we still need to "
      "build k cache』）——trace 双产物：slot 0 有量化键（scale 0.0078125）+ "
      "buffer 行 [0,-1,…]。"
  },
  "quantified":
    "pos 7/15/20/31/32 → candidates 2/4/5/8/8：选中 2/4/5/8/8 条、哨兵 "
    "6/4/3/0/0 个。批级阈值：max_seq_len 32 → 8≤8 快路径、36 → 9>8 正常打分；"
    "实尺 topk=2048 → 上下文 ≤ 4×2048=8192 的请求整批判全选（短请求 prefill "
    "与早期 decode 全走快路径——打分成本归零、建缓存照付 132B/条）。",
})

# ═══════════════ m13 V4 一核双源（figure-only） ═══════════════
M.append({
  "mechanism_id": "ch26-m13",
  "figure_specs": [{
    "figure_id": "ch26-fig-dual-source",
    "template": "flow",
    "l0_zoom": L0 + "⑥ 消费·稀疏 MLA 的 V4 臂（L2 站 14）。",
    "claim":
      "V4 decode 消费是一次 flash_mla_with_kvcache 调用吃两本 KV：k_cache=SWA "
      "滑窗缓存（indices=滑窗位）+ extra_k_cache=压缩 KV 池（extra_indices_"
      "in_kvcache=indexer 选的 top-k 位）——输出严格等于 union 上的 softmax"
      "（trace max diff 0.000000），NSA 的滑窗/压缩/选择三支路在这一次调用里"
      "结构性回归。",
    "numbers": [
      {"value": "例 pos 15、滑窗 8：SWA 位 [8..15] 共 8 条 + top-k 有效 2 条 → "
                "union 10 vs 全上下文 16；输出==union softmax（max diff "
                "0.000000）",
       "provenance": "traces/m13.json decode_dual_source"},
      {"value": "C4A 换算：buffer 行 [3,1,-1]（请求内压缩坐标）→ 全局物理 slot "
                "[323, 321]（block_table 块 5：5*64+3/5*64+1）",
       "provenance": "traces/m13.json decode_dual_source.compressed_physical_"
                     "slots"},
      {"value": "NSA 三支路映射：滑窗=k_cache+indices、压缩=extra_k_cache、"
                "选择=extra_indices_in_kvcache（indexer 选）",
       "provenance": "traces/m13.json decode_dual_source.nsa_mapping + "
                     "flashmla.py:L228-L244 调用面"},
      {"value": "prefill 并集：token0 len=2+4=6（top-k 段 [3,-1] + SWA 段 4）、"
                "token2 len=1+4=5——combine_topk_swa_indices 段拼接",
       "provenance": "traces/m13.json prefill_union + flashmla.py:L331-L363"},
      {"value": "C128A 无 indexer：metadata 期直算全选表（m10：1280≤2048）"
                "——同一消费面吃直选表",
       "provenance": "traces/m10.json c128a_full_select + flashmla.py:"
                     "L185-L187"}
    ],
    "caption_draft":
      "一核双源的合流图：中央 flash_mla_with_kvcache 核（GPU 绿），左上入口 "
      "k_cache=SWA 滑窗缓存 + indices=滑窗位（支路③），左下入口 extra_k_cache="
      "压缩 KV 池 + extra_indices_in_kvcache=top-k（支路①压缩+②选择，indexer "
      "经 C4A 换算把 [3,1] 变物理 slot [323,321]）；两路在同一次 softmax 里"
      "合算（union 10 条 vs 全上下文 16 条，输出==union softmax）。图角注 NSA "
      "谱系：三支路（arXiv:2502.11089）→ DSA 砍成一支（ch26 主线）→ V4 三支"
      "回归于这一核——谱系闭环。",
    "illustrator_notes":
      "flow 模板：两个源框（SWA 缓存=青、压缩池=青但不同深浅）各带 indices "
      "箭头汇入中央核框，出口 attn_out。C4A 换算画小框（[3,1]→[323,321]）。"
      "NSA 三支路图例角标。数字全取 numbers.provenance；不出现内部路径。"
  }]
})

out = HERE.parent / "explainer.json"
out.open("w", encoding="utf-8", newline="\n").write(
    json.dumps(doc, ensure_ascii=False, indent=1))
print("explainer.json written:", out, "| mechanisms:", len(M))
for m in M:
    we = m.get("worked_example") or {}
    print(" -", m["mechanism_id"], "| we:", bool(we),
          "| rows:", len(((we.get("table") or {}).get("rows")) or []),
          "| figs:", len(m.get("figure_specs") or []))
