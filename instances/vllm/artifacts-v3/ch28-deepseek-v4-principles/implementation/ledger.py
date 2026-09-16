"""两本账的算术工具（FLOPs 账 / KV 体积账）—— 本章 m01 的「自算」件。

**性质说明**：这不是论文机制的实现，而是把论文给的**相对比例**换成分母明确的绝对账。
论文出处（口径与断言）：
- arXiv:2606.19348 摘要：`DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs
  and 10% of KV cache compared with DeepSeek-V3.2`（分母 = **V3.2**，同为 1M 场景）；
- arXiv:2606.19348 §2.3.4：`Taking BF16 GQA8 with a head dimension of 128 as the baseline ...
  the KV cache size of DeepSeek-V4 series can be dramatically reduced to approximately 2%
  times of that baseline in the 1M-context setting`（分母 = **BF16 GQA8 / head_dim=128 基线**）。

**三个数字、三个分母**——本文件用参数把分母显式化，任何比例都从 `ratio_vs_*` 取，不允许
脱离分母引用数字。

字节数与层分布的来源（论文不给数字，全在代码/config 侧）：
- `584B/条` = `448B NoPE + 128B RoPE + 8B fp8 scale`（pin: vllm/v1/kv_cache_interface.py 的
  fp8_ds_mla 分支注释）；`656B/token` 是 V3.2 的对照值（同处注释）；
- `132B/条` = IndexCache 的 128B fp8 小头 + 4B fp32 scale（dossier m05/m20；pin 的 indexer 缓存）；
- 层分布 = config 的 `compress_ratios`（V4-Flash 44 项：前两项 0、中段 4/128 严格交替、末项 0
  对应 MTP 槽 —— 论文只说 `interleaved hybrid configuration`，数字全在 config）；
- 滑窗行宽 576B 来自 pin 的 `SlidingWindowMLASpec(window_size=128, alignment=576)`。

**结论数由 explainer 跑脚本复算**（本文件只给算术）：计不计 IndexCache、计不计滑窗、分母取
谁，都会改变结果——正文引用任何自算数字都必须写清口径。
"""
import numpy as np

# PAPER: arXiv:2606.19348 §八(config 口径) —— V4-Flash 的 compress_ratios（44 项）
V4_FLASH_COMPRESS_RATIOS = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)] + [0]


# PAPER: arXiv:2606.19348 §2.3.4 —— KV 体积账的分子：每条压缩条目的字节数
def compressed_entry_bytes():
    """fp8_ds_mla 的一条压缩 KV 条目 = **584B** = 448B NoPE(fp8) + 128B RoPE(bf16) + 8B fp8 scale。

    这是「混合精度存储」在 KV 账上的落点（论文 §2.3.4：RoPE 维 BF16、其余 FP8）。
    数值出自 pin 注释（vllm/v1/kv_cache_interface.py），不是论文给的数字。
    """
    return 448 + 128 + 8


# PAPER: arXiv:2606.19348 §2.3.1 —— 索引器的独立小头缓存（第二本账）
def indexer_cache_entry_bytes():
    """IndexCache 每条 = **132B** = 128B fp8 小头 + 4B fp32 scale。

    列宽是索引头维 c^I（config 口径 128），与主注意力 head_dim=512 无关；行数与主压缩 KV
    同步增长（都以 n/m 条为单位）——所以它按同样的压缩率摊薄（dossier m05/m20）。
    """
    return 128 + 4


# PAPER: arXiv:2606.19348 §2.3.4 —— 对照项：V3.2 的每 token 字节（分母之一）
def v32_bytes_per_token():
    """DeepSeek-V3.2 主 MLA 的 656B/token 自定义布局（pin 同处注释的对照值）。

    `10% KV` 这个数字的分母就是它（同为 1M 场景）。
    """
    return 656


# PAPER: arXiv:2606.19348 §2.3.4 —— 对照项：BF16 GQA8 基线（另一个分母）
def gqa8_baseline_bytes_per_token(kv_heads=8, head_dim=128, bytes_per_elem=2):
    """BF16 GQA8、head_dim=128 基线 = `2 × kv_heads × head_dim × bytes` = 4096B/token。

    论文原话把这一档叫 `one of the common configurations of LLM attention`；`~2% KV` 的
    分母就是它（不是 V3.2）。
    """
    return 2 * kv_heads * head_dim * bytes_per_elem


# PAPER: arXiv:2606.19348 §2.3 引言 —— 三类层：0/4/128 决定谁付哪笔账
def classify_layers(compress_ratios):
    """按 `compress_ratios` 分层型（pin: `compress_ratio <= 1` → swaonly、
    `== 4` → c4a、`== 128` → c128a；`max(1, ·)` 把配置里的 0 归一成 1——MTP 层不在表里）。

    返回 `{"swaonly": n0, "c4a": n1, "c128a": n2}`。
    """
    counts = {"swaonly": 0, "c4a": 0, "c128a": 0}
    for r in compress_ratios:
        r = max(1, int(r))
        if r <= 1:
            counts["swaonly"] += 1
        elif r == 4:
            counts["c4a"] += 1
        elif r == 128:
            counts["c128a"] += 1
        else:
            raise ValueError(f"unsupported compress_ratio={r} (pin 只认 1/4/128)")
    return counts


# PAPER: arXiv:2606.19348 §2.3.4 —— 每层每 token 的 KV 字节（压缩账 + 索引器账）
def cache_bytes_per_token(compress_ratio, count_indexer=True, count_swa=False, swa_row_bytes=576.0):
    """单层的「每 token 摊销字节」：

    - CSA 层（ratio=4）：`(584 + 132) / 4 = 179B`；不计 IndexCache 则 `584 / 4 = 146B`；
    - HCA 层（ratio=128）：`584 / 128 = 4.5625B`（没有索引器）；
    - ratio ≤ 1 的层：压缩账为 **0**（没有压缩机），滑窗账另计。

    注意这是**摊销**口径：压缩率在分母上（每 m 个 token 才存 1 条）。
    """
    r = max(1, int(compress_ratio))
    if r <= 1:
        return 0.0
    per_entry = compressed_entry_bytes()
    if count_indexer and r == 4:  # 只有 C4A 层建索引器（pin: compress_ratio==4 才建）
        per_entry += indexer_cache_entry_bytes()
    return per_entry / r


# PAPER: arXiv:2606.19348 §2.3.4 —— 按层分布加权出整机账，并把两个分母都算出来
def kv_byte_account(
    compress_ratios,
    count_indexer=True,
    count_swa_window=False,
    context_len=1_000_000,
    n_win=128,
    swa_row_bytes=576.0,
    gqa8_baseline=None,
    v32_baseline=None,
):
    """把逐层账按层数加权：`bytes_per_token = Σ_层 (条目字节 × 该层条目数 / 该层原 token 数) / 层数`。

    - `bytes_per_token`：压缩 KV 账（+ IndexCache，若计）的每层均值；
    - `swa_window_total_bytes`：滑窗是**常数窗**——`层数 × n_win × swa_row_bytes`，与上下文
      长度无关（这是"加账"而不是省账的算术形态）；`swa_bytes_per_token` 是它按 context_len
      摊销后的值（1M 下很薄，短上下文下相对更贵）；
    - `total_bytes_per_token` = 压缩账 + 滑窗摊销；
    - `ratio_vs_gqa8` / `ratio_vs_v32`：**两个分母都在这里**（论文的 ~2% 对 GQA8、10% 对 V3.2）。

    返回 dict（含 `per_type` 逐层型账目，便于对账）。
    """
    counts = classify_layers(compress_ratios)
    n_layers = len(compress_ratios)
    per_type = {
        "swaonly": cache_bytes_per_token(1, count_indexer=count_indexer),
        "c4a": cache_bytes_per_token(4, count_indexer=count_indexer),
        "c128a": cache_bytes_per_token(128, count_indexer=count_indexer),
    }
    total = sum(counts[k] * per_type[k] for k in counts)
    bytes_per_token = total / n_layers

    swa_total = n_layers * n_win * swa_row_bytes if count_swa_window else 0.0
    swa_per_token = swa_total / context_len if swa_total else 0.0
    total_per_token = bytes_per_token + swa_per_token

    gqa8 = gqa8_baseline if gqa8_baseline is not None else gqa8_baseline_bytes_per_token()
    v32 = v32_baseline if v32_baseline is not None else v32_bytes_per_token()
    return {
        "n_layers": n_layers,
        "counts": counts,
        "per_type": per_type,
        "bytes_per_token": bytes_per_token,
        "swa_window_total_bytes": swa_total,
        "swa_bytes_per_token": swa_per_token,
        "total_bytes_per_token": total_per_token,
        "ratio_vs_gqa8": total_per_token / gqa8,
        "ratio_vs_v32": total_per_token / v32,
    }


# PAPER: arXiv:2606.19348 §2.3.1 Eq.(17) —— FLOPs 账的载体：每 query 实际看多少条
def attention_visible_entries(n_tokens, compress_ratio, index_topk=None):
    """每 query 实际看到的条目数（FLOPs 账的"每 query 看多少条"）：

    - CSA 层（给 `index_topk`）：`min(topk, n/m)`——压缩把候选降 m 倍、稀疏再把实看数封顶在 k；
    - HCA 层（不给 topk）：`n/m'`——"压完不挑"，全看；
    - 候选不够 topk 时实看 = 候选数（短上下文全选快路径）。

    这条式子给出的就是论文 m02 图里"每档分辨率"的数：`m` 决定了候选墙、`topk` 决定了预算。
    """
    candidates = n_tokens // max(1, int(compress_ratio))
    if index_topk is None:
        return candidates
    return min(int(index_topk), candidates)
