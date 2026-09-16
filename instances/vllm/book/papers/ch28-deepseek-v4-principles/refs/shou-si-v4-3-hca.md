Title: 手撕 DeepSeek-V4 (3): HCA（Heavily Compressed Attention）

URL Source: https://zhuanlan.zhihu.com/p/2039632097103623588

Markdown Content:
​

目录

> 小冬瓜AIGC | X-R1[开源框架](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E5%BC%80%E6%BA%90%E6%A1%86%E6%9E%B6&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLlvIDmupDmoYbmnrYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.GnAuo_ybsJ5gzOCj1qm_rWvGXT2h6BMw_WHe1Qv8y78&zhida_source=entity)| 现高校[LLM](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=LLM&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJMTE0iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.8Hv6PCsBbtUWl5-3IYvTj1J3g3Up-FSvidaflNa02wE&zhida_source=entity)对齐研究  
> 原创课程帮助学员拿下OpenAI, Meta, [字节SEED](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E5%AD%97%E8%8A%82SEED&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLlrZfoioJTRUVEIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6Mjc1MDQ3OTkzLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.OsHbg_INzKA8ppEKceMAcZ-4TaxWm4M2aEWJYiBpWQI&zhida_source=entity)等

手撕 DeepSeek-V4 系列，含代码

[小冬瓜AIGC：手撕 DeepSeek-V4 (1) : 模型架构](https://zhuanlan.zhihu.com/p/2032043071395869284)

[小冬瓜AIGC：手撕 DeepSeek-V4 (2): 注意力方案](https://zhuanlan.zhihu.com/p/2037107284803842257)

[小冬瓜AIGC：手撕 DeepSeek-V4 (3): HCA（Heavily Compressed Attention）](https://zhuanlan.zhihu.com/p/2039632097103623588)

[小冬瓜AIGC：手撕 DeepSeek-V4 (4): CSA (Compressed Sparse Attention)](https://zhuanlan.zhihu.com/p/2046142495940145191)

[小冬瓜AIGC：手撕 DeepSeek-V4 (5): Sparse Kernel](https://zhuanlan.zhihu.com/p/2068997042303660919)

DeepSeek-V4 前瞻，含代码

[小冬瓜AIGC：【手撕NSA】DeepSeek新作-原生稀疏注意力-超长文(附代码)](https://zhuanlan.zhihu.com/p/24841366485)

[小冬瓜AIGC：【手撕 DSA】 DeepSeek-V3.2 的 Sparse Attention 比 NSA 好在哪？](https://zhuanlan.zhihu.com/p/1957032283270812718)

[小冬瓜AIGC：【手撕 mHC】详解DeepSeek残差链接mHC进化之路（超长文、附代码）](https://zhuanlan.zhihu.com/p/1990683672337223894)

[小冬瓜AIGC：【手撕Engram】DeepSeek 的 Conditional Memory 能取代 Attention 吗？](https://zhuanlan.zhihu.com/p/1994713080131772751)

* * *

## 0.前言

本系列将连载 V4 从模型架构到 [Infra](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=Infra&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJJbmZyYSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI3NTA0Nzk5MywiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.wm4r8jI948C48OZPxskL7JkLDp_f9GwdIlAumWfrI3M&zhida_source=entity) 优化的细节，并为相关技术提供 [PyTorch](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=PyTorch&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJQeVRvcmNoIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6Mjc1MDQ3OTkzLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.k0WLsh7SS0oHp-f0HnIcXpqGt7_tqP0_lnL4SK0NLCk&zhida_source=entity) 级别代码。代码仓库已开源：

本文聚焦 `V4` 中最核心的组件之一：**HCA（Heavily Compressed Attention）**。

## 1. HCA 介绍

HCA 以超高的压缩率（`V4 Pro: r=128`）将 KV 沿序列维度做块级（block-level）压缩。以 1M 上下文为例，压缩后仅有 7812 个 KV 向量，直接减少注意力计算量和 KV-Cache 量。

HCA 的设计围绕两个关键组件展开：

1.   **Compressor** — 怎么压缩
2.   **HCA 类** — 如何用压缩后的 KV 做注意力计算

HCA 也是后续 CSA（Compressed Sparse Attention）的基础。CSA 复用 Compressor；区别在于 HCA 选择全部历史压缩 KV，而 CSA 用 Indexer 做稀疏选择。

阅读本文前，引入导读思考：

*   HCA 中的 V4-MQA 是否取代了 MLA？
*   压缩 KV 中的 gated pooling 含义是什么？
*   KV 同头后为什么要 DeRotate？
*   Window KV 与 Compressed KV 信息重叠如何处理？

## 2. Compressor

### 2.1 压缩问题定义

给定长度为 L[特征维度](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E7%BB%B4%E5%BA%A6&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLnibnlvoHnu7TluqYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.sRA6Ypcroiy-NmtpjwsMp2JmUHRZb9E0JO8uFFLu-O4&zhida_source=entity)为 d 输入 X \in \mathbb{R}^{L \times d}，给定压缩率 r，将一段连续的 r 个 KV 向量压成一个向量：l = L / r。

`r` 在 HCA 中固定为 `128`。1M Context 下压缩至 7812 个向量。

![Image 1](https://pic3.zhimg.com/v2-ff48ac53b8b1f438435a6c57f0e406f4_1440w.jpg)

压缩率 r=4 时，16 个 token (t0~t15) 分成 4 个 Block，每个 Block 经 gated pooling 输出 1 个 compressed vector，shape 从 [1,16,8] 压缩至 [1,4,8]。

### 2.2 Compressor 原理

先看最简单的实现：[均值池化](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E5%9D%87%E5%80%BC%E6%B1%A0%E5%8C%96&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLlnYflgLzmsaDljJYiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.IMmZdw4JbHTOR1J0ZBrsDTtN4MQg6ojYEV8OgXI1vWQ&zhida_source=entity)。

```
# ./lc3/sec1_basic_compress.py

import torch

def compress_mean_pooling(X, ratio):
    B, L, D = X.shape
    cutoff = (L // ratio) * ratio
    X_blocks = X[:, :cutoff].reshape(B, -1, ratio, D)
    Xc = X_blocks.mean(dim=2)
    return Xc

B, L, D = 2, 16, 8; ratio = 4
X = torch.randn(B, L, D)
Xc_mean = compress_mean_pooling(X, ratio)
print(f"Input:  {list(X.shape)}")
print(f"Output: {list(Xc_mean.shape)}  ({L} -> {L//ratio})")

Input:  [2, 16, 8]
Output: [2, 4, 8]  (16 -> 4)
```

`V4` 的做法是可学习门控加权。每个 block 内，`wgate(x) + APE` 经 softmax 输出 `[ratio, head_dim]` 的[权重矩阵](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E6%9D%83%E9%87%8D%E7%9F%A9%E9%98%B5&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLmnYPph43nn6npmLUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.lk6yLySSOa7C8h5difRNmchlFBgnEtC4wr0cNCMKf0g&zhida_source=entity)，对 block 内 `r` 个位置分配不同的聚合比重。

gate 学到的本质是"这一段中哪个 token 最能代表这一段"。softmax 权重高的位置对整个 compressed vector 贡献更大。r 越大意味着更激进的语义抽象，也意味着更多细节被丢弃。

![Image 2](https://pic3.zhimg.com/v2-67ad1c9075273e67a08fd3b9cff5ce7c_1440w.jpg)

Gated Pooling 前向流程：单 block 含 4 个 token t0~t3，分两路处理。上路 w_kv 投影输出 kv [4,d]，下路 w_gate 投影加 APE 后经 softmax输出 weight [4,d]。两路元素乘后对 dim=0 求和，输出 cKV [1,d]。

```
# ./lc3/sec2_gated_compressor.py

class GatedCompressor(nn.Module):
    def __init__(self, dim, head_dim, ratio):
        super().__init__()
        self.ratio = ratio
        self.wkv = nn.Linear(dim, head_dim, bias=False)
        self.wgate = nn.Linear(dim, head_dim, bias=False)
        self.ape = nn.Parameter(torch.empty(ratio, head_dim))

    def forward(self, x):
        B, L, D = x.shape
        n_blocks = L // self.ratio
        cutoff = n_blocks * self.ratio
        kv = self.wkv(x)
        score = self.wgate(x)
        kv = kv[:, :cutoff].reshape(B, n_blocks, self.ratio, -1)
        score = score[:, :cutoff].reshape(B, n_blocks, self.ratio, -1)
        score = score + self.ape
        weight = score.softmax(dim=2)
        kv_c = (kv * weight).sum(dim=2)
        return kv_c, weight
```

输出：

```
wkv(x):   [2, 16, 16]
  wgate(x): [2, 16, 16]
  kv blocks:     [2, 4, 4, 16]
  weight: [2, 4, 4, 16]
  compressed KV: [2, 4, 16]  (16 -> 4)
```

### 2.3 触发式压缩

Prefill 和 Decode 两阶段的压缩模式不同。

**Prefill**：输入 L 个 token。切出 \lfloor L/r \rfloor 个完整 block 做压缩，剩余 `remainder` 个 token 缓存在 `kv_state`。

**Decode**：每次 1 个 token。累积到 `kv_state`，满 r 个时触发压缩，输出 1 个 compressed vector 并重置状态。

![Image 3](https://pic2.zhimg.com/v2-782ad665b0cf947466145e23f9adbebf_1440w.jpg)

Prefill 时 seqlen=10 产生 2 个完整 Block (t0~t7) 压缩为 cKV 0/1，余数 t8~t9 写入 kv_state。Decode 时逐 token 累积：pos=1 时缓存 2/4，pos=2 时 3/4，pos=3 时满 4 触发压缩输出 cKV 0。触发条件 (pos+1) % r == 0。

代码实现为

```
# ./lc3/sec3_trigger_compress.py

def prefill_compression_info(seqlen, ratio):
    n_blocks = seqlen // ratio
    remainder = seqlen % ratio
    print(f"  seqlen={seqlen}: {n_blocks} full blocks, {remainder} cached")
    return seqlen >= ratio, n_blocks

def decode_compression_info(start_pos, ratio):
    should = (start_pos + 1) % ratio == 0
    fill = start_pos % ratio + 1
    print(f"  pos={start_pos}: cache {fill}/{ratio}", "[TRIGGER]" if should else "")
    return should
```

输出

```
--- Prefill ---
  seqlen=7: 1 full blocks, 3 cached
  seqlen=8: 2 full blocks, 0 cached
--- Decode ---
  pos=3: cache 4/4 [TRIGGER]
  pos=4: cache 1/4
  pos=7: cache 4/4 [TRIGGER]
```

### 2.4 Compressor 完整实现

Compressor 维护两组缓存：`kv_state/score_state` 用于 Decode 阶段累计未满 ratio 的临时数据，`kv_cache` 存储已压缩的 KV。

```
# ./lc3/sec4_full_compressor.py — __init__ 节选

class Compressor(nn.Module):
    def __init__(self, dim, head_dim, rope_head_dim, ratio, max_batch_size=2):
        super().__init__()
        self.ratio = ratio # 压缩率 (HCA: 128, CSA: 4)
        self.wkv = nn.Linear(dim, head_dim) # KV 下投影
        self.wgate = nn.Linear(dim, head_dim) # 门控投影
        self.ape = nn.Parameter(torch.empty(ratio, head_dim)) # 块内位置编码
        self.norm = RMSNorm(head_dim)

        # Decode 阶段: 累积未满 ratio 的临时数据 (环形缓冲)
        self.kv_state = torch.zeros(max_batch_size, ratio, head_dim)
        self.score_state = torch.full((max_batch_size, ratio, head_dim), float("-inf"))

        # 压缩后 KV 存储 (由外部设置)
        self.kv_cache = None
        self.freqs_cis = None
```

`init` 定义参数；`forward` 统一 Prefill 和 Decode 的压缩逻辑。如下图所示。

![Image 4](https://pic3.zhimg.com/v2-cde7171821b33bf40530d20c3c119f2e_1440w.jpg)

Compressor.forward 在 Prefill 时 (seqlen=10)，w_kv/w_gate 投影后 cutoff=8 的前 8 个分 2 组压缩写入 kv_cache[0,1]，remainder=2 的最后 2 个写入 kv_state/score_state，之后做 RoPE 写 kv_cache。Decode 时 t10 更新状态无输出，t11 触发压缩后 softmax 加权求和输出 cKV，写入 kv_cache[2]。

```
# ./lc3/sec4_full_compressor.py — forward 节选

def forward(self, x, start_pos):
    B, L = x.shape[:2]
    kv = self.wkv(x.float())
    score = self.wgate(x.float())

    if start_pos == 0:           # Prefill
        should_compress = L >= self.ratio
        remainder = L % self.ratio; cutoff = L - remainder
        if remainder > 0:
            kv, self.kv_state[:B, :remainder] = kv.split([cutoff, remainder], dim=1)
            self.score_state[:B, :remainder] = score[:, cutoff:] + self.ape[:remainder]
            score = score[:, :cutoff]
        kv = kv.unflatten(1, (-1, self.ratio))
        score = score.unflatten(1, (-1, self.ratio)) + self.ape
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
    else:                         # Decode
        should_compress = (start_pos + 1) % self.ratio == 0
        score += self.ape[start_pos % self.ratio]
        self.kv_state[:B, start_pos % self.ratio] = kv.squeeze(1)
        self.score_state[:B, start_pos % self.ratio] = score.squeeze(1)
        if should_compress:
            w = self.score_state[:B].softmax(dim=1)
            kv = (self.kv_state[:B] * w).sum(dim=1, keepdim=True)

    if not should_compress and start_pos != 0:
        return None

    kv = self.norm(kv)
    freqs = (self.freqs_cis[:cutoff:self.ratio] if start_pos == 0
             else self.freqs_cis[start_pos + 1 - self.ratio].unsqueeze(0))
    apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs)

    # 写入压缩后 KV
    if start_pos == 0:
        self.kv_cache[:B, :L // self.ratio] = kvseqlen=7: 1 full blocks, 3 
    else:
        self.kv_cache[:B, start_pos // self.ratio] = kv.squeeze(1)
    return kv
```

输出

```
--- Compressor Demo: Decode (r=4) ---
    pos_in_block=4/4, kv_state[3] = kv_t → [TRIGGER] compressed: [2, 1, 16]
    pos_in_block=1/4, kv_state[0] = kv_t → cached (ring buffer overwrites oldest)
```

以下从四个问题展开代码细节：

1.   KV 是否分离？
2.   压缩细节？
3.   块内和块间位置编码 `pos_id` 和 `yarn_scale` 如何设定？
4.   Compressor 内部如何维护状态和存储压缩 KV-Cache？

### 2.4.1 KV Sharing

`V4` 中 K 和 V 共享同一份投影（MQA）。论文原文：

> "To reduce [KV-cache](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=KV-cache&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJLVi1jYWNoZSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI3NTA0Nzk5MywiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.bffGipeoQlq7dtsHIyxHY9cxtTWGeYsTr8GCnkz0og8&zhida_source=entity), we adopt Multi-Query Attention (MQA) where all heads share a single Key and Value head."

### 2.4.2 块内位置编码（APE）

每个 block 的 r 个位置有独立可学习编码 `APE ∈ [r × head_dim]`。作用在 `wgate(x)` 的 softmax 之前。

APE 块内位置编码热力图如下：

![Image 5](https://pic1.zhimg.com/v2-c08e16fc79fa2c5cba11eabc7da949c8_1440w.jpg)

### 2.4.3 块间位置编码（YaRN）

压缩后的 block 之间需要位置信息。该过程发生在得到压缩 kv 后，存到 kv_cache 之前。问题在于位置编码的 pos_id 是多少。

V4 压缩KV 的位置嵌入规则：

1.   位置 ID = 块的**首 token** 位置（`block_idx × r`）
2.   **YaRN [缩放因子](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E7%BC%A9%E6%94%BE%E5%9B%A0%E5%AD%90&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLnvKnmlL7lm6DlrZAiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.NX_nXL2aJ3GwroO9EZoWvifzTtvoeZVKUR3MNiEKvKA&zhida_source=entity)** = 块的**尾 token** 位置
3.   仅作用在最后 `rope_head_dim` 维（V4 Pro: 64/512）

```
# ./lc3/sec4_full_compressor.py
# Block 0: tokens=[0, 3]    pos_id=0     yarn_factor=3
# Block 1: tokens=[4, 7]    pos_id=4     yarn_factor=7
# Block 2: tokens=[8, 11]   pos_id=8     yarn_factor=11
```

选择首 token 位置的原因：block 表达一段连续 token 的"整体语义"，首位置使 RoPE 旋转角度与原始序列对齐。

![Image 6](https://pic4.zhimg.com/v2-907f71fbb85bac180296491a1151ef45_1440w.jpg)

块间位置编码：Prefill 时 16 个 token 分 4 组 (Block 0~3)，4 个 cKV 各含 content（未旋转）和 tail（RoPE 旋转）。cKV[0]~[3] 的 pos_id = 0,4,8,12（各 block 首 token 位置），R(pos_id) 作用于 tail dims，RoPE 旋转角度随 pos_id 增大而增大。

```
# Prefill: 取 block 首位置频率
freqs_cis = self.freqs_cis[:cutoff:ratio]
# Decode:  取当前 token 位置频率
freqs_cis = self.freqs_cis[start_pos + 1 - ratio]

# 嵌入位置编码: 仅对最后 rope_head_dim 维度做旋转
apply_rotary_emb(kv[..., -rope_head_dim:], freqs_cis)
```

数据形状变化：`kv [B, n_blocks, head_dim]`，取最后 `rope_head_dim=64` 维，复数旋转后写回。

Decode 时取尾 token 位置的频率。此时只有 1 个压缩 token 产生，尾位置就是该 token 在原始序列中的位置。

![Image 7](https://pic3.zhimg.com/v2-c7f4ae56917f1f477bbf8e41e31e7086_1440w.jpg)

YaRN 位置编码：Prefill 取 block 首 token 位置作为 pos_id=0,4,8,12。Decode 时 token 19 触发压缩，pos_id = start_pos + 1 - ratio = 16，取首 token t16 位置，cKV.tail 经 R(pos_id) 旋转。

总结典型的位置编码嵌入如下图所示。

![Image 8](https://pic2.zhimg.com/v2-90f272b777df91d47dd93386df612a15_1440w.jpg)

### 2.5 Compressor 维护状态

Compressor 内部维护 `kv_state/score_state`，长度为 `compress_ratio` 定量。

压缩后的 KV 写入的 `kv_cache` 并非 Compressor 独自持有，而是来自 HCA 进行管理。

![Image 9](https://pica.zhimg.com/v2-6e34282463321ced1f76fae478c2b8ac_1440w.jpg)

Compressor 与 Attention 的状态存储。Compressor 存 kv state/score state。Attention 类存 kv_cache，lazily assign 到 Compressor。

```
class Compressor(nn.Module):
    def __init__(self, ...):
        # ...
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache

        self.register_buffer(
            "kv_state",
             torch.zeros(
                 args.max_batch_size,
                 compress_ratio,
                 self.head_dim),
             persistent=False)

        self.register_buffer(
            "score_state",
            torch.full((
                args.max_batch_size,
                compress_ratio,
                self.head_dim), float("-inf"),
            persistent=False)
```

在 Prefill 时，如果长度为 10, 压缩率为 4, 以下代码中的

*   `remainder` 为 10%4 = 2。 遗留最后 2 个 token，需要 cache 起来
*   `cutoff` 为 10 - 2 = 8。前8个 token 可以压缩成 2 个 压缩KV。

```
class Compressor(nn.Module):
    def forward(self, x: torch.Tensor, start_pos: int):
        # ...
        if start_pos == 0:  # prefill stage
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            kv, self.kv_state[:bsz, :remainder] = kv.split(
                [cutoff, remainder], dim=1)
            self.score_state[:bsz, :remainder] = score[:,cutoff:] + self.ape[:remainder]
            # ...
```

该过程结束后， 所得到的压缩 kv 经过 yarn 变换再被 外部 attention 类的 `self.kv_cache` 储存。

```
class Compressor(nn.Module):
    def forward(self, x: torch.Tensor, start_pos: int):
        # ...
        # 1. get compress kv
        # 2. compress kv apply YaRN

        # 3. compress kv cache
        if not should_compress:
            return
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
```

在 Decoding 时，

*   [传参](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E4%BC%A0%E5%8F%82&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLkvKDlj4IiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.l0xpUBYQ5SvQ9YuYlOXhwNAcDYkNj45BCnc4cTEzAww&zhida_source=entity)`start_pos=10`，根据是否压缩[判别式](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E5%88%A4%E5%88%AB%E5%BC%8F&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLliKTliKvlvI8iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.mEddt2UnKP22IxPJ9BPeR5ZT6ydcF25SVhRoRCdmF9E&zhida_source=entity)`(10+1)%4!=0`，仅做 Compressor 内部状态更新。无新压缩 KV 返回。外部继续生成 next_token，此时长度为 11。
*   传参 `start_pos=11`，根据是否压缩判别式 `(11+1)%4=0`，触发压缩。

```
class Compressor(nn.Module):
    def forward(self, x: torch.Tensor, start_pos: int):
        # ...
        if start_pos == 0:  # prefill stage
            # ... do prefill stage compress
        else:  # decoding stage
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
             self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
              if should_compress:
                   kv = self.kv_state[:bsz] *
                     self.score_state[:bsz].softmax(dim=1)
                    kv = kv.sum(dim=1, keepdim=True)

        if not should_compress:
            return
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
```

### 2.6 Compressor 分析

1.   **Block 式压缩** 可能存在块间信息不平滑。Overlap 机制（拼接相邻压缩 KV）在 CSA 中使用
2.   **压缩前提**：token-level KV 表达有冗余，长上下文中大量信息可预测
3.   **Q 多头、KV 单头（MQA）**：单头 KV 是超大压缩的基础

## 3. HCA

HCA 的本质：Block-wise 压缩特征（全局）+ Window Token-Level 特征（局部），拼接后统一注意力计算。

### 3.1 HCA 类实现

HCA 自己维护一个 `kv_cache`，前 `window` 个 slot 存窗口 KV，后 `max_len/ratio` 个 slot 存压缩 KV（由 Compressor 使用）。

```
# ./lc3/sec5_hca_attention.py — HCA.__init__ 节选

class HCA(nn.Module):
    def __init__(self, dim, n_heads, head_dim, rope_head_dim, window_size, ratio):
        super().__init__()
        self.n_heads = n_heads; self.window_size = window_size; self.ratio = ratio
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)  # Q: dim → 多头
        self.wkv = nn.Linear(dim, head_dim, bias=False)           # KV: dim → 单头 (MQA)
        self.kv_norm = RMSNorm(head_dim)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)  # O 投影
        self.compressor = Compressor(dim, head_dim, rope_head_dim, ratio)

    def set_cache(self, max_seq_len, batch_size, freqs_cis):
        cache_size = self.window_size + max_seq_len // self.ratio
        self.kv_cache = torch.zeros(batch_size, cache_size, self.head_dim)
        # 将 kv_cache 的后半段交给 Compressor 管理
        self.compressor.set_kv_cache(self.kv_cache[:, self.window_size:], freqs_cis)
        self.freqs_cis = freqs_cis
```

以下分 Prefill 和 Decode 两种情况描述数据流：

![Image 10](https://pic4.zhimg.com/v2-b715d7144e3daa3da0f5b54e1591e129_1440w.jpg)

Prefill 时 token 序列过两路得到 cKV 和 win KV（通常用原始kv进行存储），逐token 有独立的 topk-ids 候选用于实际计算注意力的KV。Decoding 时仅维护 KVCache，Compressor 有条件触发压缩。

```
# ./lc3/sec5_hca_attention.py — HCA.forward 节选

def forward(self, x, start_pos):
    B, L = x.shape[:2]; win = self.window_size; ratio = self.ratio
    rd = self.rope_head_dim
    freqs_cis = self.freqs_cis[start_pos:start_pos+L]

    # 1. Q (multi-head) + KV (single-head MQA)
    q = self.wq(x).unflatten(-1, (self.n_heads, self.head_dim))
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    apply_rotary_emb(q[..., -rd:], freqs_cis)
    kv = self.wkv(x); kv = self.kv_norm(kv)
    apply_rotary_emb(kv[..., -rd:], freqs_cis)

    # 2. Assemble topk IDs: window + compressed
    topk_ids = get_window_topk_idxs(win, L, start_pos)
    offset = kv.size(1) if start_pos == 0 else win
    compress_ids = get_compress_topk_idxs(ratio, L, start_pos, offset)
    topk_ids = torch.cat([topk_ids, compress_ids], dim=-1).int()

    # 3. KV store (window circular + compressor incremental)
    if start_pos == 0:                      # Prefill
        if L <= win:
            self.kv_cache[:B, :L] = kv
        else:
            cutoff = L % win
            kv_win = kv[:, -win:]
            self.kv_cache[:B, cutoff:win], self.kv_cache[:B, :cutoff] = \
                kv_win.split([win - cutoff, cutoff], dim=1)
        kv_c = self.compressor(x, start_pos)
        kv = torch.cat([kv, kv_c], dim=1) if kv_c is not None else kv
    else:                                    # Decode
        self.kv_cache[:B, start_pos % win] = kv.squeeze(1)
        self.compressor(x, start_pos)
        kv = self.kv_cache[:B]

    # 4. Sparse attention (per-query gather)
    o = torch.zeros(B, L, self.n_heads, self.head_dim)
    for i in range(L):
        ids = topk_ids[i] if topk_ids.dim() == 2 else topk_ids[0, i]
        valid = ids >= 0
        if valid.any():
            sel = kv[0, ids[valid]]
            o[0, i] = single_query_sparse_attn(q[0, i], sel, sel)

    # 5. DeRotate + output projection
    apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
    return self.wo(o.view(B, L, -1))
```

输出

```
--- HCA Demo: Prefill ---
  kv_cache: [1, 16, 16]  (window=8, compressed=8)
  Q: [1, 16, 4, 16] → KV: [1, 16, 16]
  Window IDs: [16, 8], Compress IDs: [16, 4], Total: [16, 12]
  KV for attn: [1, 20, 16], topk_ids max: 19
  Output: [1, 16, 32]

--- HCA Demo: Decode ---
  KV for attn: [1, 16, 16], topk_ids max: 11
  Output: [1, 1, 32]
```

### 3.2 压缩 KV 的 ID 筛选

每个 query 可选的压缩 KV 受**因果掩码**限制：query `j` 只能看到 `j // r` 个历史压缩块。Prefill 时通过 `get_compress_topk_idxs` 函数生成候选索引，超过因果范围的 ID 设为 -1。

![Image 11](https://pica.zhimg.com/v2-e670a28d55808d765d6304821403164e_1440w.jpg)

因果掩码下 (L=11, r=4)，query j=0~2 全被遮断，j=3~6 仅 cid=0 可见，j=7~10 cid=0 和 cid=1 可见。规则：query j 可看到 cid 满足 end_of_block(cid) &lt; j，即 j//r 决定可选数量。

```
# ./lc3/sec5_hca_attention.py — get_compress_topk_idxs 节选

def get_compress_topk_idxs(ratio, seqlen, start_pos, offset):
    if start_pos > 0:
        n = (start_pos + 1) // ratio
        return torch.arange(n).unsqueeze(0) + offset
    else:
        n = seqlen // ratio
        matrix = torch.arange(n).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        return torch.where(mask, -1, matrix + offset)

# L=11, r=4, offset=0
idxs = get_compress_topk_idxs(4, 11, 0, 0)
print(idxs)
```

输出

```
tensor([[ 0,  1, -1, -1],   # j=0~2: j//4=0 → 全-1
        [ 0, -1, -1, -1],   # j=3~6: j//4=0 → 只能选 compressed_id=0
        [ 0,  1, -1, -1],   # j=7~10: j//4=1 → 选 compressed_id=0,1
        ...]])
```

HCA 选取**全部**历史压缩 KV。若 window kv 和 compressed kv 拼接为序列，则 compressed 候选ID `[0,1]` 跟随固定的 win size 偏移，如下 win size =8, 则有两个 compressed kv 则最终注意力则取kv id 为`[8, 9]`

![Image 12](https://picx.zhimg.com/v2-4ae88dcf3b887df5d3047d30420f2a2f_1440w.jpg)

KV-Cache 布局：窗口 8 个 slot 为 Window 区，2 个 slot 为 Compressed 区。Window KV 来自 Attention wkv，Compressed KV 来自 Compressor wkv/wgate（两套独立权重）。

### 3.3 SWA Cache 存储

### 3.3.1 环形存储原理

Window KV 做环形覆盖， 常见于 Decoding 阶段。

`self.kv_cache[:, start_pos % window_size] = kv_t`

![Image 13](https://pic3.zhimg.com/v2-233253ed721e76ea4b3d6d91e7a02ed0_1440w.jpg)

环形 Window Cache (window_size=4)：Frame 1 4 个 slot 首次填满 kv_0~kv_3。Frame 2 slot 0 被 kv_4 覆盖，当前指针回绕到 0。Frame 3 4 个 slot 为 kv_4~kv_7，当前指针为 3。

以 window=4, ratio=4 为例。Prefill 给定序列长度 10，0-7 被 cutoff，前 4 个 kv_cache 存位置 [8,9,6,7] 的 kv（8%4=0 位置）。Decode 第 11 步时（start_pos=10，从 0 开始），10%4=2，覆盖 id=2，得 [8,9,10,7]。恢复为 [7,8,9,10] 需通过以下函数获取 re-arrange id [3,0,1,2]。

### 3.3.2 PD阶段获取 win id

分析 Prefill 和 Decoding 如何获得 win idx。不分析代码实现细节，仅分析打印结果

```
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size),  torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)

# prefill stage
idx = get_window_topk_idxs(4, 1, 10, 0)
print(idx)

# decoding state
idx = get_window_topk_idxs(4, 1, 11, 10)
print(idx)
```

运行得到输出，Prefill 时每个query选的 kv id 不同，从原始kv获取window kv。 Decoding 时只有一个query，可以从 kv cache 获取kv。

```
tensor([[[ 0, -1, -1, -1],
         [ 0,  1, -1, -1],
         [ 0,  1,  2, -1],
         [ 0,  1,  2,  3],
         [ 1,  2,  3,  4],
         [ 2,  3,  4,  5],
         [ 3,  4,  5,  6],
         [ 4,  5,  6,  7],
         [ 5,  6,  7,  8],
         [ 6,  7,  8,  9]]])
tensor([[[3, 0, 1, 2]]]) # win rearange id
```

### 3.3.3 如何算 SWA

查看 forward 代码，`topk_idxs` 是所选的 kv id 的抽象，并非单纯适用于 sparse attention 的 topk_idx，用 select_idx 描述更加合适。

```
if start_pos == 0: # prefill
    # make kv[-win:] -> kv_cache
    
    kv = torch.cat([kv, kv_compress], dim=1)
    o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
else: # decoding
    # get kv_cache[ :win] and update
    
    self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
    o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
```

代码中可见 Decoding 阶段使用 Prefill 的 kv_cache 作为 kv 算 sparse attn。回到 Prefill 阶段，实际上输入的完整 kv，

1.   forward 或 prefill 时，需要驻留完整的 kv 序列，这些并不能随 query 号进行丢弃。从 forward 训练角度，需要有一定的 infra 技术来支撑，避免 HBM 塞满完整 context 的 kv 序列。
2.   Decoding 时，一个batch也没有多个 query选不同的win id ，仅由[kvcache](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=kvcache&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJrdmNhY2hlIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6Mjc1MDQ3OTkzLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.T4wQ3ugrQG39aoBeeJPyYzTnpu7gOcEdsF4qLsxAEMo&zhida_source=entity) 所维护的 window kv 就可以做swa了，非常轻量。

总结来说，[kv-cache](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=kv-cache&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJrdi1jYWNoZSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI3NTA0Nzk5MywiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.1UJhR5jpVaNU_H4iAByelbPaCy2DO9n0XHbnmu8WVv8&zhida_source=entity) 模型是确定性的，但不意味着 forward 只靠 kv-cache 就能运转。此情况将在 CSA 中进一步讨论。SWA 各 query 选不同的 win id 本身与选 top-k 的逻辑一致，差异在于 swa 选连续 kv，top-k 选随机不连续 kv。

### 3.4 DeRotate

DeRotate 因 MQA 的 KV 同源而产生。

**原始问题**：MQA 下 KV 同源（K=V）。RoPE 对 V 的 rope_head_dim 部分做逐对旋转，将位置信息编码进向量。注意力计算时，加权求和会携带这份位置编码到输出中。

![Image 14](https://pic1.zhimg.com/v2-8a901b3a9ad98a8cf8193d283c41dfa0_1440w.jpg)

RoPE 绝对位置性对比：Standard Attn 下 V 无 RoPE 输出 clean。V4-MQA 下 K=V 共享导致 V 也携带绝对位置，输出 O 因此受污染，需 DeRotate（逆旋转）消除绝对位置偏差后恢复相对位置表达。

以 2 维向量为例，RoPE 对相邻 2 维施加旋转矩阵：

R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}, \quad v_r = R(\theta) \cdot v \\

Query 和 Key 都经过同样的旋转：Q_i = R(\theta_i)\,q, K_j = R(\theta_j)\,k。注意力分数具有相对位置性：

Q_i^T K_j = q^T R(\theta_i)^T R(\theta_j) k = q^T R(\theta_i - \theta_j) k \\

但 V 的旋转会传入输出。设 V_r 的第 j 行为 R(\theta_j)\, v_j，注意力输出为：

o = \sum_j p_j \cdot R(\theta_j)\, v_j \\

p_j 是注意力权重。o 的每一行都带有不同位置的旋转，需要消除。

解法：对注意力输出施加逆旋转 R(-\theta')，抵消 V 中的旋转。

o' = R(-\theta') \cdot o = \sum_j p_j \cdot R(-\theta') R(\theta_j)\, v_j \\

当 \theta' 取 query 自身的位置 \theta_i 时，R(-\theta_i)R(\theta_i) = R(\theta_i - \theta_i)，每项变为相对旋转。由于 V 的内容本身不应依赖位置，这个相对旋转在统计上可被后续层吸收，实际实现直接取 \theta' = \theta_i 做逆旋转。

![Image 15](https://pic4.zhimg.com/v2-a512a69de4a5cb936bc4422355540c2f_1440w.jpg)

图右，对 v_rot 向量同角度逆旋转回原向量。

旋转矩阵的逆 R(-\theta) 等价于复数共轭 e^{-i\theta}。验证如下：

```
# 验证旋转矩阵逆 R(-θ) 与复数共轭 e^{-iθ} 的等价性
theta = torch.tensor(30.0 * 3.14159 / 180.0)
R = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                   [torch.sin(theta),  torch.cos(theta)]])
z = torch.polar(torch.tensor(1.0), theta)   # e^{iθ}

v = torch.tensor([1.0, 0.0])                # 任意向量
v_rot = R @ v                                # 旋转矩阵正向: v_rot = R(θ)·v
v_rot_c = torch.view_as_real(
    torch.view_as_complex(v.unsqueeze(0)) * z).flatten()

print(f"正向等价: {torch.allclose(v_rot, v_rot_c, atol=1e-6)}")

# 逆变换: R^T @ v_rot   vs   v_rot_c · e^{-iθ}
v_back_R = R.T @ v_rot
v_back_c = torch.view_as_real(
    torch.view_as_complex(v_rot_c.unsqueeze(0).float()) * z.conj()).flatten()

print(f"R^T == conj:   {torch.allclose(v_back_R, v_back_c, atol=1e-6)}")

R@v     = [0.866, 0.5]
v·e^iθ  = [0.866, 0.5]
正向等价: True

R^T@v_rot     = [1.0, 0.0]
v_rot·e^(-iθ) = [1.0, 0.0]
R^T == conj:   True
```

实现等价于复数共轭：

```
# ./lc3/sec5_hca_attention.py
apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
# inverse=True → 使用 R(θ)^T = R(-θ) 做逆旋转
```

数值验证（`./lc3/sec5_hca_attention.py` 输出）：

```
Original:     [0.707, 0.788, 0.040, -1.019]
After RoPE:   [-0.327, -0.375, 0.954, -0.600]
After DeRot:  [0.707, 0.788, 0.040, -1.019]
Match: True
```

另一种方案：存 RoPE 前后的两份 KV，用未旋转的 V 做注意力。缺点：增加一倍显存。V4 选 DeRotate。

### 3.5 MQA 分组投影

已在[系列文章注意力方案](https://zhuanlan.zhihu.com/p/2032043071395869284)中详述。HCA 与之完全一致。

### 3.6 Window token 与 Compressed token 重叠

Query 为 6 时，Window token 与 Compressed token 信息会有重叠。实现上不会过滤。信息重叠在 V4 attn 中是被接受的。 如下示例

*   去重：win id 只取 6
*   非去重：win id 取 4,5,6 （采纳）

![Image 16](https://picx.zhimg.com/v2-e64b4d357aaca11dfe07adeab04f3785_1440w.jpg)

### 3.7 混合 KV 分数归一化

window + compressed（+ sparse for CSA）拼接后统一 softmax，不分区。

```
# ./lc3/sec5_hca_attention.py
# 拼接示意（在 sparse attention 中）
K_mix = torch.cat([K_window, K_compressed], dim=0)
V_mix = torch.cat([V_window, V_compressed], dim=0)
scores = q @ K_mix.t()
probs = F.softmax(scores, dim=-1)
o = probs @ V_mix
```

![Image 17](https://pic3.zhimg.com/v2-ba214b831a6134bf6fa792dad0d837a4_1440w.jpg)

混合 KV scores 统一 softmax：8 个 window scores + 4 个 compressed scores + 1 个 attention sink，统一 softmax 归一化，总和=1。窗口和压缩不分区，统一计算注意力权重。

其中 attn_sink 是 128 头的可学习分数。

```
self.attn_sink = nn.Parameter(
    torch.empty(self.n_heads, dtype=torch.float32)
)
```

### 3.8 KV-Cache 管理

`Cache Size = window_size + max_seq_len // ratio`

HCA 的 kv_cache 前半段为窗口cache（环形覆盖更新）。HCA 的 `kv_cache` 的后半段（`self.kv_cache[:, window_size:]`）为压缩KV， 来源于 compressor 类内增量添加。以下为 HCA 的[存储模型](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E5%AD%98%E5%82%A8%E6%A8%A1%E5%9E%8B&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLlrZjlgqjmqKHlnosiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.n_L81MwHADiWu3r_gw-v8YrTGllmNSJU5m9jsNAV3nM&zhida_source=entity)。

![Image 18](https://pica.zhimg.com/v2-259ef144426de678366e6e4549869016_1440w.jpg)

KV-Cache 存储量对比：1M context 下 Full Attention 需 1,000,000 slots，HCA 仅需 7,940 slots（Window 128 + Compressed 7,812），HCA的cache 长度 仅为 标准的 attn 的 0.794%，压缩比 125.9 倍。如果考量kv共头、kv单头等因素，实际压缩率更高。

## 4. HCA 分析

### 4.1 HCA vs MLA

HCA（V4-MQA）的单头 KV 和 repeat 特性是 Memory-Efficient 的。KV-Cache 无需二次投影即可直接参与注意力计算，即所见即所得、即取即计算。

MLA 的 latent cache c 到实际运算的弊端在于 HBM 与 [SRAM](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=SRAM&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJTUkFNIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6Mjc1MDQ3OTkzLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.lKVzkIw4U00AxhZs_OZVpQi_OkyfVT8w40tzuJ9rpHs&zhida_source=entity) 的 IO 开销：需要额外加载 Wk、Wv 和 W_rope 等上投影权重。这些权重从 HBM 搬运到 SRAM 本身消耗[带宽](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E5%B8%A6%E5%AE%BD&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLluKblrr0iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.xXlzZf8q3TvXrQ9Q7goricNK1DRXBmRqaDfDcyMErdg&zhida_source=entity)，同时上投影后的多头 K、V 在 SRAM 中占用的空间也远大于 V4-MQA 的单头 KV。

这使得 MLA 更易触发 SRAM spill（SRAM 容量溢出），一旦发生，[GPU](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=GPU&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJHUFUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.yfVoHsydykXMypFOrqv_Grs5QWja3A_w6XuRHuTWY7c&zhida_source=entity) 便不得不在 HBM 与 SRAM 之间频繁分页交换注意力计算所需的[数据块](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E5%9D%97&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLmlbDmja7lnZciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.E-bCkZdFpOqtszv00pDKF3fViV8AEaQ75iDWChwPcDU&zhida_source=entity)，整段计算被彻底拖为 [memory-bound](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=memory-bound&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJtZW1vcnktYm91bmQiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.Dx9j95vofQGbqKStpdMK8iUg7VwcgDoxzwCvBNjeoOQ&zhida_source=entity)。128 头设定下，MLA 的 KV 量是 V4-MQA 单头 KV 的 256 倍。

![Image 19](https://picx.zhimg.com/v2-d7921a5655fbfae5936abfbdd308edf9_1440w.jpg)

MLA vs MQA-V4 投影路径：MLA 中 x(d) 降维到 latent C(d&#39;)，双流 W_k/W_v 上投影分多头（额外 HBM）。MQA-V4 中 x(d) 降维到 KV(d&#39;)，单流 repeat 复制到多头（无上采样）。

对比如下。

| 方案 | 压缩 | 上采样 | KV-Cache | 工程开销 |
| --- | --- | --- | --- | --- |
| MLA | dim → latent C | C → 多头 KV | C（需上投影矩阵） | 额外 RoPE 解耦投影 |
| V4-MQA | dim → head_dim | 无 | head_dim（单头） | HBM 交换少 |

在代码层面，两种方案的区别是 KV 投影后的处理：

```
# MQA: dim → 单头 head_dim
kv = self.wkv(x) # [B, L, head_dim]
# 1 个头, 1 份 KV-Cache, 直接和所有 Q 头做 attention

# MLA: dim → latent C → 上采样 → 多头
# c = self.wkv(x) # [B, L, latent_dim]
# k = self.wk_up(c).unflatten(...) # [B, L, n_heads, head_dim]
# 需要存 C + 额外的上投影矩阵, HBM 交换多一次
```

### 4.2 HCA vs NSA / DSA / CSA

*   HCA vs NSA：HCA 是 NSA 去除 Selection Attention 的版本。NSA 无 KV-Cache 减少，且多头候选 KV 不同
*   HCA vs DSA：DSA 做 token 级稀疏，不压缩 KV，不减少 KV-Cache。HCA 的 block 压缩两者兼具
*   HCA vs CSA：CSA 复用 Compressor，用 Indexer 对压缩 KV 做 top-k 二次筛选，其稀疏是块级别稀疏。 代码层面的筛选差异：

```
# HCA: 全部历史压缩块 (由因果掩码决定)
  compress_ids = get_compress_topk_idxs(ratio, L, start_pos, offset)
  # CSA: Indexer 选出 top-k
  # top-k = indexer(q, compressed_kv, topk=512)
```

### 4.3 CSA 如何继承 HCA

1.   CSA 复用 `Compressor`，`Indexer` 作用对象是压缩后的 KV
2.   HCA 的 compress 索引由因果掩码决定（全部历史块）；CSA 来自因果掩码 + Indexer top-k 候选
3.   `Compressor` 的 Overlap（ratio=4 时启用）被 CSA 使用

### 4.4 洞察

1.   **Infra**：HCA 不优化 FLOPS（计算量仍是多头 Q × 单头 KV 的矩阵乘），它优化的是 HBM 搬运。KV-Cache 减到 0.8%，Memory Wall 从瓶颈变成台阶。
2.   **训练**：原生训练的压缩权重（wkv/wgate）不是后训练 sparse finetune 学出来的，是预训练过程中和语言建模一起逐步学会的。Short-to-long 的渐进式训练中，短序列先学会 SWA，[长序列](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E9%95%BF%E5%BA%8F%E5%88%97&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLplb_luo_liJciLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.vUrCJJd3YjJZE7Z4FTOyv0rSlmzyulNejhH-uem0MOE&zhida_source=entity)才激活压缩。
3.   **局限**：Block 边界两端的 token 被分到不同的 compressed vector 里，边界信息断裂；r 越大越明显，因此需要 SWA 来兜底局部精度。
4.   **工程悖论**：MLA 每头单独表达，MQA 单头高度压缩信息。V4 选择MQA。Inference 的瓶颈从优化精度转变为优化带宽。

## 5. 总结

1.   Compressor：可学习门控池化将连续 `r` 个 KV 压成一个向量，Prefill/Decode 两阶段统一管理
2.   HCA：Window + Compressed KV 拼接，Sparse Attention 逐 query 零浪费计算，7812 个压缩 KV 覆盖 1M 上下文
3.   KV-Cache：总量为窗口大小加压缩KV大小，1M context为标准 Full Attention 的 0.8%
4.   MQA 取代 MLA：超长上下文下[显存带宽](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=%E6%98%BE%E5%AD%98%E5%B8%A6%E5%AE%BD&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiLmmL7lrZjluKblrr0iLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzUwNDc5OTMsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.UeViCrRunHY2HrTNiEnhJeo5-f3hzRYZOX0bRZXe0Rk&zhida_source=entity)是瓶颈，MQA 单头 KV 更 memory-efficient

## 代码

[http://github.com/dhcode-cpp/D eepSeek-V4-mini](https://link.zhihu.com/?target=http%3A//github.com/dhcode-cpp/DeepSeek-V4-mini)

*   `lc3/hca_simple.py`
*   `lc3/sec1_basic_compress.py`
*   `lc3/sec2_gated_compressor.py`
*   `lc3/sec3_trigger_compress.py`
*   `lc3/sec4_full_compressor.py`
*   `lc3/sec5_hca_attention.py`
*   `lc3/sec6_kv_cache_analysis.py`

## REFERENCE

*   DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
*   NSA: [Native Sparse Attention](https://zhida.zhihu.com/search?content_id=275047993&content_type=Article&match_order=1&q=Native+Sparse+Attention&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODk3MjE4MzUsInEiOiJOYXRpdmUgU3BhcnNlIEF0dGVudGlvbiIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI3NTA0Nzk5MywiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.id7kwoTO8-bx-jjcbnZw6fgCykcO4E7-rJWAW2wUrY8&zhida_source=entity): Hardware-Aligned and Natively Trainable Sparse Attention
*   DSA: DeepSeek Sparse Attention

* * *

手撕 DeepSeek-V4 系列，含代码

[小冬瓜AIGC：手撕 DeepSeek-V4 (1) : 模型架构](https://zhuanlan.zhihu.com/p/2032043071395869284)

[小冬瓜AIGC：手撕 DeepSeek-V4 (2): 注意力方案](https://zhuanlan.zhihu.com/p/2037107284803842257)

[小冬瓜AIGC：手撕 DeepSeek-V4 (3): HCA（Heavily Compressed Attention）](https://zhuanlan.zhihu.com/p/2039632097103623588)

[小冬瓜AIGC：手撕 DeepSeek-V4 (4): CSA (Compressed Sparse Attention)](https://zhuanlan.zhihu.com/p/2046142495940145191)

[小冬瓜AIGC：手撕 DeepSeek-V4 (5): Sparse Kernel](https://zhuanlan.zhihu.com/p/2068997042303660919)

DeepSeek-V4 前瞻，含代码

[小冬瓜AIGC：【手撕NSA】DeepSeek新作-原生稀疏注意力-超长文(附代码)](https://zhuanlan.zhihu.com/p/24841366485)

[小冬瓜AIGC：【手撕 DSA】 DeepSeek-V3.2 的 Sparse Attention 比 NSA 好在哪？](https://zhuanlan.zhihu.com/p/1957032283270812718)

[小冬瓜AIGC：【手撕 mHC】详解DeepSeek残差链接mHC进化之路（超长文、附代码）](https://zhuanlan.zhihu.com/p/1990683672337223894)

[小冬瓜AIGC：【手撕Engram】DeepSeek 的 Conditional Memory 能取代 Attention 吗？](https://zhuanlan.zhihu.com/p/1994713080131772751)

* * *

> 小冬瓜AIGC | X-R1开源框架 | 现高校LLM对齐研究  
> 原创课程帮助学员拿下OpenAI, Meta, 字节SEED等
