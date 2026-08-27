# v3 ch13《分页 KV：虚拟内存思想进显存》定稿交付（2026-08-31 归档）

## 状态
- **全章定稿**：narrative 1145 行 + L1-partIV/L2-ch13 + 8 张机制图
- Test APPROVED：54/54 host + 差分电池 38/38（容器 vllm-omni 真源码 + GPU Triton kernel 逐位一致）
- Review APPROVED（round 2 确认）：零 blocking，2 negotiable 已修（DAG 论断限定 + m1/m6 剧本切换提示）
- 评审升级项（dossier/concepts 错形状串 7 处）已修并提交 `ccfa1357`
- F2 伏笔已埋（图注「抢占恢复撞前缀缓存，第 15 章回收」）——**ch15 成稿后升级为跨章链接**

## 核心内容
PagedAttention 血统（三重浪费 20-38%→近满）/ 块池+逻辑块表 / slot 恒等式 / GPU 端 Triton 换算 / 侵入式双链自由队列 O(1) 摘挂 / 引用计数 touch-free 共享 / LRU 双不变量（逆序 free + 无哈希先驱逐）/ 预构空对象 GC 纪律。

## 归档注意
- dossier.json 曾三轮修复（批默认顺序归位 → subtraction 切雕 → 形状串）
- 素材真相源口径：层内真实形状由注意力后端仲裁，说明性视图 [num_blocks, 2, block_size, kv_heads, head_dim]
