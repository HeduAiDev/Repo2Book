# ch36 CUDABackend 把五段 stages 注入编译管线 定稿

- **Type**: delivery
- **Chapter**: ch36
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T17:11:08Z
- **Agents involved**: analyst, implementer, tester, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: triton, part-viii, backend, cudabackend, foreshadow-payoff, f1, f6, capability-gating

## What happened

ch36《CUDABackend:把五段 stages 注入编译管线》定稿,Part VIII 硬件后端开篇,reviewer APPROVED。CUDABackend=BaseBackend 具体实现:add_stages 五个 lambda 注册 ttir→ttgir→llir→ptx→cubin(闭包藏 options/capability、对外统一 (src,metadata) 签名),compile() 单调切片逐段跑并落缓存、末段 cubin 产 bytes;parse_options 按 capability 动态拼 fp8 清单(≥89 补 fp8e4nv、≥90 标 deprecated fp8e4b15),组装 CUDAOptions(__post_init__ 注入 libdevice + assert num_warps 是 2 的幂);make_ttgir 按 capability//10 分档注入 pass 序列(17 基线/≥sm80 +9/≥sm90 +2,总数 17→26→28);capability 一自变量喂两把尺(粗档 //10 控 pass、细阈 89/90 控 fp8);C++/Python 双语接缝(load_dialects/init_triton_nvidia 经 pybind)。skip_impl 章,无精简版接口。回收伏笔 f1(ch01 埋后端接缝/BaseBackend 契约)+f6(ch05 埋 fp8 后端能力接缝),二者经正文兑现已 resolve。Lead 派 writer 补 4 处、illustrator 修 fig-m1 记号。

## Why it matters

Part VIII 开篇把散落 Part V-VII 的 pass 收成 CUDABackend 一条真实编译序列,兑现 ch01(f1 后端接缝配对脊柱 CUDA 样板端,姊妹篇 ascend 对位锚)与 ch05(f6 fp8 能力接缝另一端),回答『一块新卡怎么接进来』;给读者两个性能抓手(pass 门控看卡分档、num_warps 必 2 的幂约束)。

## What to remember

f1+f6 已 resolve(bible.py payoff --resolve,in ch36)。glossary 新增 9 词(CUDAOptions/parse_options/make_ttgir/capability/load_dialects(后端)/init_triton_nvidia/ClusterInfo/TRITON_PLUGIN_DIRS/后端发现机制)+更新 3 词(CUDABackend/BaseBackend/add_stages 补 ch36 落地)。concepts 新增 11 条、figures 登记 2 图(fig-m1/fig-m3)。注:dossier foreshadow_due 实际已填 f1/f6(Lead note 说漏填,核验为已填),bible.py due 亦兜住。skip_impl 无 interfaces。
