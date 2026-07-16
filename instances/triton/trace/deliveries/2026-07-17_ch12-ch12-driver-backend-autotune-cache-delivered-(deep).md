# ch12-driver-backend-autotune-cache-delivered-(deep)

- **Type**: delivery
- **Chapter**: 12
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T00:00:00Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch12, deep, part-3, driver-discovery, LazyProxy, autotune, disk-cache, FileCacheManager, f11-payoff

## What happened

第十二章《driver 抽象、后端发现、autotune 与磁盘缓存》交付（Part III，kind=deep/mode=skip_impl，pin triton==3.2.0 真编译取证；并行发车 skip_archive 模式，Review+Map 已 APPROVED/PASS，本次由 Lead 手工串行补归档）。全章三条支线：① 后端配对脊柱——`_discover_backends` 扫 `backends/<name>/` 各取唯一 concrete 子类建 `{name: Backend(compiler, driver)}` 表；`LazyProxy` 惰性代理让 `driver = DriverConfig()` 的 `default` 字段延后到首次属性访问才 `_create_driver`（筛唯一 `is_active()` 的 driver），保证 `import triton` 不碰 CUDA；`DriverBase`/`GPUDriver` 契约把 `torch.cuda` 桥接点钉死为 ch11 标出的 host 无 GPU 断裂点（回收 f11）。② autotune 完整操作面——`Autotuner.run` 以 key 参数值+带 dtype 实参的 dtype 字符串组缓存键，miss 走 `prune_configs`（early_config_prune 硬筛 + perf_model top_k）+`_bench`（pre/post hook 装配 + `do_bench` 分位数计时，异常记 inf）+取最快存缓存；`reset_to_zero`/`restore_value` 用 pre/post hook 保护被 kernel 改写的张量；计时器经 `driver.active.get_benchmarker()` 走后端接缝。③ 磁盘缓存——`FileCacheManager` 目录布局 `<TRITON_CACHE_DIR 或 ~/.triton/cache>/<key>`，`put()` 走 tmp 目录 + `os.replace` 原子落盘（POSIX rename 原子性论证），`get_group`/`put_group` 用 `__grp__` 索引校验一次编译多产物文件是否齐全；`compile()` 里的编译缓存键 `triton_key+src.hash+backend.hash+options.hash+env_vars` 经 sha256/`_base64` 落到该缓存；点明与 ch11 内存 launch 缓存正交（不同粒度/生命周期）。14 机制(6 core+8 supporting)。5 张机制图(m1-lazy-state/m2-discover-flow/m5-key-cache/m7-hook-guard/m11-atomic-put) + chapter-map 全 blind PASS。review APPROVED(7 negotiable/non-blocking issues：1 条省略注释描述失实(内嵌 Autotuner.run 截断处) + 1 条格式类(m1/m2/m11 缺显式『不变量』标签) + 1 条图文行号误差(m1-lazy-state.png 把 `self._obj=None` 错标 L15，应为 L16) + 4 条 reader-comprehension 小卡点：前后两处『四个旋钮』指代不同集合未统一/KernelInterface 首现未释/CudaDriver 类名未与 GPUDriver 体系接上/`Config.all_kwargs` 源码里六个进阶旋钮字段未加『可跳过』注解)；write_review_rounds=1、blind_rounds=1(0 failures)、map_rounds=1(PASS)、无 escalation。禁区遵守：不重讲 ch11 已建立的 run() 六段编排/内存 launch 缓存键构成，不展开 compile() 内部五段驱动主循环（留给 ch14）。

## Why it matters

本章是 ch11 埋下的 f11（driver 子系统边界断裂点）唯一正式回收处——把「host 无 GPU 会在哪里炸」从边界调用面（ch11）展开成完整机制（LazyProxy 惰性代理 + `_discover_backends` 配对脊柱 + `DriverBase`/`GPUDriver` 契约），也是姊妹篇《Triton-Ascend 源码解读》后端接入方式的直接参照点（新增 `backends/ascend/` 目录零改动接入）。autotune 与磁盘缓存两条支线补全了「一次 kernel 调用」在 launch 主脊之外的全部性能可配置面，供后续任何涉及 kernel 调优/编译产物复用的章节回指。

## What to remember

ch12 done（kind=deep/mode=skip_impl，Part III）。glossary.json 120→135（新增 15 条：LazyProxy/DriverConfig/_discover_backends/DriverBase/GPUDriver/CudaDriver/Autotuner/@triton.autotune/autotune 缓存键(key 参数+张量 dtype)/_bench/prune_configs/reset_to_zero·restore_value/Config/FileCacheManager/编译磁盘缓存键(triton_key+src+backend+options+env)）。concepts.json 新增 4 条→ch12（惰性 driver 选择、后端目录发现配对脊柱、autotune 完整操作面、磁盘缓存与内存 launch 缓存正交）。interfaces.json 新增 ch12 键（源码接口，非精简版）：`DriverBase` 三契约方法、`GPUDriver.__init__`、`Autotuner.run`/`_bench`/`prune_configs`、`Config.all_kwargs`、`FileCacheManager.put`/`get_group`/`put_group`、`get_cache_manager`。arc-map.json：**f11 回收**（status open→resolved，resolved_in=ch12，`bible.py due ch12` 已验证不再列出待回收项，确认无遗漏）；dossier `foreshadow_due.plant` 为空，本章未新开正式伏笔（autotune 与后续性能 pass 章的关系已由既有 f2/f3/f9/f6 覆盖，无需新条）。reviews/review-report.json 与 run-ledger.json 由 Lead 预写，本次未改动；narrative/chapter.md 与 diagrams/ 由 writer/illustrator 并行定点修，archivist 未触碰。
