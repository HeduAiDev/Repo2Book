# ch28-bishengir-boundary

- **Type**: delivery
- **Chapter**: ch28
- **Date**: 2026-07-24
- **Timestamp**: 2026-07-24T03:00:00Z
- **Agents involved**: writer, reviewer, archivist
- **User present**: False
- **Tags**: backend-runtime, part-6, bishengir-compile, subprocess, closed-source-boundary, metadata-regex, ub-bits, dlopen, 910_95, A2_A3

## What happened

Part 6「后端与运行时」第三站，承 ch27（三段下降链编排站，本章深入 ch27 分叉点、把 `npubin` 段内部讲透）·kind=meta（对位基座 ch37 `ptxas`→cubin→发射，闭源边界主题）·deps=ch27。12 个机制全覆盖，三段主线：①**第①段 抠元数据**——`_parse_linalg_metadata`（compiler.py:L188-227）用 6 条正则（MIX_MODE_REGEX/PARALLEL_MODE_REGEX/KERNEL_NAME_REGEX/TENSOR_KIND_REGEX/BITCODES_REGEX/DISABLE_AUTO_TILE_REGEX）从 Linalg IR 文本 re.search/re.findall 抠出 mix_mode/parallel_mode/kernel_name/tensor_kinds/bitcodes 等字段，填进 metadata dict；`TENSOR_KIND_REGEX` 靠非贪婪限定符锁在单花括号块内单独拆解；`kernel_name + '_' + mix_mode` 编码进 `metadata['name']`，靠 `rsplit('_', 1)` 从右还原。②**第②段 拼命令行**——按 `if metadata[x] is not None` 逐项拼接，None 留白交给编译器默认，910_95 分支约 30 个条件开关，本例 9 个里落地 6 个；拼之前 `_get_npucompiler_path` 定位二进制、`--help` grep 探能力与版本，产物名 kernel.o vs kernel_reloc.o 是探出来的。③**闭源边界**——`subprocess.run(cmd_list, check=True)`（compiler.py:L465-L471）把 IR 文件和命令行喂给 `bishengir-compile`，边界精确停在这一行，闭源内部不猜、无源码可读，对位基座 `ptxas`。④**第③段 三通道回收**——`read_bytes` 拿 npubin 字节给 driver、stdout 正则抠 `required_ub_bits` 给 inductor autotune、dlopen+ctypes 从 libkernel.so 抠 4 个同步/任务字段（bs_task_type/workspace_size/lock_num/lock_init_val）。⑤**910_95 vs A2_A3 两候选差异**——骨架同构，差在 `--target` 取法/regbased 分叉/`sync_solver` 挂载数/A2_A3 独有开关，由导入期常量 `is_compile_on_910_95` 一次性决定全局分叉；补上一章 `force_simt_only` 快路径命令行细节。write↔review 2 轮收敛，blind 1 轮 0 failure，map 1 轮 PASS。多维评审 APPROVED，0 blocking + 12 non-blocking（2 条 fidelity 精确性：代码块行号标注差 1 行/`pack_metadata` 省略函数说明注释未标省略号，均 negotiable 留存量回修；3 条 algorithm-pedagogy：三个 core 机制的不变量论证散文化未加显式标签、subprocess-closed-boundary 量化环节复用 §28.3 数字未自带新数字；2 条图面：regex-extract/cmdline-assembly 两图基本复刻紧邻表格无额外信息增益、branch-divergence 图插在术语讲解之前；5 条 reader-comprehension：**「f7 的答案」「f7 回收」两处内部追踪 ID 裸漏进正文小标题**（零脚手架泄漏，需修）/`is_compile_on_910_95` 全局常量与 `options.compile_on_910_95` 选项字段关系未打通/`bin_file`·`bin_path`·`callback_path` 三个名字跨小节未衔接/`torch inductor` 外部概念未括注/表格「展平」措辞先于正文解释）——均 negotiable 且 non-blocking，留存量回修批次处理，未做退回重写。**回收伏笔 f7**（ch27→ch28：npubin 段命令行拼接、subprocess 调用、910_95 vs A2_A3 两候选差异，本章完整兑现，arc-map 状态改 resolved）；本章未埋新伏笔（dossier.foreshadow_due.should_plant 为空）。kind=meta，无 implementation/tests 目录（该章无精简版产物，闭源边界章天然无可跑代码，交叉验证走 pin 源码逐行核对 `_parse_linalg_metadata`/`linalg_to_bin_enable_npu_compile_910_95`/`linalg_to_bin_enable_npu_compile_A2_A3`/`ttir_to_npubin` 四处真实函数体，不伪造运行 dump）。5 个机制图（regex-extract/cmdline-assembly/closed-boundary/three-channel/branch-divergence）+ 本章地图共 6 张图，blind_review 全 5 张 PASS。

## Why it matters

backend-runtime 子系统第三站，是全书下降链的终点站：从 ch01 鸟瞰图上「ttir→ttadapter→npubin」三段最后一段，到 ch26 装配、ch27 编排，本章第一次把 `npubin` 段内部——从文本正则抠元数据到 subprocess 调用到产物回收——完整摊开，并诚实标出「开源侧读到哪、闭源边界在哪」。与基座《Triton 源码解读》ch37 `ptxas`→cubin→发射构成全书方法论最工整的一次跨书对位：两边都是「拼命令行、subprocess 喂、读回产物」的同款套路，只是闭源二进制换了一个。收尾处「再往后离开编译期看运行时怎么发射」的过渡句，把全书叙事从「怎么编出来」正式移交给「怎么跑起来」。

## What to remember

ch28 APPROVED，backend-runtime 子系统第三站（承 ch27）。12 机制全覆盖，主脊是抠元数据→拼命令行→subprocess 闭源边界→三通道回收四段论证，收尾讲透 910_95 vs A2_A3 两候选差异。评审 0 blocking + 12 non-blocking，其中最值得关注的一条是**「（f7 的答案）」「（f7 回收）」两处内部追踪 ID 裸漏进正文小标题**——典型零脚手架泄漏，与本书历史上已修过的同类问题同源，留存量回修批次定点删除/改写，不影响本次归档判定（negotiable 且 non-blocking）。**回收伏笔 f7**（npubin 段命令行/subprocess/两候选差异完整兑现），本章未埋新伏笔。Bible 回写：glossary+6（`_parse_linalg_metadata`/`required_ub_bits`·UB-bits/libkernel.so dlopen 回调/`is_compile_on_910_95` 全局常量/`_get_npucompiler_path`·bishengir 能力探测/`kernel_name`+`mix_mode` 编码约定；`bishengir-compile`/`A2_A3`/`910_95`/`force_simt_only`/`compile_on_910_95` 均已在 ch26/ch27 首现，本章确认沿用未重复登记）/concepts+7（对应 7 条机制级要点）/figures+6（5 机制图+chapter-map）；interfaces 不新增（无精简版）。**下一站**：编译期收官，故事转向运行时——二进制怎么发射上核跑起来。
