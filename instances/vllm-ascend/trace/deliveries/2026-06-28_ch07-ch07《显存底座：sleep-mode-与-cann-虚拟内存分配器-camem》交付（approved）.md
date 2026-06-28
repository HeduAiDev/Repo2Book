# ch07《显存底座：sleep-mode 与 CANN 虚拟内存分配器 camem》交付（APPROVED）

- **Type**: delivery
- **Chapter**: 07
- **Date**: 2026-06-28
- **Timestamp**: 2026-06-28T19:35:58Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch07, delivery, sleep-mode, camem, virtual-memory, f4-payoff, APPROVED

## What happened

reviewer APPROVED（无阻断 issue）。正文逐字内嵌 4 个真实文件（vllm_ascend/device_allocator/camem.py · vllm_ascend/worker/worker.py · vllm_ascend/patch/platform/patch_camem_allocator.py · vllm_ascend/platform.py），对照基座 vllm/device_allocator/cumem.py 逐行同构，仅换符号(cudart→acl.rt, libcudart→vllm_ascend_C)。lint_fidelity / narrative_vllm_refs / implementation SOURCE 阻断闸门全过；唯一遗留为 lint_source_grounding 非阻断告警(impl-notes.md 只列 1 个 vllm_ascend 路径 <3)，及 5 项声线/可读性小修(Roadmap≤25字、裸 chNN 指代、7.6 表 level2 拷贝代价口径、图表 VA 示例不一致、配套精简版前向引用、dest_max*2 / 双重 null 检查 / gc.collect+empty_cache 说明)，均 negotiable+非阻断。

## Why it matters

ch07 是 vllm-ascend 显存底座章，撑 sleep mode；交付后 Part 推进，f4 伏笔在此回收一半(ctypes 范式)。

## What to remember

reviewer APPROVED（无阻断 issue）。正文逐字内嵌 4 个真实文件（vllm_ascend/device_allocator/camem.py · vllm_ascend/worker/worker.py · vllm_ascend/patch/platform/patch_camem_allocator.py · vllm_ascend/platform.py），对照基座 v...
