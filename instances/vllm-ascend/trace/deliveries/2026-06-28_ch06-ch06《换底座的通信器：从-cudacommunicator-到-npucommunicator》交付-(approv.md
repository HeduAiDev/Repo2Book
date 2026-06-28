# ch06《换底座的通信器：从 CudaCommunicator 到 NPUCommunicator》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 06
- **Date**: 2026-06-28
- **Timestamp**: 2026-06-28T18:43:44Z
- **Agents involved**: analyst,  implementer,  tester,  writer,  reviewer,  archivist
- **User present**: False
- **Tags**: get_device_communicator_cls,  NPUCommunicator,  DeviceCommunicatorBase,  all_to_all,  pyhccl,  pyhccl_wrapper,  ctypes,  HCCL,  patch_distributed,  310p,  OOT-换通信器,  foreshadow-f3-plant,  foreshadow-f4-plant

## What happened

本章把「通信器是 OOT 换底座最干净样本」讲透，四条并排线：①platform 回调注入——NPUPlatform.get_device_communicator_cls 覆写返回 NPUCommunicator qualname，parallel_state 经 resolve_obj_by_qualname 字符串→类→统一签名实例化，一个字符串换底座；②子类化 DeviceCommunicatorBase——绝大多数集合通信直接继承(底层走 torch.distributed + HCCL backend，故复用基类即可)，只 __init__ 微调(self.device=torch.npu.current_device()/ca_comm=None 占位)+ 新增唯一差异点 all_to_all(均分 tensor_split / 不均分逐 rank 按 gather_sizes 预分配 output)；③手写 pyhccl ↔ pynccl 的 ctypes 通信器逐段对位移植(unique_id 建组 / CommInitRank / warmup all_reduce / disabled 降级；HCCLLibrary 对位 NCCLLibrary，hcclUniqueId 4108B、枚举按 hccl_types.h，照搬范式只换符号)，当前未接入(npu_communicator.py:L32 TODO，真实使用仅单测)；④patch_distributed 仅 310P 把 broadcast / int64 all_reduce 猴补成 all_gather 模拟补硬件能力缺口(复用 ch03 技法④)，降级直通、A2/A3/A5 不触发。Reviewer 判 APPROVED，13 条 issue 全部 non-blocking(7 条 reader-comprehension 维度补名词解释 NCCL/HCCL/.so/310P 等 + 6 条保真/导航/节奏小瑕：§6.3.4 TODO 注释也命中 grep 的措辞精度、总览图 alt 三/四条与正文打架、6.1 收束后两段旁白缺路标、qualname 译名与 glossary「限定名」分叉、all_to_all 不均分分支与 6.4 int64 all_reduce 缺数值追踪、6.2.3「与 all_gather 同阶」量化口径与 6.4 冲突)。

## Why it matters

通信器章证成 OOT 换底座最干净样本的论点——接口窄、基类已抽象好、只露一个 all_to_all 差异点；为后续 FusedMoE 章(all_to_all 路由侧)与显存/图捕获章(custom all-reduce/ca_comm 占位)建立共享词汇与接口注册。同时把 GPU→NPU 的 ctypes 绑定范式照搬(pyhccl↔pynccl)沉淀为可复用样本。

## What to remember

ch06 APPROVED 交付。7 个新接口入 bible interfaces.json(get_device_communicator_cls 覆写/resolve_obj_by_qualname/NPUCommunicator 类+all_to_all/PyHcclCommunicator/HCCLLibrary/communication_adaptation_310p)。本章埋两条伏笔(均 open)：f3=all_to_all 真实路由用途→ch26(FusedMoE)；f4=ca_comm=None 占位 + pyhccl 未接入的 ctypes 预留→ch07(custom all-reduce/graph capture)。本章无应回收伏笔。13 条非阻断 issue 留待 writer 定点小修(不退章)；待 Lead/Archivist 拍板 qualname 译名(glossary「限定名」vs 正文「全限定类名」)统一方向。
