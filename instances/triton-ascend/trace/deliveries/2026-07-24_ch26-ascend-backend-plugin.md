# ch26-ascend-backend-plugin

- **Type**: delivery
- **Chapter**: ch26
- **Date**: 2026-07-24
- **Timestamp**: 2026-07-24T01:13:25Z
- **Agents involved**: writer, reviewer, archivist
- **User present**: False
- **Tags**: backend-runtime, part-6, BaseBackend, NPUOptions, hacc.target

## What happened

Part 6「后端与运行时」开篇站(承 hivm-hfusion 子系统 ch20-25 收官后新子系统)·deps=ch14(对照基座编译驱动)：昇腾后端如何挂进 Triton——third_party/ascend/backend/compiler.py(977 行)。7 core+4 supporting 共 11 机制全覆盖：**backend-discovery**(triton/backends/__init__.py:_discover_backends 扫目录、显式跳过发行包不带的 nvidia/amd、_find_concrete_subclasses 要求恰好 1 个非抽象子类装配 backends['ascend']=Backend(AscendBackend,NPUDriver))；**basebackend-contract**(BaseBackend(metaclass=ABCMeta) 6 个抽象方法——supports_target/hash/parse_options/add_stages/load_dialects/get_module_map，ABCMeta 强制未实现即无法实例化)；**target-resolution**(NPUDriver.get_current_target 造 GPUTarget('npu',arch,0)，arch 来自环境变量或 self.utils.get_arch() 硬件探测)；**parse-options**(按 NPUOptions.__dataclass_fields__ 白名单过滤 opts，arch 缺省灌 self.target.arch)；**npuoptions-postinit**(compile_mode 三态派生 parallel_mode/force_simt_only/force_simt_template 及 shared_mem_dynamic_size 122880 vs 221184，frozen dataclass 用 object.__setattr__ 绕过冻结写入派生字段)；**npuoptions-hash**(全字段拼接 sha256 = 选项维度缓存键)；**backend-hash**(str(target) = 目标维度缓存键，与 NPUOptions.hash 正交构成两级缓存键，改 arch 同时影响两者)；**add-stages-pipeline**(注册 stages['ttir']=make_ttir→['ttadapter']=ttir_to_linalg(named_ops=True)→['npubin']=按 compile_on_910_95 二选一的 linalg_to_bin_*；force_simt_only 时跳过 ttadapter 直编 npubin，PLAN_CORRECTIONS 纠正章计划'三段 make_ttir/make_ttgir/make_npubin'措辞错误——昇腾没有 ttgir 这一阶段)；**stage-metadata-channel**(stage 契约 (src,metadata)->str|bytes，末段返回 bytes，metadata dict 串接下降管线)；**hacc-target-injection**(get_codegen_implementation 触发 _apply_ascend_patch()，幂等 monkey-patch ASTSource.make_ir——原始 IR 生成后读 options.arch、把 #hacc.target<"arch"> set_attr 到 module，PLAN_CORRECTIONS 纠正章计划'make_ttir/make_ttgir 阶段注入'错误措辞——真实位置是 patch 后的 make_ir，非任何 add_stages 注册的 stage)。真实 pytest_ut 夹具 third_party/ascend/unittest/pytest_ut/test_arch.py 佐证 arch='Ascend950' 字符串形态与 Options 字段对齐(PLAN_CORRECTIONS 同时纠正章计划臆造的'310'型号——本章源码只见 Ascend910B*/Ascend910*/Ascend950/Ascend910_95)。dossier 三条 PLAN_CORRECTIONS 均已在正文体现，无杜撰。kind=code(非 primer)，无 implementation/tests 目录(该章无精简版产物)。write↔review 3 轮收敛，blind 1 轮 0 failure，map 1 轮 PASS。多维评审 APPROVED，0 blocking + 5 non-blocking(1 条 algorithm-pedagogy：backend-discovery/basebackend-contract 两个 core 机制的『不变量』论证层比其余 5 个 core 机制单薄，只在旁白句提及未独立成段；4 条 reader-comprehension：§26.2 配图脚注'nvidia/amd 也各自实现一份'与 §26.1 'skip nvidia/amd' 措辞表面打架、要到 §26.5 才消歧/正文说'6 个 @abstractmethod' 但代码里 supports_target 实际是 @abstractclassmethod/叙述引入代码不存在的名字 npu_utils(代码是 self.utils)/§26.5 llir·ptx·cubin 三缩写对比时未加注解)——均 negotiable 且 non-blocking，留存量回修批次，未做退回重写。旁记：cartography outline-final.json 记录的 slug 是 ch26-ascend-backend-contract，与实际归档目录 ch26-ascend-backend-plugin 不一致(疑似出稿时改了 slug 未回写大纲)，如实记录未擅自改大纲。Bible 登记 glossary+8(NPUOptions/hacc.target/NPUDriver/parse_options/supports_target/后端自动发现(_discover_backends)/monkey-patch(补丁式注入)/get_codegen_implementation；AscendBackend/BaseBackend·GPUTarget/add_stages/bishengir-compile/ttadapter/npubin/compile_mode 等 7 个术语已在 ch01 首现，本章确认沿用未重复登记)/concepts+7(对应 7 个 core 机制)；interfaces 不新增(无精简版)；**新埋伏笔 f6**(ch26→ch27：ttadapter 段内部 triton_adapter pass 编排与 npubin 段 bishengir-compile 命令行拼接留给下一章站在同一 add_stages 上做后端 POV 编排细读)；bible.py due ch26 两清单本章均为空(无到期伏笔)。

## Why it matters

backend-runtime 子系统开篇章，把此前 25 章逐层建立的下降链事实(Part1-5 全部内容)收进'一个真实存在、可被 Triton 通用编译驱动发现并调用'的插件契约里——是全书从'讲昇腾怎么改 Triton IR'转向'讲昇腾怎么合法挂进 Triton 框架'的关键转折站，也是两级编译缓存键(target 维度+选项维度)与 hacc.target 硬件属性注入两条后续贯穿全书 backend-runtime 部分的机制首次完整登场。

## What to remember

ch26 APPROVED，backend-runtime 子系统开篇。7 core+4 supporting 共 11 机制全覆盖 AscendBackend 契约/NPUOptions/hacc.target monkey-patch。评审 0 blocking+5 non-blocking(留存量回修)。Bible glossary+8/concepts+7，新埋伏笔 f6(→ch27 ttadapter/npubin 内部编排)。旁记 outline slug 与归档目录名不一致(ch26-ascend-backend-contract vs -plugin)，未擅自改大纲。
