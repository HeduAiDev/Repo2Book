# ch14《NPUModelRunner(GPUModelRunner)：继承 244KB 父类 + 运行时猴补设备符号让父类原样跑昇腾》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 14
- **Date**: 2026-06-29
- **Timestamp**: 2026-06-29T17:44:30Z
- **Agents involved**: analyst
- **User present**: False
- **Tags**: ch14

## What happened

ch14 多维评审 APPROVED 交付。旗舰章：NPUModelRunner 继承 244KB 的 GPUModelRunner，只 override 设备相关方法（_init_device_properties/num_sms=None、_sync_device/torch.npu.synchronize），庞大父类逻辑原样复用——与 ch13 NPUWorker「重写而非继承」恰成对照（设备层基座钉死 cuda 走不通则重写；ModelRunner 父类 cuda 散落在 torch.cuda.* 调用与模块级 graph_capture/CUDAGraphWrapper、没钉死在 else-raise，故可继承+运行时猴补）。核心揭秘两个上下文管理器成对装卸：_torch_cuda_wrapper 临时换 torch 设备符号 8 个（7 个 torch.cuda.* Event/Stream/default_stream/current_stream/stream/synchronize/mem_get_info + 1 个顶层 torch.Event）→npu 版，try/except(placeholder 兜底+re-raise)/finally 还原、finally 把 synchronize/mem_get_info 留 npu 版作稳态缺省；_replace_gpu_model_runner_function_wrapper 经 _get_gpu_model_runner_module_name 走 MRO 取父模块、setattr 把 graph_capture/CUDAGraphWrapper 换 NPU/ACLGraph 版、finally 还原。双 wrapper 包父类 capture_model/profile_cudagraph_memory 一行不改跑昇腾。再讲 _use_aclgraph 三条件决策、CUDAGraph→ACLGraph 对位、AscendSampler/AscendAttentionState 替换。横切点名：注意力后端实体(AscendAttentionBackend/MLA)留 ch18/19、采样器内部留后续。

## Why it matters

确立 OOT 插件「继承父类 + 运行时设备符号猴补」这一最具插件味的方法对位（与 ch13 重写形成两面），讲透「成对进出、作用域内才生效」为何安全/临时——为 Part V 图捕获(ACLGraph)与注意力后端章铺垫。

## What to remember

ch14 无伏笔应埋/应回收（bible.py due 空；arc-map 无 ch14 plant/payoff）。新接口 8 条已登记 interfaces.json（NPUModelRunner 类 + _torch_cuda_wrapper/_replace_gpu_model_runner_function_wrapper/_use_aclgraph/_get_gpu_model_runner_module_name/capture_model+profile_cudagraph_memory/graph_capture+GraphCaptureContext/ACLGraphWrapper）。评审 APPROVED，11 条均 non-blocking 商榷（4 条声线/算法精度小修：§14.5③末句主语漂移断句、§14.8 t5 破折号占位释义、torch.cuda.* 8 个应拆 7+1、§14.8 t0 mem_get_info 与 §14.5④ 稳态缺省张力；7 条 reader-comprehension：最重歧义/CUDA 图概念/mem_get_info 返回值/精简版术语/except placeholder 逻辑/未绑定方法过时术语）。
