# vllm_ascend/worker/model_runner_v1.py —— subtract-only 精简版（ch14 主角）
#
# 本章主线：NPUModelRunner 继承 244KB 的 GPUModelRunner 父类，只 override 设备相关
# 方法 + 在窄接缝处用两个"成对进出"的上下文管理器临时把 torch.cuda.* 与父类模块级
# graph_capture/CUDAGraphWrapper 换成 NPU/ACL 版，让父类巨方法一行不改地跑在昇腾上。
#
# 精简版只保留"设备层猴补 + 图捕获接缝"控制流；与之正交的 PCP/DCP/EPLB/sparse/
# multimodal/spec_decode 等大段字段初始化按 subtraction_plan.delete 折叠。
# 真实 torch.cuda/torch.npu 符号与 ACLGraph 捕获不在 host 真跑——测试用 sys.modules
# 注入桩验证纯 Python 的"装/卸"与决策控制流（见 ../tests）。
import gc
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

# SOURCE: vllm_ascend/worker/model_runner_v1.py:L39
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
# SOURCE: vllm_ascend/worker/model_runner_v1.py:L104
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
# SOURCE: vllm_ascend/worker/model_runner_v1.py:L113
from vllm_ascend.attention.attention_v1 import AscendAttentionState
# SOURCE: vllm_ascend/worker/model_runner_v1.py:L121-L123
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, reset_graph_params
# SOURCE: vllm_ascend/worker/model_runner_v1.py:L136
from vllm_ascend.sample.sampler import AscendSampler
# SUBTRACTED: 数百行其它 import（attention 后端实体/quantization/spec_decode 各 proposer/
#   distributed/buffer 工具等，model_runner_v1.py:L20-L155）—— 与本章设备猴补/图捕获接缝正交。


@dataclass
class GraphCaptureContext:
    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L196
    # 与父类 GraphCaptureContext.stream 字段同形（只是这里是 torch.npu.Stream）——
    # 父方法 `with graph_capture(...) as ctx: ctx.stream` 不改即可用。
    stream: "torch.npu.Stream"


@contextmanager
def graph_capture(device: torch.device):
    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L201
    # NPU 版 graph_capture：与父类同签名 graph_capture(device)、同返回 GraphCaptureContext，
    # 被 _replace_gpu_model_runner_function_wrapper 塞进父类模块替换原 cuda 版。
    # SUBTRACTED: 原 docstring 大段说明（model_runner_v1.py:L204-L216），删后控制流不变。
    graph_capture_context = GraphCaptureContext(torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


# SOURCE: vllm_ascend/worker/model_runner_v1.py:L255
class NPUModelRunner(GPUModelRunner):
    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L256
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # TODO(qcs): These manual pad and unpad for GPUModelRunner are
        # used to expand some buffers ... (https://github.com/vllm-project/vllm/pull/28988)
        max_pcp_pad_tokens = (
            vllm_config.parallel_config.prefill_context_parallel_size * 2 * vllm_config.scheduler_config.max_num_seqs
        )
        vllm_config.scheduler_config.max_num_batched_tokens += max_pcp_pad_tokens

        # Must be set before super().__init__() because parent init may call
        # _allocate_kv_cache_tensors which accesses self.use_compress.
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None) if model_config else None
        self.use_compress = (
            hf_config is not None and hasattr(hf_config, "compress_ratios")
        )

        # 关键：父类 244KB 的 __init__ 在 _torch_cuda_wrapper() 作用域内运行——其内部
        # 所有 torch.cuda.Stream()/Event() 创建被临时改向 NPU；退出 with 即卸载。
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # SUBTRACTED: NPUPrefetchOffloader 替换、query_start_loc / gdn buffer、Ascend 各项
        #   config、PCP/DCP/EPLB/sparse/multimodal 等大段字段初始化（model_runner_v1.py:L276-L489
        #   中与设备猴补正交的部分，subtraction_plan.delete）。
        vllm_config.scheduler_config.max_num_batched_tokens -= max_pcp_pad_tokens

        # super().__init__() 把 self.sampler 设成 GPU 版 Sampler、attn 状态用 vLLM 枚举；
        # 这里在父构造器之后把设备相关实体字段覆盖成昇腾版。本章只点名替换这一事实，
        # AscendSampler 内部留采样器章节、AscendAttentionState 实体留 Part V（ch18/ch19）。
        # SOURCE: vllm_ascend/worker/model_runner_v1.py:L317-L318
        self.sampler = AscendSampler()
        self.attn_state: AscendAttentionState | None = None

        # SOURCE: vllm_ascend/worker/model_runner_v1.py:L490
        self.use_aclgraph = self._use_aclgraph()
        # SUBTRACTED: eplb / 其余 Ascend-specific 字段（model_runner_v1.py:L492+）。

    # Note: used for model runner override.（父类预留的设备钩子，see gpu_model_runner.py:L1056）
    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L580
    def _init_device_properties(self) -> None:
        self.num_sms = None

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L583
    def _sync_device(self) -> None:
        torch.npu.synchronize()

    # SUBTRACTED: _set_up_drafter / _get_drafter（model_runner_v1.py:L586-L619）——
    #   spec_decode drafter 装配，与设备猴补/图捕获接缝正交。

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L620
    def _use_aclgraph(self) -> bool:
        # 三条件齐备才用 ACLGraph：图模式非 NONE、编译模式 VLLM_COMPILE、未强制 eager。
        return (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and self.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not self.model_config.enforce_eager
        )

    # SUBTRACTED: NPUModelRunner 其余 ~4000 行 override（prepare_inputs / execute_model /
    #   _dummy_run / sample 等），均原样复用或与本章主题正交——只在此保留图捕获接缝两法。

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4798
    def profile_cudagraph_memory(self) -> int:
        parent_module_name = _get_gpu_model_runner_module_name(self)
        # 双 wrapper 包住父方法：作用域内 torch.cuda.* 指向 npu、父模块 graph_capture/
        # CUDAGraphWrapper 已是 NPU/ACL 版 → 父方法 mem_get_info() 实测 NPU 显存。
        with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(parent_module_name):
            result = GPUModelRunner.profile_cudagraph_memory(self)

        reset_graph_params()

        # SUBTRACTED: profiling 善后——手动清两份多余 KV cache 副本 + gc.collect() +
        #   torch.accelerator.empty_cache()（model_runner_v1.py:L4805-L4816，用到模块级
        #   import gc），与"双 wrapper 包父方法"主线无关。
        return result

    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4820
    def capture_model(self) -> int:
        """Capture NPU graphs and return actual graph pool memory bytes consumed."""
        parent_module_name = _get_gpu_model_runner_module_name(self)
        with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(parent_module_name):
            # 以未绑定方法显式调父类巨方法，传 self——"就是要父类那份一行不改地跑"。
            return GPUModelRunner.capture_model(self)


# SOURCE: vllm_ascend/worker/model_runner_v1.py:L4876
def _get_gpu_model_runner_module_name(model_runner) -> str:
    """Return the module name of GPUModelRunner found in the MRO."""
    # 沿 MRO 找到 GPUModelRunner 类，取它所在的模块名（vllm.v1.worker.gpu_model_runner）。
    # 关键：graph_capture/CUDAGraphWrapper 被 import 进"父类那个模块的命名空间"，要让
    # 父方法看见替换，必须改父类模块的属性，而非本模块。
    gpu_model_runner_cls = next(
        (cls for cls in model_runner.__class__.__mro__ if cls.__name__ == "GPUModelRunner"),
        None,
    )
    if gpu_model_runner_cls is None:
        raise TypeError(
            "Could not find GPUModelRunner in the MRO. "
            "The class hierarchy may have changed."
        )
    return gpu_model_runner_cls.__module__


@contextmanager
def _torch_cuda_wrapper():
    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4890
    # 进程级临时替换 torch.cuda 模块属性（不是实例级）——所以必须配 try/finally 成对装卸。
    class _EventPlaceholder:
        # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4892
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda *a, **kw: None
            self.synchronize = lambda *a, **kw: None
            self.wait = lambda *a, **kw: None
            self.query = lambda *a, **kw: True

    class _StreamPlaceholder:
        # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4899
        def __init__(self, *args, **kwargs) -> None:
            pass

    try:
        # replace cuda APIs with npu APIs, this should work by default
        torch.Event = torch.npu.Event
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        # SUBTRACTED: default_stream/current_stream/stream 三处同模式替换
        #   （model_runner_v1.py:L4908-L4910）——保留代表性四个符号即可说明"装"。
        yield
    except Exception as e:
        # except 分支先装 placeholder 再 re-raise：init 失败时不让残留 NPU 绑定坏掉别处。
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        torch.cuda.synchronize = _StreamPlaceholder
        torch.cuda.mem_get_info = _StreamPlaceholder
        # SUBTRACTED: 同 except 分支里 default_stream/current_stream/stream 的 placeholder 赋值。
        raise RuntimeError(f"NPUModelRunner init failed, error is {e}")
    finally:
        # if anything goes wrong, just patch it with a placeholder
        # 退出后 torch.cuda 并非原样还原，而是落到一组安全缺省/NPU 直通（稳态选择）。
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = torch.cuda.Stream
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        # SUBTRACTED: finally 分支里 default_stream/current_stream/stream 的还原赋值。


# TODO: This method will be removed subsequently and implemented in platform.
@contextmanager
def _replace_gpu_model_runner_function_wrapper(target_module_name):
    # SOURCE: vllm_ascend/worker/model_runner_v1.py:L4934
    target_module = None
    original_attrs = {}
    try:
        # 先存 original_attrs（旧值），再 setattr 父类模块的 graph_capture→NPU 版、
        # CUDAGraphWrapper→ACLGraphWrapper；yield 期间父方法引用的就是替换版。
        target_module = sys.modules[target_module_name]
        if hasattr(target_module, "graph_capture"):
            original_attrs["graph_capture"] = target_module.graph_capture
        setattr(target_module, "graph_capture", graph_capture)  # noqa: B010
        if hasattr(target_module, "CUDAGraphWrapper"):
            original_attrs["CUDAGraphWrapper"] = target_module.CUDAGraphWrapper
            setattr(target_module, "CUDAGraphWrapper", ACLGraphWrapper)  # noqa: B010
        yield
    except Exception as e:
        raise RuntimeError(f"NPUModelRunner failed, error is {e}")
    finally:
        # finally 逐一 setattr 还原——"临时/可逆"的物证。
        if target_module is not None:
            for attr_name, attr_value in original_attrs.items():
                setattr(target_module, attr_name, attr_value)  # noqa: B010
