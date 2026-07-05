# 精简版（只做减法）— 对照真实源码 vllm_ascend/compilation/compiler_interface.py
#
# AscendCompiler 是 vLLM CompilerInterface 的昇腾自定义子类，platform.get_compile_backend()
# 返回它的字符串路径，整体顶替 vLLM 默认的 InductorAdaptor。compile() 按
# ascend_compilation_config.enable_npugraph_ex 二选一：
#   True  → npugraph_ex_compile（npugraph_ex 图编译，ImportError 回退 torchair，两者都经
#           _configure_backend 设成 aclgraph 模式）
#   False → fusion_pass_compile（compile_fx + aot_autograd，inner_compile 里跑自家
#           GraphFusionPassManager 做手写算子融合）
#
# 减法：编译产物缓存（patched_get_compiled_gm 写盘巧思 + load() 的缓存读写/cache-miss 重编译）
# 与 enable_static_kernel 静态 kernel 加速开关按 dossier 批准删除——它们只影响二次启动/吞吐，
# 不改变「二分编译 + tuple 适配」的主线语义。
import copy
import functools
import os
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from torch._inductor.decomposition import select_decomp_table
from torch.fx import GraphModule
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import logger

from vllm_ascend.ascend_config import AscendCompilationConfig, get_ascend_config
from vllm_ascend.utils import COMPILATION_PASS_KEY


# SOURCE: vllm_ascend/compilation/compiler_interface.py:L39
def compile_fx(graph: GraphModule, example_inputs: list, inner_compile: Callable, decompositions: dict) -> Callable:
    recursive_compile_fx = functools.partial(compile_fx, inner_compile=inner_compile, decompositions=decompositions)

    # torch.compile 后端协议要求 fx 图输出是 flat tuple；不是则先包一层再递归编译。
    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, recursive_compile_fx)
    return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)


# SOURCE: vllm_ascend/compilation/compiler_interface.py:L47
def fusion_pass_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    compile_range: Range,
    key: str | None = None,
) -> tuple[Callable | None, Any | None]:
    def compile_inner(graph, example_inputs):
        # SOURCE: vllm_ascend/compilation/compiler_interface.py:L54
        # compiler_config[COMPILATION_PASS_KEY] 取出经 inductor 自定义 pass 机制注入的
        # GraphFusionPassManager 实例，对图跑融合 pass。
        current_pass_manager = compiler_config[COMPILATION_PASS_KEY]
        graph = current_pass_manager(graph)
        return graph

    decompositions = select_decomp_table()

    compiled_fn = compile_fx(
        graph=graph,
        example_inputs=example_inputs,
        inner_compile=compile_inner,
        decompositions=decompositions,
    )

    return compiled_fn, None


# SUBTRACTED: _compute_decode_cudagraph_batch_sizes（原 :L71-L79）——它由 max_num_seqs×
#             (num_speculative_tokens+1) 与 cudagraph_capture_sizes 求交，限定 enable_static_kernel
#             下静态 kernel 编译的 batch 范围；static_kernel 是可选加速开关，与 aclgraph 模式主干无关。


# SOURCE: vllm_ascend/compilation/compiler_interface.py:L82
def _configure_backend(
    config: Any,
    ascend_compilation_config: AscendCompilationConfig,
    vllm_config: VllmConfig,
    process_kwargs_options: Callable | None = None,
) -> None:
    if process_kwargs_options is not None:
        # npugraph_ex (both old and new): build options dict and use _process_kwargs_options.
        # force_eager=True: execute FX graph in eager mode before graph capture.
        # inplace_pass=False: disable reinplace pass to avoid gelu fallback to CPU.
        options: dict[str, Any] = {
            "force_eager": True,
            "inplace_pass": False,
        }
        # SUBTRACTED: if ascend_compilation_config.enable_static_kernel 块（原 :L98-L106）——
        #             开 static_kernel_compile + _vllm_aclnn_static_kernel_sym_range（可选加速）。
        process_kwargs_options(config, {"options": options})
    else:
        # torchair (reduce-overhead): use nested config structure directly.
        # mode="reduce-overhead": use aclgraph mode, avoid fx graph to Ascend IR transformation.
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        # Disable reinplace pass to avoid gelu fallback to CPU causing host-device copy error.
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        # SUBTRACTED: if ascend_compilation_config.enable_static_kernel 块（原 :L114-L122）——
        #             开 _aclnn_static_shape_kernel + sym_value_range（可选加速）。


# SOURCE: vllm_ascend/compilation/compiler_interface.py:L125
def npugraph_ex_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    vllm_config: VllmConfig,
    ascend_compilation_config: AscendCompilationConfig,
    compile_range: Range,
    key: str | None = None,
    cache_dir: str | None = None,
) -> tuple[Callable | None, Any | None]:
    # Try npugraph_ex first, fall back to torchair for backward compatibility.
    try:
        import npugraph_ex as nge

        cache_path = os.path.join(cache_dir, key) if (cache_dir and key) else None
        torch.npu.set_compile_mode(jit_compile=False)
        config = nge.CompilerConfig()
        # _process_kwargs_options exists in both old and new npugraph_ex,
        # but in different modules: new -> compiler_config, old -> npugraphex_config.
        try:
            from npugraph_ex.configs.compiler_config import _process_kwargs_options
        except ImportError:
            from npugraph_ex.configs.npugraphex_config import _process_kwargs_options
        _configure_backend(
            config, ascend_compilation_config, vllm_config, process_kwargs_options=_process_kwargs_options
        )
        # SUBTRACTED: patched_get_compiled_gm 缓存巧思整段（原 :L151-L185）——猴补
        #             nfx._NpuFxCompiler._get_compiled_gm 把编译产物 py_code 写盘（含 triton_kernel_wrapper
        #             的图因 kernel_side_table 进程内本地、不可跨进程序列化而跳过）。缓存只影响二次启动速度。
        backend = nge.get_npu_backend(compiler_config=config)
        # torch.compile requires the output of the fx graph to be a tuple
        if not graph_returns_tuple(graph):
            compiled_fn = make_graph_return_tuple(graph, example_inputs, backend)
        else:
            compiled_fn = backend(graph, example_inputs)
        return compiled_fn, (key, cache_path)
    except ImportError:
        import torchair

        torch.npu.set_compile_mode(jit_compile=False)
        config = torchair.CompilerConfig()
        _configure_backend(config, ascend_compilation_config, vllm_config)
        backend = torchair.get_npu_backend(compiler_config=config)
        # torch.compile requires the output of the fx graph to be a tuple
        if not graph_returns_tuple(graph):
            compiled_fn = make_graph_return_tuple(graph, example_inputs, backend)
        else:
            compiled_fn = backend(graph, example_inputs)
        return compiled_fn, None


# SOURCE: vllm_ascend/compilation/compiler_interface.py:L202
class AscendCompiler(CompilerInterface):
    """
    AscendCompiler is a custom compiler interface for the Ascend platform.
    This class provides a method to compile a PyTorch FX graph module with
    specific configurations for graph fusion and decomposition.
    """

    name = "AscendCompiler"

    # TODO(wxs): add passes related to compilation in compute_hash
    # SOURCE: vllm_ascend/compilation/compiler_interface.py:L212
    def compute_hash(self, vllm_config: VllmConfig) -> str:
        self.vllm_config = vllm_config
        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        from hashlib import sha256

        import torch_npu

        factors = {
            "torch_npu_version": torch_npu.__version__,
            "enable_npugraph_ex": ascend_compilation_config.enable_npugraph_ex,
            "enable_static_kernel": ascend_compilation_config.enable_static_kernel,
        }
        logger.info("AscendCompiler hash factors: %s", factors)
        return sha256(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]

    # SOURCE: vllm_ascend/compilation/compiler_interface.py:L227
    def initialize_cache(self, cache_dir, disable_cache=False, prefix=""):
        self.cache_dir = cache_dir
        self.disable_cache = disable_cache

    # SOURCE: vllm_ascend/compilation/compiler_interface.py:L231
    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:
        # inductor can inplace modify the graph, so we need to copy it
        # see https://github.com/pytorch/pytorch/issues/138980
        graph = copy.deepcopy(graph)

        from torch._guards import detect_fake_mode

        current_fake_mode = detect_fake_mode()
        if current_fake_mode is not None:
            example_inputs = [
                current_fake_mode.from_tensor(inp)
                if (
                    isinstance(inp, torch.Tensor)
                    and hasattr(inp, "fake_mode")
                    and inp.fake_mode is not current_fake_mode
                )
                else inp
                for inp in example_inputs
            ]

        ascend_compilation_config = get_ascend_config().ascend_compilation_config
        if ascend_compilation_config.enable_npugraph_ex:
            cache_dir = None if getattr(self, "disable_cache", False) else getattr(self, "cache_dir", None)
            logger.info_once(
                "enable_npugraph_ex is enabled, which will bring graph compilation optimization.",
                scope="global",
            )
            assert hasattr(self, "vllm_config")
            return npugraph_ex_compile(
                graph,
                example_inputs,
                compiler_config,
                self.vllm_config,
                ascend_compilation_config,
                compile_range,
                key,
                cache_dir,
            )
        else:
            return fusion_pass_compile(graph, example_inputs, compiler_config, compile_range, key)

    # SUBTRACTED: load(self, handle, graph, ...)（原 :L279-L344）——编译产物缓存的读回：
    #             cache-miss（文件缺失/含 Triton kernel 未存）回退 npugraph_ex_compile 重编译；
    #             命中则从 py_code 反序列化 _CompiledFxGraph 并按需重建 unflatten 包装。
    #             缓存属正交优化，删除不改变 compile() 的二分编译语义。
