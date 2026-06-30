# 精简版（只做减法）— 对照真实源码 vllm_ascend/compilation/graph_fusion_pass_manager.py
#
# GraphFusionPassManager 对位 vLLM 的 PostGradPassManager（platform.get_pass_manager_cls()
# 返回它、注册到 current_platform.pass_key=COMPILATION_PASS_KEY）。注释自陈：torch_npu 暂不
# 支持 triton，故昇腾自定义一个只跑 pattern matcher 的精简 pass manager。
# 本文件按真实源码原样保留（无删减）。
from torch import fx as fx
from vllm.compilation.passes.inductor_pass import get_pass_context
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass
from vllm.config import VllmConfig


# SOURCE: vllm_ascend/compilation/graph_fusion_pass_manager.py:L25
class GraphFusionPassManager:
    """
    A pass manager for graph fusion passes.
    It handles the configuration and execution of passes.
    The counterpart in vllm is PostGradPassManager. Since torch_npu
    does not support triton for now, we define our own pass manager.
    """

    def __init__(self):
        # SOURCE: vllm_ascend/compilation/graph_fusion_pass_manager.py:L33
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph) -> fx.Graph:
        # SOURCE: vllm_ascend/compilation/graph_fusion_pass_manager.py:L36
        compile_range = get_pass_context().compile_range

        for pass_ in self.passes:
            if pass_.is_applicable_for_range(compile_range):
                pass_(graph)
        graph.recompile()
        return graph

    def add(self, pass_: VllmInductorPass):
        # SOURCE: vllm_ascend/compilation/graph_fusion_pass_manager.py:L45
        assert isinstance(pass_, VllmInductorPass)
        self.passes.append(pass_)

    def configure(self, config: VllmConfig):
        # SOURCE: vllm_ascend/compilation/graph_fusion_pass_manager.py:L49
        from vllm_ascend.utils import is_310p

        # By default, we enable the graph fusion and quantization fusion pass.
        self.ascend_compilation_config: dict = config.additional_config.get("ascend_compilation_config", {})
        if self.ascend_compilation_config.get("fuse_norm_quant", True) and not is_310p():
            from .passes.norm_quant_fusion_pass import AddRMSNormQuantFusionPass

            self.passes.append(AddRMSNormQuantFusionPass(config))

        if self.ascend_compilation_config.get("fuse_qknorm_rope", True):
            from .passes.qknorm_rope_fusion_pass import QKNormRopeFusionPass

            self.passes.append(QKNormRopeFusionPass(config))

        if self.ascend_compilation_config.get("fuse_allreduce_rms", True):
            from .passes.allreduce_rmsnorm_fusion_pass import MatmulAllReduceAddRMSNormPass

            self.passes.append(MatmulAllReduceAddRMSNormPass(config))

        if self.ascend_compilation_config.get("fuse_muls_add", True) and not is_310p():
            from .passes.muls_add_pass import MulsAddFusionPass

            self.passes.append(MulsAddFusionPass(config))

        if config.compilation_config.pass_config.enable_sp:
            from .passes.sequence_parallelism import SequenceParallelismPass
            from .passes.sequence_parallelism_moe import SequenceParallelismMoePass

            self.passes.append(SequenceParallelismPass(config))
            self.passes.append(SequenceParallelismMoePass(config))
