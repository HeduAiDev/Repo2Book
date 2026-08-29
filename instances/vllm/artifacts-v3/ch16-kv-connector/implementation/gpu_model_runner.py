# SOURCE: vllm/v1/worker/gpu_model_runner.py
# worker 集成点三处（m7 的宿主挂点；执行主体归 ch17-20）：
#   execute_model 入口 handle_preemptions（L4197-L4200——覆写前抢救，
#   在 _update_states（块被复用/清零/CoW）之前）；
#   无 token 步分支（L4231-L4234——kv_connector_no_forward：无工作也走
#   KV 收发）；
#   _model_forward 的 connector 包裹（L4420-L4456——maybe_get_kv_connector_
#   output 与 set_forward_context 一起包住前向：start_load_kv 在首层前、
#   wait_for_save 在整个前向后；spec decode defer_finalize）；
#   register_kv_caches 装配点（L7669-L7681——池张量按层名注册给
#   connector，m14 的『worker 从此能直写 GPU 内存』）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 11 条执行主体：cudagraph/ubatch/DP/pooling/PP 广播/EPLB/
#     spec 提案器/持久批处理/routed experts/编码器——本章 worker 面只
#     保留 connector 生命周期挂点；
#   第 5 条 cross_layers 分支（L7671-L7678——布局族）与 set_host_xfer_
#     buffer_ops（L7681）；_update_states 的块表增量主体（零清账位以
#     注释保留——ch15 的清零面）。
from typing import TYPE_CHECKING, Any

from .forward_context import set_forward_context
from .kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from .kv_transfer_state import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)

if TYPE_CHECKING:
    from .config import VllmConfig
    from .kv_cache_interface import KVCacheConfig
    from .output import SchedulerOutput
    from .outputs import ModelRunnerOutput


# SOURCE: vllm/v1/worker/gpu_model_runner.py GPUModelRunner（切面：KV
#   connector 挂点宿主——真实类为完整 runner，ch17-20 全文）
class GPUModelRunner(KVConnectorModelRunnerMixin):
    # SOURCE: vllm/v1/worker/gpu_model_runner.py（__init__ 切面账位）
    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        # SUBTRACTED: 模型装载/attention backend/持久批/cudagraph/池化
        #   装配面（ch17-20）。speculative_config 账位 None（defer 判定）。
        self.speculative_config = None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4166 execute_model（切面：
    #   connector 三挂点）
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput":
        # SUBTRACTED: spec 修正/spec 采样重排（L4190-L4195——ch33）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4197-L4200 挂点①：
        #   handle_preemptions 在 _update_states（块被复用/覆写）之前
        if has_kv_transfer_group():
            kv_connector_metadata = scheduler_output.kv_connector_metadata
            assert kv_connector_metadata is not None
            get_kv_transfer_group().handle_preemptions(kv_connector_metadata)

        # SUBTRACTED: preprocess/_update_states/输入装配主体（L4201-L4419
        #   ——ch17/18；零清/CoW 的 worker 侧执行归 ch15）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4231-L4234 挂点②：
        #   无 token 步——无工作也走 KV 收发（P/D push 型传输的保活步）
        if scheduler_output.total_num_scheduled_tokens == 0:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if no work to do.
                from .outputs import EMPTY_MODEL_RUNNER_OUTPUT

                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4420-L4456 挂点③：
        #   connector 包裹住 _model_forward（spec decode 时 defer_finalize）
        return self._forward_with_connector(scheduler_output)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4420-L4456（ENGINE SEAM：
        #   从 execute_model 内联块抽出为方法以便单测，控制流逐字——
        #   EPLB 段删（dossier elide 批准））
    def _forward_with_connector(
        self, scheduler_output: "SchedulerOutput"
    ) -> "ModelRunnerOutput":
        # Run the model.
        # Use persistent buffers for CUDA graphs.
        # When spec decode is enabled, defer connector finalization
        # (wait_for_save + clear metadata) until after draft model runs.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4424-L4425
        defer_kv_connector_finalize = self.speculative_config is not None
        # SUBTRACTED: EPLB prepare_forward（L4427-L4431——与本机制无关）。
        with (
            set_forward_context(
                None,
                self.vllm_config,
            ),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                defer_finalize=defer_kv_connector_finalize,
            ) as kv_connector_output,
        ):
            self._model_forward(scheduler_output)

        # SUBTRACTED: postprocess/输出装配主体（L4457 起——ch17）。
        from .outputs import ModelRunnerOutput as _MRO

        return _MRO.with_kv_conn_output_only(kv_connector_output)

    # _model_forward（ENGINE SEAM：真实为模型执行主体——本章切面以逐层
    #   钩子装饰过的层回放替身驱动（host CPU）；控制流面 = execute_model
    #   的调用序）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py _model_forward
    def _model_forward(self, scheduler_output: "SchedulerOutput") -> None:
        # SUBTRACTED: 模型执行主体（ch17-20）。本章 worker 面的层回放由
        #   测试/explainer 以 maybe_transfer_kv_layer 装饰的层函数驱动。
        return None


# SOURCE: vllm/v1/worker/gpu_worker.py:L662 Worker 装配点（ENGINE SEAM：
#   ensure_kv_transfer_initialized 在 KV cache 配置就绪后、initialize_kv_
#   cache 之前调用——worker 侧 connector 的出生点）
class Worker:
    # SOURCE: vllm/v1/worker/gpu_worker.py（__init__ 切面）
    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.model_runner = GPUModelRunner(vllm_config)

    # SOURCE: vllm/v1/worker/gpu_worker.py:L655-L663 initialize_from_config
    #   （切面：connector 装配序——先于 initialize_kv_cache）
    def initialize_from_config(self, kv_cache_config: "KVCacheConfig") -> None:
        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g., kv cache sharing layers).
        # SOURCE: vllm/v1/worker/gpu_worker.py:L662
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        # SUBTRACTED: initialize_kv_cache 编排（L664-L679——ch14/17）。

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7669-L7681 register_kv_
    #   caches 装配点（m14：池张量按层名注册——非 cross-layer 路径）
    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        # SUBTRACTED: cross_layers 分支与 set_host_xfer_buffer_ops
        #   （L7671-L7681——第 5 条布局族）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7679-L7680
        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            kv_transfer_group.register_kv_caches(kv_caches)
