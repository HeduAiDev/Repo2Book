# 两段式 monkey-patch —— 单一入口 + 四个触发点（subtract-only 精简版）
#
# 本文件汇集 spine 上的「入口/触发」骨架：
#   vllm_ascend/utils.py        : adapt_patch          —— 唯一入口
#   vllm_ascend/platform.py     : pre_register_and_update —— platform 段触发点①
#   vllm_ascend/__init__.py     : _ensure_global_patch + general_plugins —— platform 段触发点②
#   vllm_ascend/worker/worker.py: NPUWorker.__init__   —— worker 段触发点
#
# SUBTRACTED: 每个源文件中与 patch 主线无关的成员/语句已删（详见各处 # SUBTRACTED 注释）。
# 注：本文件 import 真实 vllm_ascend.patch 包，需在装有 vLLM 的环境/容器内才能 import；
#     纯语言级的「重绑定语义」由 tests/ 在 host 上验证。


# ============================ 单一入口 ============================
def adapt_patch(is_global_patch: bool = False):
    # SOURCE: vllm_ascend/utils.py:L511-L515
    # 不调用任何 apply()：patch 全靠 import 这两个包时执行的「模块级副作用」完成。
    if is_global_patch:
        from vllm_ascend.patch import platform  # noqa: F401
    else:
        from vllm_ascend.patch import worker  # noqa: F401


# ===================== platform 段触发点①：构图前、进程级 =====================
class NPUPlatform:
    # SOURCE: vllm_ascend/platform.py (NPUPlatform)
    # SUBTRACTED: 平台协议其余方法（device_type / get_attn_backend_cls /
    #   check_and_update_config 等数十个）已删，只留 pre_register_and_update。

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        # SOURCE: vllm_ascend/platform.py:L181-L186
        # Adapt the global patch here.
        from vllm_ascend.utils import adapt_patch

        adapt_patch(is_global_patch=True)
        # SUBTRACTED: 后续把 "ascend" quantization 注入 argparser、按 is_310p 导入量化
        #   配置、跑 config_deprecated_logging() —— 与 patch 主线无关 (platform.py:L187+)


# ===================== platform 段触发点②：general_plugins 入口（ch02 f1 回收） =====================
# SOURCE: vllm_ascend/__init__.py:L20
_GLOBAL_PATCH_APPLIED = False


def _ensure_global_patch():
    # SOURCE: vllm_ascend/__init__.py:L23-L37
    """Apply process-wide vLLM patches before engine-core initialization.

    vLLM loads general plugins in engine-core subprocesses. E2E test
    conftest hooks do not run there, so global patches that affect scheduler
    and engine code must also be applied through these plugin entry points.
    """
    # 进程级幂等守卫：多个 general_plugins 入口都会抢着触发，守卫保证 platform 段只跑一次。
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register_connector():
    # SOURCE: vllm_ascend/__init__.py:L46-L51
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector

    register_connector()


def register_model_loader():
    # SOURCE: vllm_ascend/__init__.py:L54-L61
    _ensure_global_patch()

    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    # SOURCE: vllm_ascend/__init__.py:L64-L69
    _ensure_global_patch()

    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()

# SUBTRACTED: register()（返回平台类路径）与 register_model()（只注册模型、不触发 patch）
#   不调 _ensure_global_patch，与本章主线无关，已删 (__init__.py:L40-L43, L72-L75)。


# ===================== worker 段触发点：每个 worker 进程构造时 =====================
class NPUWorker:
    # SOURCE: vllm_ascend/worker/worker.py (NPUWorker, 继承 WorkerBase)
    # SUBTRACTED: 仅保留 __init__ 中的 adapt_patch() 触发；其余 worker 方法已删。

    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        # Additional parameters for compatibility with vllm
        **kwargs,
    ):
        # SOURCE: vllm_ascend/worker/worker.py:L82-L102
        """Initialize the worker for Ascend."""
        # SUBTRACTED: COMPILE_CUSTOM_KERNELS 告警 (worker.py:L90-L97)
        # register patch for vllm
        from vllm_ascend.utils import adapt_patch

        adapt_patch()  # 默认 is_global_patch=False → 走 worker 包
        # SUBTRACTED: 后续 ops.register_dummy_fusion_op() / register_ascend_customop /
        #   init_ascend_config(vllm_config) 等 worker 构造逻辑 (worker.py:L104+)
