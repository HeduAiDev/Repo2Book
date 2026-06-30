# 精简版（只做减法）— 对照真实源码 vllm_ascend/platform.py（节选三个编译相关平台钩子）
#
# 本章立意的核心：昇腾不改 vLLM 编译框架一行，只靠 NPUPlatform 上几个返回「类路径字符串」的
# 钩子，把 vLLM 的编译后端 / 图捕获包装器 / pass manager 整体换成 NPU 版。vLLM 编译框架启动期经
# current_platform.get_compile_backend() / get_static_graph_wrapper_cls() / get_pass_manager_cls()
# 拿到这些字符串，import 出对应类来用。
#
# 减法：NPUPlatform 类体其余数百行（设备能力探测、内存、通信器、attention backend 选择等，分散在
# 他章）全部 SUBTRACTED，仅截取这三个钩子 + pass_key 属性。
from vllm_ascend.utils import COMPILATION_PASS_KEY


# SUBTRACTED: NPUPlatform 类的其余成员（设备/内存/通信/attention 等，原 platform.py 全文）——
#             非本章编译栈焦点。下面用一个空壳类承载三个编译钩子，签名/返回值与真实 NPUPlatform 一致。
class NPUPlatform:
    # SOURCE: vllm_ascend/platform.py:L156
    @property
    def pass_key(self) -> str:
        # SOURCE: vllm_ascend/platform.py:L157
        """
        Inductor config key for the PassManager custom pass, for example 'post_grad_custom_post_pass'.
        It is a parameter of inductor_config used to register custom passes.
        Currently, we only use Inductor's 'pattern matcher' functionality, so we define our own pass_key.
        """
        return COMPILATION_PASS_KEY

    # SOURCE: vllm_ascend/platform.py:L165
    @classmethod
    def get_pass_manager_cls(cls) -> str:
        # SOURCE: vllm_ascend/platform.py:L166
        """
        Get the pass manager class for this platform.
        It will be registered as a custom pass under the current_platform.pass_key.
        """
        return "vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager"

    # SOURCE: vllm_ascend/platform.py:L173
    @classmethod
    def get_compile_backend(self) -> str:
        # SOURCE: vllm_ascend/platform.py:L174
        """
        Get the custom compile backend. Previously, we used EagerAdaptor by default.
        To use graph fusion operations, we defined our own backend compiler.
        """
        return "vllm_ascend.compilation.compiler_interface.AscendCompiler"

    # SOURCE: vllm_ascend/platform.py:L816
    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        # SOURCE: vllm_ascend/platform.py:L817
        """
        Get piecewise backend class for piecewise graph.
        """
        return "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"  # noqa
