# vllm_ascend/compilation/acl_graph.py —— subtract-only 精简版（ch14 配角）
#
# 本章只需要 ACLGraphWrapper 与父类 CUDAGraphWrapper "同形"（同构造签名、同
# _all_instances WeakSet、同 clear_all_graphs 语义）——正是这种鸭子兼容让
# _replace_gpu_model_runner_function_wrapper 能把父模块里的 CUDAGraphWrapper
# setattr 换成 ACLGraphWrapper，而父方法 `CUDAGraphWrapper(self.model, ...)` 不报错。
# 真正的 ACLGraph 捕获/重放实现属图模式章节，本章只展示"同形可热替换"。
import weakref
from typing import Callable, ClassVar


# SOURCE: vllm_ascend/compilation/acl_graph.py:L64
class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability."""

    # SOURCE: vllm_ascend/compilation/acl_graph.py:L89
    _all_instances: ClassVar["weakref.WeakSet[ACLGraphWrapper]"] = weakref.WeakSet()

    @classmethod
    def clear_all_graphs(cls) -> None:
        # SOURCE: vllm_ascend/compilation/acl_graph.py:L91
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    # SOURCE: vllm_ascend/compilation/acl_graph.py:L96
    def __init__(
        self,
        runnable: Callable,
        vllm_config,
        runtime_mode,
        cudagraph_options=None,
        *,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ):
        # 与父类 CUDAGraphWrapper.__init__ 逐字同形的构造签名——热替换成立的依据：
        # 父方法 `CUDAGraphWrapper(self.model, vllm_config, runtime_mode, ...)` 拿到
        # 的其实是 ACLGraphWrapper，因签名同形而不报错。
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        # SUBTRACTED: graph_pool / aclgraph_options / concrete_aclgraph_entries /
        #   first_run_finished / is_debugging_mode 等捕获状态初始化（acl_graph.py:L110-L130）
        #   —— ACLGraph 捕获细节属图模式章节，与"同形可替换"主题正交。
        self.concrete_aclgraph_entries: dict = {}
        ACLGraphWrapper._all_instances.add(self)

    # SOURCE: vllm_ascend/compilation/acl_graph.py:L153
    def clear_graphs(self) -> None:
        self.concrete_aclgraph_entries.clear()

    # SUBTRACTED: __getattr__ / unwrap / cudagraph_wrapper / __call__ 捕获重放派发
    #   （acl_graph.py:L127-L183）—— torch.npu.NPUGraph 捕获/重放属图模式章节。


# SOURCE: vllm_ascend/compilation/acl_graph.py:L333
def reset_graph_params():
    global _graph_params, _draft_graph_params, _draft_graph_prefill_params
    _graph_params = None
    _draft_graph_params = None
    _draft_graph_prefill_params = None


_graph_params = None
_draft_graph_params = None
_draft_graph_prefill_params = None
