# 技法①：整类替换 —— vllm_ascend/patch/platform/patch_multiproc_executor.py（subtract-only）
#
# 招式：子类继承原类整体重写关键方法，末行用子类直接覆盖模块属性 MultiprocExecutor。
#
# SOURCE: vllm_ascend/patch/platform/patch_multiproc_executor.py:L8-L20
import vllm.v1.executor.multiproc_executor
from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    WorkerProc,
)
# SUBTRACTED: weakref / deque / Lock / envs / VllmConfig / MessageQueue / 网络工具 /
#   FutureWrapper / UnreadyWorkerProcHandle 等被复制类体所需的 import (patch:L1-L20)。


class AscendMultiprocExecutor(MultiprocExecutor):
    # SOURCE: vllm_ascend/patch/platform/patch_multiproc_executor.py:L24-L209
    # SUBTRACTED: _init_executor / _distribute_work / make_worker_process 等约 180 行
    #   从 vLLM 原 MultiprocExecutor 大段复制的执行器内部实现（消息队列、worker 拉起、
    #   ready/death pipe、failure 清理）—— 与「整类替换」招式本身无关，是被替换类的业务体。
    #   核心差异只有一处：make_worker_process 里子进程以 daemon=False 启动。

    def make_worker_process(self, *args, **kwargs):
        # SOURCE: vllm_ascend/patch/platform/patch_multiproc_executor.py:L195-L209
        # SUBTRACTED: 组装 process_kwargs / pipe / 返回 UnreadyWorkerProcHandle 的细节。
        # Run EngineCore busy loop in background process.
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=False,  # 实质差异：daemon=True 不允许 EPLB 再 fork 子进程
        )
        ...


# 招式核心：用子类直接覆盖模块属性 —— 此后任意 `import ...multiproc_executor` 拿到的就是 Ascend 版。
vllm.v1.executor.multiproc_executor.MultiprocExecutor = AscendMultiprocExecutor
