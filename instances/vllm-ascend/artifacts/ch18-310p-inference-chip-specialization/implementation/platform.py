# vllm_ascend/platform.py —— subtract-only 精简版（ch17 横切回收：platform 入口 + attention 分流）
#
# 两个横切落点，由 is_310p() 在运行期分流：
#   (1) check_and_update_config 里把 parallel_config.worker_cls 指向 NPUWorker310——
#       这是"入口选择点"：单一 if 决定后面 ModelRunner310 / InputBatch310 / BlockTable310
#       全部被链式装配；且 310P 不启用 custom_ops=['all']（310P 无 custom op）。
#   (2) get_attn_backend_cls 里 backend_map_310 把 attention backend 在 is_310p 时改用
#       AscendAttentionBackend310——MLA/SFA 条目被注释掉 = 不支持，与 model_runner
#       initialize_kv_cache_tensors 主动 raise use_mla 形成闭环。
#
# 这里只摘 NPUPlatform 类的这两段；platform.py 其余配置校验/环境变量/算子注册等数百行
# 与本章正交，整体折叠。真实方法挂在 NPUPlatform 上、依赖 VllmConfig，host 不真跑；本文件
# 把两段抽成可读控制流（用桩 config 即可走分支，见 ../tests）。
from vllm.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, is_310p


# SOURCE: vllm_ascend/platform.py:L602-L618
def select_worker_cls_and_custom_ops(vllm_config, parallel_config, compilation_config, ascend_config):
    """check_and_update_config 中 worker_cls 选择 + custom_ops 开关的可读切片。"""
    if parallel_config and parallel_config.worker_cls == "auto":
        # TODO: this is a tricky way to disable `use_sequence_parallel_moe` in vllm.
        if not vllm_config.compilation_config.pass_config.enable_sp:
            parallel_config.all2all_backend = "flashinfer_all2allv"
        if is_310p():
            # 入口选择点：worker_cls 一旦指向 NPUWorker310，整条 310 栈被链式拉起。
            parallel_config.worker_cls = "vllm_ascend._310p.worker_310p.NPUWorker310"
        # SUBTRACTED: elif xlite_graph_config.enabled 分支（platform.py:L608-L610）——
        #   openEuler Xlite 与 310P 主题正交。
        else:
            parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"

    # SUBTRACTED: refresh_block_size(vllm_config)（L614）—— block size 刷新细节正交。

    # Activate custom ops for v1, except on 310P
    if get_ascend_device_type() != AscendDeviceType._310P:
        compilation_config.custom_ops = ["all"]


# SOURCE: vllm_ascend/platform.py:L738-L765
def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads: int | None = None):
    use_compress = getattr(attn_selector_config, "use_compress", False)
    key = (attn_selector_config.use_mla, attn_selector_config.use_sparse)

    # SUBTRACTED: FLASH_ATTN/FA3 早返回分支（platform.py:L743-L744）—— 与 310P 分流正交。

    backend_map = {
        (True, False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",
        (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
        (True, True, False): "vllm_ascend.attention.sfa_v1.AscendSFABackend",
        (True, False, True): "vllm_ascend.attention.dsa_v1.AscendDSABackend",
    }
    backend_map_310 = {
        (
            False,
            False,
        ): "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310",
        # TODO If MLA/SFA is supported in the future, consider implementing the logic described in these comments.
        # (True, False): "...AscendMLABackend310",
        # (True, True):  "...AscendSFABackend310",
    }

    if is_310p():
        # 310P 只有一个 (False,False) 条目，MLA/SFA 被注释掉 = 不支持。
        return backend_map_310.get(key, backend_map_310[(False, False)])

    return backend_map[(attn_selector_config.use_mla, attn_selector_config.use_sparse, use_compress)]
