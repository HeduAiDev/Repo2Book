# vllm_ascend/platform.py —— subtract-only 精简版（昇腾后端路由总入口）
#
# 本章点(1)「后端路由」：NPUPlatform.get_attn_backend_cls 用 (use_mla, use_sparse, use_compress)
# 三元 key 查 backend_map，返回 4 个昇腾后端之一的**点分类路径字符串**（不是类对象——交给
# vLLM selector 的 resolve_obj_by_qualname 延迟解析，避免在平台层 import 全部重依赖后端）。
#   (True,  False, False) → AscendMLABackend        （MLA）
#   (False, False, False) → AscendAttentionBackend   （标准 MHA）
#   (True,  True,  False) → AscendSFABackend         （稀疏 SFA）
#   (True,  False, True ) → AscendDSABackend         （DSA）
#
# platform.py 其余数百行（环境/配置校验、worker 选择、内存统计等）与本章正交，整体折叠；
# 这里只摘 get_attn_backend_cls 这一段，host 用桩 attn_selector_config 即可走分支（见 ../tests）。
from vllm.v1.attention.backends.registry import AttentionBackendEnum


# SOURCE: vllm_ascend/platform.py:L738-L765
class NPUPlatform:
    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads: int | None = None):
        # SOURCE: vllm_ascend/platform.py:L738-L765
        # 第三维 use_compress 经 getattr 取——基座 AttentionSelectorConfig 当前只声明 use_mla/use_sparse，
        # 故 (True,False,True)→DSA 仅在带 use_compress 字段的更新版 vLLM 上可达（前向兼容写法）。
        use_compress = getattr(attn_selector_config, "use_compress", False)
        key = (attn_selector_config.use_mla, attn_selector_config.use_sparse)  # 喂给下方 FA3/310p 旁支

        # SUBTRACTED: FA3 早返回旁支（platform.py:L743-L744）及 _validate_fa3_backend 整个方法（L767-L792）——
        #   if selected_backend == AttentionBackendEnum.FLASH_ATTN and cls._validate_fa3_backend(key, ...):
        #       return "vllm_ascend.attention.fa3_v1.AscendFABackend"
        #   FA3 是训推一致场景的旁支后端，不在本章「4 后端选择 + 契约」主线；保留此注释占位即可。

        backend_map = {
            (True, False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",
            (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
            (True, True, False): "vllm_ascend.attention.sfa_v1.AscendSFABackend",
            (True, False, True): "vllm_ascend.attention.dsa_v1.AscendDSABackend",
        }

        # SUBTRACTED: 310p 推理卡旁支（platform.py:L752-L763）——backend_map_310 + `if is_310p(): return ...`。
        #   310p 是特定硬件型号特例，主路由（非 310p）走下方 backend_map；详见 ch17。保留此注释占位即可。

        return backend_map[(attn_selector_config.use_mla, attn_selector_config.use_sparse, use_compress)]
