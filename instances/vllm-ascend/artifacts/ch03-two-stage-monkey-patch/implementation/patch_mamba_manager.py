# 技法②：工厂(注册表)替换 —— vllm_ascend/patch/platform/patch_mamba_manager.py（subtract-only）
#
# 招式：除把类名 MambaManager 重绑（整类替换）外，还必须改写 vLLM 按 KVCacheSpec 类型
#   派发 manager 的「工厂派发表」spec_manager_map[MambaSpec]——否则工厂仍按旧表 new 出原版。
#   「谁来 new 这个类」决定了必须连派发表一起改。
#
# SOURCE: vllm_ascend/patch/platform/patch_mamba_manager.py:L7-L15
import vllm.v1.core.single_type_kv_cache_manager as single_type_kv_cache_manager
from vllm.v1.core.single_type_kv_cache_manager import (
    BlockPool,
    MambaManager,
    MambaSpec,
)
# SUBTRACTED: BlockHashList / KVCacheBlock / KVCacheSpec import —— 仅 find_longest_cache_hit
#   形参/实现需要，随其一并折叠 (patch_mamba_manager.py:L9-L15)。


class AscendMambaManager(MambaManager):
    # SOURCE: vllm_ascend/patch/platform/patch_mamba_manager.py:L18-L49
    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs) -> None:
        # SOURCE: vllm_ascend/patch/platform/patch_mamba_manager.py:L19-L22
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        if self.enable_caching:
            self.block_size = kv_cache_spec.block_size

    @classmethod
    def find_longest_cache_hit(cls, *args, **kwargs):
        # SOURCE: vllm_ascend/patch/platform/patch_mamba_manager.py:L24-L49
        # SUBTRACTED: 块命中扫描实现（约 12 行：按 alignment 过滤、null_block 填充）与其完整
        #   形参列表 —— 属被替换类的业务体，与技法②无关 (patch_mamba_manager.py:L25-L49)。
        ...


# 招式核心：类名重绑 + 工厂派发表改写「双管齐下」。
single_type_kv_cache_manager.MambaManager = AscendMambaManager
single_type_kv_cache_manager.spec_manager_map[MambaSpec] = AscendMambaManager
