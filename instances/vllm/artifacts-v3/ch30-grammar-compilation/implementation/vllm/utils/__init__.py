# SOURCE: vllm/utils/__init__.py
# HOST SEAM：包标记。真实 vllm/utils/__init__.py 从子模块聚合数百个工具
# （vllm/utils/__init__.py 的巨型 re-export 面）——本章消费面走精确模块路径
# （vllm.utils.import_utils.LazyLoader / vllm.utils.mistral.is_mistral_tokenizer），
# 顶层聚合面不进精简版。
