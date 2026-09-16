# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/utils/mistral.py —— HOST SEAM（退化承载）：真实文件探测
# mistral_common tokenizer 并提供 mt 序列化工具族；dossier delete[5] 已把
# Mistral 特例（maybe_serialize_tool_calls/truncate_tool_call_ids/
# validate_request_params/_grammar_from_parser 分支/is_mistral_* 判定内支路）
# 从本章精简版删去——普通 HF tokenizer 路径不经过。保留的唯一消费点是
# render_chat 的 tool_parsing_unavailable 条件（online_renderer.py:L148）里
# 的 is_mistral_tokenizer(tokenizer) 调用：host 精简环境恒走 False 支路
# （HF tokenizer 场景的真实行为）。
# SUBTRACTED: mt（maybe_serialize_tool_calls 等）与 is_mistral_tool_parser
# ——delete[5] Mistral 特例。


# SOURCE: vllm/utils/mistral.py —— HOST SEAM：is_mistral_tokenizer 退化位
# （真实：isinstance(tokenizer, MistralTokenizer)；精简环境无 mistral
# tokenizer 类型，恒 False——与普通 HF tokenizer 行为一致）
def is_mistral_tokenizer(tokenizer) -> bool:
    return False
