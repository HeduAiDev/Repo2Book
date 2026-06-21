"""InputPreprocessor —— deprecated raw-prompt 兜底路径（精简版）。

真实主路径已由 Renderer (render_cmpl/render_chat) 完成 tokenize/多模态/embeds 渲染，
InputProcessor 主线吃**已渲染**的 EngineInput dict。只有把 raw prompt 直接喂给
InputProcessor 的 deprecated 路径才会现场调 InputPreprocessor.preprocess() 做 tokenize
（将于 v0.18 移除）。本文件保留这条兜底路径的**控制流骨架**：enc-dec / decoder-only
分流 → _prompt_to_llm_inputs 按 embeds/tokens/text 分派 → _process_text 调
_tokenize_prompt 出 token ids，封装 TokensInput。
"""

from __future__ import annotations

from typing import Any

from messages import length_from_prompt_token_ids_or_embeds  # noqa: F401  (供叙事引用)


def tokens_input(prompt_token_ids: list[int]) -> dict:
    # SOURCE: vllm/inputs/engine.py (tokens_input)
    return {"type": "token", "prompt_token_ids": prompt_token_ids}


class InputPreprocessor:
    # SOURCE: vllm/inputs/preprocess.py (class InputPreprocessor)
    def __init__(self, vllm_config, renderer=None, mm_registry=None):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.renderer = renderer
        # SUBTRACTED: mm_registry/tokenizer 持有等 — 兜底路径细节，
        #   原 vllm/inputs/preprocess.py:__init__。

    # SOURCE: vllm/inputs/preprocess.py (_tokenize_prompt)
    def _tokenize_prompt(self, prompt_text: str,
                         tokenization_kwargs: dict[str, Any] | None = None) -> list[int]:
        # 真实版委托 Renderer/tokenizer 编码；精简版用 renderer.tokenizer.encode。
        tok = None if self.renderer is None else self.renderer.tokenizer
        if tok is None:
            raise ValueError("raw-prompt fallback requires a tokenizer")
        return tok.encode(prompt_text, add_special_tokens=True)

    # SOURCE: vllm/inputs/preprocess.py:L161 (_process_text)
    def _process_text(self, parsed_content: dict,
                      tokenization_kwargs: dict[str, Any] | None = None) -> dict:
        prompt_text = parsed_content["prompt"]
        # SUBTRACTED: multi_modal_data 分支走 _process_multimodal（委托 Renderer）
        #   — 兜底路径的多模态委托，原 vllm/inputs/preprocess.py:L169-L175。
        prompt_token_ids = self._tokenize_prompt(
            prompt_text, tokenization_kwargs=tokenization_kwargs
        )
        inputs = tokens_input(prompt_token_ids)
        inputs["prompt"] = prompt_text
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt
        return inputs

    # SOURCE: vllm/inputs/preprocess.py:L211 (_prompt_to_llm_inputs)
    def _prompt_to_llm_inputs(self, prompt: dict,
                              tokenization_kwargs: dict[str, Any] | None = None) -> dict:
        if "prompt_embeds" in prompt:
            # SUBTRACTED: _process_embeds（委托 Renderer），原 preprocess.py:L216-L217。
            return {"type": "embeds", "prompt_embeds": prompt["prompt_embeds"]}
        if "prompt_token_ids" in prompt:
            # SUBTRACTED: _process_tokens 内部细节，原 preprocess.py:L219-L220。
            return tokens_input(list(prompt["prompt_token_ids"]))
        if "prompt" in prompt:
            return self._process_text(
                prompt, tokenization_kwargs=tokenization_kwargs
            )
        raise AssertionError(f"Unrecognized prompt: {prompt!r}")

    # SOURCE: vllm/inputs/preprocess.py:L264 (_process_decoder_only_prompt)
    def _process_decoder_only_prompt(self, prompt: dict,
                                     tokenization_kwargs=None) -> dict:
        return self._prompt_to_llm_inputs(
            prompt, tokenization_kwargs=tokenization_kwargs
        )

    # SOURCE: vllm/inputs/preprocess.py:L274 (preprocess)
    def preprocess(self, prompt, tokenization_kwargs: dict[str, Any] | None = None) -> dict:
        """Preprocess the input prompt."""
        if getattr(self.model_config, "is_encoder_decoder", False):
            # SUBTRACTED: _process_encoder_decoder_prompt 完整实现（enc/dec 复合 +
            #   decoder_start_token） — 兜底路径的 enc-dec 分支，
            #   原 vllm/inputs/preprocess.py:L230-L262。保留分流判断本身。
            raise NotImplementedError(
                "enc-dec raw-prompt fallback omitted in companion"
            )
        # 精简版接受 dict 形式的 raw prompt（{'prompt': str} / {'prompt_token_ids': ...}）。
        return self._process_decoder_only_prompt(
            prompt, tokenization_kwargs=tokenization_kwargs
        )
