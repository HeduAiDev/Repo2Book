# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/reasoning/abs_reasoning_parsers.py —— 忠实承载：ReasoningParser
# 抽象基类 + ReasoningParserManager 注册表逐字（统一 Parser 层 WC6 的两轴
# 之一；adjust_initial_state_from_prompt 是 m10「思考内续写的初态校正」
# 钩子）。ToolServer 导入与 prepare_structured_tag 的 MCP 面删（MCP 工具
# 服务器域，本章不触达）。
import importlib
import os
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, cast

from vllm.logger import init_logger
from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import import_from_path

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.tokenizers import TokenizerLike

logger = init_logger(__name__)

# SUBTRACTED: vllm/reasoning/abs_reasoning_parsers.py:L11 mcp ToolServer 导入
# ——prepare_structured_tag 的 MCP 参数注解随该方法一并删（MCP 域）；
# ResponsesRequest 注解随 delete[13] 退化为 ChatCompletionRequest。


# SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L26-L210 —— ReasoningParser 逐字
class ReasoningParser:
    """
    Abstract reasoning parser class that should not be used directly.
    Provided and methods should be used in derived classes.

    It is used to extract reasoning content from the model output.
    """

    engine_based_streaming: bool = False

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L36-L40
    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        self.model_tokenizer = tokenizer
        # Optional vLLM ModelConfig from the server. Use get (not pop) so composite
        # parsers can forward **kwargs to nested parsers.
        self._model_config: ModelConfig | None = kwargs.get("model_config")

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L42-L46
    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only TokenizersBackend is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L48-L54
    @property
    def reasoning_start_str(self) -> str | None:
        """Set `reasoning_start_str` to the strings that delimit
        the reasoning block (e.g. `""<seed:think>""` and `"<think>"`).
        """
        return None

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L56-L62
    @property
    def reasoning_end_str(self) -> str | None:
        """Set `reasoning_end_str` to the strings that delimit
        the reasoning block (e.g. `""</seed:think>""` and `"</think>"`).
        """
        return None

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L62-L71
    def has_engine_confirmed_reasoning_end(self) -> bool:
        """Whether the engine has confirmed the reasoning end transition.

        Engine-based parsers may defer terminal processing when the
        detokenizer holds back text.  This method returns the engine's
        *processed* state, not a raw token-ID check.

        Only called for parsers with ``engine_based_streaming = True``.
        """
        return False

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L73-L88
    @abstractmethod
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """
        Check if the reasoning content ends in the input_ids.

        It is used in structured engines like `xgrammar` to check if
        the reasoning content ends in the model output.

        Parameters:
        input_ids: list[int]
            The input_ids of the model output.

        Returns:
        bool
            True if the reasoning content ends in the input_ids.
        """

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L90-L113
    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        """
        Check if the reasoning content ends in the input_ids on a
        decode step.

        It is used in structured engines like `xgrammar` to check if
        the reasoning content ends in the model output during a decode step.
        `input_ids` the entire model output and `delta_ids` are the last few
        computed tokens of the model output (like during a decode step).

        Parameters:
        input_ids: list[int]
            The entire model output.
        delta_ids: list[int]
            The last few computed tokens of the model output at the current decode step.

        Returns:
        bool
            True if the reasoning content ends in the `delta_ids` on a decode step.
        """
        return self.is_reasoning_end(input_ids)

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L115-L125
    @abstractmethod
    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        Parameters:
        input_ids: list[int]
            The input_ids of the model output.
        Returns:
        list[int]
            The extracted content from the input_ids.
        """

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L127-L144
    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        """Count the number of reasoning tokens in a sequence.

        Text-based reasoning models typically wrap their chain-of-thought
        between special start/end tokens (e.g., ``<think> ... </think>``).
        Implementations that support reasoning token counting should override
        this method. The default implementation returns ``0`` so existing
        parsers remain unchanged unless they explicitly opt in.

        Args:
            token_ids: Sequence of generated token ids (excluding prompt).

        Returns:
            int: Number of tokens that belong to reasoning content.
        """

        # By default, assume the parser cannot detect reasoning spans.
        return 0

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L146-L164
    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest",
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
            model_output: The model-generated string to extract reasoning content from.
            request: The request object that was used to generate the model_output.

        Returns:
            A tuple containing the reasoning content and the content.
        """

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L166-L182
    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        """
        Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L184-L188
    def adjust_request(
        self, request: "ChatCompletionRequest"
    ) -> "ChatCompletionRequest":
        """Adjust request parameters; override in subclasses as needed."""
        return request

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L190-L200 —— m10 初态校正钩子
    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        """Hook called once at the start of streaming with the prompt tokens.

        Gives parsers a chance to adjust their initial parsing state based on
        the prompt — for example, when the chat template leaves the prompt
        inside an open reasoning channel and the engine's default initial
        state would otherwise misclassify the first generated tokens.

        Default is a no-op; override in subclasses as needed.
        """
        return

    # SUBTRACTED: vllm/reasoning/abs_reasoning_parsers.py:L202-L210
    # prepare_structured_tag（MCP 工具服务器的结构化标签预处理）——MCP 域。


# SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L213-L375 —— ReasoningParserManager 逐字
class ReasoningParserManager:
    """
    Central registry for ReasoningParser implementations.

    Supports two registration modes:
      - Eager registration via `register_module`
      - Lazy registration via `register_lazy_module`

    Each reasoning parser must inherit from `ReasoningParser`.
    """

    reasoning_parsers: dict[str, type[ReasoningParser]] = {}
    lazy_parsers: dict[str, tuple[str, str]] = {}  # name -> (module_path, class_name)

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L227-L247
    @classmethod
    def get_reasoning_parser(cls, name: str) -> type[ReasoningParser]:
        """
        Retrieve a registered or lazily registered ReasoningParser class.

        If the parser is lazily registered, it will be imported and cached
        on first access.

        Raises:
            KeyError: if no parser is found under the given name.
        """
        if name in cls.reasoning_parsers:
            return cls.reasoning_parsers[name]

        if name in cls.lazy_parsers:
            return cls._load_lazy_parser(name)

        registered = ", ".join(cls.list_registered())
        raise KeyError(
            f"Reasoning parser '{name}' not found. Available parsers: {registered}"
        )

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L249-L252
    @classmethod
    def list_registered(cls) -> list[str]:
        """Return names of all eagerly and lazily registered reasoning parsers."""
        return sorted(set(cls.reasoning_parsers.keys()) | set(cls.lazy_parsers.keys()))

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L254-L275
    @classmethod
    def _load_lazy_parser(cls, name: str) -> type[ReasoningParser]:
        """Import and register a lazily loaded reasoning parser."""
        module_path, class_name = cls.lazy_parsers[name]
        try:
            mod = importlib.import_module(module_path)
            parser_cls = getattr(mod, class_name)
            if not issubclass(parser_cls, ReasoningParser):
                raise TypeError(
                    f"{class_name} in {module_path} is not a ReasoningParser subclass."
                )

            cls.reasoning_parsers[name] = parser_cls  # cache
            return parser_cls
        except Exception as e:
            logger.exception(
                "Failed to import lazy reasoning parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L277-L303
    @classmethod
    def _register_module(
        cls,
        module: type[ReasoningParser],
        module_name: str | list[str] | None = None,
        force: bool = True,
    ) -> None:
        """Register a ReasoningParser class immediately."""
        if not issubclass(module, ReasoningParser):
            raise TypeError(
                f"module must be subclass of ReasoningParser, but got {type(module)}"
            )

        if module_name is None:
            module_names = [module.__name__]
        elif isinstance(module_name, str):
            module_names = [module_name]
        elif is_list_of(module_name, str):
            module_names = module_name
        else:
            raise TypeError("module_name must be str, list[str], or None.")

        for name in module_names:
            if not force and name in cls.reasoning_parsers:
                existed = cls.reasoning_parsers[name]
                raise KeyError(f"{name} is already registered at {existed.__module__}")
            cls.reasoning_parsers[name] = module

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L305-L317
    @classmethod
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:
        """
        Register a lazy module mapping for delayed import.

        Example:
            ReasoningParserManager.register_lazy_module(
                name="qwen3",
                module_path="vllm.reasoning.qwen3_engine_reasoning_parser",
                class_name="Qwen3ParserReasoningAdapter",
            )
        """
        cls.lazy_parsers[name] = (module_path, class_name)

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L319-L358
    @classmethod
    def register_module(
        cls,
        name: str | list[str] | None = None,
        force: bool = True,
        module: type[ReasoningParser] | None = None,
    ) -> (
        type[ReasoningParser] | Callable[[type[ReasoningParser]], type[ReasoningParser]]
    ):
        """
        Register module with the given name or name list. it can be used as a
        decoder(with module as none) or normal function(with module as not
        None).
        """
        if not isinstance(force, bool):
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # Immediate registration (explicit call)
        if module is not None:
            cls._register_module(module=module, module_name=name, force=force)
            return module

        # Decorator usage
        def _decorator(obj: type[ReasoningParser]) -> type[ReasoningParser]:
            module_path = obj.__module__
            class_name = obj.__name__

            if isinstance(name, str):
                names = [name]
            elif is_list_of(name, str):
                names = cast(list[str], name)
            else:
                names = [class_name]

            for n in names:
                cls.lazy_parsers[n] = (module_path, class_name)

            return obj

        return _decorator

    # SOURCE: vllm/reasoning/abs_reasoning_parsers.py:L360-L374
    @classmethod
    def import_reasoning_parser(cls, plugin_path: str) -> None:
        """
        Import a user-defined reasoning parser by the path
        of the reasoning parser define file.
        """
        module_name = os.path.splitext(os.path.basename(plugin_path))[0]

        try:
            import_from_path(module_name, plugin_path)
        except Exception:
            logger.exception(
                "Failed to load module '%s' from %s.", module_name, plugin_path
            )
            return
