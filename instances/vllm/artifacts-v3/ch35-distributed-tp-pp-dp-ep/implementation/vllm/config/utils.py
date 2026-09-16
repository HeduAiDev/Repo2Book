# SOURCE: vllm/config/utils.py —— @config 装饰器逐字（pydantic dataclass 化）。
# 其余（hash/get_field/检查器族）属 ch03 装配线域，不携带。

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar, overload

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from typing_extensions import dataclass_transform

ConfigT = TypeVar("ConfigT")


# SOURCE: vllm/config/utils.py:L41-L80 config 装饰器 —— 逐字
@overload
# SOURCE: vllm/config/utils.py:L41-L41（锚点双置）
def config(cls: type[ConfigT]) -> type[ConfigT]: ...


@overload
@dataclass_transform(field_specifiers=(Any,))
def config(
    cls: None = None,
    *,
    config: Optional[ConfigDict] = None,
    **kwargs: Any,
# SOURCE: vllm/config/utils.py:L41-L41（锚点双置）
) -> Callable[[type[ConfigT]], type[ConfigT]]: ...


# SOURCE: vllm/config/utils.py config 本体 —— 逐字（默认 extra="forbid"）
def config(
    cls: type[ConfigT] | None = None,
    *,
    config: ConfigDict | None = None,
    **kwargs: Any,
) -> type[ConfigT] | Callable[[type[ConfigT]], type[ConfigT]]:
    """Decorator to create a pydantic dataclass with default config. The default config
    for the dataclass forbids extra fields.

    All config classes in vLLM should use this decorator.

    Args:
        cls: The class to decorate
        config: The pydantic ConfigDict to use for the dataclass. If provided, it will
            be merged with the default config.
        **kwargs: Additional arguments to pass to pydantic.dataclass."""
    # Extra fields are forbidden by default
    merged_config = ConfigDict(extra="forbid")
    if config is not None:
        merged_config.update(config)

    def decorator(cls: type[ConfigT]) -> type[ConfigT]:
        # SOURCE: vllm/config/utils.py:L73-L74 decorator（锚点双置）
        return dataclass(cls, config=merged_config, **kwargs)  # type: ignore[return-value]

    # Called with arguments: @config(config=...)
    if cls is None:
        return decorator
    # Called without arguments: @config
    return decorator(cls)
