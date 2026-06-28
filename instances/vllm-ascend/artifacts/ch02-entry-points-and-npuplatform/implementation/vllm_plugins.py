"""Subtract-only companion — entry-point 发现机制。

规范源码：vllm/plugins/__init__.py

关键点：plugin.load() 只把 entry-point 指向的对象（这里是 register 函数）导进来，
**并不调用它**——调用发生在上层 resolve_current_platform_cls_qualname 的 func()。
"""
from typing import Any, Callable


# SOURCE: vllm/plugins/__init__.py:L28-L66
def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    """Load plugins registered under the given entry point group."""
    from importlib.metadata import entry_points

    # SUBTRACTED: 原为 `allowed_plugins = envs.VLLM_PLUGINS`（vllm.envs 的可选白名单；
    #   None=加载全部，非 None=只加载名字在白名单里的插件）。原 vllm/plugins/__init__.py:L31
    allowed_plugins = None

    discovered_plugins = entry_points(group=group)
    if len(discovered_plugins) == 0:
        # SUBTRACTED: logger.debug("No plugins for group %s found.", group)
        return {}

    # SUBTRACTED: 原 L40-L52 一段纯日志（打印发现到的插件清单 plugin.name->plugin.value、
    #   是否受 VLLM_PLUGINS 过滤），不影响控制流。原 vllm/plugins/__init__.py:L40-L52
    plugins = dict[str, Callable[[], Any]]()
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            try:
                func = plugin.load()
                plugins[plugin.name] = func
            except Exception:
                # SUBTRACTED: logger.exception("Failed to load plugin %s", plugin.name)
                pass

    return plugins
