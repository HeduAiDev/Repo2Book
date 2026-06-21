import pytest


def pytest_configure(config):
    # 让 @pytest.mark.asyncio 生效，无需外部 pyproject 配置。
    config.option.asyncio_mode = "auto"
