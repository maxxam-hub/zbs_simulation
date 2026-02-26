from __future__ import annotations

from .models import BaseConfig


def default_config() -> BaseConfig:
    """
    Возвращает базовую конфигурацию модели.

    Роль в проекте:
    - Единая точка с дефолтными параметрами для быстрого запуска MVP.
    """
    return BaseConfig()
