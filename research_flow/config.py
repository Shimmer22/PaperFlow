from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

from research_flow.models import AppConfig, ProviderConfig


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_app_config(path: Union[str, Path]) -> AppConfig:
    return AppConfig.model_validate(load_yaml(path))


def load_provider_config(path: Union[str, Path]) -> ProviderConfig:
    return ProviderConfig.model_validate(load_yaml(path))
