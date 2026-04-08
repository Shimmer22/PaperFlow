from __future__ import annotations

from pathlib import Path

from research_flow.models import ProviderConfig
from research_flow.providers.api_provider import OpenAICompatibleAPIProvider
from research_flow.providers.base import BaseCLIProvider


def create_provider(config: ProviderConfig, workdir: Path) -> BaseCLIProvider:
    if config.provider_type != "openai_compatible_api":
        raise ValueError("Only openai_compatible_api provider_type is supported.")
    return OpenAICompatibleAPIProvider(config, workdir=workdir)
