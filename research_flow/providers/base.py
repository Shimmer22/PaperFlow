from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel

from research_flow.models import ProviderCallResult, ProviderCapabilities


class BaseCLIProvider(ABC):
    """Abstract adapter for CLI-based agent providers."""

    @abstractmethod
    def check_available(self) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def describe_provider_capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    @abstractmethod
    def run_task(
        self,
        prompt: str,
        context: dict[str, Any],
        expected_output_schema: Optional[Type[BaseModel]],
        output_path: Optional[Path],
        timeout: int,
        extra_args: Optional[list[str]] = None,
        runtime_options: Optional[dict[str, str]] = None,
    ) -> ProviderCallResult:
        raise NotImplementedError

    def run_subtask(
        self,
        prompt: str,
        context: dict[str, Any],
        expected_output_schema: Optional[Type[BaseModel]],
        output_path: Optional[Path],
        timeout: int,
        extra_args: Optional[list[str]] = None,
        runtime_options: Optional[dict[str, str]] = None,
    ) -> ProviderCallResult:
        return self.run_task(prompt, context, expected_output_schema, output_path, timeout, extra_args, runtime_options)
