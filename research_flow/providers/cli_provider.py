from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

from research_flow.models import ProviderCallResult, ProviderCapabilities, ProviderConfig
from research_flow.providers.base import BaseCLIProvider
from research_flow.utils import command_exists, env_with_updates, write_json


class GenericCLIProvider(BaseCLIProvider):
    """Config-driven provider adapter for external CLI agents."""

    def __init__(self, config: ProviderConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir

    def check_available(self) -> tuple[bool, str]:
        check_cmd = self.config.availability_check or [self.config.command, "--help"]
        if not command_exists(check_cmd[0]):
            return False, f"Command not found: {check_cmd[0]}"
        try:
            result = subprocess.run(
                check_cmd,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=20,
                env=self._provider_env(),
            )
        except Exception as exc:
            return False, str(exc)
        if result.returncode == 0:
            text = (result.stdout or result.stderr).strip()
            return True, text
        return False, (result.stderr or result.stdout or "availability check failed").strip()

    def describe_provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.config.name,
            supports_subtasks=self.config.supports_subtasks,
            supports_parallel_invocations=self.config.supports_parallel_invocations,
            supports_output_schema=self.config.supports_output_schema,
            prompt_mode=self.config.prompt_mode,
            output_mode=self.config.output_mode,
        )

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
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
        with tempfile.TemporaryDirectory(prefix="research-flow-provider-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            context_path = temp_dir / "context.json"
            write_json(context_path, context)

            schema_path: Optional[Path] = None
            if expected_output_schema and self.config.supports_output_schema and self.config.schema_flag:
                schema_path = temp_dir / "schema.json"
                write_json(schema_path, expected_output_schema.model_json_schema())

            final_output_path = output_path or (self.workdir / ".cache" / "research-flow" / "provider_last_output.txt")
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            command = [self.config.command, *self.config.args, *self.config.default_extra_args]
            if runtime_options:
                command.extend(self._build_runtime_args(runtime_options))
            if extra_args:
                command.extend(extra_args)
            if self.config.output_mode == "file" and self.config.output_flag:
                command.extend([self.config.output_flag, str(final_output_path)])
            if schema_path and self.config.schema_flag:
                command.extend([self.config.schema_flag, str(schema_path)])
            if self.config.prompt_mode == "file":
                prompt_path = temp_dir / "prompt.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                command.append(str(prompt_path))
            elif self.config.prompt_mode == "arg":
                command.append(prompt)

            try:
                completed = subprocess.run(
                    command,
                    cwd=self.workdir,
                    input=prompt if self.config.prompt_mode == "stdin" else None,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=self._provider_env(),
                )
            except subprocess.TimeoutExpired as exc:
                raw_output = ""
                if self.config.output_mode == "file" and final_output_path.exists():
                    raw_output = final_output_path.read_text(encoding="utf-8").strip()
                return ProviderCallResult(
                    success=False,
                    raw_output=raw_output,
                    stderr=(exc.stderr or "").strip() if isinstance(exc.stderr, str) else "",
                    command=command,
                    error=f"provider timed out after {timeout} seconds",
                    output_path=str(final_output_path),
                )
            raw_output = completed.stdout.strip()
            if self.config.output_mode == "file" and final_output_path.exists():
                raw_output = final_output_path.read_text(encoding="utf-8").strip()
            if completed.returncode != 0:
                return ProviderCallResult(
                    success=False,
                    raw_output=raw_output,
                    stderr=completed.stderr.strip(),
                    command=command,
                    error=f"provider exited with code {completed.returncode}",
                    output_path=str(final_output_path),
                )

            parsed_output = None
            if expected_output_schema:
                parsed_output = self._parse_schema_output(raw_output, expected_output_schema)
            return ProviderCallResult(
                success=True,
                raw_output=raw_output,
                parsed_output=parsed_output,
                stderr=completed.stderr.strip(),
                command=command,
                output_path=str(final_output_path),
            )

    def _parse_schema_output(
        self,
        raw_output: str,
        expected_output_schema: Type[BaseModel],
    ) -> Optional[dict[str, Any]]:
        if not raw_output:
            return None
        parsed = None
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(raw_output[start : end + 1])
        if parsed is None:
            return None
        return expected_output_schema.model_validate(parsed).model_dump()

    def _build_runtime_args(self, runtime_options: dict[str, str]) -> list[str]:
        args: list[str] = []
        model = (runtime_options.get("model") or "").strip()
        reasoning = (runtime_options.get("reasoning_effort") or "").strip()
        if model and self.config.model_args_template:
            args.extend(self._expand_template(self.config.model_args_template, model))
        if reasoning and self.config.reasoning_args_template:
            args.extend(self._expand_template(self.config.reasoning_args_template, reasoning))
        return args

    @staticmethod
    def _expand_template(template: list[str], value: str) -> list[str]:
        return [item.replace("{value}", value) for item in template]

    def _provider_env(self) -> dict[str, str]:
        env = env_with_updates(self.config.env)
        for key in [
            "CODEX_SANDBOX_NETWORK_DISABLED",
            "CODEX_SANDBOX",
            "CODEX_THREAD_ID",
            "CODEX_CI",
        ]:
            env.pop(key, None)
        return env
