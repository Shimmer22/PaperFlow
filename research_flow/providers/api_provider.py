from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional, Type

import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

from research_flow.models import ProviderCallResult, ProviderCapabilities, ProviderConfig
from research_flow.providers.base import BaseCLIProvider
from research_flow.utils import set_nested_value, write_json


class OpenAICompatibleAPIProvider(BaseCLIProvider):
    """Provider adapter for OpenAI-compatible HTTP APIs."""

    def __init__(self, config: ProviderConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir

    def check_available(self) -> tuple[bool, str]:
        api_key = self._load_api_key()
        if not self._resolved_base_url():
            return False, "Provider base_url is not configured."
        if not api_key:
            return False, f"Missing API key in env var {self.config.api_key_env_var}."
        return True, f"API provider ready: {self._resolved_base_url()}"

    def describe_provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name=self.config.name,
            supports_subtasks=self.config.supports_subtasks,
            supports_parallel_invocations=self.config.supports_parallel_invocations,
            supports_output_schema=self.config.supports_output_schema,
            prompt_mode="http",
            output_mode="http",
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
        del extra_args
        raw_response = ""
        payload = self._build_payload(prompt=prompt, expected_output_schema=expected_output_schema, runtime_options=runtime_options)
        final_output_path = output_path or (self.workdir / ".cache" / "research-flow" / "provider_last_output.json")
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(
                    self._endpoint_url(),
                    headers=self._headers(),
                    json=payload,
                )
            raw_response = response.text
            response.raise_for_status()
        except httpx.TimeoutException:
            return ProviderCallResult(
                success=False,
                raw_output="",
                stderr="",
                command=[self._endpoint_url()],
                error=f"provider timed out after {timeout} seconds",
                output_path=str(final_output_path),
            )
        except httpx.HTTPStatusError as exc:
            return ProviderCallResult(
                success=False,
                raw_output=raw_response,
                stderr=str(exc),
                command=[self._endpoint_url()],
                error=self._humanize_http_error(exc.response.status_code, raw_response),
                output_path=str(final_output_path),
            )
        except Exception as exc:
            return ProviderCallResult(
                success=False,
                raw_output=raw_response,
                stderr=str(exc),
                command=[self._endpoint_url()],
                error=str(exc),
                output_path=str(final_output_path),
            )

        write_json(final_output_path, response.json())
        raw_output = self._extract_text(response.json())
        parsed_output = self._parse_schema_output(raw_output, expected_output_schema) if expected_output_schema else None
        return ProviderCallResult(
            success=True,
            raw_output=raw_output,
            parsed_output=parsed_output,
            stderr="",
            command=[self._endpoint_url()],
            output_path=str(final_output_path),
        )

    def _build_payload(
        self,
        prompt: str,
        expected_output_schema: Optional[Type[BaseModel]],
        runtime_options: Optional[dict[str, str]],
    ) -> dict[str, Any]:
        payload = dict(self.config.default_body)
        model = (runtime_options or {}).get("model") or (self.config.supported_models[0] if self.config.supported_models else None)
        reasoning_effort = (runtime_options or {}).get("reasoning_effort") or ""
        if model:
            set_nested_value(payload, self.config.model_field, model)
        if reasoning_effort and self.config.reasoning_effort_field:
            set_nested_value(payload, self.config.reasoning_effort_field, reasoning_effort)
        if self.config.temperature is not None and self.config.temperature_field:
            set_nested_value(payload, self.config.temperature_field, self.config.temperature)
        if self.config.supports_thinking_controls:
            thinking_type = self._thinking_type(runtime_options)
            payload["thinking"] = {"type": thinking_type}
        if self.config.supports_clear_thinking and self.config.clear_thinking is not None:
            payload["clear_thinking"] = self.config.clear_thinking

        schema_notice = ""
        if expected_output_schema:
            schema_json = json.dumps(expected_output_schema.model_json_schema(), ensure_ascii=False, indent=2)
            schema_notice = (
                "\n\nOUTPUT REQUIREMENT:\n"
                "Return only one JSON object that matches this schema exactly.\n"
                f"{schema_json}\n"
            )

        user_content = prompt + schema_notice
        if self.config.api_style == "responses":
            payload.setdefault("input", user_content)
            if expected_output_schema and self.config.json_mode == "json_object":
                payload["text"] = {"format": {"type": "json_object"}}
            return payload

        payload["messages"] = [
            {
                "role": "system",
                "content": "你是一个严格遵守输出格式的研究工作流助手。除 JSON 外不要输出任何额外文本。",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        if expected_output_schema and self.config.json_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _extract_text(self, data: dict[str, Any]) -> str:
        if self.config.api_style == "responses":
            output_text = data.get("output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()
            chunks: list[str] = []
            for item in data.get("output", []):
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        chunks.append(content["text"])
            return "\n".join(chunks).strip()

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return ""

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

    def _endpoint_url(self) -> str:
        base_url = self._resolved_base_url()
        if self.config.api_path:
            return f"{base_url.rstrip('/')}/{self.config.api_path.lstrip('/')}"
        return str(base_url)

    def _resolved_base_url(self) -> str:
        if self.config.base_url_env_var:
            env_value = str(os.environ.get(self.config.base_url_env_var, "")).strip()
            if env_value:
                return env_value
        return str(self.config.base_url or "").strip()

    def _load_api_key(self) -> str:
        return str(os.environ.get(self.config.api_key_env_var, "")).strip()

    def _thinking_type(self, runtime_options: Optional[dict[str, str]]) -> str:
        enabled_value = str((runtime_options or {}).get("thinking_enabled", "")).strip().lower()
        if enabled_value in {"true", "1", "yes", "on"}:
            return "enabled"
        if enabled_value in {"false", "0", "no", "off"}:
            return "disabled"
        return self.config.thinking_type

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._load_api_key()}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.headers)
        return headers

    def _provider_label(self) -> str:
        display_name = str(self.config.display_name or "").strip()
        if display_name:
            return display_name
        if self.config.name == "nvidia":
            return "NVIDIA"
        if self.config.name == "glm_api":
            return "GLM API"
        return self.config.name or "Provider"

    def _humanize_http_error(self, status_code: int, raw_response: str) -> str:
        provider_label = self._provider_label()
        error_message = self._extract_error_message(raw_response)
        error_detail = self._extract_error_detail(raw_response)
        if status_code == 429:
            suffix = f" 原始响应: {error_message}" if error_message else ""
            return f"{provider_label} 触发速率限制，请降低请求频率或稍后重试。{suffix}".strip()
        if status_code == 401:
            return f"{provider_label} 鉴权失败，请检查环境变量 {self.config.api_key_env_var} 是否正确。"
        if status_code == 410:
            detail_text = error_detail or error_message
            suggestion = ""
            if "minimaxai/minimax-m2.1" in detail_text:
                suggestion = " 建议切换到 minimaxai/minimax-m2.5 或 z-ai/glm4.7。"
            suffix = f" 原始响应: {detail_text}" if detail_text else ""
            return f"{provider_label} 模型不可用（可能已下线/EOL）。{suggestion}{suffix}".strip()
        if status_code == 403:
            suffix = f" 原始响应: {error_message}" if error_message else ""
            return f"{provider_label} 拒绝了当前请求，请检查账号权限、模型权限或地区限制。{suffix}".strip()
        if status_code >= 500:
            suffix = f" 原始响应: {error_message}" if error_message else ""
            return f"{provider_label} 服务端暂时异常，请稍后重试。{suffix}".strip()
        suffix = f": {error_message}" if error_message else ""
        return f"{provider_label} HTTP {status_code}{suffix}"

    @staticmethod
    def _extract_error_message(raw_response: str) -> str:
        if not raw_response:
            return ""
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return raw_response.strip()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str):
                    return message.strip()
            message = payload.get("message")
            if isinstance(message, str):
                return message.strip()
        return raw_response.strip()

    @staticmethod
    def _extract_error_detail(raw_response: str) -> str:
        if not raw_response:
            return ""
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return ""
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, str):
                return detail.strip()
        return ""
