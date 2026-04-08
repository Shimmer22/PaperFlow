import unittest
from pathlib import Path

from research_flow.models import ProviderConfig
from research_flow.providers.api_provider import OpenAICompatibleAPIProvider
from research_flow.providers.factory import create_provider


class ApiOnlyAndThinkingTests(unittest.TestCase):
    def _base_cfg(self) -> ProviderConfig:
        return ProviderConfig.model_validate(
            {
                "provider_type": "openai_compatible_api",
                "name": "glm_api",
                "supported_models": ["glm-4.7-flash"],
                "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                "api_key_env_var": "API_KEY",
                "default_body": {"stream": False},
                "json_mode": "json_object",
            }
        )

    def test_factory_rejects_non_api_provider(self) -> None:
        cfg = ProviderConfig.model_validate(
            {
                "provider_type": "cli",
                "name": "legacy_cli",
                "command": "echo",
            }
        )
        with self.assertRaises(ValueError):
            create_provider(cfg, workdir=Path.cwd())

    def test_main_task_defaults_to_thinking_enabled(self) -> None:
        provider = OpenAICompatibleAPIProvider(self._base_cfg(), workdir=Path.cwd())
        payload = provider._build_payload(
            prompt="hello",
            expected_output_schema=None,
            runtime_options={"thinking_enabled": "true"},
        )
        self.assertIn("thinking", payload)
        self.assertEqual(payload["thinking"].get("type"), "enabled")

    def test_subtask_can_disable_thinking(self) -> None:
        provider = OpenAICompatibleAPIProvider(self._base_cfg(), workdir=Path.cwd())
        payload = provider._build_payload(
            prompt="hello",
            expected_output_schema=None,
            runtime_options={"thinking_enabled": "false"},
        )
        self.assertIn("thinking", payload)
        self.assertEqual(payload["thinking"].get("type"), "disabled")

    def test_humanize_429_error_for_glm(self) -> None:
        provider = OpenAICompatibleAPIProvider(self._base_cfg(), workdir=Path.cwd())
        message = provider._humanize_http_error(
            429,
            '{"error":{"code":"1302","message":"您的账户已达到速率限制，请您控制请求频率"}}',
        )
        self.assertIn("GLM API", message)
        self.assertIn("速率限制", message)

    def test_humanize_401_error_for_nvidia(self) -> None:
        config = ProviderConfig.model_validate(
            {
                "provider_type": "openai_compatible_api",
                "name": "nvidia",
                "supported_models": ["z-ai/glm4.7"],
                "base_url": "https://integrate.api.nvidia.com/v1/chat/completions",
                "api_key_env_var": "NV_API_KEY",
                "default_body": {"stream": False},
                "json_mode": "json_object",
            }
        )
        provider = OpenAICompatibleAPIProvider(config, workdir=Path.cwd())
        message = provider._humanize_http_error(
            401,
            '{"error":{"message":"Unauthorized"}}',
        )
        self.assertIn("NVIDIA", message)
        self.assertIn("NV_API_KEY", message)

    def test_humanize_410_eol_for_minimax(self) -> None:
        config = ProviderConfig.model_validate(
            {
                "provider_type": "openai_compatible_api",
                "name": "nvidia",
                "supported_models": ["minimaxai/minimax-m2.5"],
                "base_url": "https://integrate.api.nvidia.com/v1/chat/completions",
                "api_key_env_var": "NV_API_KEY",
                "default_body": {"stream": False},
                "json_mode": "json_object",
            }
        )
        provider = OpenAICompatibleAPIProvider(config, workdir=Path.cwd())
        message = provider._humanize_http_error(
            410,
            '{"type":"about:blank","title":"Gone","status":410,"detail":"The model \'minimaxai/minimax-m2.1\' has reached its end of life."}',
        )
        self.assertIn("模型不可用", message)
        self.assertIn("minimaxai/minimax-m2.5", message)


if __name__ == "__main__":
    unittest.main()
