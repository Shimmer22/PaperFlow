import unittest
from pathlib import Path

from research_flow.config import load_app_config, load_provider_config


class DeepSeekDefaultsTests(unittest.TestCase):
    def test_deepseek_provider_config_defaults(self) -> None:
        config = load_provider_config(Path("providers/deepseek.example.yaml"))

        self.assertEqual(config.name, "deepseek")
        self.assertEqual(config.display_name, "DeepSeek")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.api_path, "chat/completions")
        self.assertEqual(config.api_key_env_var, "DS_API_KEY")
        self.assertEqual(config.supported_models[:2], ["deepseek-v4-pro", "deepseek-v4-flash"])
        self.assertTrue(config.supports_thinking_controls)

    def test_app_defaults_use_deepseek_provider(self) -> None:
        config = load_app_config(Path("config.example.yaml"))

        self.assertEqual(config.provider["name"], "deepseek")
        self.assertEqual(config.provider["config_path"], "providers/deepseek.example.yaml")


if __name__ == "__main__":
    unittest.main()
