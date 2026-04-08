import unittest

from research_flow.clarification import enforce_unsure_option


class ClarificationTurnTests(unittest.TestCase):
    def test_enforce_unsure_option_appends_when_missing(self) -> None:
        options = [
            {"id": "opt_a", "label": "方向A", "description": "A"},
            {"id": "opt_b", "label": "方向B", "description": "B"},
        ]
        fixed = enforce_unsure_option(options)
        ids = [item["id"] for item in fixed]
        self.assertIn("unsure_model_think", ids)

    def test_enforce_unsure_option_keeps_existing(self) -> None:
        options = [
            {"id": "unsure_model_think", "label": "我不确定", "description": "请模型先推理"},
        ]
        fixed = enforce_unsure_option(options)
        self.assertEqual(len([x for x in fixed if x["id"] == "unsure_model_think"]), 1)


if __name__ == "__main__":
    unittest.main()
