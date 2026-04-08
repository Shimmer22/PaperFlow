import json
import unittest

from research_flow.cli import format_json_output


class CliJsonOutputTests(unittest.TestCase):
    def test_format_json_output_round_trips(self) -> None:
        payload = {
            "turn": {
                "question": "line1\nline2",
                "options": [{"id": "a", "label": "A", "description": "B"}],
            },
            "provider_result": {
                "success": False,
                "raw_output": "{\"a\":\"b\"}",
            },
        }
        formatted = format_json_output(payload)
        reparsed = json.loads(formatted)
        self.assertEqual(reparsed["turn"]["question"], "line1\nline2")
        self.assertEqual(reparsed["provider_result"]["success"], False)


if __name__ == "__main__":
    unittest.main()
