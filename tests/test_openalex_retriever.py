import json
import unittest

from research_flow.models import QueryGroup
from research_flow.retrievers.openalex import OpenAlexRetriever


class _DummyResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _DummyClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def get(self, url: str) -> _DummyResponse:
        self.calls.append(url)
        return _DummyResponse(self.payloads[len(self.calls) - 1])


class OpenAlexRetrieverTests(unittest.TestCase):
    def test_retrieve_uses_multiple_openalex_strategies_and_dedupes(self) -> None:
        retriever = OpenAlexRetriever(timeout=5, user_agent="test-agent")
        retriever.client = _DummyClient(
            [
                {
                    "results": [
                        {"id": "https://openalex.org/W1", "title": "EC-DIT", "publication_year": 2024, "authorships": [], "concepts": []}
                    ]
                },
                {
                    "results": [
                        {"id": "https://openalex.org/W1", "title": "EC-DIT", "publication_year": 2024, "authorships": [], "concepts": []},
                        {"id": "https://openalex.org/W2", "title": "Expert Race", "publication_year": 2025, "authorships": [], "concepts": []},
                    ]
                },
                {
                    "results": [
                        {"id": "https://openalex.org/W2", "title": "Expert Race", "publication_year": 2025, "authorships": [], "concepts": []}
                    ]
                },
            ]
        )
        query = QueryGroup(
            label="precise",
            intent="precise problem statement",
            query_text="EC-DIT adaptive expert-choice routing diffusion transformers",
            target_sources=["openalex"],
            rationale="test",
        )
        papers = retriever.retrieve(query, limit=5)
        self.assertEqual([paper.paper_id for paper in papers], ["openalex:https://openalex.org/W1", "openalex:https://openalex.org/W2"])
        self.assertEqual(len(retriever.client.calls), 3)
        serialized_calls = json.dumps(retriever.client.calls)
        self.assertIn("filter=title.search", serialized_calls)
        self.assertIn("filter=title_and_abstract.search", serialized_calls)


if __name__ == "__main__":
    unittest.main()
