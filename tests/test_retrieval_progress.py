import logging
import unittest
from unittest.mock import patch

from research_flow.models import PaperRecord, QueryGroup, QueryPlan
from research_flow.retrieval import retrieve_candidates


class _StubRetriever:
    def __init__(self, source: str) -> None:
        self.source = source

    def retrieve(self, query, per_source_limit):
        return [
            PaperRecord(
                paper_id=f"{self.source}:{query.label}",
                title=f"{self.source}-{query.label}",
                source=self.source,
                retrieved_by_query=[query.query_text],
            )
        ][:per_source_limit]


class RetrievalProgressTests(unittest.TestCase):
    def test_retrieval_reports_progress_per_source_attempt(self) -> None:
        plan = QueryPlan(
            broad_queries=[
                QueryGroup(
                    label="q1",
                    intent="broad coverage",
                    query_text="foo",
                    target_sources=["openalex", "arxiv"],
                    rationale="",
                )
            ],
            precise_queries=[
                QueryGroup(
                    label="q2",
                    intent="precise",
                    query_text="bar",
                    target_sources=["openalex"],
                    rationale="",
                )
            ],
        )
        events = []

        with patch(
            "research_flow.retrieval.build_retrievers",
            return_value={
                "openalex": _StubRetriever("openalex"),
                "arxiv": _StubRetriever("arxiv"),
            },
        ):
            papers, warnings = retrieve_candidates(
                query_plan=plan,
                enabled_sources=["openalex", "arxiv"],
                per_source_limit=1,
                timeout=5,
                user_agent="test-agent",
                cache_dir=".cache/test-retrieval-progress",
                logger=logging.getLogger("test_retrieval_progress"),
                progress_callback=lambda completed, total, source, query_text, action: events.append(
                    (completed, total, source, query_text, action)
                ),
            )

        self.assertEqual(len(warnings), 0)
        self.assertEqual(len(papers), 3)
        self.assertEqual(
            events,
            [
                (1, 3, "openalex", "foo", "retrieved"),
                (2, 3, "arxiv", "foo", "retrieved"),
                (3, 3, "openalex", "bar", "retrieved"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
