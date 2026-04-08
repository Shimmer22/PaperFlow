from __future__ import annotations

from urllib.parse import quote_plus

import httpx

from research_flow.models import PaperRecord, ProvenanceRecord, QueryGroup
from research_flow.retrievers.base import BaseRetriever


class SemanticScholarRetriever(BaseRetriever):
    source_name = "semanticscholar"

    def __init__(self, timeout: int, user_agent: str) -> None:
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": user_agent}, follow_redirects=True)

    def retrieve(self, query: QueryGroup, limit: int) -> list[PaperRecord]:
        fields = ",".join(
            [
                "title",
                "authors",
                "year",
                "abstract",
                "venue",
                "citationCount",
                "influentialCitationCount",
                "fieldsOfStudy",
                "externalIds",
                "url",
                "openAccessPdf",
            ]
        )
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search?"
            f"query={quote_plus(query.query_text)}&limit={limit}&fields={fields}"
        )
        response = self.client.get(url)
        response.raise_for_status()
        payload = response.json()
        papers: list[PaperRecord] = []
        for item in payload.get("data", []):
            external_ids = item.get("externalIds") or {}
            paper = PaperRecord(
                paper_id=f"semanticscholar:{item.get('paperId', '')}",
                title=item.get("title") or "Untitled",
                authors=[author.get("name", "") for author in item.get("authors", []) if author.get("name")],
                year=item.get("year"),
                abstract=item.get("abstract") or "",
                venue=item.get("venue"),
                source=self.source_name,
                source_id=item.get("paperId"),
                doi=external_ids.get("DOI"),
                arxiv_id=external_ids.get("ArXiv"),
                url=item.get("url"),
                pdf_url=(item.get("openAccessPdf") or {}).get("url"),
                citation_count=item.get("citationCount"),
                influential_citation_count=item.get("influentialCitationCount"),
                fields_of_study=item.get("fieldsOfStudy") or [],
                retrieved_by_query=[query.query_text],
                raw_score=0.0,
                metadata_completeness=self._metadata_score(item),
                provenance=[
                    ProvenanceRecord(
                        source=self.source_name,
                        source_id=item.get("paperId"),
                        matched_queries=[query.query_text],
                        raw_payload={"paperId": item.get("paperId"), "title": item.get("title")},
                    )
                ],
            )
            papers.append(paper)
        return papers

    @staticmethod
    def _metadata_score(item: dict) -> float:
        keys = ["title", "year", "abstract", "venue", "authors"]
        present = sum(1 for key in keys if item.get(key))
        return round(present / len(keys), 2)
