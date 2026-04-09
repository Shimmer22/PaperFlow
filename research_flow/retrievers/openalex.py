from __future__ import annotations

from urllib.parse import quote_plus

import httpx

from research_flow.models import PaperRecord, ProvenanceRecord, QueryGroup
from research_flow.retrievers.base import BaseRetriever


class OpenAlexRetriever(BaseRetriever):
    source_name = "openalex"

    def __init__(self, timeout: int, user_agent: str) -> None:
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": user_agent}, follow_redirects=True)
        self.user_agent = user_agent

    def retrieve(self, query: QueryGroup, limit: int) -> list[PaperRecord]:
        payloads = [self._fetch_search_payload(query.query_text, limit)]
        if self._should_add_fielded_queries(query):
            payloads.append(self._fetch_filter_payload("title.search", query.query_text, limit))
            payloads.append(self._fetch_filter_payload("title_and_abstract.search", query.query_text, limit))
        by_id: dict[str, PaperRecord] = {}
        for payload in payloads:
            for item in payload.get("results", []):
                paper = self._paper_from_item(item, query)
                existing = by_id.get(paper.paper_id)
                if existing is None:
                    by_id[paper.paper_id] = paper
                    continue
                existing.retrieved_by_query = list(dict.fromkeys(existing.retrieved_by_query + paper.retrieved_by_query))
                existing.provenance.extend(paper.provenance)
                if paper.abstract and not existing.abstract:
                    existing.abstract = paper.abstract
                existing.raw_score = max(float(existing.raw_score or 0.0), float(paper.raw_score or 0.0))
                existing.metadata_completeness = max(existing.metadata_completeness, paper.metadata_completeness)
        return list(by_id.values())

    def _fetch_search_payload(self, query_text: str, limit: int) -> dict:
        encoded = quote_plus(query_text)
        url = f"https://api.openalex.org/works?search={encoded}&per-page={limit}&mailto=research-flow@example.local"
        response = self.client.get(url)
        response.raise_for_status()
        return response.json()

    def _fetch_filter_payload(self, filter_name: str, query_text: str, limit: int) -> dict:
        encoded = quote_plus(query_text)
        url = (
            "https://api.openalex.org/works?"
            f"filter={filter_name}:{encoded}&per-page={limit}&sort=relevance_score:desc&mailto=research-flow@example.local"
        )
        response = self.client.get(url)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _should_add_fielded_queries(query: QueryGroup) -> bool:
        lowered = f"{query.label} {query.intent}".lower()
        return "precise" in lowered or "title" in lowered or len(query.query_text.split()) >= 4

    def _paper_from_item(self, item: dict, query: QueryGroup) -> PaperRecord:
        papers: list[PaperRecord] = []
        authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
        primary_location = item.get("primary_location") or {}
        primary_source = primary_location.get("source") or {}
        paper = PaperRecord(
            paper_id=f"openalex:{item.get('id', '')}",
            title=item.get("title") or "Untitled",
            authors=[author for author in authors if author],
            year=item.get("publication_year"),
            abstract="",
            venue=primary_source.get("display_name"),
            source=self.source_name,
            source_id=item.get("id"),
            doi=(item.get("doi") or "").replace("https://doi.org/", "") or None,
            url=item.get("id"),
            pdf_url=primary_location.get("pdf_url"),
            citation_count=item.get("cited_by_count"),
            fields_of_study=[topic.get("display_name", "") for topic in item.get("concepts", [])[:5]],
            retrieved_by_query=[query.query_text],
            raw_score=float(item.get("relevance_score", 0.0) or 0.0),
            metadata_completeness=self._metadata_score(item),
            provenance=[
                ProvenanceRecord(
                    source=self.source_name,
                    source_id=item.get("id"),
                    matched_queries=[query.query_text],
                    raw_payload={"id": item.get("id"), "title": item.get("title")},
                )
            ],
        )
        abstract_index = item.get("abstract_inverted_index")
        if abstract_index:
            paper.abstract = self._reconstruct_abstract(abstract_index)
        return paper

    @staticmethod
    def _reconstruct_abstract(index: dict[str, list[int]]) -> str:
        words = []
        for token, positions in index.items():
            for pos in positions:
                words.append((pos, token))
        return " ".join(token for _, token in sorted(words))

    @staticmethod
    def _metadata_score(item: dict) -> float:
        keys = ["title", "doi", "publication_year", "primary_location", "authorships"]
        present = sum(1 for key in keys if item.get(key))
        return round(present / len(keys), 2)
