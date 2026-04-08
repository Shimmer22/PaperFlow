from __future__ import annotations

import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import httpx

from research_flow.models import PaperRecord, ProvenanceRecord, QueryGroup
from research_flow.retrievers.base import BaseRetriever


class ArxivRetriever(BaseRetriever):
    source_name = "arxiv"

    def __init__(self, timeout: int, user_agent: str) -> None:
        self.client = httpx.Client(timeout=timeout, headers={"User-Agent": user_agent}, follow_redirects=True)

    def retrieve(self, query: QueryGroup, limit: int) -> list[PaperRecord]:
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{quote_plus(query.query_text)}&start=0&max_results={limit}"
        )
        response = self.client.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[PaperRecord] = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            entry_id = entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = [node.findtext("atom:name", default="", namespaces=ns) for node in entry.findall("atom:author", ns)]
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
                    break
            arxiv_id = entry_id.rsplit("/", maxsplit=1)[-1] if entry_id else None
            paper = PaperRecord(
                paper_id=f"arxiv:{arxiv_id or title}",
                title=title or "Untitled",
                authors=[author for author in authors if author],
                year=year,
                abstract=summary,
                venue="arXiv",
                source=self.source_name,
                source_id=arxiv_id,
                arxiv_id=arxiv_id,
                url=entry_id,
                pdf_url=pdf_url,
                fields_of_study=[],
                retrieved_by_query=[query.query_text],
                raw_score=0.0,
                metadata_completeness=0.8 if title and summary else 0.5,
                provenance=[
                    ProvenanceRecord(
                        source=self.source_name,
                        source_id=arxiv_id,
                        matched_queries=[query.query_text],
                        raw_payload={"id": arxiv_id, "title": title},
                    )
                ],
            )
            papers.append(paper)
        return papers
