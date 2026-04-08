from __future__ import annotations

from collections import defaultdict

from research_flow.models import PaperRecord
from research_flow.utils import normalize_title


def merge_and_dedupe(papers: list[PaperRecord]) -> list[PaperRecord]:
    merged: dict[str, PaperRecord] = {}
    title_index: dict[str, str] = {}
    for paper in papers:
        key = paper.doi or (f"arxiv:{paper.arxiv_id}" if paper.arxiv_id else None)
        if not key:
            normalized = normalize_title(paper.title)
            key = title_index.get(normalized) or f"title:{normalized}"
            title_index[normalized] = key
        if key not in merged:
            merged[key] = paper.model_copy(deep=True)
            continue
        target = merged[key]
        target.authors = list(dict.fromkeys(target.authors + paper.authors))
        target.fields_of_study = list(dict.fromkeys(target.fields_of_study + paper.fields_of_study))
        target.retrieved_by_query = list(dict.fromkeys(target.retrieved_by_query + paper.retrieved_by_query))
        target.provenance.extend(paper.provenance)
        target.metadata_completeness = max(target.metadata_completeness, paper.metadata_completeness)
        target.citation_count = max(filter(lambda x: x is not None, [target.citation_count, paper.citation_count]), default=None)
        target.influential_citation_count = max(
            filter(lambda x: x is not None, [target.influential_citation_count, paper.influential_citation_count]),
            default=None,
        )
        if not target.abstract and paper.abstract:
            target.abstract = paper.abstract
        if not target.pdf_url and paper.pdf_url:
            target.pdf_url = paper.pdf_url
        if not target.url and paper.url:
            target.url = paper.url
        if not target.venue and paper.venue:
            target.venue = paper.venue
        if not target.doi and paper.doi:
            target.doi = paper.doi
        if not target.arxiv_id and paper.arxiv_id:
            target.arxiv_id = paper.arxiv_id
    return list(merged.values())


def build_source_hit_map(papers: list[PaperRecord]) -> dict[str, int]:
    counts = defaultdict(int)
    for paper in papers:
        counts[paper.paper_id] = len({record.source for record in paper.provenance})
    return dict(counts)

