from __future__ import annotations

import re

from research_flow.models import IdeaSpec, PaperRecord


GENERIC_TERMS = {
    "efficient",
    "efficiency",
    "model",
    "models",
    "generation",
    "generative",
    "architecture",
    "architectures",
    "method",
    "methods",
    "task",
    "tasks",
    "benchmark",
    "score",
    "latency",
    "throughput",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9+\- ]+", " ", text.lower())


def _idea_anchor_terms(idea: IdeaSpec) -> list[str]:
    anchors: list[str] = []
    for term in idea.keywords + idea.related_tasks:
        normalized = " ".join(_normalize(term).split())
        if len(normalized) < 3 or normalized in GENERIC_TERMS:
            continue
        if normalized not in anchors:
            anchors.append(normalized)
    return anchors


def _benchmark_terms(idea: IdeaSpec) -> list[str]:
    terms: list[str] = []
    for term in idea.benchmark_methods:
        normalized = " ".join(_normalize(term).split())
        if len(normalized) < 3:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms


def anchor_match_breakdown(idea: IdeaSpec, paper: PaperRecord) -> tuple[int, int, int]:
    text = _normalize(f"{paper.title} {paper.abstract}")
    anchor_terms = _idea_anchor_terms(idea)
    benchmark_terms = _benchmark_terms(idea)

    exact_anchor_hits = sum(1 for term in anchor_terms if term in text)
    token_anchor_hits = 0
    for term in anchor_terms:
        tokens = [token for token in term.split() if len(token) > 2 and token not in GENERIC_TERMS]
        if len(tokens) >= 2 and all(token in text for token in tokens):
            token_anchor_hits += 1
    benchmark_hits = sum(1 for term in benchmark_terms if term in text)
    return exact_anchor_hits, token_anchor_hits, benchmark_hits


def anchor_relevance_score(idea: IdeaSpec, paper: PaperRecord) -> float:
    exact_anchor_hits, token_anchor_hits, benchmark_hits = anchor_match_breakdown(idea, paper)
    anchor_terms = _idea_anchor_terms(idea)
    if not anchor_terms:
        return 0.0
    anchor_mass = max(3, min(6, len(anchor_terms)))
    score = (exact_anchor_hits * 1.0 + token_anchor_hits * 0.6 + benchmark_hits * 0.15) / anchor_mass
    return round(min(1.0, score), 3)
