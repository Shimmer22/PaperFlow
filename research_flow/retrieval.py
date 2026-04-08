from __future__ import annotations

import logging

from research_flow.cache import JsonCache
from research_flow.models import PaperRecord, QueryPlan
from research_flow.retrievers import ArxivRetriever, OpenAlexRetriever, SemanticScholarRetriever


def build_retrievers(timeout: int, user_agent: str) -> dict[str, object]:
    return {
        "openalex": OpenAlexRetriever(timeout=timeout, user_agent=user_agent),
        "semanticscholar": SemanticScholarRetriever(timeout=timeout, user_agent=user_agent),
        "arxiv": ArxivRetriever(timeout=timeout, user_agent=user_agent),
    }


def _should_block_source(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "timed out" in text
        or "ssl" in text
        or "eof occurred in violation of protocol" in text
        or "nodename nor servname" in text
        or "name or service not known" in text
    )


def retrieve_candidates(
    query_plan: QueryPlan,
    enabled_sources: list[str],
    per_source_limit: int,
    timeout: int,
    user_agent: str,
    cache_dir: str,
    logger: logging.Logger,
) -> tuple[list[PaperRecord], list[str]]:
    retrievers = build_retrievers(timeout=timeout, user_agent=user_agent)
    cache = JsonCache(cache_dir)
    all_papers: list[PaperRecord] = []
    warnings: list[str] = []
    blocked_sources: dict[str, str] = {}
    for query in query_plan.iter_queries():
        for source in query.target_sources:
            if source not in enabled_sources:
                continue
            if source in blocked_sources:
                logger.info("Skipping %s for query '%s' because source is temporarily blocked: %s", source, query.query_text, blocked_sources[source])
                continue
            retriever = retrievers.get(source)
            if retriever is None:
                warnings.append(f"Retriever not configured: {source}")
                continue
            cache_key = f"{source}|{query.query_text}|{per_source_limit}"
            try:
                cached = cache.get("retrieval", cache_key)
                if cached is not None:
                    logger.info("Using cached retrieval for %s query '%s'", source, query.query_text)
                    all_papers.extend(PaperRecord.model_validate(item) for item in cached)
                    continue
                logger.info("Retrieving from %s with query '%s'", source, query.query_text)
                papers = retriever.retrieve(query, per_source_limit)
                cache.set("retrieval", cache_key, [paper.model_dump() for paper in papers])
                all_papers.extend(papers)
            except Exception as exc:
                warning = f"{source} failed for query '{query.query_text}': {exc}"
                logger.warning(warning)
                warnings.append(warning)
                if _should_block_source(exc):
                    blocked_sources[source] = str(exc)
                    logger.warning("Blocking source %s for remaining queries: %s", source, exc)
    return all_papers, warnings
