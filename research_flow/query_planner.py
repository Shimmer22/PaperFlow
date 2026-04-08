from __future__ import annotations

import json
from itertools import islice
from typing import Optional

from research_flow.models import IdeaSpec, QueryGroup, QueryPlan
from research_flow.providers.base import BaseCLIProvider
from research_flow.prompts_loader import PromptLibrary


def _join_terms(terms: list[str], limit: int = 4) -> str:
    return " ".join(list(islice((term for term in terms if term), limit)))


def _pick_terms(idea: IdeaSpec, candidates: list[str], limit: int = 5) -> list[str]:
    haystacks = [
        idea.raw_idea.lower(),
        " ".join(idea.keywords).lower(),
        " ".join(idea.related_tasks).lower(),
        " ".join(idea.benchmark_methods).lower(),
    ]
    picked: list[str] = []
    for candidate in candidates:
        normalized = candidate.lower()
        if any(normalized in haystack for haystack in haystacks) and candidate not in picked:
            picked.append(candidate)
    return picked[:limit]


def _english_term_pool(idea: IdeaSpec) -> list[str]:
    preferred_terms = [
        "beamforming",
        "neural beamforming",
        "hybrid beamforming",
        "self-attention",
        "attention mechanism",
        "sparse attention",
        "model compression",
        "parameter reduction",
        "parameter pruning",
        "weight pruning",
        "low-rank compression",
        "wireless communication",
        "antenna array",
        "array signal processing",
        "edge-cloud collaboration",
        "distributed inference",
        "split inference",
        "communication-efficient inference",
        "bandwidth reduction",
        "latency reduction",
        "vision-language model",
        "multimodal inference",
        "token pruning",
        "token compression",
    ]
    return _pick_terms(idea, preferred_terms, limit=8)


def _compact_terms(terms: list[str], limit: int = 5) -> list[str]:
    compact: list[str] = []
    seen_tokens: set[str] = set()
    for term in terms:
        normalized = term.strip().lower()
        if not normalized or normalized in seen_tokens:
            continue
        if any(normalized in existing.lower() or existing.lower() in normalized for existing in compact):
            continue
        compact.append(term.strip())
        seen_tokens.add(normalized)
        if len(compact) >= limit:
            break
    return compact


def _build_precise_variants(idea: IdeaSpec) -> list[str]:
    keywords = [term for term in idea.keywords if len(term) > 1]
    variants = []
    if any("beamforming" in term.lower() or "波束成形" in term for term in keywords):
        variants.append("beamforming self-attention model compression")
        variants.append("neural beamforming attention mechanism parameter reduction")
        variants.append("attention-based beamforming")
        variants.append("transformer-based beamforming")
        variants.append("beamforming parameter reduction")
    if any("self-attention" in term.lower() or "自注意力" in term for term in keywords):
        variants.append("self-attention parameter-efficient architecture")
    if any("model compression" in term.lower() or "减少模型权重" in term or "parameter reduction" in term.lower() for term in keywords):
        variants.append("model compression parameter reduction attention mechanism")
    if any("wireless" in term.lower() or "通信" in term or "antenna" in term.lower() for term in keywords):
        variants.append("wireless communication beamforming attention")
    if any("vlm" in term.lower() or "vision-language" in term.lower() for term in keywords):
        variants.append("VLM token compression edge-cloud collaborative inference")
        variants.append("vision-language model token pruning bandwidth latency")
    if any("edge-cloud" in term.lower() or "split computing" in term.lower() for term in keywords):
        variants.append("edge-cloud multimodal split inference communication-efficient serving")
    if any("bandwidth" in term.lower() or "latency" in term.lower() for term in keywords):
        variants.append("latency-aware bandwidth-aware multimodal inference")
    if any("token" in term.lower() for term in keywords):
        variants.append("token pruning token reduction token sparsification multimodal inference")
    return list(dict.fromkeys(variants))


def build_query_plan(idea: IdeaSpec, enabled_sources: list[str]) -> QueryPlan:
    english_terms = _english_term_pool(idea)
    compact_keywords = _compact_terms(english_terms, limit=5)
    keyword_line = _join_terms(compact_keywords, 5) or _join_terms(idea.keywords, 5) or idea.core_problem
    related_line = _join_terms([term for term in idea.related_tasks if term.isascii()], 4) or _join_terms(idea.related_tasks, 4)
    method_line = _join_terms([term for term in idea.benchmark_methods if term.isascii()], 4) or _join_terms(idea.benchmark_methods, 4)
    precise_variants = _build_precise_variants(idea)
    broad_query = " ".join(_compact_terms(compact_keywords + ["parameter reduction", "wireless communication"], limit=5)).strip()
    if len(broad_query.split()) <= 1:
        broad_query = keyword_line if len(keyword_line.split()) > 1 else f"{keyword_line} {related_line}".strip()
    broad_query = broad_query.strip() or idea.core_problem
    precise_query = precise_variants[0] if precise_variants else f"{keyword_line} efficient inference".strip()
    method_terms = _compact_terms(
        [
            "attention-based beamforming" if any("beamforming" in keyword.lower() or "波束成形" in keyword for keyword in idea.keywords) else "",
            "neural beamforming" if any("beamforming" in keyword.lower() or "波束成形" in keyword for keyword in idea.keywords) else "",
            "self-attention" if any("self-attention" in keyword.lower() or "自注意力" in keyword for keyword in idea.keywords) else "",
            "parameter pruning" if any("model compression" in keyword.lower() or "减少模型权重" in keyword or "parameter reduction" in keyword.lower() for keyword in idea.keywords) else "",
            "low-rank compression" if any("model compression" in keyword.lower() or "参数" in keyword for keyword in idea.keywords) else "",
        ],
        limit=5,
    )
    method_query = " ".join(method_terms or [method_line or keyword_line]).strip()
    application_terms = _compact_terms(
        [
            "wireless communication" if any("波束成形" in keyword or "beamforming" in keyword.lower() for keyword in idea.keywords) else "",
            "antenna array" if any("波束成形" in keyword or "beamforming" in keyword.lower() for keyword in idea.keywords) else "",
            "transformer-based beamforming" if any("波束成形" in keyword or "beamforming" in keyword.lower() for keyword in idea.keywords) and any("self-attention" in keyword.lower() or "自注意力" in keyword for keyword in idea.keywords) else "",
            "communication-efficient inference" if any("通信" in keyword or "wireless" in keyword.lower() for keyword in idea.keywords) else "",
            "parameter reduction" if any("model compression" in keyword.lower() or "减少模型权重" in keyword for keyword in idea.keywords) else "",
        ],
        limit=5,
    )
    application_query = " ".join(application_terms or [related_line or keyword_line]).strip()
    return QueryPlan(
        broad_queries=[
            QueryGroup(
                label="broad-core",
                intent="broad coverage",
                query_text=broad_query,
                target_sources=enabled_sources,
                rationale="Capture broad surface area around the idea.",
            )
        ],
        precise_queries=[
            QueryGroup(
                label="precise-problem",
                intent="precise problem statement",
                query_text=precise_query,
                target_sources=enabled_sources,
                rationale="Anchor retrieval to the real systems problem rather than generic token papers.",
            ),
            *[
                QueryGroup(
                    label=f"precise-variant-{index + 1}",
                    intent="precise problem variant",
                    query_text=variant,
                    target_sources=enabled_sources,
                    rationale="Add nearby phrasings of the same problem to improve recall.",
                )
                for index, variant in enumerate(precise_variants[1:3])
            ],
        ],
        method_centric_queries=[
            QueryGroup(
                label="method-centric",
                intent="similar methods",
                query_text=method_query or method_line or keyword_line,
                target_sources=enabled_sources,
                rationale="Find papers that use compression, pruning, split inference, or scheduling methods relevant to the idea.",
            )
        ],
        application_centric_queries=[
            QueryGroup(
                label="application-centric",
                intent="application focus",
                query_text=application_query or related_line or keyword_line,
                target_sources=enabled_sources,
                rationale="Find papers grounded in the same deployment and systems setting.",
            )
        ],
        semantic_queries=[
            keyword_line,
            f"{keyword_line} scholarly papers",
        ],
        synonym_expansions=[
            "neural beamforming",
            "attention-guided beamforming",
            "parameter-efficient attention",
            "model weight reduction",
            "edge-cloud collaborative inference",
            "split computing for VLM",
            "communication-efficient multimodal inference",
            "token pruning for VLM serving",
        ],
        alternative_method_terms=[
            "hybrid beamforming",
            "sparse attention",
            "parameter pruning",
            "low-rank compression",
            "token merging",
            "adaptive token reduction",
            "visual token sparsification",
            "bandwidth-aware offloading",
        ],
        citation_expansion_strategy="Prefer papers surfaced by multiple query families, especially those linking the core mechanism terms, system setting, and optimization target in the idea.",
    )


def maybe_build_query_plan_with_provider(
    idea: IdeaSpec,
    enabled_sources: list[str],
    provider: Optional[BaseCLIProvider],
    prompt_library: PromptLibrary,
    timeout: int,
    runtime_options: Optional[dict[str, str]] = None,
) -> tuple[QueryPlan, Optional[dict]]:
    local_plan = build_query_plan(idea, enabled_sources)
    if provider is None:
        return local_plan, None
    prompt = prompt_library.load("query_planner.txt")
    context = {
        "idea_spec": idea.model_dump(),
        "enabled_sources": enabled_sources,
        "fallback_local_plan": local_plan.model_dump(),
        "instructions": "Return only a JSON object matching the schema. Use retrieval-friendly technical English in query_text.",
    }
    result = provider.run_task(
        prompt=prompt + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, indent=2),
        context=context,
        expected_output_schema=QueryPlan,
        output_path=None,
        timeout=timeout,
        runtime_options=runtime_options,
    )
    if result.success and result.parsed_output:
        plan = QueryPlan.model_validate(result.parsed_output)
        for group in plan.iter_queries():
            if not group.target_sources:
                group.target_sources = enabled_sources
        return plan, result.model_dump()
    return local_plan, result.model_dump()
