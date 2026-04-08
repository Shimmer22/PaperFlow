from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Optional

from research_flow.models import IdeaSpec
from research_flow.providers.base import BaseCLIProvider
from research_flow.prompts_loader import PromptLibrary


DOMAIN_PATTERNS = OrderedDict(
    [
        ("vlm", ["vlm", "vision-language model", "vision language model", "多模态", "视觉语言"]),
        ("edge-cloud collaboration", ["端云协同", "edge-cloud", "edge cloud", "cloud-edge", "cloud edge"]),
        ("token compression", ["压缩token", "token compression", "token pruning", "token reduction", "token sparsification"]),
        ("beamforming", ["波束成形", "beamforming"]),
        ("self-attention", ["自注意力", "self-attention", "self attention", "attention mechanism"]),
        ("model compression", ["模型压缩", "减少模型权重", "parameter reduction", "parameter-efficient", "model compression", "weight pruning"]),
        ("wireless communication", ["无线", "wireless", "通信", "communication", "antenna", "阵列"]),
        ("bandwidth optimization", ["带宽", "bandwidth"]),
        ("latency optimization", ["延迟", "latency"]),
        ("multimodal inference", ["推理", "inference", "serving"]),
        ("split computing", ["split computing", "split inference", "分层推理", "协同推理"]),
    ]
)

CANONICAL_TERM_GROUPS = OrderedDict(
    [
        ("beamforming", ["beamforming", "neural beamforming", "hybrid beamforming", "attention-guided beamforming"]),
        ("self-attention", ["self-attention", "attention mechanism", "sparse attention", "structured attention"]),
        ("model compression", ["model compression", "parameter reduction", "parameter pruning", "weight pruning", "low-rank compression"]),
        ("wireless communication", ["wireless communication", "antenna array", "array signal processing"]),
        ("edge-cloud collaboration", ["edge-cloud collaboration", "distributed inference", "split inference"]),
        ("token compression", ["token compression", "token pruning", "token sparsification"]),
        ("vlm", ["vision-language model", "multimodal inference"]),
        ("latency optimization", ["latency reduction", "low-latency inference"]),
        ("bandwidth optimization", ["bandwidth reduction", "communication-efficient inference"]),
    ]
)


def _extract_keywords(raw_idea: str) -> list[str]:
    ascii_tokens = [token.lower() for token in re.split(r"[^a-zA-Z0-9+-]+", raw_idea) if len(token) >= 3]
    chinese_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", raw_idea)
    candidates: list[str] = []
    for token in ascii_tokens + chinese_phrases:
        if token not in candidates:
            candidates.append(token)
    lowered = raw_idea.lower()
    for canonical, variants in DOMAIN_PATTERNS.items():
        if any(variant.lower() in lowered for variant in variants):
            if canonical not in candidates:
                candidates.append(canonical)
    return candidates[:14]


def _domain_matches(raw_idea: str) -> dict[str, bool]:
    lowered = raw_idea.lower()
    return {
        canonical: any(variant.lower() in lowered for variant in variants)
        for canonical, variants in DOMAIN_PATTERNS.items()
    }


def _canonical_terms(matches: dict[str, bool]) -> list[str]:
    terms: list[str] = []
    for domain, present in matches.items():
        if not present:
            continue
        for term in CANONICAL_TERM_GROUPS.get(domain, [domain]):
            if term not in terms:
                terms.append(term)
    return terms


def clarify_idea_locally(raw_idea: str) -> IdeaSpec:
    """Build a stable first-pass idea spec without replacing the user's domain."""
    lowered = raw_idea.lower()
    keywords = _extract_keywords(raw_idea)
    matches = _domain_matches(raw_idea)
    canonical_terms = _canonical_terms(matches)

    related_tasks: list[str] = []
    application_scenarios: list[str] = []
    benchmark_methods: list[str] = []
    excluded_directions = ["与原始研究方向无关的泛化论文检索系统话题"]

    if matches["vlm"]:
        related_tasks.extend(["vision-language model inference", "multimodal inference optimization"])
        benchmark_methods.extend(["visual token pruning", "adaptive token reduction", "multimodal token compression"])
        application_scenarios.extend(["multimodal serving", "resource-constrained VLM deployment"])
    if matches["edge-cloud collaboration"]:
        related_tasks.extend(["edge-cloud collaborative inference", "split computing", "distributed inference"])
        benchmark_methods.extend(["split inference", "hierarchical offloading", "edge-cloud scheduling"])
        application_scenarios.extend(["端侧设备与云侧协同推理", "带宽受限场景下的多模态服务"])
    if matches["token compression"]:
        related_tasks.extend(["token compression", "token pruning", "token sparsification"])
        benchmark_methods.extend(["token pruning", "token merging", "token sparsification"])
    if matches["beamforming"]:
        related_tasks.extend(["beamforming-aware model design", "communication-efficient signal processing"])
        benchmark_methods.extend(["neural beamforming", "attention-guided beamforming", "hybrid beamforming"])
        application_scenarios.extend(["无线通信系统建模", "阵列信号处理与模型协同优化"])
    if matches["self-attention"]:
        related_tasks.extend(["attention mechanism design", "parameter-efficient attention"])
        benchmark_methods.extend(["self-attention", "sparse attention", "structured attention"])
    if matches["model compression"]:
        related_tasks.extend(["model compression", "parameter reduction", "efficient architecture design"])
        benchmark_methods.extend(["parameter pruning", "low-rank compression", "weight sharing"])
    if matches["wireless communication"]:
        related_tasks.extend(["wireless communication modeling", "beamforming optimization"])
        benchmark_methods.extend(["communication-aware learning", "resource-constrained optimization"])
        application_scenarios.extend(["低开销无线通信", "边缘通信与推理协同"])
    if matches["bandwidth optimization"] or matches["latency optimization"]:
        related_tasks.extend(["communication-efficient inference", "latency-aware serving"])
        benchmark_methods.extend(["bandwidth-aware compression", "latency-aware scheduling"])
    if matches["multimodal inference"]:
        related_tasks.extend(["efficient multimodal serving", "multimodal systems optimization"])

    if "agent" in lowered and not related_tasks:
        related_tasks.extend(["agent systems", "tool use", "task planning"])
    if ("retriev" in lowered or "search" in lowered or "paper" in lowered) and not matches["vlm"]:
        related_tasks.extend(["information retrieval", "paper recommendation"])
        benchmark_methods.extend(["retrieval-augmented generation", "citation-based ranking"])
        application_scenarios.extend(["research ideation support", "paper scouting and screening"])

    if not related_tasks:
        related_tasks = ["efficient model inference", "systems optimization"]
    if not benchmark_methods:
        benchmark_methods = ["token compression", "split inference", "latency-aware serving"]
    if not application_scenarios:
        application_scenarios = ["系统优化研究", "低带宽与低延迟推理场景"]

    for term in canonical_terms:
        if term not in keywords:
            keywords.append(term)

    return IdeaSpec(
        raw_idea=raw_idea,
        core_problem=raw_idea.strip().splitlines()[0][:240],
        application_scenarios=list(dict.fromkeys(application_scenarios)),
        related_tasks=list(dict.fromkeys(related_tasks)),
        keywords=keywords,
        benchmark_methods=list(dict.fromkeys(benchmark_methods)),
        excluded_directions=excluded_directions,
        time_preference="recent + foundational",
        preferred_research_type="systems + applied research" if matches["edge-cloud collaboration"] else "balanced",
        clarification_notes=[
            "保留了原始研究方向，并补充了更适合论文检索的技术术语。",
            "优先加入了英文规范术语，避免只拿中文短句直接去检索。",
            "避免把题目误改写成泛化的文献工作流或论文推荐问题。",
        ],
    )


def maybe_clarify_idea_with_provider(
    raw_idea: str,
    provider: Optional[BaseCLIProvider],
    prompt_library: PromptLibrary,
    timeout: int,
    clarification_history: Optional[list[dict]] = None,
    runtime_options: Optional[dict[str, str]] = None,
) -> tuple[IdeaSpec, Optional[dict]]:
    local_spec = clarify_idea_locally(raw_idea)
    if clarification_history:
        local_spec.clarification_notes.extend(
            [
                "已结合用户在 GUI 澄清阶段给出的补充信息。",
                f"澄清轮次: {len(clarification_history)}",
            ]
        )
    if provider is None:
        return local_spec, None
    prompt = prompt_library.load("clarify_idea.txt")
    context = {
        "raw_idea": raw_idea,
        "clarification_history": clarification_history or [],
        "fallback_local_spec": local_spec.model_dump(),
        "instructions": "Return only a JSON object matching the schema. Use the fallback only as support, not as a constraint.",
    }
    result = provider.run_task(
        prompt=prompt + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, indent=2),
        context=context,
        expected_output_schema=IdeaSpec,
        output_path=None,
        timeout=timeout,
        runtime_options=runtime_options,
    )
    if result.success and result.parsed_output:
        return IdeaSpec.model_validate(result.parsed_output), result.model_dump()
    return local_spec, result.model_dump()
