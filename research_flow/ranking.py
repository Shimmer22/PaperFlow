from __future__ import annotations

from research_flow.models import IdeaSpec, PaperRecord, RankedPaper, ScoreBreakdown
from research_flow.utils import normalize_title


MIN_IDEA_RELEVANCE = 0.35
MIN_METHOD_RELEVANCE_FOR_SELECTION = 0.1
MIN_TOTAL_FOR_SELECTION = 0.42


def rank_candidates(
    idea: IdeaSpec,
    papers: list[PaperRecord],
    max_selected: int,
    weights: dict[str, float],
) -> list[RankedPaper]:
    ranked: list[RankedPaper] = []
    selected_titles: list[str] = []
    for paper in papers:
        breakdown = score_paper(idea, paper, papers, weights)
        reason = build_selection_reason(paper, breakdown)
        ranked.append(
            RankedPaper(
                paper=paper,
                score_breakdown=breakdown,
                selection_reason=reason,
            )
        )
    ranked.sort(key=lambda item: item.score_breakdown.total, reverse=True)
    for item in ranked:
        similar = [title for title in selected_titles if title_overlap(item.paper.title, title) > 0.72]
        item.similar_to = similar
        if len([p for p in ranked if p.selected]) >= max_selected:
            item.selected = False
            item.rejection_reason = "未入选：shortlist 已达到上限。"
            continue
        if item.score_breakdown.idea_relevance < MIN_IDEA_RELEVANCE:
            item.selected = False
            item.rejection_reason = (
                f"未入选：与用户想法的直接相关性过低（idea_relevance={item.score_breakdown.idea_relevance:.2f}）。"
            )
            continue
        if item.score_breakdown.method_relevance < MIN_METHOD_RELEVANCE_FOR_SELECTION and item.score_breakdown.idea_relevance < 0.5:
            item.selected = False
            item.rejection_reason = (
                f"未入选：缺少与核心方法相关的证据（method_relevance={item.score_breakdown.method_relevance:.2f}）。"
            )
            continue
        if item.score_breakdown.total < MIN_TOTAL_FOR_SELECTION:
            item.selected = False
            item.rejection_reason = f"未入选：综合得分过低（total={item.score_breakdown.total:.2f}）。"
            continue
        if similar and len([p for p in ranked if p.selected]) >= max_selected - 1:
            item.selected = False
            item.rejection_reason = "未入选：与已选论文过于相似。"
            continue
        item.selected = True
        selected_titles.append(item.paper.title)
    return ranked


def score_paper(
    idea: IdeaSpec,
    paper: PaperRecord,
    all_papers: list[PaperRecord],
    weights: dict[str, float],
) -> ScoreBreakdown:
    keywords = {term.lower() for term in idea.keywords + idea.related_tasks + idea.benchmark_methods}
    text = f"{paper.title} {paper.abstract}".lower()
    overlap = sum(1 for keyword in keywords if keyword and keyword in text)

    critical_clusters = []
    if any(token in keyword for keyword in keywords for token in ["vlm", "vision-language", "multimodal"]):
        critical_clusters.append(["vlm", "vision-language", "multimodal"])
    if any(token in keyword for keyword in keywords for token in ["edge-cloud", "split computing", "split inference", "distributed inference", "offloading", "communication-efficient"]):
        critical_clusters.append(["edge-cloud", "split computing", "split inference", "distributed inference", "offloading", "communication-efficient"])
    if any(token in keyword for keyword in keywords for token in ["token compression", "token pruning", "token reduction", "token sparsification", "token merging"]):
        critical_clusters.append(["token compression", "token pruning", "token reduction", "token sparsification", "token merging"])
    if any(token in keyword for keyword in keywords for token in ["bandwidth", "latency"]):
        critical_clusters.append(["bandwidth", "latency"])
    if any(token in keyword for keyword in keywords for token in ["beamforming", "wireless", "antenna"]):
        critical_clusters.append(["beamforming", "wireless", "antenna"])
    if any(token in keyword for keyword in keywords for token in ["self-attention", "attention mechanism", "transformer"]):
        critical_clusters.append(["self-attention", "attention", "transformer"])
    if any(token in keyword for keyword in keywords for token in ["model compression", "parameter reduction", "parameter pruning", "low-rank", "weight pruning"]):
        critical_clusters.append(["compression", "parameter", "pruning", "low-rank", "lightweight"])
    cluster_hits = 0
    for cluster in critical_clusters:
        if any(term in keyword for keyword in keywords for term in cluster) and any(term in text for term in cluster):
            cluster_hits += 1

    idea_relevance = min(1.0, (overlap / max(5, len(keywords) * 0.35)) + cluster_hits * 0.12)
    method_overlap = sum(1 for method in idea.benchmark_methods if method.lower() in text)
    method_relevance = min(1.0, method_overlap / max(1, len(idea.benchmark_methods)))
    citation_count = float(paper.citation_count or 0)
    influential = float(paper.influential_citation_count or 0)
    importance = min(1.0, (citation_count / 300.0) + (influential / 80.0))
    year = paper.year or 2018
    novelty = max(0.2, min(1.0, 1 - (2026 - year) / 12.0))
    venue_bonus = 0.2 if paper.venue and paper.venue.lower() != "arxiv" else 0.0
    evidence_quality = min(1.0, paper.metadata_completeness * 0.6 + venue_bonus + (0.2 if paper.abstract else 0.0))
    titles = [normalize_title(other.title) for other in all_papers]
    diversity_value = 1.0 - min(0.9, titles.count(normalize_title(paper.title)) / max(1, len(all_papers)))
    total = (
        idea_relevance * weights["idea_relevance"]
        + method_relevance * weights["method_relevance"]
        + importance * weights["importance"]
        + novelty * weights["novelty_or_recency"]
        + diversity_value * weights["diversity_value"]
        + evidence_quality * weights["evidence_quality"]
    )
    return ScoreBreakdown(
        idea_relevance=round(idea_relevance, 3),
        method_relevance=round(method_relevance, 3),
        importance=round(importance, 3),
        novelty_or_recency=round(novelty, 3),
        diversity_value=round(diversity_value, 3),
        evidence_quality=round(evidence_quality, 3),
        total=round(total, 3),
    )


def build_selection_reason(paper: PaperRecord, breakdown: ScoreBreakdown) -> str:
    reasons: list[str] = [f"综合得分 {breakdown.total:.3f}"]
    if breakdown.idea_relevance >= 0.7:
        reasons.append("和原始想法高度相关")
    elif breakdown.idea_relevance >= 0.45:
        reasons.append("和原始想法有一定相关性")
    else:
        reasons.append("和原始想法相关性偏弱")
    if breakdown.method_relevance >= 0.45:
        reasons.append("方法路线贴近")
    elif breakdown.method_relevance >= 0.15:
        reasons.append("方法上有部分重叠")
    else:
        reasons.append("方法贴合度不高")
    if breakdown.importance >= 0.6:
        reasons.append("有一定影响力")
    if breakdown.evidence_quality >= 0.8:
        reasons.append("元数据较完整")
    return "入选原因：" + "；".join(reasons) + "。"


def title_overlap(left: str, right: str) -> float:
    left_tokens = set(normalize_title(left).split())
    right_tokens = set(normalize_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
