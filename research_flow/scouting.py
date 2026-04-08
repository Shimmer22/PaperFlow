from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from research_flow.models import IdeaSpec, PaperRecord, RankedPaper, ScoreBreakdown, ScoutReport, ScoutShortlistDecision
from research_flow.providers.base import BaseCLIProvider
from research_flow.prompts_loader import PromptLibrary


def _score_report(report: ScoutReport) -> float:
    return (
        report.relevance_score * 0.4
        + report.novelty_score * 0.25
        + report.feasibility_score * 0.2
        + report.confidence_score * 0.15
    )


def local_select_from_scout(reports: list[ScoutReport], max_selected: int) -> list[str]:
    sorted_reports = sorted(
        reports,
        key=lambda item: (
            item.worth_reading,
            _score_report(item),
        ),
        reverse=True,
    )
    return [item.paper_id for item in sorted_reports[:max_selected]]


def build_ranked_from_scout(
    papers: list[PaperRecord],
    reports: list[ScoutReport],
    selected_ids: list[str],
) -> list[RankedPaper]:
    by_id = {item.paper_id: item for item in reports}
    selected_set = set(selected_ids)
    ranked_items: list[RankedPaper] = []

    for paper in papers:
        report = by_id.get(paper.paper_id)
        if report is None:
            score = ScoreBreakdown(total=0.0)
            ranked_items.append(
                RankedPaper(
                    paper=paper,
                    score_breakdown=score,
                    selected=False,
                    selection_reason="缺少 subagent 报告。",
                    rejection_reason="未产出 subagent 报告。",
                )
            )
            continue

        total = round(_score_report(report), 3)
        score = ScoreBreakdown(
            idea_relevance=round(report.relevance_score, 3),
            method_relevance=round(report.feasibility_score, 3),
            importance=round(report.novelty_score, 3),
            novelty_or_recency=round(report.novelty_score, 3),
            diversity_value=round(report.feasibility_score, 3),
            evidence_quality=round(report.confidence_score, 3),
            total=total,
        )
        selected = paper.paper_id in selected_set
        ranked_items.append(
            RankedPaper(
                paper=paper,
                score_breakdown=score,
                selected=selected,
                selection_reason=(report.main_findings[0] if report.main_findings else "subagent 认为值得跟进"),
                rejection_reason=None if selected else (report.risk_notes[0] if report.risk_notes else "主模型未选入 topk"),
            )
        )

    ranked_items.sort(key=lambda item: item.score_breakdown.total, reverse=True)
    return ranked_items


def _local_scout_report(idea: IdeaSpec, paper: PaperRecord) -> ScoutReport:
    text = f"{paper.title} {paper.abstract}".lower()
    keywords = [term.lower() for term in (idea.keywords + idea.related_tasks + idea.benchmark_methods)]
    overlap = sum(1 for keyword in keywords if keyword and keyword in text)
    relevance = min(1.0, overlap / max(3, len(keywords) * 0.4))
    novelty = 0.5 if (paper.year or 2018) >= 2023 else 0.35
    feasibility = 0.6 if paper.abstract else 0.3
    confidence = min(1.0, paper.metadata_completeness + (0.1 if paper.abstract else 0.0))
    worth = relevance >= 0.45
    return ScoutReport(
        paper_id=paper.paper_id,
        title=paper.title,
        relevance_score=round(relevance, 3),
        novelty_score=round(novelty, 3),
        feasibility_score=round(feasibility, 3),
        confidence_score=round(confidence, 3),
        worth_reading=worth,
        main_findings=["与研究主题有一定重叠，可作为候选。" if worth else "与主题重叠有限。"],
        risk_notes=[] if worth else ["主题贴合度不足"],
        relation_to_idea="基于标题与摘要的本地评估。",
    )


def maybe_generate_scout_report_with_provider(
    provider: Optional[BaseCLIProvider],
    prompt_library: PromptLibrary,
    idea: IdeaSpec,
    paper: PaperRecord,
    timeout: int,
    runtime_options: Optional[dict[str, str]] = None,
) -> tuple[ScoutReport, Optional[dict]]:
    local_report = _local_scout_report(idea, paper)
    if provider is None:
        return local_report, None

    prompt = prompt_library.load("paper_scout_subagent.txt")
    context = {
        "idea_spec": idea.model_dump(),
        "paper": paper.model_dump(),
        "instructions": "Return only one JSON object matching the schema.",
    }
    result = provider.run_subtask(
        prompt=prompt + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, indent=2),
        context=context,
        expected_output_schema=ScoutReport,
        output_path=None,
        timeout=timeout,
        runtime_options=runtime_options,
    )
    if result.success and result.parsed_output:
        return ScoutReport.model_validate(result.parsed_output), result.model_dump()
    return local_report, result.model_dump()


def maybe_select_shortlist_with_provider(
    provider: Optional[BaseCLIProvider],
    prompt_library: PromptLibrary,
    idea: IdeaSpec,
    reports: list[ScoutReport],
    max_selected: int,
    timeout: int,
    runtime_options: Optional[dict[str, str]] = None,
) -> tuple[ScoutShortlistDecision, Optional[dict]]:
    local_selected = local_select_from_scout(reports, max_selected=max_selected)
    local_decision = ScoutShortlistDecision(
        selected_paper_ids=local_selected,
        selection_notes=["使用本地规则按 subagent 评分排序。"],
        rejection_notes=[],
    )
    if provider is None:
        return local_decision, None

    prompt = prompt_library.load("scout_shortlist_synthesizer.txt")
    context = {
        "idea_spec": idea.model_dump(),
        "scout_reports": [item.model_dump() for item in reports],
        "max_selected": max_selected,
        "instructions": "Return only one JSON object matching the schema.",
    }
    result = provider.run_task(
        prompt=prompt + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, indent=2),
        context=context,
        expected_output_schema=ScoutShortlistDecision,
        output_path=None,
        timeout=timeout,
        runtime_options=runtime_options,
    )
    if result.success and result.parsed_output:
        decision = ScoutShortlistDecision.model_validate(result.parsed_output)
        decision.selected_paper_ids = [pid for pid in decision.selected_paper_ids if pid in {item.paper_id for item in reports}][:max_selected]
        if not decision.selected_paper_ids:
            decision.selected_paper_ids = local_selected
        return decision, result.model_dump()
    return local_decision, result.model_dump()


def scout_candidates(
    papers: list[PaperRecord],
    idea_spec: IdeaSpec,
    provider: Optional[BaseCLIProvider],
    prompt_library: PromptLibrary,
    timeout: int,
    parallelism: int,
    runtime_options: Optional[dict[str, str]] = None,
) -> tuple[list[ScoutReport], list[dict]]:
    reports: list[ScoutReport] = []
    provider_results: list[dict] = []

    def worker(paper: PaperRecord) -> tuple[ScoutReport, Optional[dict]]:
        return maybe_generate_scout_report_with_provider(
            provider=provider,
            prompt_library=prompt_library,
            idea=idea_spec,
            paper=paper,
            timeout=timeout,
            runtime_options=runtime_options,
        )

    if len(papers) <= 1:
        for paper in papers:
            report, result = worker(paper)
            reports.append(report)
            if result:
                provider_results.append(result)
        return reports, provider_results

    with ThreadPoolExecutor(max_workers=max(1, parallelism)) as executor:
        futures = {executor.submit(worker, paper): paper for paper in papers}
        for future in as_completed(futures):
            report, result = future.result()
            reports.append(report)
            if result:
                provider_results.append(result)
    return reports, provider_results
