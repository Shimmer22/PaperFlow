from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from research_flow.briefing import maybe_generate_brief_with_provider
from research_flow.config import load_app_config, load_provider_config
from research_flow.dedupe import merge_and_dedupe
from research_flow.idea import maybe_clarify_idea_with_provider
from research_flow.models import (
    IdeaSpec,
    PaperBrief,
    PaperRecord,
    QueryPlan,
    RunManifest,
    RunSummary,
)
from research_flow.pdf_processing import fetch_and_extract_pdf
from research_flow.providers import BaseCLIProvider, create_provider
from research_flow.prompts_loader import PromptLibrary
from research_flow.query_planner import maybe_build_query_plan_with_provider
from research_flow.ranking import rank_candidates
from research_flow.rendering import compile_typst_if_available, render_brief_typst
from research_flow.retrieval import retrieve_candidates
from research_flow.scouting import build_ranked_from_scout, maybe_select_shortlist_with_provider, scout_candidates
from research_flow.synthesis import maybe_synthesize_with_provider
from research_flow.utils import ensure_dir, now_run_id, safe_excerpt, setup_logging, slugify, write_json, write_text
from research_flow.validation import validate_run_dir


def _pre_rank_candidates(papers: list[PaperRecord], idea: IdeaSpec) -> list[PaperRecord]:
    idea_terms = {
        term.lower()
        for term in (idea.keywords + idea.related_tasks + idea.benchmark_methods)
        if term and len(term) > 2
    }

    def lexical_relevance(paper: PaperRecord) -> tuple[int, int]:
        text = f"{paper.title} {paper.abstract}".lower()
        overlap = sum(1 for term in idea_terms if term in text)
        anchor_hits = 0
        if any("beamforming" in term or "波束成形" in term for term in idea_terms):
            if any(token in text for token in ["beamforming", "beam", "antenna", "wireless"]):
                anchor_hits += 1
        if any("self-attention" in term or "自注意力" in term or "attention mechanism" in term for term in idea_terms):
            if any(token in text for token in ["self-attention", "attention", "transformer"]):
                anchor_hits += 1
        if any("compression" in term or "reduction" in term or "pruning" in term or "减少模型权重" in term for term in idea_terms):
            if any(token in text for token in ["compression", "pruning", "parameter", "lightweight", "low-rank"]):
                anchor_hits += 1
        return anchor_hits, overlap

    return sorted(
        papers,
        key=lambda paper: (
            *lexical_relevance(paper),
            len(paper.retrieved_by_query),
            paper.metadata_completeness,
            float(paper.raw_score or 0.0),
            float(paper.influential_citation_count or 0),
            float(paper.citation_count or 0),
        ),
        reverse=True,
    )


def _scout_candidate_pool(papers: list[PaperRecord], idea: IdeaSpec, max_candidates: int) -> list[PaperRecord]:
    ranked = _pre_rank_candidates(papers, idea)
    scout_pool_limit = min(max(max_candidates * 2, 12), 15)
    return ranked[: min(len(ranked), scout_pool_limit)]


def _provider_stage_label(provider_result: Optional[dict], provider_active: bool) -> str:
    if not provider_active:
        return "provider_unavailable_local_fallback"
    if not provider_result:
        return "local_only"
    return "provider_success" if provider_result.get("success") else "provider_failed_local_fallback"


def _retrieval_stage_label(warnings: list[str], candidate_count: int) -> str:
    if candidate_count == 0 and warnings:
        return "no_candidates_with_source_failures"
    if warnings:
        return "completed_with_partial_source_failures"
    if candidate_count == 0:
        return "no_candidates"
    return "completed"


def _build_key_notes(
    provider_active: bool,
    clarify_stage: str,
    query_stage: str,
    retrieval_stage: str,
    warnings: list[str],
) -> list[str]:
    notes: list[str] = []
    notes.append("主模型可用" if provider_active else "主模型不可用，已退回本地策略")
    if clarify_stage == "provider_success":
        notes.append("idea 澄清由主模型完成")
    elif clarify_stage == "provider_failed_local_fallback":
        notes.append("idea 澄清阶段主模型失败，已退回本地规则")
    if query_stage == "provider_success":
        notes.append("query 规划由主模型完成")
    elif query_stage == "provider_failed_local_fallback":
        notes.append("query 规划阶段主模型失败，已退回本地规则")
    if retrieval_stage == "completed_with_partial_source_failures":
        notes.append("检索阶段部分论文源失败，但保留了其余结果")
    elif retrieval_stage == "no_candidates_with_source_failures":
        notes.append("检索阶段没有拿到候选，且伴随论文源故障")
    elif retrieval_stage == "no_candidates":
        notes.append("检索阶段没有拿到候选论文")
    if any("429" in warning for warning in warnings):
        notes.append("部分论文源触发限流")
    if any("nodename nor servname" in warning or "Name or service not known" in warning for warning in warnings):
        notes.append("当前网络或 DNS 存在问题")
    return notes


class ResearchFlowOrchestrator:
    def __init__(self, config_path: Union[str, Path], provider_config_path: Optional[Union[str, Path]] = None) -> None:
        self.app_config = load_app_config(config_path)
        provider_path = provider_config_path or self.app_config.provider["config_path"]
        self.provider_config = load_provider_config(provider_path)
        self.prompt_library = PromptLibrary(self.app_config.app["prompts_dir"])

    def run(
        self,
        idea_text: str,
        clarification_history: Optional[list[dict]] = None,
        outdir: Optional[Union[str, Path]] = None,
        provider_name: Optional[str] = None,
        candidate_limit: Optional[int] = None,
        max_papers: Optional[int] = None,
        download_pdf: Optional[bool] = None,
        parallel: bool = True,
        sources: Optional[list[str]] = None,
        main_model: Optional[str] = None,
        main_reasoning_effort: Optional[str] = None,
        sub_model: Optional[str] = None,
        sub_reasoning_effort: Optional[str] = None,
    ) -> Path:
        run_id = now_run_id()
        output_root = Path(outdir) if outdir else Path(self.app_config.app["default_output_root"]) / run_id
        run_dir = ensure_dir(output_root)
        logger = setup_logging(run_dir / "run.log")

        provider = create_provider(self.provider_config, workdir=Path.cwd())
        available, availability_detail = provider.check_available()
        if not available and not self.app_config.provider.get("allow_fallback_without_provider", True):
            raise RuntimeError(f"Provider unavailable: {availability_detail}")
        active_provider = provider if available else None
        logger.info("Provider active: %s", bool(active_provider))
        if availability_detail:
            logger.info("Provider check: %s", availability_detail)

        enabled_sources = sources or list(self.app_config.retrieval["enabled_sources"])
        max_candidates = candidate_limit or int(self.app_config.ranking.get("candidate_limit", 5))
        max_selected = max_papers or int(self.app_config.ranking["max_selected_papers"])
        allow_download_pdf = self.app_config.pdf["download_pdf"] if download_pdf is None else download_pdf
        timeout = int(self.app_config.execution["provider_timeout_seconds"])
        planning_timeout = int(self.app_config.execution.get("planning_timeout_seconds", min(timeout, 45)))
        planning_timeout = max(10, min(planning_timeout, timeout))
        main_runtime_options = {
            "model": main_model or "",
            "reasoning_effort": main_reasoning_effort or "",
            "thinking_enabled": "true",
        }
        sub_runtime_options = {
            "model": sub_model or "",
            "reasoning_effort": sub_reasoning_effort or "",
            "thinking_enabled": "false",
        }

        idea_path = run_dir / "idea.txt"
        write_text(idea_path, idea_text)

        manifest = RunManifest(
            run_id=run_id,
            created_at=datetime.now(),
            status="running",
            idea_path=str(idea_path),
            output_dir=str(run_dir),
            provider_name=provider_name or self.provider_config.name,
            enabled_sources=enabled_sources,
            artifacts={"log": str(run_dir / "run.log")},
        )
        write_json(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))

        warnings: list[str] = []
        try:
            idea_spec, clarify_provider_result = maybe_clarify_idea_with_provider(
                raw_idea=idea_text,
                clarification_history=clarification_history or [],
                provider=active_provider,
                prompt_library=self.prompt_library,
                timeout=planning_timeout,
                runtime_options=main_runtime_options,
            )
            write_json(run_dir / "clarified_idea.json", idea_spec.model_dump())
            if clarify_provider_result is not None:
                write_json(run_dir / "clarify_provider_result.json", clarify_provider_result)
            if clarification_history:
                write_json(run_dir / "clarification_dialogue.json", clarification_history)

            query_plan, query_provider_result = maybe_build_query_plan_with_provider(
                idea=idea_spec,
                enabled_sources=enabled_sources,
                provider=active_provider,
                prompt_library=self.prompt_library,
                timeout=planning_timeout,
                runtime_options=main_runtime_options,
            )
            write_json(run_dir / "query_plan.json", query_plan.model_dump())
            if query_provider_result is not None:
                write_json(run_dir / "query_plan_provider_result.json", query_provider_result)

            raw_candidates, retrieval_warnings = retrieve_candidates(
                query_plan=query_plan,
                enabled_sources=enabled_sources,
                per_source_limit=int(self.app_config.retrieval["per_source_limit"]),
                timeout=int(self.app_config.retrieval["request_timeout_seconds"]),
                user_agent=self.app_config.retrieval["user_agent"],
                cache_dir=self.app_config.app["cache_dir"],
                logger=logger,
            )
            warnings.extend(retrieval_warnings)
            write_json(run_dir / "candidate_papers_raw.json", [paper.model_dump() for paper in raw_candidates])

            merged_candidates = merge_and_dedupe(raw_candidates)
            scout_candidates_pool = _scout_candidate_pool(merged_candidates, idea_spec, max_candidates=max_candidates)
            logger.info(
                "Prepared %s merged candidates; sending top %s into scout stage (ui candidate_limit=%s)",
                len(merged_candidates),
                len(scout_candidates_pool),
                max_candidates,
            )
            write_json(run_dir / "candidate_papers_merged.json", [paper.model_dump() for paper in scout_candidates_pool])

            logger.info(
                "Starting scout stage for %s candidates with parallelism=%s",
                len(scout_candidates_pool),
                int(self.app_config.execution["parallelism"]),
            )
            scout_reports, scout_provider_results = scout_candidates(
                papers=scout_candidates_pool,
                idea_spec=idea_spec,
                provider=active_provider,
                prompt_library=self.prompt_library,
                timeout=timeout,
                parallelism=int(self.app_config.execution["parallelism"]),
                runtime_options=sub_runtime_options,
            )
            logger.info("Scout stage completed with %s reports", len(scout_reports))
            write_json(run_dir / "scout_reports.json", [item.model_dump() for item in scout_reports])
            if scout_provider_results:
                write_json(run_dir / "scout_provider_results.json", scout_provider_results)

            shortlist_decision, shortlist_provider_result = maybe_select_shortlist_with_provider(
                provider=active_provider,
                prompt_library=self.prompt_library,
                idea=idea_spec,
                reports=scout_reports,
                max_selected=max_selected,
                timeout=planning_timeout,
                runtime_options=main_runtime_options,
            )
            write_json(run_dir / "shortlist_decision.json", shortlist_decision.model_dump())
            if shortlist_provider_result is not None:
                write_json(run_dir / "shortlist_provider_result.json", shortlist_provider_result)

            ranked = build_ranked_from_scout(
                papers=scout_candidates_pool,
                reports=scout_reports,
                selected_ids=shortlist_decision.selected_paper_ids,
            )
            if not ranked:
                ranked = rank_candidates(
                    idea=idea_spec,
                    papers=scout_candidates_pool,
                    max_selected=max_selected,
                    weights=self.app_config.ranking["weights"],
                )
            write_json(run_dir / "ranked_candidates.json", [item.model_dump() for item in ranked])

            selected = [item.paper for item in ranked if item.selected][:max_selected]
            write_json(run_dir / "selected_papers.json", [paper.model_dump() for paper in selected])

            briefs = self._deep_read_selected_papers(
                papers=selected,
                idea_spec=idea_spec,
                provider=active_provider,
                timeout=timeout,
                run_dir=run_dir,
                allow_download_pdf=allow_download_pdf,
                parallel=parallel,
                runtime_options=sub_runtime_options,
                logger=logger,
            )
            final_discussion = maybe_synthesize_with_provider(
                provider=active_provider,
                prompt_library=self.prompt_library,
                idea=idea_spec,
                briefs=briefs,
                timeout=timeout,
                runtime_options=main_runtime_options,
            )
            write_json(run_dir / "final_discussion.json", final_discussion.model_dump())
            write_text(run_dir / "final_discussion.md", self._final_discussion_markdown(final_discussion))

            clarify_stage = _provider_stage_label(clarify_provider_result, bool(active_provider))
            query_plan_stage = _provider_stage_label(query_provider_result, bool(active_provider))
            retrieval_stage = _retrieval_stage_label(warnings, len(scout_candidates_pool))
            key_notes = _build_key_notes(
                provider_active=bool(active_provider),
                clarify_stage=clarify_stage,
                query_stage=query_plan_stage,
                retrieval_stage=retrieval_stage,
                warnings=warnings,
            )

            summary = RunSummary(
                run_id=run_id,
                status="running",
                idea_excerpt=safe_excerpt(idea_text),
                candidate_count=len(scout_candidates_pool),
                selected_count=len(selected),
                selected_titles=[paper.title for paper in selected],
                final_discussion_path=str(run_dir / "final_discussion.md"),
                output_dir=str(run_dir),
                warnings=warnings,
                provider_active=bool(active_provider),
                clarify_stage=clarify_stage,
                query_plan_stage=query_plan_stage,
                retrieval_stage=retrieval_stage,
                key_notes=key_notes,
            )
            write_json(run_dir / "run_summary.json", summary.model_dump())

            validation_report = validate_run_dir(run_dir)
            write_json(run_dir / "validation_report.json", validation_report.model_dump())

            manifest.status = "completed" if validation_report.success else "completed_with_warnings"
            manifest.selected_paper_ids = [paper.paper_id for paper in selected]
            manifest.warnings = warnings
            manifest.artifacts.update(
                {
                    "clarified_idea": str(run_dir / "clarified_idea.json"),
                    "clarification_dialogue": str(run_dir / "clarification_dialogue.json"),
                    "clarify_provider_result": str(run_dir / "clarify_provider_result.json"),
                    "query_plan": str(run_dir / "query_plan.json"),
                    "query_plan_provider_result": str(run_dir / "query_plan_provider_result.json"),
                    "scout_reports": str(run_dir / "scout_reports.json"),
                    "shortlist_decision": str(run_dir / "shortlist_decision.json"),
                    "ranked_candidates": str(run_dir / "ranked_candidates.json"),
                    "selected_papers": str(run_dir / "selected_papers.json"),
                    "final_discussion": str(run_dir / "final_discussion.md"),
                    "validation_report": str(run_dir / "validation_report.json"),
                }
            )
            write_json(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))

            summary.status = manifest.status
            write_json(run_dir / "run_summary.json", summary.model_dump())
        except Exception as exc:
            manifest.status = "failed"
            warnings.append(str(exc))
            manifest.warnings = warnings
            write_json(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))
            raise
        return run_dir

    def _deep_read_selected_papers(
        self,
        papers: list[PaperRecord],
        idea_spec: IdeaSpec,
        provider: Optional[BaseCLIProvider],
        timeout: int,
        run_dir: Path,
        allow_download_pdf: bool,
        parallel: bool,
        runtime_options: dict[str, str],
        logger,
    ) -> list[PaperBrief]:
        papers_root = ensure_dir(run_dir / "papers")
        template_path = Path(self.app_config.typst["template_path"])
        parallelism = int(self.app_config.execution["parallelism"])

        def worker(paper: PaperRecord) -> PaperBrief:
            paper_dir = ensure_dir(papers_root / slugify(paper.paper_id.replace(":", "-"), max_length=140))
            pdf_result = fetch_and_extract_pdf(
                paper,
                papers_root,
                timeout=int(self.app_config.retrieval["request_timeout_seconds"]),
                max_chars=int(self.app_config.pdf["max_chars_from_pdf"]),
            ) if allow_download_pdf else self._skip_pdf_fetch()
            write_json(paper_dir / "pdf_read_result.json", pdf_result.model_dump())
            brief = maybe_generate_brief_with_provider(
                provider=provider,
                prompt_library=self.prompt_library,
                paper=paper,
                idea=idea_spec,
                pdf_result=pdf_result,
                timeout=timeout,
                output_path=paper_dir / "provider_brief_output.json",
                runtime_options=runtime_options,
            )
            write_json(paper_dir / "paper_brief.json", brief.model_dump())
            typ_path = paper_dir / "paper_brief.typ"
            render_brief_typst(template_path, brief, typ_path)
            compiled, detail = compile_typst_if_available(typ_path)
            write_text(paper_dir / "typst_compile.log", detail)
            logger.info("Rendered brief for %s (compiled=%s)", paper.title, compiled)
            return brief

        if not parallel or len(papers) <= 1:
            return [worker(paper) for paper in papers]
        briefs: list[PaperBrief] = []
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = {executor.submit(worker, paper): paper for paper in papers}
            for future in as_completed(futures):
                briefs.append(future.result())
        return briefs

    @staticmethod
    def _skip_pdf_fetch():
        from research_flow.models import PDFReadResult

        return PDFReadResult(status="not_attempted", reading_depth="abstract-only", notes=["PDF download disabled by configuration."])

    @staticmethod
    def _final_discussion_markdown(final_discussion) -> str:
        sections = [
            "# 最终讨论",
            "## 更精确的问题定义",
            final_discussion.refined_problem_definition,
            "## 文献覆盖情况",
            final_discussion.literature_coverage,
            "## 已较为拥挤的方向",
            "\n".join(f"- {item}" for item in final_discussion.saturated_areas) or "- None recorded.",
            "## 仍可能存在空间的方向",
            "\n".join(f"- {item}" for item in final_discussion.open_spaces) or "- None recorded.",
            "## 建议优先阅读的论文",
            "\n".join(f"- {item}" for item in final_discussion.priority_papers) or "- None recorded.",
            "## 可能的创新切入点",
            "\n".join(f"- {item}" for item in final_discussion.innovation_angles) or "- None recorded.",
            "## 风险与常见失败模式",
            "\n".join(f"- {item}" for item in final_discussion.risks_and_failure_modes) or "- None recorded.",
            "## 下一步建议",
            "\n".join(f"- {item}" for item in final_discussion.next_steps) or "- None recorded.",
            "## 证据说明",
            "\n".join(f"- {item}" for item in final_discussion.evidence_notes) or "- None recorded.",
        ]
        return "\n\n".join(sections) + "\n"
