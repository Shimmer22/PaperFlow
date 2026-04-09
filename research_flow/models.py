from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class IdeaSpec(BaseModel):
    raw_idea: str
    core_problem: str
    application_scenarios: list[str] = Field(default_factory=list)
    related_tasks: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    benchmark_methods: list[str] = Field(default_factory=list)
    excluded_directions: list[str] = Field(default_factory=list)
    time_preference: str = "recent + foundational"
    preferred_research_type: str = "balanced"
    clarification_notes: list[str] = Field(default_factory=list)


class QueryGroup(BaseModel):
    label: str
    intent: str
    query_text: str
    target_sources: list[str]
    rationale: str


class QueryPlan(BaseModel):
    broad_queries: list[QueryGroup] = Field(default_factory=list)
    precise_queries: list[QueryGroup] = Field(default_factory=list)
    method_centric_queries: list[QueryGroup] = Field(default_factory=list)
    application_centric_queries: list[QueryGroup] = Field(default_factory=list)
    semantic_queries: list[str] = Field(default_factory=list)
    synonym_expansions: list[str] = Field(default_factory=list)
    alternative_method_terms: list[str] = Field(default_factory=list)
    citation_expansion_strategy: str = ""

    def iter_queries(self) -> list[QueryGroup]:
        return (
            self.broad_queries
            + self.precise_queries
            + self.method_centric_queries
            + self.application_centric_queries
        )


class ProvenanceRecord(BaseModel):
    source: str
    source_id: Optional[str] = None
    matched_queries: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    venue: Optional[str] = None
    source: str
    source_id: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    fields_of_study: list[str] = Field(default_factory=list)
    retrieved_by_query: list[str] = Field(default_factory=list)
    raw_score: Optional[float] = None
    metadata_completeness: float = 0.0
    provenance: list[ProvenanceRecord] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    idea_relevance: float = 0.0
    method_relevance: float = 0.0
    importance: float = 0.0
    novelty_or_recency: float = 0.0
    diversity_value: float = 0.0
    evidence_quality: float = 0.0
    total: float = 0.0


class RankedPaper(BaseModel):
    paper: PaperRecord
    score_breakdown: ScoreBreakdown
    selected: bool = False
    selection_reason: str
    rejection_reason: Optional[str] = None
    similar_to: list[str] = Field(default_factory=list)


class PDFReadResult(BaseModel):
    status: Literal["not_attempted", "downloaded", "download_failed", "parsed", "parse_failed"] = "not_attempted"
    pdf_path: Optional[str] = None
    extracted_text_path: Optional[str] = None
    reading_depth: Literal["abstract-only", "partial-fulltext", "fulltext-major-sections"] = "abstract-only"
    notes: list[str] = Field(default_factory=list)


class BibliographicInfo(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    sources: list[str] = Field(default_factory=list)


class PaperBrief(BaseModel):
    bibliographic_info: BibliographicInfo
    one_sentence_summary: str
    research_problem: str
    core_method: str
    key_innovations: list[str] = Field(default_factory=list)
    experiment_summary: str
    main_results: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    relation_to_user_idea: str
    reusable_parts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confidence_notes: list[str] = Field(default_factory=list)
    reading_depth: Literal["abstract-only", "partial-fulltext", "fulltext-major-sections"] = "abstract-only"


class ScoutReport(BaseModel):
    paper_id: str
    title: str
    relevance_score: float = 0.0
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    confidence_score: float = 0.0
    worth_reading: bool = False
    main_findings: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    relation_to_idea: str = ""


class ScoutShortlistDecision(BaseModel):
    selected_paper_ids: list[str] = Field(default_factory=list)
    selection_notes: list[str] = Field(default_factory=list)
    rejection_notes: list[str] = Field(default_factory=list)


class FinalDiscussion(BaseModel):
    refined_problem_definition: str
    literature_coverage: str
    saturated_areas: list[str] = Field(default_factory=list)
    open_spaces: list[str] = Field(default_factory=list)
    priority_papers: list[str] = Field(default_factory=list)
    innovation_angles: list[str] = Field(default_factory=list)
    risks_and_failure_modes: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    evidence_notes: list[str] = Field(default_factory=list)


class ValidationItem(BaseModel):
    name: str
    success: bool
    detail: str


class ValidationReport(BaseModel):
    run_dir: str
    success: bool
    checks: list[ValidationItem] = Field(default_factory=list)


class RunManifest(BaseModel):
    run_id: str
    created_at: datetime
    status: Literal["running", "completed", "completed_with_warnings", "failed"]
    idea_path: str
    output_dir: str
    provider_name: str
    enabled_sources: list[str]
    selected_paper_ids: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    run_id: str
    status: str
    idea_excerpt: str
    candidate_count: int
    scout_pool_count: int = 0
    selected_count: int
    selected_titles: list[str]
    final_discussion_path: Optional[str] = None
    output_dir: str
    warnings: list[str] = Field(default_factory=list)
    provider_active: bool = False
    clarify_stage: str = "local"
    query_plan_stage: str = "local"
    retrieval_stage: str = "unknown"
    key_notes: list[str] = Field(default_factory=list)


class ProviderCallResult(BaseModel):
    success: bool
    raw_output: Optional[str] = None
    parsed_output: Optional[dict[str, Any]] = None
    stderr: Optional[str] = None
    command: list[str] = Field(default_factory=list)
    attempts: int = 1
    error: Optional[str] = None
    output_path: Optional[str] = None


class ProviderCapabilities(BaseModel):
    name: str
    supports_subtasks: bool = False
    supports_parallel_invocations: bool = True
    supports_output_schema: bool = False
    prompt_mode: str
    output_mode: str


class ProviderConfig(BaseModel):
    provider_type: Literal["cli", "openai_compatible_api"] = "cli"
    name: str
    display_name: Optional[str] = None
    command: str = ""
    args: list[str] = Field(default_factory=list)
    prompt_mode: Literal["stdin", "arg", "file"] = "stdin"
    output_mode: Literal["stdout", "file"] = "stdout"
    output_flag: Optional[str] = None
    schema_flag: Optional[str] = None
    availability_check: list[str] = Field(default_factory=list)
    supports_subtasks: bool = False
    supports_parallel_invocations: bool = True
    supports_output_schema: bool = False
    default_extra_args: list[str] = Field(default_factory=list)
    model_args_template: list[str] = Field(default_factory=list)
    reasoning_args_template: list[str] = Field(default_factory=list)
    supported_models: list[str] = Field(default_factory=list)
    supported_reasoning_efforts: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    base_url: Optional[str] = None
    base_url_env_var: Optional[str] = None
    api_path: Optional[str] = None
    api_style: Literal["chat_completions", "responses"] = "chat_completions"
    api_key_env_var: str = "API_KEY"
    dotenv_path: Optional[str] = ".env"
    headers: dict[str, str] = Field(default_factory=dict)
    default_body: dict[str, Any] = Field(default_factory=dict)
    model_field: str = "model"
    reasoning_effort_field: Optional[str] = None
    temperature_field: Optional[str] = None
    temperature: Optional[float] = None
    json_mode: Literal["off", "json_object"] = "json_object"
    thinking_type: Literal["enabled", "disabled"] = "enabled"
    clear_thinking: Optional[bool] = None
    supports_thinking_controls: bool = True
    supports_clear_thinking: bool = True


class ClarificationOption(BaseModel):
    id: str
    label: str
    description: str = ""


class ClarificationTurn(BaseModel):
    question: str
    options: list[ClarificationOption] = Field(default_factory=list)
    ready_for_research: bool = False
    research_aspects: list[str] = Field(default_factory=list)
    assistant_notes: str = ""


class AppConfig(BaseModel):
    app: dict[str, Any]
    provider: dict[str, Any]
    retrieval: dict[str, Any]
    ranking: dict[str, Any]
    pdf: dict[str, Any]
    execution: dict[str, Any]
    typst: dict[str, Any]
    ui: dict[str, Any]


def model_schema_path(base_dir: Path, name: str) -> Path:
    return base_dir / f"{name}.schema.json"
