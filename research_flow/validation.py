from __future__ import annotations

from pathlib import Path
from typing import Union

from research_flow.models import ValidationItem, ValidationReport


def validate_run_dir(run_dir: Union[str, Path]) -> ValidationReport:
    path = Path(run_dir)
    checks = []
    required_files = [
        "clarified_idea.json",
        "query_plan.json",
        "candidate_papers_raw.json",
        "candidate_papers_merged.json",
        "ranked_candidates.json",
        "selected_papers.json",
        "final_discussion.md",
        "run_manifest.json",
        "run_summary.json",
    ]
    for filename in required_files:
        exists = (path / filename).exists()
        checks.append(
            ValidationItem(
                name=filename,
                success=exists,
                detail="found" if exists else "missing",
            )
        )
    briefs_dir = path / "papers"
    brief_json_count = len(list(briefs_dir.rglob("paper_brief.json")))
    brief_typ_count = len(list(briefs_dir.rglob("paper_brief.typ")))
    checks.append(
        ValidationItem(
            name="brief_json_count",
            success=brief_json_count > 0,
            detail=f"found {brief_json_count} brief json files",
        )
    )
    checks.append(
        ValidationItem(
            name="brief_typ_count",
            success=brief_typ_count > 0,
            detail=f"found {brief_typ_count} typst files",
        )
    )
    checks.append(
        ValidationItem(
            name="selected_limit",
            success=brief_json_count <= 5,
            detail=f"selected {brief_json_count} papers",
        )
    )
    success = all(check.success for check in checks)
    return ValidationReport(run_dir=str(path), success=success, checks=checks)
