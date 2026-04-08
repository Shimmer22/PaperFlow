import unittest

from research_flow.models import IdeaSpec, PaperRecord, ScoutReport
from research_flow.scouting import build_ranked_from_scout, local_select_from_scout


class ScoutingChainTests(unittest.TestCase):
    def _idea(self) -> IdeaSpec:
        return IdeaSpec(
            raw_idea="波束成形和自注意力结合",
            core_problem="波束成形和自注意力结合",
            keywords=["beamforming", "self-attention"],
            related_tasks=["beamforming-aware model design"],
            benchmark_methods=["self-attention"],
        )

    def _paper(self, pid: str, title: str) -> PaperRecord:
        return PaperRecord(
            paper_id=pid,
            title=title,
            abstract="test abstract",
            source="openalex",
            retrieved_by_query=["beamforming self-attention"],
            metadata_completeness=0.8,
        )

    def test_local_select_from_scout_uses_total_score(self) -> None:
        reports = [
            ScoutReport(paper_id="p1", title="A", relevance_score=0.9, novelty_score=0.5, feasibility_score=0.5, confidence_score=0.8, worth_reading=True, main_findings=["a"], risk_notes=[], relation_to_idea="x"),
            ScoutReport(paper_id="p2", title="B", relevance_score=0.8, novelty_score=0.9, feasibility_score=0.8, confidence_score=0.8, worth_reading=True, main_findings=["b"], risk_notes=[], relation_to_idea="y"),
        ]
        selected = local_select_from_scout(reports, max_selected=1)
        self.assertEqual(selected, ["p2"])

    def test_build_ranked_from_scout_marks_selected(self) -> None:
        papers = [self._paper("p1", "A"), self._paper("p2", "B")]
        reports = [
            ScoutReport(paper_id="p1", title="A", relevance_score=0.7, novelty_score=0.7, feasibility_score=0.7, confidence_score=0.7, worth_reading=True, main_findings=["a"], risk_notes=[], relation_to_idea="x"),
            ScoutReport(paper_id="p2", title="B", relevance_score=0.6, novelty_score=0.6, feasibility_score=0.6, confidence_score=0.7, worth_reading=True, main_findings=["b"], risk_notes=[], relation_to_idea="y"),
        ]
        ranked = build_ranked_from_scout(papers, reports, selected_ids=["p1"])
        picked = [item.paper.paper_id for item in ranked if item.selected]
        self.assertEqual(picked, ["p1"])


if __name__ == "__main__":
    unittest.main()
