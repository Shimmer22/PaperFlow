import unittest

from research_flow.models import IdeaSpec, PaperRecord, RankedPaper, ScoreBreakdown
from research_flow.orchestrator import _display_ranked_candidates, _scout_candidate_pool


class CandidatePoolingTests(unittest.TestCase):
    def _idea(self) -> IdeaSpec:
        return IdeaSpec(
            raw_idea="DiT与MoE的结合",
            core_problem="DiT with MoE",
            keywords=["DiT", "MoE"],
            related_tasks=["diffusion transformer", "expert routing"],
            benchmark_methods=["FID"],
        )

    def _paper(self, index: int) -> PaperRecord:
        title = f"Paper {index}"
        if index == 11:
            title = "EC-DIT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing"
        return PaperRecord(
            paper_id=f"p{index}",
            title=title,
            abstract="Diffusion transformer mixture of experts routing for image generation.",
            source="openalex",
            retrieved_by_query=["DiT MoE Diffusion Transformer Mixture of Experts"],
            metadata_completeness=0.9,
        )

    def test_scout_pool_keeps_more_than_ui_topk(self) -> None:
        papers = [self._paper(index) for index in range(12)]
        pool = _scout_candidate_pool(papers, self._idea(), max_candidates=5)
        self.assertGreater(len(pool), 5)
        self.assertLessEqual(len(pool), 15)
        self.assertIn(
            "EC-DIT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing",
            [paper.title for paper in pool],
        )

    def test_scout_pool_is_capped_for_large_candidate_limit(self) -> None:
        papers = [self._paper(index) for index in range(60)]
        pool = _scout_candidate_pool(papers, self._idea(), max_candidates=10)
        self.assertEqual(len(pool), 15)

    def test_display_ranked_candidates_respects_limit(self) -> None:
        ranked = [
            RankedPaper(
                paper=self._paper(index),
                score_breakdown=ScoreBreakdown(total=1.0 - index * 0.01),
                selected=index < 2,
                selection_reason="test",
            )
            for index in range(12)
        ]
        displayed = _display_ranked_candidates(ranked, max_candidates=10)
        self.assertEqual(len(displayed), 10)
        self.assertEqual(displayed[0].paper.paper_id, "p0")


if __name__ == "__main__":
    unittest.main()
