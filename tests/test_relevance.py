import unittest

from research_flow.models import IdeaSpec, PaperRecord
from research_flow.scouting import _local_scout_report


class RelevanceScoringTests(unittest.TestCase):
    def test_local_scout_marks_ec_dit_as_worth_reading_for_dit_moe(self) -> None:
        idea = IdeaSpec(
            raw_idea="DiT与MoE的结合",
            core_problem="Combine Diffusion Transformer with Mixture of Experts",
            keywords=["Diffusion Transformer", "DiT", "Mixture of Experts", "MoE"],
            related_tasks=["expert routing", "text-to-image generation"],
            benchmark_methods=["FID", "inference latency"],
        )
        paper = PaperRecord(
            paper_id="p1",
            title="EC-DIT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing",
            abstract=(
                "We develop a new family of Mixture-of-Experts models for diffusion transformers "
                "with expert-choice routing for text-to-image synthesis."
            ),
            source="openalex",
            metadata_completeness=0.9,
        )
        report = _local_scout_report(idea, paper)
        self.assertTrue(report.worth_reading)
        self.assertGreaterEqual(report.relevance_score, 0.45)

    def test_local_scout_preserves_communication_anchor_matching(self) -> None:
        idea = IdeaSpec(
            raw_idea="波束成形与自注意力结合",
            core_problem="Combine beamforming with self-attention",
            keywords=["beamforming", "self-attention"],
            related_tasks=["wireless communication", "antenna array optimization"],
            benchmark_methods=["latency", "throughput"],
        )
        paper = PaperRecord(
            paper_id="p2",
            title="Transformer Beamforming for Wireless Communication",
            abstract="A self-attention architecture for beamforming over antenna arrays in wireless systems.",
            source="openalex",
            metadata_completeness=0.9,
        )
        report = _local_scout_report(idea, paper)
        self.assertTrue(report.worth_reading)
        self.assertGreaterEqual(report.relevance_score, 0.45)


if __name__ == "__main__":
    unittest.main()
