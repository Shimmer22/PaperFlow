import unittest
from pathlib import Path

from research_flow.models import IdeaSpec, ProviderCallResult, ProviderCapabilities, QueryPlan
from research_flow.prompts_loader import PromptLibrary
from research_flow.providers.base import BaseCLIProvider
from research_flow.query_planner import build_query_plan, maybe_build_query_plan_with_provider


class _FakeProvider(BaseCLIProvider):
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def check_available(self) -> tuple[bool, str]:
        return True, "ok"

    def describe_provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="fake",
            supports_subtasks=True,
            supports_parallel_invocations=True,
            supports_output_schema=True,
            prompt_mode="http",
            output_mode="http",
        )

    def run_task(
        self,
        prompt: str,
        context: dict,
        expected_output_schema,
        output_path,
        timeout: int,
        extra_args=None,
        runtime_options=None,
    ) -> ProviderCallResult:
        del prompt, context, expected_output_schema, output_path, timeout, extra_args, runtime_options
        return ProviderCallResult(success=True, raw_output="", parsed_output=self.payload)


class QueryPlannerRecallTests(unittest.TestCase):
    def _idea(self) -> IdeaSpec:
        return IdeaSpec(
            raw_idea="DiT与MoE的结合",
            core_problem="Combining Diffusion Transformer (DiT) with Mixture of Experts (MoE) for scalable image generation",
            keywords=["Diffusion Transformer", "DiT", "Mixture of Experts", "MoE"],
            related_tasks=["scalable transformer-based diffusion", "sparse MoE for generative models"],
            benchmark_methods=["FID", "inference latency"],
        )

    def test_local_plan_adds_dit_moe_title_like_variants(self) -> None:
        plan = build_query_plan(self._idea(), ["openalex", "arxiv"])
        all_queries = [group.query_text for group in plan.iter_queries()]
        joined_queries = " || ".join(all_queries).lower()
        joined_semantic = " || ".join(plan.semantic_queries).lower()
        self.assertIn("expert-choice routing", joined_queries)
        self.assertIn("dit-moe", joined_queries)
        self.assertIn("ec-dit", joined_semantic)

    def test_provider_plan_is_enriched_with_local_recall_variants(self) -> None:
        sparse_provider_plan = QueryPlan(
            precise_queries=[],
            broad_queries=[],
            method_centric_queries=[],
            application_centric_queries=[],
            semantic_queries=["Diffusion Transformer MoE"],
            synonym_expansions=[],
            alternative_method_terms=[],
            citation_expansion_strategy="provider only",
        )
        plan, _ = maybe_build_query_plan_with_provider(
            idea=self._idea(),
            enabled_sources=["openalex", "arxiv"],
            provider=_FakeProvider(sparse_provider_plan.model_dump()),
            prompt_library=PromptLibrary(Path("prompts")),
            timeout=10,
            runtime_options={"thinking_enabled": "true"},
        )
        all_queries = [group.query_text for group in plan.iter_queries()]
        joined_queries = " || ".join(all_queries).lower()
        joined_semantic = " || ".join(plan.semantic_queries).lower()
        self.assertIn("expert-choice routing", joined_queries)
        self.assertIn("ec-dit", joined_semantic)
        self.assertIn("Diffusion Transformer MoE", plan.semantic_queries)


if __name__ == "__main__":
    unittest.main()
