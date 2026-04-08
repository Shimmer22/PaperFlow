from __future__ import annotations

import json
from typing import Any, Optional

from research_flow.models import ClarificationTurn
from research_flow.providers.base import BaseCLIProvider
from research_flow.prompts_loader import PromptLibrary

UNSURE_OPTION = {
    "id": "unsure_model_think",
    "label": "我不确定，让模型先思考并给出建议",
    "description": "当你暂时无法判断时，主模型应先基于现有信息提出可执行假设。",
}


def enforce_unsure_option(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixed = [dict(item) for item in options]
    if any(str(item.get("id", "")).strip() == UNSURE_OPTION["id"] for item in fixed):
        return fixed
    fixed.append(dict(UNSURE_OPTION))
    return fixed


def build_local_turn(raw_idea: str, history: list[dict[str, Any]]) -> ClarificationTurn:
    base_question = "你更希望优先调研哪一部分？"
    if not history:
        base_question = "先把方向对齐：你这次更想优先解决哪类问题？"
    options = enforce_unsure_option(
        [
            {
                "id": "impl_path",
                "label": "实现路径",
                "description": "关注可落地架构、模块边界、训练或部署路线。",
            },
            {
                "id": "research_gap",
                "label": "Research Gaps",
                "description": "关注现有文献空白、未覆盖场景和潜在创新点。",
            },
            {
                "id": "evaluation",
                "label": "评测与验证",
                "description": "关注数据集、指标、对照方法与实验设计。",
            },
        ]
    )
    return ClarificationTurn(
        question=base_question,
        options=options,
        ready_for_research=len(history) >= 2,
        research_aspects=["实现路径", "research gaps", "评测与风险"],
        assistant_notes="可继续补充 1-2 轮后进入检索。",
    )


def maybe_generate_clarification_turn_with_provider(
    raw_idea: str,
    history: list[dict[str, Any]],
    provider: Optional[BaseCLIProvider],
    prompt_library: PromptLibrary,
    timeout: int,
    runtime_options: Optional[dict[str, str]] = None,
) -> tuple[ClarificationTurn, Optional[dict[str, Any]]]:
    local_turn = build_local_turn(raw_idea, history)
    if provider is None:
        return local_turn, None

    prompt = prompt_library.load("clarification_turn.txt")
    context = {
        "raw_idea": raw_idea,
        "history": history,
        "fallback_local_turn": local_turn.model_dump(),
        "instructions": "Return only one JSON object matching the schema.",
    }
    result = provider.run_task(
        prompt=prompt + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, indent=2),
        context=context,
        expected_output_schema=ClarificationTurn,
        output_path=None,
        timeout=timeout,
        runtime_options=runtime_options,
    )

    if result.success and result.parsed_output:
        turn = ClarificationTurn.model_validate(result.parsed_output)
        turn.options = [
            type(turn.options[0]).model_validate(item)
            for item in enforce_unsure_option([opt.model_dump() for opt in turn.options])
        ] if turn.options else [type(local_turn.options[0]).model_validate(UNSURE_OPTION)]
        return turn, result.model_dump()

    local_turn.options = [type(local_turn.options[0]).model_validate(item) for item in enforce_unsure_option([opt.model_dump() for opt in local_turn.options])]
    return local_turn, result.model_dump()
