from __future__ import annotations

import json

from typing import Optional

from research_flow.models import FinalDiscussion, IdeaSpec, PaperBrief
from research_flow.providers.base import BaseCLIProvider
from research_flow.prompts_loader import PromptLibrary


def _summarize_brief_cluster(briefs: list[PaperBrief]) -> str:
    text = " ".join(
        " ".join(
            [
                brief.bibliographic_info.title,
                brief.core_method,
                " ".join(brief.main_results),
            ]
        ).lower()
        for brief in briefs
    )
    themes: list[str] = []
    if "pruning" in text or "剪枝" in text:
        themes.append("visual/token 剪枝")
    if "spars" in text or "稀疏" in text:
        themes.append("token 稀疏化")
    if "merge" in text or "合并" in text:
        themes.append("token 合并")
    if "latency" in text or "延迟" in text:
        themes.append("降低推理延迟")
    if "bandwidth" in text or "通信" in text:
        themes.append("控制通信或带宽成本")
    if "accuracy" in text or "性能" in text:
        themes.append("尽量保持精度")
    if not themes:
        return "这些论文大多围绕 token 级效率优化展开。"
    return f"这些论文主要集中在{'、'.join(dict.fromkeys(themes))}。"


def _idea_text(idea: IdeaSpec) -> str:
    return " ".join([idea.core_problem, *idea.keywords, *idea.related_tasks, *idea.benchmark_methods]).lower()


def _fallback_open_spaces(idea: IdeaSpec) -> list[str]:
    text = _idea_text(idea)
    spaces = [f"围绕“{idea.core_problem}”做更细粒度的问题拆分、约束建模与评价协议设计"]
    if "beamforming" in text or "波束成形" in text:
        spaces.append("明确区分传统波束成形优化、神经波束成形与注意力增强波束成形三类路线")
    if "self-attention" in text or "自注意力" in text or "attention" in text:
        spaces.append("分析自注意力究竟用于波束生成、特征选择还是参数共享，并分别建立对照实验")
    if "model compression" in text or "parameter reduction" in text or "减少模型权重" in text:
        spaces.append("联合考虑参数量、推理时延与性能损失，而不只优化单一压缩指标")
    return spaces[:3]


def _fallback_innovation_angles(idea: IdeaSpec) -> list[str]:
    text = _idea_text(idea)
    angles = [f"围绕“{idea.core_problem}”找到更明确的系统瓶颈定义与评价协议。"]
    if "beamforming" in text or "波束成形" in text:
        angles.append("把注意力机制显式映射到波束选择、通道建模或阵列特征聚合的具体环节。")
    if "self-attention" in text or "自注意力" in text:
        angles.append("比较标准自注意力、稀疏注意力和结构化注意力在参数缩减上的真实收益。")
    if "model compression" in text or "parameter reduction" in text or "减少模型权重" in text:
        angles.append("把方法分成“减少参数量”与“减少推理计算量”两类，避免概念混淆。")
    angles.append("在自动化之外，显式保留置信说明与 provenance。")
    return angles[:4]


def synthesize_fallback(idea: IdeaSpec, briefs: list[PaperBrief]) -> FinalDiscussion:
    titles = [brief.bibliographic_info.title for brief in briefs]
    limitations = [lim for brief in briefs for lim in brief.limitations[:1]]
    joined_keywords = "、".join(idea.keywords[:5]) if idea.keywords else idea.core_problem
    saturated_areas = [f"{joined_keywords} 相关路线已有不少工作"] if titles else [f"{joined_keywords} 方向目前缺少可靠 shortlist 证据"]
    open_spaces = _fallback_open_spaces(idea)
    relation_notes = [brief.relation_to_user_idea for brief in briefs[:3]]
    method_notes = [brief.core_method for brief in briefs[:2]]
    cluster_summary = _summarize_brief_cluster(briefs)
    next_steps = [
        "检查 top brief JSON，继续收窄问题定义。",
        "当初始候选可信后，再做 citation chasing 扩展。",
        "在单篇论文 deep-read 阶段测试不同 provider 的并行效果。",
    ]
    evidence_notes = ["当前综合讨论来自结构化 brief 与元数据降级结果。"]
    if not briefs:
        next_steps = [
            "先恢复网络访问或使用缓存检索结果，然后重新运行。",
            "也可以先手工给一小批 seed papers，用来验证 deep-read 与 Typst 产物。",
            "拿到可靠论文后，再开启并行单篇论文处理重跑。",
        ]
        evidence_notes = ["当前没有 shortlist，因此综合讨论只能基于澄清后的 idea spec。"]
    return FinalDiscussion(
        refined_problem_definition=idea.core_problem,
        literature_coverage=(
            f"当前 shortlist 包含 {len(briefs)} 篇论文，代表性论文包括：{', '.join(titles[:3])}。"
            f" 从现有 brief 看，{cluster_summary}"
            if titles
            else "当前没有可用的入选论文。"
        ),
        saturated_areas=saturated_areas,
        open_spaces=open_spaces,
        priority_papers=titles[:3],
        innovation_angles=[
            * _fallback_innovation_angles(idea),
        ],
        risks_and_failure_modes=list(dict.fromkeys(limitations + relation_notes[:2])) or ["网络或论文源 API 不稳定会直接限制召回效果。"],
        next_steps=next_steps,
        evidence_notes=evidence_notes + method_notes[:2],
    )


def maybe_synthesize_with_provider(
    provider: Optional[BaseCLIProvider],
    prompt_library: PromptLibrary,
    idea: IdeaSpec,
    briefs: list[PaperBrief],
    timeout: int,
    runtime_options: Optional[dict[str, str]] = None,
) -> FinalDiscussion:
    if not briefs:
        return synthesize_fallback(idea, briefs)
    if provider is None:
        return synthesize_fallback(idea, briefs)
    prompt = prompt_library.load("final_synthesizer.txt")
    context = {
        "idea_spec": idea.model_dump(),
        "paper_briefs": [brief.model_dump() for brief in briefs],
        "instructions": "Return only a JSON object matching the output schema.",
    }
    result = provider.run_task(
        prompt=prompt + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, indent=2),
        context=context,
        expected_output_schema=FinalDiscussion,
        output_path=None,
        timeout=timeout,
        runtime_options=runtime_options,
    )
    if result.success and result.parsed_output:
        return FinalDiscussion.model_validate(result.parsed_output)
    return synthesize_fallback(idea, briefs)
