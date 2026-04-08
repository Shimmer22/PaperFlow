from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from research_flow.models import BibliographicInfo, IdeaSpec, PDFReadResult, PaperBrief, PaperRecord
from research_flow.providers.base import BaseCLIProvider
from research_flow.prompts_loader import PromptLibrary
from research_flow.utils import ensure_dir, safe_excerpt, write_json


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。；;])\s+", text.strip()) if part.strip()]


def _pick_sentence(sentences: list[str], keywords: list[str], fallback: str) -> str:
    lowered_keywords = [keyword.lower() for keyword in keywords if keyword]
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in lowered_keywords):
            return sentence
    return fallback


def _abstract_analysis(paper: PaperRecord, idea: IdeaSpec) -> dict[str, object]:
    sentences = _split_sentences(paper.abstract or "")
    title_lower = paper.title.lower()
    abstract_lower = (paper.abstract or "").lower()
    method_keywords = [
        "propose",
        "introduce",
        "framework",
        "method",
        "approach",
        "pruning",
        "compression",
        "sparsification",
        "reduction",
        "merge",
        "split",
    ]
    result_keywords = [
        "improves",
        "reduce",
        "reduction",
        "latency",
        "bandwidth",
        "accuracy",
        "speedup",
        "flops",
        "throughput",
    ]
    idea_keywords = idea.keywords + idea.related_tasks + idea.benchmark_methods
    one_sentence = sentences[0] if sentences else paper.title
    method_line = _pick_sentence(
        sentences,
        method_keywords,
        safe_excerpt(paper.abstract or paper.title, 220),
    )
    result_line = _pick_sentence(
        sentences,
        result_keywords,
        "摘要中未明确给出足够具体的量化结果。",
    )
    relation_hits = [keyword for keyword in idea_keywords if keyword.lower() in f"{title_lower} {abstract_lower}"]
    return {
        "one_sentence": safe_excerpt(one_sentence, 180),
        "method_line": safe_excerpt(method_line, 260),
        "result_line": safe_excerpt(result_line, 220),
        "relation_hits": relation_hits[:6],
        "sentences": sentences,
    }


def _infer_method_summary_cn(paper: PaperRecord, analysis: dict[str, object]) -> str:
    text = f"{paper.title} {paper.abstract}".lower()
    method_tags: list[str] = []
    if "adaptive" in text:
        method_tags.append("自适应策略")
    if "token pruning" in text or "prune" in text:
        method_tags.append("token 剪枝")
    if "token reduction" in text or "reduction" in text:
        method_tags.append("token 缩减")
    if "sparsification" in text or "sparse" in text:
        method_tags.append("稀疏化")
    if "merge" in text:
        method_tags.append("token 合并")
    if "text-guided" in text or "prompt-aware" in text:
        method_tags.append("文本/提示引导")
    if "noise gating" in text or "gating" in text:
        method_tags.append("门控选择")
    if "split" in text or "edge-cloud" in text or "communication" in text:
        method_tags.append("通信/协同约束")

    if method_tags:
        return f"这篇论文的核心方法可概括为：通过{'、'.join(dict.fromkeys(method_tags))}来减少视觉或多模态 token 开销，从而提升推理效率。"
    return f"这篇论文大致方法是：{analysis['method_line']}"


def _infer_result_summary_cn(paper: PaperRecord) -> str:
    text = f"{paper.title} {paper.abstract}".lower()
    benefits: list[str] = []
    if "latency" in text:
        benefits.append("降低推理延迟")
    if "bandwidth" in text or "communication" in text:
        benefits.append("降低通信或带宽压力")
    if "flops" in text or "compute" in text or "efficient" in text:
        benefits.append("减少计算开销")
    if "accuracy" in text or "performance" in text:
        benefits.append("尽量保持任务性能")
    if benefits:
        return f"摘要表明它主要试图{'、'.join(dict.fromkeys(benefits))}。"
    return "摘要说明它主要关注 token 级效率优化，但缺少足够具体的量化结果。"


def build_paper_brief_fallback(
    paper: PaperRecord,
    idea: IdeaSpec,
    pdf_result: PDFReadResult,
) -> PaperBrief:
    evidence = "仅使用摘要与元数据。"
    if pdf_result.reading_depth != "abstract-only":
        evidence = "使用了摘要、元数据以及部分 PDF 文本。"
    analysis = _abstract_analysis(paper, idea)
    relation_terms = "、".join(analysis["relation_hits"]) if analysis["relation_hits"] else "VLM、token 压缩、带宽/延迟优化中的部分主题"
    return PaperBrief(
        bibliographic_info=BibliographicInfo(
            title=paper.title,
            authors=paper.authors,
            year=paper.year,
            venue=paper.venue,
            url=paper.url,
            pdf_url=paper.pdf_url,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            sources=sorted({entry.source for entry in paper.provenance} or {paper.source}),
        ),
        one_sentence_summary=analysis["one_sentence"],
        research_problem=(
            safe_excerpt(analysis["sentences"][0], 300)
            if analysis["sentences"]
            else safe_excerpt(paper.abstract or "暂无摘要可用。", 300)
        ),
        core_method=_infer_method_summary_cn(paper, analysis),
        key_innovations=[
            "从摘要中抽取了论文声称的方法主线。",
            "保留 provenance 与阅读深度，便于审计与后续重跑。",
        ],
        experiment_summary=(
            "当前主要依据摘要推断实验设置，缺少表格、消融和实现细节。"
            if pdf_result.reading_depth == "abstract-only"
            else "已读取部分 PDF，可获得有限实验线索，但仍建议结合原文确认。"
        ),
        main_results=[_infer_result_summary_cn(paper)],
        strengths=[
            "即使 provider 或 PDF 解析失败，也能从摘要中提取出比模板更具体的方法与结果线索。"
        ],
        limitations=[
            "当前不是基于完整正文的深读结果，可能遗漏关键实现、实验设定和失败案例。",
            *pdf_result.notes,
        ],
        assumptions=["默认摘要能够基本反映论文主问题与主要贡献。"],
        relation_to_user_idea=(
            f"它和你的想法有交集，主要都涉及 {relation_terms}；"
            "但它是否真的建模了端云协同里的通信约束，还需要阅读全文才能确认。"
        ),
        reusable_parts=["可继续扩展的检索关键词", "可能可复用的压缩/剪枝思路", "候选 baseline 方向"],
        open_questions=["该论文是否真的处理了端云协同中的通信约束，还是只优化了本地推理 token 计算？"],
        confidence_notes=[evidence, *pdf_result.notes],
        reading_depth=pdf_result.reading_depth,
    )


def maybe_generate_brief_with_provider(
    provider: Optional[BaseCLIProvider],
    prompt_library: PromptLibrary,
    paper: PaperRecord,
    idea: IdeaSpec,
    pdf_result: PDFReadResult,
    timeout: int,
    output_path: Path,
    runtime_options: Optional[dict[str, str]] = None,
) -> PaperBrief:
    if provider is None:
        return build_paper_brief_fallback(paper, idea, pdf_result)
    prompt = prompt_library.load("paper_briefer.txt")
    context = {
        "idea_spec": idea.model_dump(),
        "paper": paper.model_dump(),
        "pdf_result": pdf_result.model_dump(),
        "instructions": "Return only a JSON object matching the output schema.",
    }
    result = provider.run_subtask(
        prompt=prompt + "\n\nCONTEXT:\n" + json.dumps(context, ensure_ascii=False, indent=2),
        context=context,
        expected_output_schema=PaperBrief,
        output_path=output_path,
        timeout=timeout,
        runtime_options=runtime_options,
    )
    ensure_dir(output_path.parent)
    write_json(output_path.parent / "provider_call_result.json", result.model_dump())
    if result.success and result.parsed_output:
        return PaperBrief.model_validate(result.parsed_output)
    return build_paper_brief_fallback(paper, idea, pdf_result)
