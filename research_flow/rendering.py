from __future__ import annotations

import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from research_flow.models import PaperBrief
from research_flow.utils import detect_typst, write_text


def render_brief_typst(template_path: Path, brief: PaperBrief, output_path: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)
    content = template.render(brief=brief.model_dump())
    write_text(output_path, content)


def compile_typst_if_available(typ_path: Path) -> tuple[bool, str]:
    typst_bin = detect_typst()
    if not typst_bin:
        return False, "Typst binary not found; kept .typ source only."
    pdf_path = typ_path.with_suffix(".pdf")
    try:
        subprocess.run([typst_bin, "compile", str(typ_path), str(pdf_path)], check=True, capture_output=True, text=True)
        return True, str(pdf_path)
    except Exception as exc:
        return False, f"Typst compile failed: {exc}"

