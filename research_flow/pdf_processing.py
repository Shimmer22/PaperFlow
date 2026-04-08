from __future__ import annotations

from pathlib import Path

import httpx
from pypdf import PdfReader

from research_flow.models import PDFReadResult, PaperRecord
from research_flow.utils import write_text


def fetch_and_extract_pdf(
    paper: PaperRecord,
    papers_dir: Path,
    timeout: int,
    max_chars: int,
) -> PDFReadResult:
    if not paper.pdf_url:
        return PDFReadResult(status="not_attempted", reading_depth="abstract-only", notes=["No pdf_url available."])
    paper_dir = papers_dir / paper.paper_id.replace(":", "_")
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / "paper.pdf"
    text_path = paper_dir / "paper_excerpt.txt"
    try:
        response = httpx.get(paper.pdf_url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)
    except Exception as exc:
        return PDFReadResult(
            status="download_failed",
            reading_depth="abstract-only",
            notes=[f"PDF download failed: {exc}"],
        )
    try:
        reader = PdfReader(str(pdf_path))
        chunks: list[str] = []
        for page in reader.pages[:10]:
            chunks.append(page.extract_text() or "")
        text = "\n\n".join(chunks).strip()[:max_chars]
        if not text:
            raise ValueError("No text extracted from PDF.")
        write_text(text_path, text)
        return PDFReadResult(
            status="parsed",
            pdf_path=str(pdf_path),
            extracted_text_path=str(text_path),
            reading_depth="partial-fulltext",
            notes=["Read the first several pages of the PDF."],
        )
    except Exception as exc:
        return PDFReadResult(
            status="parse_failed",
            pdf_path=str(pdf_path),
            reading_depth="abstract-only",
            notes=[f"PDF parse failed: {exc}"],
        )

