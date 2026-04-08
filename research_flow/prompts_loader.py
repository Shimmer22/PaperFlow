from __future__ import annotations

from pathlib import Path
from typing import Union


class PromptLibrary:
    def __init__(self, prompts_dir: Union[str, Path]) -> None:
        self.prompts_dir = Path(prompts_dir)

    def load(self, name: str) -> str:
        path = self.prompts_dir / name
        return path.read_text(encoding="utf-8")
