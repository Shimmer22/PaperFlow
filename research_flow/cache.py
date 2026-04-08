from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Union

from research_flow.utils import read_json, write_json


class JsonCache:
    """Very small file-based cache for retrieval results and provider artifacts."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        digest = sha256(key.encode("utf-8")).hexdigest()
        return self.root / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        return read_json(path)

    def set(self, namespace: str, key: str, value: Any) -> Path:
        path = self._path(namespace, key)
        write_json(path, value)
        return path
