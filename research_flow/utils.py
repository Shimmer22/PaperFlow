from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union


def setup_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("research_flow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def ensure_dir(path: Union[str, Path]) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def write_json(path: Union[str, Path], data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def read_json(path: Union[str, Path]) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Union[str, Path], content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(content)


def slugify(text: str, max_length: int = 80) -> str:
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    text = re.sub(r"[-\s]+", "-", text)
    return text[:max_length].strip("-") or "item"


def normalize_title(title: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return re.sub(r"\s+", " ", text).strip()


def now_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def detect_typst() -> Optional[str]:
    return shutil.which("typst")


def safe_excerpt(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def env_with_updates(updates: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(updates)
    return env


def load_dotenv_if_present(path: Union[str, Path]) -> dict[str, str]:
    env_updates: dict[str, str] = {}
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return env_updates
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            env_updates[key] = value
    os.environ.update(env_updates)
    return env_updates


def set_nested_value(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = target
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        return
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value
