from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import httpx


def load_dotenv(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser(description="Temporary GLM API smoke test.")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--url", default="https://open.bigmodel.cn/api/paas/v4/chat/completions")
    parser.add_argument("--url-env-var", default="OPENAI_BASE_URL")
    parser.add_argument("--model", default="glm-4.7-flash")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    load_dotenv(Path(args.dotenv_path))
    api_key = os.environ.get("API_KEY", "").strip()
    url = os.environ.get(args.url_env_var, "").strip() or args.url
    if not api_key:
        print(json.dumps({"ok": False, "error": "API_KEY not found in env or .env"}, ensure_ascii=False))
        return 1

    payload = {
        "model": args.model,
        "stream": False,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "你是一个严格输出 JSON 的助手。",
            },
            {
                "role": "user",
                "content": '请只返回 JSON：{"ok": true, "model_echo": "<你看到的模型名>", "message": "hello"}',
            },
        ],
    }

    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        response.raise_for_status()
        body = response.json()
        content = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content", "")
        print(json.dumps({"ok": True, "url": url, "http_status": response.status_code, "content": content}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
