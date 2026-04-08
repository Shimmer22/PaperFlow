from __future__ import annotations

import sys
from pathlib import Path

from research_flow.utils import write_json
from research_flow.validation import validate_run_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python validate/validate_outputs.py <run_dir>")
        return 1
    run_dir = Path(sys.argv[1])
    report = validate_run_dir(run_dir)
    out_path = run_dir / "validation_report.json"
    write_json(out_path, report.model_dump())
    print(report.model_dump_json(indent=2))
    return 0 if report.success else 2


if __name__ == "__main__":
    raise SystemExit(main())

