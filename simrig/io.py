"""Input/output helpers for reports and run directories."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from simrig.core import report_markdown, to_dict


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "simrig"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_run_dir(env_name: str, preset: str, root: Path | str = "runs") -> Path:
    return Path(root) / f"{timestamp()}-{slugify(env_name)}-{slugify(preset)}"


def save_json(path: Path | str, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_dict(value), indent=2, sort_keys=True) + "\n")
    return output


def save_report_pair(
    report: Any,
    *,
    name: str,
    title: str,
    root: Path | str = "reports",
) -> tuple[Path, Path]:
    base = Path(root)
    stem = f"{slugify(name)}_inspection"
    json_path = base / f"{stem}.json"
    md_path = base / f"{stem}.md"
    save_json(json_path, report)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report_markdown(title, report), encoding="utf-8")
    return md_path, json_path

