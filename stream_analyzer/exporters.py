"""Reports and saved runs."""
from __future__ import annotations

import json
import re
from typing import Optional

from .config import OUT


def safe(name: str) -> str:
    return re.sub(r"[^\w.\-() ]", "_", name) or "document"


def to_markdown(r: dict) -> str:
    rows = "\n".join(f"| {b['start']} → {b['end']} | {b['topic']} | {b['summary']} |" for b in r["timeline"])
    return f"# {r['name']}\n\n{r['overview']}\n\n## Topic timeline\n\n| When | Topic | Summary |\n|---|---|---|\n{rows}\n"


def load_last_run() -> Optional[dict]:
    files = sorted(OUT.glob("*/results.json")) if OUT.exists() else []
    try:
        return json.loads(files[-1].read_text(encoding="utf-8")) if files else None
    except Exception:
        return None
