"""Optional crawl4ai snippet enrichment for grounding results.

Off by default. Enable with LAST30DAYS_CRAWL4AI=1. Requires `crwl` on PATH
or an importable `crawl4ai` package. Never a required dependency.

Caps at a handful of short snippets so a research run cannot spawn a crawl farm.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from . import env, log

# ponytail: global cap, per-run lock if crawl4ai ever becomes default-on.
MAX_ENRICH = 5
SNIPPET_CHARS = 1200
TIMEOUT_SEC = 25
_MIN_SNIPPET = 80


def _log(msg: str) -> None:
    log.source_log("crawl4ai", msg, tty_only=False)


def is_enabled(config: dict[str, Any] | None = None) -> bool:
    config = config or {}
    return env._truthy(config.get("LAST30DAYS_CRAWL4AI") or config.get("CRAWL4AI_ENABLED"))


def is_available() -> bool:
    if shutil.which("crwl"):
        return True
    try:
        import crawl4ai  # noqa: F401
        return True
    except Exception:
        return False


def _extract_markdown(payload: Any) -> str:
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text
        else:
            return text
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return ""
    for key in ("markdown", "fit_markdown", "cleaned_html", "extracted_content"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            inner = val.get("raw_markdown") or val.get("fit_markdown") or val.get("markdown")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    result = payload.get("result") or payload.get("results")
    if result:
        return _extract_markdown(result)
    return ""


def _crwl_fetch(url: str) -> str:
    crwl = shutil.which("crwl")
    if not crwl:
        return ""
    try:
        proc = subprocess.run(
            [crwl, url, "-o", "markdown"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log(f"crwl failed for {url}: {exc}")
        return ""
    if proc.returncode != 0:
        _log(f"crwl exit {proc.returncode} for {url}: {(proc.stderr or '')[:200]}")
        return ""
    return _extract_markdown(proc.stdout)


def enrich_items(items: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Replace thin grounding snippets with crawl4ai markdown when enabled."""
    if not items or not is_enabled(config):
        return items
    if not is_available():
        _log("LAST30DAYS_CRAWL4AI set but crwl/crawl4ai not installed; skipping")
        return items

    enriched = 0
    for item in items:
        if enriched >= MAX_ENRICH:
            break
        url = str(item.get("url") or "")
        if not url.startswith("http"):
            continue
        snippet = str(item.get("snippet") or "")
        if len(snippet) >= _MIN_SNIPPET:
            continue
        markdown = _crwl_fetch(url)
        if not markdown:
            continue
        item["snippet"] = markdown[:SNIPPET_CHARS]
        item["enriched_via"] = "crawl4ai"
        enriched += 1
    if enriched:
        _log(f"Enriched {enriched} grounding snippet(s)")
    return items
