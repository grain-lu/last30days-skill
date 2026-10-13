"""Weixin (公众号) public article search for /last30days.

Opt-in only (INCLUDE_SOURCES=weixin or --search weixin / wx). Uses Sogou's
public Weixin article search — no WeChat login cookies. Optional
WEIXIN_SOURCES (comma-separated account names) scopes the query; without it
the topic is searched as-is.

This is not weixin_crawler. Do not scrape logged-in mp.weixin.qq.com sessions.
"""

from __future__ import annotations

import html as html_lib
import re
from typing import Any
from urllib.parse import urlencode, urljoin

from . import dates, http, log
from .relevance import token_overlap_relevance as _compute_relevance

SOGOU_SEARCH = "https://weixin.sogou.com/weixin"

DEPTH_PAGE_CAPS = {
    "quick": 1,
    "default": 2,
    "deep": 3,
}

_TITLE_RE = re.compile(
    r'<a[^>]*href="([^"]+)"[^>]*uigs="article_title_\d+"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_ACCOUNT_RE = re.compile(
    r'<a[^>]*uigs="account_name_\d+"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TIME_RE = re.compile(r"timeConvert\('(\d+)'\)")
_TAG_RE = re.compile(r"<[^>]+>")


def _log(msg: str) -> None:
    log.source_log("Weixin", msg, tty_only=False)


def parse_account_sources(raw: str) -> list[str]:
    """Parse comma-separated Weixin account names. Dedupes case-insensitively."""
    accounts: list[str] = []
    seen: set[str] = set()
    for part in (raw or "").split(","):
        name = part.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        accounts.append(name)
    return accounts


def _get_account_sources(config: dict[str, Any] | None) -> list[str]:
    config = config or {}
    return parse_account_sources(str(config.get("WEIXIN_SOURCES") or ""))


def is_weixin_configured(config: dict[str, Any] | None = None) -> bool:
    """Public search needs no key. Always True; opt-in is the pipeline gate."""
    return True


def _strip_tags(raw: str) -> str:
    text = html_lib.unescape(_TAG_RE.sub("", raw or ""))
    return re.sub(r"\s+", " ", text).strip()


def _parse_html(page: str, topic: str, start_index: int) -> list[dict[str, Any]]:
    if not page:
        return []
    titles = list(_TITLE_RE.finditer(page))
    accounts = [_strip_tags(m.group(1)) for m in _ACCOUNT_RE.finditer(page)]
    times = [m.group(1) for m in _TIME_RE.finditer(page)]
    items: list[dict[str, Any]] = []
    for idx, match in enumerate(titles):
        href = html_lib.unescape(match.group(1)).strip()
        title = _strip_tags(match.group(2))
        if not title:
            continue
        url = href if href.startswith("http") else urljoin("https://weixin.sogou.com", href)
        date_str = None
        if idx < len(times):
            try:
                date_str = dates.timestamp_to_date(float(times[idx]))
            except (TypeError, ValueError):
                date_str = None
        account = accounts[idx] if idx < len(accounts) else ""
        text = f"{title} {account}".strip()
        relevance = min(1.0, max(0.2, _compute_relevance(topic, text) * 0.7 + 0.3 - idx * 0.02))
        items.append({
            "id": f"WX{start_index + idx + 1}",
            "title": title,
            "url": url,
            "source_domain": "mp.weixin.qq.com",
            "snippet": title,
            "date": date_str,
            "date_confidence": "high" if date_str else "low",
            "relevance": round(relevance, 2),
            "why_relevant": f"Weixin {account}: {title[:60]}" if account else f"Weixin: {title[:60]}",
            "author": account or None,
            "engagement": {},
        })
    return items


def _fetch_page(query: str, page: int) -> str:
    params = {"type": "2", "query": query, "ie": "utf8", "page": page}
    url = f"{SOGOU_SEARCH}?{urlencode(params)}"
    return http.get_text(
        url,
        timeout=20,
        retries=1,
        accept="text/html",
        headers={"Referer": "https://weixin.sogou.com/"},
    ) or ""


def search_weixin(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search public Weixin articles. Returns {items} or {items, error}."""
    config = config or {}
    accounts = _get_account_sources(config)
    queries = [f"{account} {topic}".strip() for account in accounts] if accounts else [topic]
    max_pages = DEPTH_PAGE_CAPS.get(depth, DEPTH_PAGE_CAPS["default"])
    _log(
        f"Searching {len(queries)} query(ies) for '{topic}' "
        f"(depth={depth}, max_pages={max_pages})"
    )

    all_items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for query in queries:
        for page in range(1, max_pages + 1):
            html = _fetch_page(query, page)
            if not html:
                _log(f"Empty Sogou page for {query!r} p{page}")
                break
            parsed = _parse_html(html, topic, len(all_items))
            if not parsed:
                break
            page_all_old = True
            for item in parsed:
                url = item.get("url") or ""
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                all_items.append(item)
                if not item.get("date") or item["date"] >= from_date:
                    page_all_old = False
            if page_all_old:
                break

    in_range = [
        item for item in all_items
        if item.get("date") and from_date <= item["date"] <= to_date
    ]
    if in_range:
        items = in_range
        dropped = len(all_items) - len(in_range)
        if dropped:
            _log(f"Filtered {dropped} articles outside date range")
    else:
        items = all_items
        if all_items:
            _log(f"No articles within date range, keeping all {len(items)}")

    items.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    _log(f"Found {len(items)} Weixin articles")
    return {"items": items}


def parse_weixin_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    return response.get("items", [])
