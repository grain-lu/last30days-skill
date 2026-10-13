"""Bilibili public video search for /last30days.

Opt-in only (INCLUDE_SOURCES=bilibili or --search bilibili / bili).
Uses the public web search API; no Agent-Reach / login required.
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlencode

from . import dates, http, log
from .relevance import token_overlap_relevance as _compute_relevance

# Unsigned public type search. WBI-signed endpoint 412s without cookies.
SEARCH_URL_PLAIN = "https://api.bilibili.com/x/web-interface/search/type"

DEPTH_PAGE_CAPS = {
    "quick": 1,
    "default": 2,
    "deep": 3,
}

BROWSER_HEADERS = {
    "User-Agent": http.BROWSER_USER_AGENT,
    "Referer": "https://search.bilibili.com/",
    "Origin": "https://search.bilibili.com",
}


def _log(msg: str) -> None:
    log.source_log("Bilibili", msg, tty_only=False)


def is_bilibili_configured(config: dict[str, Any] | None = None) -> bool:
    return True


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0
    try:
        if text.endswith("万"):
            return int(float(text[:-1]) * 10000)
        if text.endswith("亿"):
            return int(float(text[:-1]) * 100000000)
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _parse_date(raw: dict[str, Any]) -> str | None:
    for key in ("pubdate", "created", "senddate"):
        val = raw.get(key)
        if val in (None, "", 0):
            continue
        try:
            return dates.timestamp_to_date(float(val))
        except (TypeError, ValueError):
            continue
    return None


def _parse_video(raw: dict[str, Any], topic: str, index: int) -> dict[str, Any]:
    bvid = str(raw.get("bvid") or "").strip()
    aid = raw.get("aid") or raw.get("id")
    title = str(raw.get("title") or "").replace('<em class="keyword">', "").replace("</em>", "").strip()
    desc = str(raw.get("description") or raw.get("desc") or "").strip()
    author = str(raw.get("author") or raw.get("uname") or "").strip()
    if bvid:
        url = f"https://www.bilibili.com/video/{bvid}"
        item_id = bvid
    elif aid:
        url = f"https://www.bilibili.com/video/av{aid}"
        item_id = f"av{aid}"
    else:
        url = ""
        item_id = f"BL{index + 1}"

    stat = raw.get("stat") if isinstance(raw.get("stat"), dict) else {}
    views = _to_int(raw.get("play") or raw.get("view") or stat.get("view"))
    likes = _to_int(raw.get("like") or stat.get("like"))
    comments = _to_int(raw.get("review") or raw.get("video_review") or stat.get("reply"))
    date_str = _parse_date(raw)
    text = f"{title} {desc} {author}"
    text_relevance = _compute_relevance(topic, text)
    rank_score = max(0.3, 1.0 - (index * 0.02))
    engagement_boost = min(0.2, math.log1p(views + likes * 10) / 50)
    relevance = min(1.0, text_relevance * 0.5 + rank_score * 0.3 + engagement_boost + 0.1)
    return {
        "id": item_id,
        "title": title,
        "url": url,
        "source_domain": "bilibili.com",
        "snippet": desc[:400] or title,
        "date": date_str,
        "date_confidence": "high" if date_str else "low",
        "relevance": round(relevance, 2),
        "why_relevant": f"Bilibili {author}: {title[:60]}" if author else f"Bilibili: {title[:60]}",
        "author": author or None,
        "engagement": {
            "views": views,
            "likes": likes,
            "comments": comments,
        },
    }


def _fetch_page(query: str, page: int) -> dict[str, Any]:
    params = {
        "search_type": "video",
        "keyword": query,
        "page": page,
        "order": "pubdate",
    }
    url = f"{SEARCH_URL_PLAIN}?{urlencode(params)}"
    try:
        data = http.get(url, headers=BROWSER_HEADERS, timeout=20, retries=1)
    except http.HTTPError as exc:
        _log(f"HTTP error searching Bilibili: {exc}")
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def search_bilibili(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search public Bilibili videos. Returns {items} or {items, error}."""
    max_pages = DEPTH_PAGE_CAPS.get(depth, DEPTH_PAGE_CAPS["default"])
    _log(f"Searching '{topic}' (depth={depth}, max_pages={max_pages})")
    all_items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        data = _fetch_page(topic, page)
        if not data:
            if page == 1:
                return {"items": [], "error": "Bilibili search returned empty"}
            break
        payload = data.get("data") or {}
        results = payload.get("result") or payload.get("list") or []
        if isinstance(results, dict):
            results = results.get("result") or []
        if not results:
            break
        page_all_old = True
        for raw in results:
            if not isinstance(raw, dict):
                continue
            parsed = _parse_video(raw, topic, len(all_items))
            if parsed["id"] in seen:
                continue
            seen.add(parsed["id"])
            all_items.append(parsed)
            if not parsed.get("date") or parsed["date"] >= from_date:
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
            _log(f"Filtered {dropped} videos outside date range")
    else:
        items = all_items
        if all_items:
            _log(f"No videos within date range, keeping all {len(items)}")

    items.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    _log(f"Found {len(items)} Bilibili videos")
    return {"items": items}


def parse_bilibili_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    return response.get("items", [])
