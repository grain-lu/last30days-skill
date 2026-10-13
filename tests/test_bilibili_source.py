"""Bilibili opt-in source tests. HTTP is fully mocked."""

from __future__ import annotations

from unittest import mock

import last30days as cli
from lib import pipeline, bilibili


MOCK_RESPONSE = {
    "code": 0,
    "data": {
        "result": [
            {
                "bvid": "BV1xx411c7mD",
                "title": 'OpenClaw <em class="keyword">agent</em> tutorial',
                "description": "A recent walkthrough of agent workflows.",
                "author": "up主",
                "pubdate": 1755900000,
                "play": "1.2万",
                "like": "345",
                "review": "12",
            },
            {
                "bvid": "BV1old",
                "title": "Old video",
                "description": "stale",
                "author": "old",
                "pubdate": 1710000000,
                "play": 10,
                "like": 1,
                "review": 0,
            },
        ]
    },
}


def test_search_flag_accepts_bili_alias():
    assert cli.parse_search_flag("bili") == ["bilibili"]
    assert pipeline.normalize_requested_sources(["bili"]) == ["bilibili"]


def test_not_available_without_opt_in():
    assert "bilibili" not in pipeline.available_sources({}, None, x_pending=False)


def test_available_via_include_sources():
    avail = pipeline.available_sources(
        {"INCLUDE_SOURCES": "bilibili"}, None, x_pending=False
    )
    assert "bilibili" in avail


def test_excluded_wins():
    avail = pipeline.available_sources(
        {"INCLUDE_SOURCES": "bilibili", "EXCLUDE_SOURCES": "bilibili"},
        None,
        x_pending=False,
    )
    assert "bilibili" not in avail


def test_search_parses_and_filters_by_date(monkeypatch):
    monkeypatch.setattr(bilibili.http, "get", lambda *a, **k: MOCK_RESPONSE)
    result = bilibili.search_bilibili(
        "OpenClaw agent",
        "2025-08-01",
        "2025-08-31",
        depth="quick",
    )
    items = result["items"]
    assert len(items) == 1
    item = items[0]
    assert item["id"] == "BV1xx411c7mD"
    assert item["title"] == "OpenClaw agent tutorial"
    assert item["url"] == "https://www.bilibili.com/video/BV1xx411c7mD"
    assert item["engagement"]["views"] == 12000
    assert item["date"] == "2025-08-22"


def test_search_empty_http_returns_error(monkeypatch):
    monkeypatch.setattr(bilibili.http, "get", mock.Mock(side_effect=bilibili.http.HTTPError("boom")))
    result = bilibili.search_bilibili("topic", "2025-08-01", "2025-08-31", depth="quick")
    assert result["items"] == []
    assert "error" in result
