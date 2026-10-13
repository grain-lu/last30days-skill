"""Weixin (公众号) opt-in source tests. HTTP is fully mocked."""

from __future__ import annotations

from unittest import mock

import last30days as cli
from lib import pipeline, weixin


SOGOU_HTML = """
<a href="/link?url=http%3A%2F%2Fmp.weixin.qq.com%2Fs%3Fsrc%3D1" uigs="article_title_0">AI 智能体落地</a>
<a href="" uigs="account_name_0">机器之心</a>
<script>timeConvert('1755900000')</script>
<a href="https://mp.weixin.qq.com/s/abc" uigs="article_title_1">Old article</a>
<a href="" uigs="account_name_1">量子位</a>
<script>timeConvert('1710000000')</script>
"""


def test_search_flag_accepts_wx_alias():
    assert cli.parse_search_flag("wx") == ["weixin"]
    assert cli.parse_search_flag("wechat") == ["weixin"]
    assert pipeline.normalize_requested_sources(["wx"]) == ["weixin"]


def test_not_available_without_opt_in():
    assert "weixin" not in pipeline.available_sources({}, None, x_pending=False)


def test_available_via_include_sources():
    avail = pipeline.available_sources(
        {"INCLUDE_SOURCES": "weixin"}, None, x_pending=False
    )
    assert "weixin" in avail


def test_available_when_requested():
    avail = pipeline.available_sources({}, requested_sources=["weixin"], x_pending=False)
    assert "weixin" in avail


def test_excluded_wins():
    avail = pipeline.available_sources(
        {"INCLUDE_SOURCES": "weixin", "EXCLUDE_SOURCES": "weixin"},
        None,
        x_pending=False,
    )
    assert "weixin" not in avail


def test_parse_account_sources_dedupes():
    assert weixin.parse_account_sources("机器之心, 机器之心,量子位") == ["机器之心", "量子位"]
    assert weixin.parse_account_sources("") == []


def test_search_parses_sogou_html(monkeypatch):
    monkeypatch.setattr(weixin.http, "get_text", lambda *a, **k: SOGOU_HTML)
    result = weixin.search_weixin(
        "AI 智能体",
        "2025-08-01",
        "2025-08-31",
        depth="quick",
        config={},
    )
    items = result["items"]
    assert len(items) == 1
    assert items[0]["title"] == "AI 智能体落地"
    assert items[0]["author"] == "机器之心"
    assert items[0]["source_domain"] == "mp.weixin.qq.com"
    assert items[0]["date"] == "2025-08-22"


def test_weixin_sources_scopes_query(monkeypatch):
    seen = []

    def fake_get_text(url, **kwargs):
        seen.append(url)
        return SOGOU_HTML

    monkeypatch.setattr(weixin.http, "get_text", fake_get_text)
    weixin.search_weixin(
        "agent",
        "2025-08-01",
        "2025-08-31",
        depth="quick",
        config={"WEIXIN_SOURCES": "机器之心"},
    )
    assert any("query=" in url and "agent" in url for url in seen)
    assert any("%E6%9C%BA%E5%99%A8%E4%B9%8B%E5%BF%83" in url or "机器之心" in url for url in seen)


def test_activate_weixin_flag_adds_include():
    config = {"INCLUDE_SOURCES": ""}
    requested = cli.activate_weixin_for_explicit_sources(
        config, ["reddit"], accounts="机器之心"
    )
    assert config["WEIXIN_SOURCES"] == "机器之心"
    assert "weixin" in config["INCLUDE_SOURCES"]
    assert "weixin" in requested
