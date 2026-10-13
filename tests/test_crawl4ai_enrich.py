"""Optional crawl4ai grounding enrichment. Fully mocked; no crawl4ai install."""

from unittest import mock

from lib import crawl4ai_enrich


def test_disabled_by_default():
    items = [{"url": "https://example.com", "snippet": "short"}]
    assert crawl4ai_enrich.enrich_items(items, {}) is items


def test_skips_when_enabled_but_unavailable(monkeypatch):
    monkeypatch.setattr(crawl4ai_enrich, "is_available", lambda: False)
    items = [{"url": "https://example.com", "snippet": "x"}]
    out = crawl4ai_enrich.enrich_items(items, {"LAST30DAYS_CRAWL4AI": "1"})
    assert out[0]["snippet"] == "x"
    assert "enriched_via" not in out[0]


def test_enriches_thin_snippets_only(monkeypatch):
    monkeypatch.setattr(crawl4ai_enrich, "is_available", lambda: True)
    monkeypatch.setattr(crawl4ai_enrich, "_crwl_fetch", lambda url: "# Title\n\nBody from crawl.")
    items = [
        {"url": "https://example.com/a", "snippet": "tiny"},
        {"url": "https://example.com/b", "snippet": "x" * 200},
        {"url": "ftp://skip", "snippet": "tiny"},
    ]
    out = crawl4ai_enrich.enrich_items(items, {"LAST30DAYS_CRAWL4AI": "1"})
    assert out[0]["snippet"].startswith("# Title")
    assert out[0]["enriched_via"] == "crawl4ai"
    assert out[1]["snippet"] == "x" * 200
    assert "enriched_via" not in out[1]
    assert "enriched_via" not in out[2]


def test_extract_markdown_from_json_blob():
    blob = '{"markdown": {"raw_markdown": "hello world"}}'
    assert crawl4ai_enrich._extract_markdown(blob) == "hello world"
