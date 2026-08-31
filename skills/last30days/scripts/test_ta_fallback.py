"""Minimal checks for the TranscriptAPI (youtube-skills) fallback in youtube_yt.

Run: python3 scripts/test_ta_fallback.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import youtube_yt

_ORIG = {
    "is_ytdlp_installed": youtube_yt.is_ytdlp_installed,
    "_fetch_transcript_direct": youtube_yt._fetch_transcript_direct,
    "_ta_fetch_transcript": youtube_yt._ta_fetch_transcript,
    "http_get": youtube_yt.http.get,
}


def _restore():
    """Put back any monkeypatched function so tests are order-independent."""
    for name, orig in _ORIG.items():
        setattr(youtube_yt, name, orig)
    youtube_yt._TA_KEY_RESOLVED = False


def _key_set():
    os.environ["TRANSCRIPT_API_KEY"] = "sk_test"


def _key_clear():
    os.environ.pop("TRANSCRIPT_API_KEY", None)


def test_ta_key_from_env():
    _restore()
    _key_set()
    assert youtube_yt._ta_key() == "sk_test"
    _key_clear()


def test_ta_fetch_transcript_clean_and_cap():
    _restore()
    _key_set()
    seen = {}
    def fake_get(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs.get("headers", {})
        return {"transcript": "  hello   world\nhello\n" + "word " * 5000 + "end"}
    youtube_yt.http.get = fake_get

    out = youtube_yt._ta_fetch_transcript("dQw4w9WgXcQ")

    assert youtube_yt.TRANSCRIPTAPI_BASE in seen["url"]
    assert "Bearer sk_test" in seen["headers"].get("Authorization", "")
    assert "User-Agent" in seen["headers"]
    assert out is not None
    assert len(out.split()) <= youtube_yt.TRANSCRIPT_MAX_WORDS  # word-capped
    assert "hello world" in out  # VTT-ish whitespace collapsed
    _key_clear()


def test_ta_fetch_transcript_error_returns_none():
    _restore()
    _key_set()
    def boom(url, **kwargs):
        raise RuntimeError("403 Cloudflare")
    youtube_yt.http.get = boom
    assert youtube_yt._ta_fetch_transcript("abc") is None
    _key_clear()


def test_fetch_transcript_uses_ta_fallback_when_keyless_cascade_fails():
    _restore()
    _key_set()
    youtube_yt.is_ytdlp_installed = lambda: False
    youtube_yt._fetch_transcript_direct = lambda vid, status=None, timeout=30: None
    youtube_yt._ta_fetch_transcript = lambda vid: "server-side transcript"
    assert youtube_yt.fetch_transcript("xyz", "/tmp", status={}) == "server-side transcript"
    _key_clear()


def test_fetch_transcript_skips_ta_when_captions_proven_disabled():
    _restore()
    _key_set()
    youtube_yt.is_ytdlp_installed = lambda: False
    def direct_no_captions(vid, status=None, timeout=30):
        if status is not None:
            status["no_caption_tracks"] = True
        return None
    youtube_yt._fetch_transcript_direct = direct_no_captions
    called = {"ta": False}
    def ta_fetch(vid):
        called["ta"] = True
        return "x"
    youtube_yt._ta_fetch_transcript = ta_fetch

    out = youtube_yt.fetch_transcript("xyz", "/tmp", status={})

    assert out is None
    assert not called["ta"]  # no credit spent on a proven caption-less video
    _key_clear()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")


if __name__ == "__main__":
    main()
