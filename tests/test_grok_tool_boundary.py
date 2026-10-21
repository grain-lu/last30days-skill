import os
import re
import subprocess
from pathlib import Path

import pytest

from lib import grok_x

# The real audit, captured before the autouse fixture stubs it.
_SELF_AUDIT = grok_x._self_audit


@pytest.fixture(autouse=True)
def isolated_audit(monkeypatch, tmp_path):
    """Keep audit records out of the real config dir; unknown builds fail closed."""
    monkeypatch.setenv("LAST30DAYS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(grok_x, "_AUDIT_SESSION", {})
    monkeypatch.setattr(grok_x, "_binary_sha256", lambda binary: "fixture-sha")
    monkeypatch.setattr(grok_x, "_self_audit", lambda binary: ("failed", "fixture"))


@pytest.mark.parametrize("version", [
    "grok 1.0.41 (old) [stable]",
    "grok 1.0.47 (future) [stable]",
    "grok 2.0.0 (future) [stable]",
    "grok 1.0.46 (unknown) [stable]",
    "grok 1.0.46 (2765805b9442) [beta]",
    "grok 1.0.46-dev (modified)",
    "grok 1.0.62 (unknown) [stable]",
    "grok 1.0.62 (1474f9cbdd91) [beta]",
    "unexpected version output",
    "",
])
def test_unaudited_cli_never_receives_credentials_or_research(monkeypatch, version):
    calls = []
    staged = []
    monkeypatch.setattr(grok_x, "binary_path", lambda: "/usr/bin/grok")
    monkeypatch.setattr(grok_x, "_stage_child_home", lambda path: staged.append(path))

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, version, "")

    monkeypatch.setattr(grok_x.subprocess, "run", fake_run)
    response = grok_x._invoke("private research query", 60)
    assert "unsupported grok CLI version" in response.get("error", "")
    assert staged == []
    assert [cmd for cmd, _ in calls] == [["/usr/bin/grok", "--version"]]
    assert calls[0][1]["timeout"] <= 5
    assert "private research query" not in str(calls)


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "oserror"])
def test_failed_version_check_never_stages_credentials(monkeypatch, failure):
    staged = []
    monkeypatch.setattr(grok_x, "binary_path", lambda: "/usr/bin/grok")
    monkeypatch.setattr(grok_x, "_stage_child_home", lambda path: staged.append(path))

    def fake_run(cmd, **kwargs):
        assert cmd == ["/usr/bin/grok", "--version"]
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 5)
        if failure == "oserror":
            raise OSError("cannot launch fixture")
        return subprocess.CompletedProcess(cmd, 1, "grok 1.0.46 (2765805b9442) [stable]", "failure")

    monkeypatch.setattr(grok_x.subprocess, "run", fake_run)
    response = grok_x._invoke("private research query", 60)
    assert "version" in response.get("error", "")
    assert staged == []


@pytest.mark.parametrize("version", [
    "grok 1.0.46 (2765805b9442)",
    "grok 1.0.46 (2765805b9442) [stable]",
    "grok 1.0.62 (1474f9cbdd91)",
    "grok 1.0.62 (1474f9cbdd91) [stable]",
])
def test_audited_build_runs_x_with_only_staged_auth(monkeypatch, tmp_path, version):
    store = tmp_path / ".grok" / "auth.json"
    store.parent.mkdir()
    store.write_text('{"key": "dummy-offline-key"}')
    monkeypatch.setattr(grok_x, "token_store_path", lambda: store)
    monkeypatch.setattr(grok_x, "binary_path", lambda: "/usr/bin/grok")
    monkeypatch.setenv("XAI_API_KEY", "dummy-unrelated-key")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        child_home = Path(kwargs["env"]["HOME"])
        assert "XAI_API_KEY" not in kwargs["env"]
        if cmd == ["/usr/bin/grok", "--version"]:
            assert not (child_home / ".grok" / "auth.json").exists()
            return subprocess.CompletedProcess(cmd, 0, version + "\n", "")
        assert (child_home / ".grok" / "auth.json").read_text() == store.read_text()
        assert cmd[1:3] == ["-p", "Use x_keyword_search for OpenAI"]
        assert "--tools" not in cmd
        assert "--disable-web-search" in cmd
        assert "x_search" not in cmd[cmd.index("--disallowed-tools") + 1].split(",")
        return subprocess.CompletedProcess(cmd, 0, '{"text":"fixture X response"}', "")

    monkeypatch.setattr(grok_x.subprocess, "run", fake_run)
    assert grok_x._invoke("Use x_keyword_search for OpenAI", 60) == {
        "text": '{"text":"fixture X response"}',
    }
    assert len(calls) == 2


@pytest.mark.parametrize("spent", [4, 10])
def test_preflight_time_counts_toward_research_timeout(monkeypatch, tmp_path, spent):
    clock = [100.0]
    calls = []
    monkeypatch.setattr(grok_x.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(grok_x, "binary_path", lambda: "/usr/bin/grok")
    monkeypatch.setattr(grok_x, "token_store_path", lambda: tmp_path / ".grok" / "auth.json")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--version" in cmd:
            assert kwargs["timeout"] == 5
            clock[0] += spent
            return subprocess.CompletedProcess(cmd, 0, "grok 1.0.46 (2765805b9442) [stable]", "")
        assert kwargs["timeout"] == 10 - spent
        return subprocess.CompletedProcess(cmd, 0, "fixture response", "")

    monkeypatch.setattr(grok_x.subprocess, "run", fake_run)
    response = grok_x._invoke("X query", 10)
    if spent < 10:
        assert response == {"text": "fixture response"}
        assert len(calls) == 2
    else:
        assert "timed out" in response.get("error", "")
        assert len(calls) == 1


def test_search_reports_unsupported_build_as_source_failure(monkeypatch):
    monkeypatch.setattr(grok_x, "binary_path", lambda: "/usr/bin/grok")

    def fake_run(cmd, **kwargs):
        assert cmd == ["/usr/bin/grok", "--version"]
        return subprocess.CompletedProcess(cmd, 0, "grok 1.0.47 (future) [stable]", "")

    monkeypatch.setattr(grok_x.subprocess, "run", fake_run)
    result = grok_x.search_x("OpenAI", "2026-07-14", "2026-08-13", depth="quick")
    assert result["items"] == []
    assert "unsupported grok CLI version" in result.get("error", "")


def test_search_reports_profile_write_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(grok_x, "binary_path", lambda: "/usr/bin/grok")
    monkeypatch.setattr(grok_x, "token_store_path", lambda: tmp_path / "missing" / "auth.json")

    def fake_run(cmd, **kwargs):
        assert cmd == ["/usr/bin/grok", "--version"]
        return subprocess.CompletedProcess(cmd, 0, "grok 1.0.46 (2765805b9442) [stable]", "")

    def denied(*args, **kwargs):
        raise PermissionError("cannot write agent profile")

    monkeypatch.setattr(grok_x.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "write_text", denied)
    result = grok_x.search_x("OpenAI", "2026-07-14", "2026-08-13", depth="quick")
    assert result["items"] == []
    assert "cannot write agent profile" in result.get("error", "")


@pytest.mark.skipif(os.name == "nt", reason="The POSIX Grok installer uses a versioned symlink")
def test_updater_symlink_cannot_change_build_after_version_check(monkeypatch, tmp_path):
    audited = tmp_path / "grok-1.0.46"
    future = tmp_path / "grok-future"
    audited.touch()
    future.touch()
    launcher = tmp_path / "grok"
    launcher.symlink_to(audited)
    monkeypatch.setattr(grok_x, "binary_path", lambda: str(launcher))
    monkeypatch.setattr(grok_x, "token_store_path", lambda: tmp_path / ".grok" / "auth.json")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "--version" in cmd:
            launcher.unlink()
            launcher.symlink_to(future)
            return subprocess.CompletedProcess(cmd, 0, "grok 1.0.46 (2765805b9442)", "")
        assert cmd[0] == str(audited)
        return subprocess.CompletedProcess(cmd, 0, "fixture response", "")

    monkeypatch.setattr(grok_x.subprocess, "run", fake_run)
    assert grok_x._invoke("X query", 10) == {"text": "fixture response"}
    assert calls[0] == [str(audited), "--version"]


def _fake_cli(version, calls):
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:] == ["--version"]:
            return subprocess.CompletedProcess(cmd, 0, version + "\n", "")
        return subprocess.CompletedProcess(cmd, 0, "fixture response", "")
    return fake_run


def _use_cli(monkeypatch, tmp_path, version, calls):
    monkeypatch.setattr(grok_x, "binary_path", lambda: "/usr/bin/grok")
    monkeypatch.setattr(grok_x, "token_store_path", lambda: tmp_path / ".grok" / "auth.json")
    monkeypatch.setattr(grok_x.subprocess, "run", _fake_cli(version, calls))


def test_unknown_build_runs_after_passing_audit_and_is_remembered(monkeypatch, tmp_path):
    audits = []
    monkeypatch.setattr(grok_x, "_self_audit", lambda binary: audits.append(binary) or ("passed", ""))
    calls = []
    _use_cli(monkeypatch, tmp_path, "grok 9.9.9 (future) [stable]", calls)
    assert grok_x._invoke("X query", 60) == {"text": "fixture response"}
    assert audits == ["/usr/bin/grok"]

    monkeypatch.setattr(grok_x, "_AUDIT_SESSION", {})
    assert grok_x._invoke("X query", 60) == {"text": "fixture response"}
    assert len(audits) == 1
    record = grok_x._load_audit_records()["grok 9.9.9 (future) [stable] sha256:fixture-sha"]
    assert record["status"] == "passed"


def test_failed_audit_is_remembered_and_blocks_research(monkeypatch, tmp_path):
    audits = []
    monkeypatch.setattr(grok_x, "_self_audit", lambda binary: audits.append(binary) or ("failed", "probe 1 leaked"))
    calls = []
    _use_cli(monkeypatch, tmp_path, "grok 9.9.9 (future)", calls)
    for _ in range(2):
        monkeypatch.setattr(grok_x, "_AUDIT_SESSION", {})
        response = grok_x._invoke("private research query", 60)
        assert "unsupported grok CLI version" in response["error"]
        assert "probe 1 leaked" in response["error"]
    assert len(audits) == 1
    assert "private research query" not in str(calls)


def test_inconclusive_audit_blocks_without_recording(monkeypatch, tmp_path):
    audits = []
    monkeypatch.setattr(grok_x, "_self_audit", lambda binary: audits.append(binary) or ("inconclusive", "probe 2 timed out"))
    calls = []
    _use_cli(monkeypatch, tmp_path, "grok 9.9.9 (future)", calls)
    for _ in range(2):
        assert "will retry next run" in grok_x._invoke("X query", 60)["error"]
    assert len(audits) == 1  # once per process
    assert grok_x._load_audit_records() == {}
    monkeypatch.setattr(grok_x, "_AUDIT_SESSION", {})
    grok_x._invoke("X query", 60)
    assert len(audits) == 2  # a new run retries


def test_changed_binary_under_known_version_is_reaudited(monkeypatch, tmp_path):
    audits = []
    monkeypatch.setattr(grok_x, "_self_audit", lambda binary: audits.append(binary) or ("passed", ""))
    calls = []
    _use_cli(monkeypatch, tmp_path, "grok 9.9.9 (future)", calls)
    grok_x._invoke("X query", 60)
    monkeypatch.setattr(grok_x, "_AUDIT_SESSION", {})
    monkeypatch.setattr(grok_x, "_binary_sha256", lambda binary: "other-sha")
    grok_x._invoke("X query", 60)
    assert len(audits) == 2


def test_auto_audit_off_rejects_unknown_build(monkeypatch, tmp_path):
    monkeypatch.setenv("LAST30DAYS_GROK_AUTO_AUDIT", "off")
    monkeypatch.setattr(grok_x, "_self_audit", lambda binary: pytest.fail("audit must not run"))
    calls = []
    _use_cli(monkeypatch, tmp_path, "grok 9.9.9 (future)", calls)
    assert "LAST30DAYS_GROK_AUTO_AUDIT is off" in grok_x._invoke("X query", 60)["error"]
    assert len(calls) == 1


def _probe_runner(monkeypatch, tmp_path, behave):
    """Patch subprocess.run so each audit probe is answered by behave(prompt, paths, kwargs)."""
    monkeypatch.setattr(grok_x, "token_store_path", lambda: tmp_path / ".grok" / "auth.json")
    prompts = []

    def fake_run(cmd, **kwargs):
        prompt = cmd[cmd.index("-p") + 1]
        prompts.append(prompt)
        assert cmd == grok_x._research_argv("/usr/bin/grok", prompt, Path(cmd[cmd.index("--agent") + 1]))
        paths = re.findall(r"(/\S+?\.txt)", prompt)
        return behave(prompt, paths, kwargs)

    monkeypatch.setattr(grok_x.subprocess, "run", fake_run)
    return prompts


def _refuse(prompt, paths, kwargs):
    return subprocess.CompletedProcess([], 0, '{"text":"Those tools are not available."}', "")


def test_self_audit_passes_when_every_probe_is_refused(monkeypatch, tmp_path):
    prompts = _probe_runner(monkeypatch, tmp_path, _refuse)
    assert _SELF_AUDIT("/usr/bin/grok") == ("passed", "")
    assert len(prompts) == 5


@pytest.mark.parametrize("leak", ["outside_read", "home_read", "shell", "write"])
def test_self_audit_fails_on_any_local_access(monkeypatch, tmp_path, leak):
    def behave(prompt, paths, kwargs):
        if leak == "outside_read" and paths and "read" in prompt.lower() and "~" not in prompt:
            return subprocess.CompletedProcess([], 0, Path(paths[0]).read_text(), "")
        if leak == "home_read" and "~/canary.txt" in prompt:
            home = kwargs["env"]["HOME"]
            return subprocess.CompletedProcess([], 0, Path(home, "canary.txt").read_text(), "")
        if leak == "shell" and "shell" in prompt:
            return subprocess.CompletedProcess([], 0, "uid=1000(ops)", "")
        if leak == "write" and "Create the file" in prompt:
            Path(paths[0]).write_text("done")
        return _refuse(prompt, paths, kwargs)

    _probe_runner(monkeypatch, tmp_path, behave)
    status, reason = _SELF_AUDIT("/usr/bin/grok")
    assert status == "failed", reason


def test_self_audit_is_inconclusive_when_a_probe_errors(monkeypatch, tmp_path):
    _probe_runner(monkeypatch, tmp_path,
                  lambda prompt, paths, kwargs: subprocess.CompletedProcess([], 1, "", "network down"))
    assert _SELF_AUDIT("/usr/bin/grok")[0] == "inconclusive"
