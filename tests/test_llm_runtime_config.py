from __future__ import annotations

from types import SimpleNamespace

from llm import gemini as gemini_backend
from llm import openai as openai_backend


def test_openai_retry_policy_uses_stage_config(monkeypatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    class Completions:
        def create(self, **_kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise TimeoutError("temporary timeout")
            return "ok"

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    stage = SimpleNamespace(
        network_retry_max_attempts=3,
        network_retry_base_sleep_seconds=0.25,
        network_retry_max_sleep_seconds=0.4,
    )
    monkeypatch.setattr(openai_backend.time, "sleep", sleeps.append)

    result = openai_backend._create_with_retry(
        client,
        {"model": "demo"},
        label="test",
        stage=stage,
    )

    assert result == "ok"
    assert calls["count"] == 3
    assert sleeps == [0.25, 0.4]


def test_continuation_overlap_window_is_configurable() -> None:
    overlap = "0123456789abcdefghij"
    assert openai_backend._append_with_overlap(f"abc{overlap}", f"{overlap}def", 20) == f"abc{overlap}def"
    assert openai_backend._append_with_overlap(f"abc{overlap}", f"{overlap}def", 15) == f"abc{overlap}{overlap}def"


def test_gemini_client_uses_stage_endpoint_and_timeout(monkeypatch) -> None:
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(gemini_backend.genai, "Client", fake_client)
    stage = SimpleNamespace(
        api_key="test-key",
        base_url="https://example.invalid",
        request_timeout_seconds=12.5,
    )

    gemini_backend._setup_gemini_client(stage)

    assert captured["api_key"] == "test-key"
    assert captured["http_options"]["base_url"] == "https://example.invalid"
    assert captured["http_options"]["timeout"] == 12500
