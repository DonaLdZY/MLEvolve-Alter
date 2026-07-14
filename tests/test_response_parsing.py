from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

from agents.coder import base_coder
from utils.response import extract_plan_and_code


def test_code_first_response_is_valid_without_prose_plan() -> None:
    plan, code = extract_plan_and_code(
        "```python\nprint('Final Validation Score: 1.0')\n```",
        default_plan="Default implementation plan.",
    )

    assert plan == "Default implementation plan."
    assert "Final Validation Score: 1.0" in code


def test_base_coder_does_not_regenerate_valid_code_first_response(monkeypatch) -> None:
    calls = 0

    def fake_generate(**kwargs):
        nonlocal calls
        calls += 1
        return "```python\nscore = 1.0\nprint(f'Final Validation Score: {score}')\n```"

    monkeypatch.setattr(base_coder, "generate", fake_generate)
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            code=SimpleNamespace(temp=0.0),
            retries=SimpleNamespace(code_generation_extract_max_attempts=2),
        ),
        cfg=SimpleNamespace(),
    )

    plan, code = base_coder.plan_and_code_query(agent, {"user": "demo"})

    assert calls == 1
    assert plan
    assert "Final Validation Score" in code


def test_full_code_prompts_do_not_request_conflicting_long_or_fenced_plans() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt_sources = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in [
            "agents/draft_agent.py",
            "agents/improve_agent.py",
            "agents/evolution_agent.py",
            "agents/fusion_agent.py",
        ]
    )

    assert "Natural length: around 8-12 sentences" not in prompt_sources
    assert "You MUST structure your plan using the following EXACT format" not in prompt_sources
