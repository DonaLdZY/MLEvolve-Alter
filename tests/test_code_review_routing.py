from __future__ import annotations

from types import SimpleNamespace

from agents import code_review_agent
from engine.search_node import SearchNode


def _agent(*, escalate: bool = True):
    retries = SimpleNamespace(
        code_review_max_attempts=2,
        code_review_delay_seconds=0.0,
        code_review_model_role="feedback",
        code_review_escalate_to_code=escalate,
    )
    return SimpleNamespace(
        task_desc="Demo task",
        autorealize_context="",
        cfg=SimpleNamespace(pretrain_model_dir=""),
        acfg=SimpleNamespace(
            use_diff_mode=True,
            generate_submission=False,
            retries=retries,
            feedback=SimpleNamespace(model="fast-review", temp=0.0),
            code=SimpleNamespace(model="main-code", temp=0.0),
        ),
    )


def _node() -> SearchNode:
    return SearchNode(code="print(1)", plan="demo", parent=None, stage="draft")


def test_code_review_uses_feedback_role_first(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {"needs_revision": False, "reasoning": "No concrete P0/P1 issue."}

    monkeypatch.setattr(code_review_agent, "query", fake_query)
    code = code_review_agent.run(_agent(), _node())

    assert code == "print(1)"
    assert [call["stage_name"] for call in calls] == ["feedback"]
    assert calls[0]["model"] == "fast-review"


def test_unpatchable_feedback_review_escalates_once(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        if kwargs["stage_name"] == "feedback":
            return {
                "needs_revision": True,
                "reasoning": "A critical issue exists, but the patch is missing.",
                "revised_code": None,
            }
        return {"needs_revision": False, "reasoning": "Main review approves the code."}

    monkeypatch.setattr(code_review_agent, "query", fake_query)
    code = code_review_agent.run(_agent(), _node())

    assert code == "print(1)"
    assert [call["stage_name"] for call in calls] == ["feedback", "code"]
