from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("dataclasses_json")
pytest.importorskip("humanize")
pytest.importorskip("coolname")
pytest.importorskip("omegaconf")
pytest.importorskip("llm")

from agents import result_parse_agent
from engine.search_node import SearchNode
from utils.metric import MetricValue


def test_llm_review_summary_still_generates_ui_insight(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        return {
            "insight": (
                "This node has a trusted score, but the validation summary still "
                "shows task-specific bottlenecks. Keep the evaluator and target "
                "the named diagnostics next."
            )
        }

    monkeypatch.setattr(result_parse_agent, "query", fake_query)
    agent = SimpleNamespace(
        task_desc="decision optimization task",
        coldstart_description="",
        cfg=SimpleNamespace(),
        acfg=SimpleNamespace(feedback=SimpleNamespace(model="fake-feedback", temp=0.0)),
    )
    node = SearchNode(
        parent=None,
        plan="greedy baseline",
        code="",
        stage="draft",
        _term_out=[],
        analysis="Execution completed, printed a scorable Decision Validation Summary, and printed a final validation score.",
        parser_analysis="Execution completed, printed a scorable Decision Validation Summary, and printed a final validation score.",
        decision_signals={
            "trusted_score_source": True,
            "final_score_source": "score_solution",
            "score_component_count": 3,
        },
        metric=MetricValue(35344613.5, maximize=False),
        is_buggy=False,
        is_valid=False,
    )

    result_parse_agent._generate_human_node_insight(node=node, agent=agent, parser_generated=False, force=True)

    assert calls
    assert calls[0]["func_spec"].name == "submit_node_insight"
    assert node.llm_insight == (
        "This node has a trusted score, but the validation summary still "
        "shows task-specific bottlenecks. Keep the evaluator and target "
        "the named diagnostics next."
    )
    assert node.llm_insight != node.parser_analysis
