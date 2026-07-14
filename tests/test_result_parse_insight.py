from __future__ import annotations

import copy
from types import SimpleNamespace
import threading

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


def test_async_insight_keeps_parser_fallback_then_replaces_it(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def fake_request(agent, payload):
        started.set()
        assert release.wait(timeout=2)
        return "LLM explanation generated in the background."

    monkeypatch.setattr(result_parse_agent, "_request_human_node_insight", fake_request)
    agent = SimpleNamespace(
        task_desc="standard regression task",
        coldstart_description="",
        journal_lock=threading.Lock(),
        cfg=SimpleNamespace(),
        acfg=SimpleNamespace(
            feedback=SimpleNamespace(model="fake-feedback", temp=0.0),
            retries=SimpleNamespace(human_insight_async=True),
        ),
    )
    node = SearchNode(
        parent=None,
        plan="train model",
        code="",
        stage="draft",
        _term_out=[],
        analysis="Parser facts are immediately available.",
        parser_analysis="Parser facts are immediately available.",
        metric=MetricValue(0.5, maximize=True),
        is_buggy=False,
        is_valid=True,
    )

    result_parse_agent.refresh_human_node_insight(agent, node)
    assert started.wait(timeout=2)
    assert "Parser facts" in node.llm_insight

    release.set()
    deadline = threading.Event()
    for _ in range(100):
        with result_parse_agent._HUMAN_INSIGHT_THREADS_LOCK:
            pending = result_parse_agent._HUMAN_INSIGHT_THREADS.get(node.id)
        if pending is None:
            break
        pending.join(timeout=0.02)
    else:
        deadline.wait(timeout=0.01)
    assert node.llm_insight == "LLM explanation generated in the background."


def test_async_insight_runtime_thread_is_not_attached_to_serializable_node(monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def fake_request(agent, payload):
        started.set()
        assert release.wait(timeout=2)
        return "Background insight."

    monkeypatch.setattr(result_parse_agent, "_request_human_node_insight", fake_request)
    agent = SimpleNamespace(
        task_desc="standard regression task",
        coldstart_description="",
        journal_lock=threading.Lock(),
        cfg=SimpleNamespace(),
        acfg=SimpleNamespace(
            feedback=SimpleNamespace(model="fake-feedback", temp=0.0),
            retries=SimpleNamespace(human_insight_async=True),
        ),
    )
    node = SearchNode(
        parent=None,
        plan="train model",
        code="print(1)",
        stage="draft",
        _term_out=[],
        analysis="Parser facts.",
        parser_analysis="Parser facts.",
        metric=MetricValue(0.5, maximize=True),
        is_buggy=False,
        is_valid=True,
    )

    result_parse_agent.refresh_human_node_insight(agent, node)
    assert started.wait(timeout=2)
    copied = copy.deepcopy(node)

    assert copied.id == node.id
    assert not hasattr(node, "_llm_insight_thread")

    release.set()
    with result_parse_agent._HUMAN_INSIGHT_THREADS_LOCK:
        pending = result_parse_agent._HUMAN_INSIGHT_THREADS.get(node.id)
    if pending is not None:
        pending.join(timeout=2)


def test_metric_direction_uses_autorealize_contract_without_llm(monkeypatch) -> None:
    def fail_query(**kwargs):
        raise AssertionError("metric direction LLM should not be called")

    monkeypatch.setattr(result_parse_agent, "query", fail_query)
    agent = SimpleNamespace(
        autorealize_context=(
            "## AutoRealize Structured Context\n"
            "## Evaluation Contract Reference\n"
            "- metric_direction: minimize\n"
        ),
        task_desc="demo",
        acfg=SimpleNamespace(retries=SimpleNamespace()),
    )

    result_parse_agent.determine_metric_direction(agent)

    assert agent.metric_maximize is False
    assert "AutoRealize evaluation contract" in agent.metric_maximize_reasoning
