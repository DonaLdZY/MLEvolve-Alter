from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_USAGE_PATH = Path(__file__).resolve().parents[1] / "llm" / "usage.py"
_SPEC = importlib.util.spec_from_file_location("mlevolve_llm_usage_for_test", _USAGE_PATH)
assert _SPEC and _SPEC.loader
_USAGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_USAGE)
log_llm_usage = _USAGE.log_llm_usage


def test_log_llm_usage_writes_jsonl_summary_and_prompt_parts(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)
    response = SimpleNamespace(
        usage={
            "prompt_tokens": 120,
            "completion_tokens": 12,
            "total_tokens": 132,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 40,
        }
    )

    row = log_llm_usage(
        cfg=cfg,
        prompt_name="draft_agent",
        mode="generate_stream",
        provider="openai_compatible",
        model="deepseek-chat",
        response=response,
        seconds=1.25,
        max_tokens=4096,
        parsed_ok=True,
        prompt_parts=[
            {"name": "system_prompt", "role": "system", "content": "stable"},
            {"name": "task_context", "role": "user", "content": "x" * 400},
            {"name": "dynamic_feedback", "role": "user", "content": "fix this"},
        ],
        estimated_completion_text="answer",
    )

    assert row["prompt_tokens"] == 120
    assert row["completion_tokens"] == 12
    assert row["prompt_cache_hit_tokens"] == 80
    usage_rows = [json.loads(line) for line in (tmp_path / "llm_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert usage_rows[-1]["prompt_parts"][1]["name"] == "task_context"
    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["calls"] == 1
    assert summary["prompt_tokens"] == 120
    assert summary["provider_cache_hit_ratio"] == round(80 / 120, 6)
    assert summary["deepseek_cost_breakdown_rmb"]["cache_miss_input_tokens"] == 40
    assert summary["deepseek_cost_breakdown_rmb"]["output_tokens"] == 12
    assert summary["by_prompt"]["draft_agent"]["by_part"]["task_context"]["estimated_tokens"] > 0
    assert summary["by_prompt_part_ranked"][0]["estimated_tokens"] >= summary["by_prompt_part_ranked"][-1]["estimated_tokens"]
    brief = json.loads((tmp_path / "llm_usage_brief.json").read_text(encoding="utf-8"))
    assert brief["deepseek_pricing_rmb_per_1m"]["cache_miss_input"] == 3.0
    assert brief["deepseek_cost_breakdown_rmb"]["output_tokens"] == 12
    assert brief["top_prompts_by_estimated_cost"][0]["stage"] == "draft_or_code_generation"
    assert brief["by_stage"][0]["stage"] == "draft_or_code_generation"


def test_log_llm_usage_marks_missing_provider_usage(tmp_path):
    cfg = SimpleNamespace(log_dir=tmp_path)

    row = log_llm_usage(
        cfg=cfg,
        prompt_name="stream_without_usage",
        mode="generate_stream",
        provider="gemini",
        model="gemini-test",
        usage={},
        prompt_parts=[{"name": "prompt", "role": "user", "content": "hello world"}],
        estimated_completion_text="ok",
    )

    assert row["usage_available"] is False
    assert row["estimated_prompt_tokens"] > 0
    summary = json.loads((tmp_path / "llm_usage_summary.json").read_text(encoding="utf-8"))
    assert summary["provider_usage_missing_calls"] == 1
    assert summary["estimated_completion_tokens"] > 0
