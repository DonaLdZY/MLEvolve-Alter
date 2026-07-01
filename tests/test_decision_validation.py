from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "decision_validation.py"
_SPEC = importlib.util.spec_from_file_location("mlevolve_decision_validation_for_test", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_DV = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DV)

decision_summary_defects = _DV.decision_summary_defects
decision_summary_is_scorable = _DV.decision_summary_is_scorable
decision_signal_summary = _DV.decision_signal_summary
trusted_decision_score_source = _DV.trusted_decision_score_source


def test_trusted_penalized_decision_summary_is_scorable_without_task_specific_fields():
    summary = {
        "evaluator_self_tests_passed": True,
        "final_score_source": "total_penalized_cost = objective_cost + invalid_case_count * M",
        "score_components": {
            "objective_cost": 51836.856,
            "invalid_case_count": 1902,
            "penalty_m": 1e9,
        },
    }

    assert trusted_decision_score_source(summary)
    assert decision_summary_is_scorable(summary)
    assert decision_summary_defects(summary) == []


def test_summary_without_self_test_flag_is_kept_for_improvement():
    summary = {
        "final_score_source": "score_solution",
        "notes": "PPO trained (or greedy fallback); validated.",
        "score_components": {
            "objective_cost": 726.912109375,
            "penalty_value": 2102000000000.0,
        },
    }

    assert trusted_decision_score_source(summary)
    assert decision_summary_is_scorable(summary)
    assert decision_summary_defects(summary) == []


def test_task_specific_diagnostics_are_optional_not_blocking():
    summary = {
        "evaluator_self_tests_passed": True,
        "final_score_source": "score_solution",
        "notes": "Greedy baseline with task-specific validation diagnostics.",
        "validator_report": {"invalid_records": 22, "dominant_reason": "example"},
        "score_components": {
            "total_penalized_cost": 1763000248491.841,
            "objective_cost": 248491.84106445312,
            "invalid_case_count": 1763,
        },
    }

    assert trusted_decision_score_source(summary)
    assert decision_summary_is_scorable(summary)
    assert decision_summary_defects(summary) == []


def test_empty_or_diagnostic_solution_can_be_scorable_when_official_score_handles_it():
    summary = {
        "evaluator_self_tests_passed": True,
        "final_score_source": "total_penalized_cost",
        "score_components": {
            "objective_cost": 0,
            "invalid_case_count": 2104,
            "penalty_m": 1e9,
        },
    }

    assert trusted_decision_score_source(summary)
    assert decision_summary_is_scorable(summary)
    assert decision_summary_defects(summary) == []


def test_total_penalty_score_alias_is_trusted():
    summary = {
        "evaluator_self_tests_passed": True,
        "final_score_source": "total_penalty_score",
        "notes": "Diagnostic baseline.",
        "score_components": {
            "M_estimate": 301696.34394749993,
            "objective_cost": 0,
            "invalid_case_count": 2104,
            "decision_count": 0,
        },
    }

    defects = decision_summary_defects(summary)

    assert trusted_decision_score_source(summary)
    assert decision_summary_is_scorable(summary)
    assert "final_score_source does not identify an official deterministic score source" not in defects
    assert defects == []


def test_formula_node_is_kept():
    summary = {
        "evaluator_self_tests_passed": True,
        "final_score_source": "total_penalized_cost = objective_cost + invalid_case_count * M",
        "notes": "Task-specific diagnostic details are allowed but not required by the parser.",
        "score_components": {
            "penalty_m": 1_000_000_000.0,
            "objective_cost": 51836.8560374,
            "invalid_case_count": 1902,
        },
    }

    assert trusted_decision_score_source(summary)
    signals = decision_signal_summary(summary)
    assert "objective_cost" not in signals
    assert "invalid_case_count" not in signals
    assert signals["trusted_score_source"] is True
    assert signals["score_component_count"] == 3
    assert decision_summary_is_scorable(summary)
    assert decision_summary_defects(summary) == []


def test_optional_diagnostic_fields_do_not_block_acceptance():
    summary = {
        "evaluator_self_tests_passed": False,
        "is_feasible": False,
        "notes": "Optional diagnostics report problems, but parser acceptance is score-based.",
    }

    defects = decision_summary_defects(summary)

    assert not trusted_decision_score_source(summary)
    assert decision_summary_is_scorable(summary)
    assert defects == []


def test_missing_decision_summary_is_not_a_blocking_defect():
    assert decision_summary_defects(None) == []
    assert decision_summary_is_scorable(None)
