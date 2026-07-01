"""Lightweight parsing helpers for optimization/RL decision validation output."""

from __future__ import annotations

import json
import re


DECISION_SUMMARY_PREFIX_RE = re.compile(
    r"^\s*Decision\s+Validation\s+Summary\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)

OPTIMIZATION_RL_KEYWORDS = (
    "reinforcement learning",
    "offline rl",
    "online rl",
    "mdp",
    "markov decision",
    "policy learning",
    "reward function",
    "gymnasium",
    "gym env",
    "environment step",
    "simulator",
    "sequential decision",
    "dynamic decision",
    "routing",
    "vehicle routing",
    "scheduling",
    "assignment problem",
    "resource allocation",
    "portfolio optimization",
    "knapsack",
    "combinatorial optimization",
    "optimization problem",
    "constraint solver",
    "cp-sat",
    "mixed integer",
    "integer programming",
    "linear programming",
    "decision problem",
    "decision optimization",
    "vehicle dispatch",
    "dispatching",
    "capacity constraint",
    "feasible solution",
    "hard constraint",
    "local search",
    "simulated annealing",
    "tabu search",
    "large neighborhood search",
    "minimize",
    "maximize",
    "objective",
    "constraint",
    "penalty",
    "强化学习",
    "离线强化学习",
    "在线强化学习",
    "马尔可夫决策",
    "状态空间",
    "动作空间",
    "奖励函数",
    "策略学习",
    "仿真环境",
    "序贯决策",
    "路径规划",
    "路径优化",
    "车辆路径",
    "车辆调度",
    "配送调度",
    "调度",
    "排程",
    "分配问题",
    "资源分配",
    "组合优化",
    "运筹优化",
    "整数规划",
    "线性规划",
    "约束求解",
    "可行解",
    "硬约束",
    "优化",
    "决策",
    "最小化",
    "最大化",
    "目标函数",
    "罚分",
    "惩罚",
    "约束",
)


def is_optimization_or_rl_text(task_desc: str = "", coldstart_description: str = "") -> bool:
    """Lightweight task-type detector that does not import prompt/LLM modules."""
    raw = f"{task_desc}\n{coldstart_description}"
    text = raw.lower()
    if not text.strip():
        return False
    if "model" in text and "optimization" in text:
        return True
    return any(keyword in text for keyword in OPTIMIZATION_RL_KEYWORDS)


def parse_bool_like(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "ok", "pass", "passed", "1"}:
            return True
        if normalized in {"false", "no", "fail", "failed", "0"}:
            return False
    return None


def extract_decision_validation_summary(text: str) -> dict | None:
    """Extract the last JSON Decision Validation Summary line from execution output."""
    summaries: list[dict] = []
    for line in (text or "").splitlines():
        match = DECISION_SUMMARY_PREFIX_RE.match(line)
        if not match:
            continue
        payload = match.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            summaries.append(parsed)
    return summaries[-1] if summaries else None


def decision_signal_summary(summary: dict | None) -> dict:
    """Return compact generic decision-output signals for logs/prompts.

    Keep this parser task-agnostic. Do not infer universal progress metrics
    here. Task-specific diagnostics belong in score_components/reports and
    should be passed through as evidence, not interpreted as system-level
    checks.
    """
    if not isinstance(summary, dict):
        return {}
    score_components = summary.get("score_components")
    if not isinstance(score_components, dict):
        score_components = {}
    signals = {
        "trusted_score_source": trusted_decision_score_source(summary),
    }
    score_source = summary.get("final_score_source")
    if score_source not in (None, "", [], {}):
        signals["final_score_source"] = score_source
    if score_components:
        signals["score_component_count"] = len(score_components)
    for key in ("evaluator_self_tests_passed", "is_feasible"):
        if key in summary:
            signals[key] = summary.get(key)
    return signals


def trusted_decision_score_source(summary: dict | None) -> bool:
    """Whether the reported decision score is grounded in a deterministic evaluator."""
    if not isinstance(summary, dict):
        return False
    score_source = str(summary.get("final_score_source", "") or "").strip().lower()
    score_components = summary.get("score_components")
    if not isinstance(score_components, dict):
        score_components = {}

    trusted_source_markers = (
        "score_solution",
        "official",
        "evaluation contract",
        "autorealize",
        "deterministic evaluator",
    )
    if any(marker in score_source for marker in trusted_source_markers):
        return True

    component_keys = {str(k).lower() for k in score_components}
    if "total_penalized_cost" in score_source:
        return True
    if "total_penalty_score" in score_source or "penalty_score" in score_source:
        return True
    if "penalized" in score_source and ("cost" in score_source or "score" in score_source):
        return True
    if "penalty" in score_source and "score" in score_source:
        return True
    if any(k in component_keys for k in ("total_penalized_cost", "penalized_cost", "total_penalty_score", "penalty_score")):
        return True
    return False


def decision_summary_defects(summary: dict | None) -> list[str]:
    """Compatibility no-op: decision summaries are diagnostic, not blockers."""
    return []


def decision_summary_is_scorable(summary: dict | None) -> bool:
    """Decision/RL acceptance is based on final score, not summary fields."""
    return True
