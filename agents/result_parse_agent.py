import logging
import json
import math
import re
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from engine.executor import ExecutionResult
from utils.metric import MetricValue, WorstMetricValue
from utils.response import trim_long_string, wrap_code
from utils.decision_validation import (
    decision_signal_summary as _dv_decision_signal_summary,
    decision_summary_defects as _dv_decision_summary_defects,
    decision_summary_is_scorable as _dv_decision_summary_is_scorable,
    extract_decision_validation_summary as _dv_extract_decision_validation_summary,
    parse_bool_like as _dv_parse_bool_like,
    trusted_decision_score_source as _dv_trusted_decision_score_source,
)
from engine.validation import call_validate, _validate_submission_with_retry, validate_submission_content_quality
from agents import data_leakage_agent
from agents.triggers import should_check_data_leakage
from agents.prompt_cache import task_section
from agents.prompts import is_optimization_or_rl_task

logger = logging.getLogger("MLEvolve")

FINAL_SCORE_RE = re.compile(
    r"Final\s+Validation\s+Score\s*[:=]\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
DECISION_SUMMARY_PREFIX_RE = re.compile(
    r"^\s*Decision\s+Validation\s+Summary\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)

NODE_INSIGHT_FUNC_SPEC = FunctionSpec(
    name="submit_node_insight",
    json_schema={
        "type": "object",
        "properties": {
            "insight": {
                "type": "string",
                "description": (
                    "A concise human-readable insight for the UI. Use 2-4 sentences. "
                    "Explain the outcome, main bottleneck or failure cause, and the most relevant next step. "
                    "Do not invent metrics or override parser facts."
                ),
            }
        },
        "required": ["insight"],
    },
    description="Submit a concise human-readable node insight.",
)


def _resolve_exp_id(agent) -> str:
    explicit = str(getattr(agent.cfg, "exp_id", "") or "").strip()
    if explicit:
        return explicit
    exp_name = str(getattr(agent.cfg, "exp_name", "") or "").strip()
    parts = exp_name.split("_", 2)
    if len(parts) >= 3 and parts[2].strip():
        return parts[2].strip()
    return exp_name or "task"


def _is_optimization_or_rl_agent(agent) -> bool:
    return is_optimization_or_rl_task(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
    )


def _set_parser_analysis(node: SearchNode, text: str | None) -> None:
    node.analysis = text
    node.parser_analysis = text


def _decision_signals_for_node(summary: dict | None, metric=None) -> dict | None:
    if not isinstance(summary, dict):
        if metric is None:
            return None
        return {"final_score": metric}
    signals = dict(_dv_decision_signal_summary(summary))
    if metric is not None:
        signals["final_score"] = metric
    return signals or None


def _fallback_human_insight(node: SearchNode, parser_analysis: str | None) -> str:
    parser = (parser_analysis or "").strip()
    if parser:
        return trim_long_string(parser.replace("\n", " "), threshold=500, k=250)
    return "???????????????????????????????"


def _human_insight_fingerprint(node: SearchNode, parser_analysis: str | None) -> str:
    metric = getattr(node, "metric", None)
    payload = {
        "parser_analysis": parser_analysis or "",
        "decision_signals": node.decision_signals,
        "is_buggy": node.is_buggy,
        "is_valid": node.is_valid,
        "metric": getattr(metric, "value", None) if metric is not None else None,
        "maximize": getattr(metric, "maximize", None) if metric is not None else None,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _generate_human_node_insight(agent, node: SearchNode, parser_generated: bool, *, force: bool = False) -> None:
    """Generate a human-facing LLM insight without replacing parser facts."""
    parser_analysis = node.parser_analysis or node.analysis
    fingerprint = _human_insight_fingerprint(node, parser_analysis)
    if (
        not force
        and node.llm_insight
        and getattr(node, "_llm_insight_fingerprint", None) == fingerprint
    ):
        return

    payload = {
        "task_type": "optimization_or_rl" if _is_optimization_or_rl_agent(agent) else "standard_ml",
        "parser_source": "deterministic_parser" if parser_generated else "llm_review_summary",
        "stage": node.stage,
        "is_buggy": node.is_buggy,
        "is_valid": node.is_valid,
        "metric": getattr(node.metric, "value", None) if node.metric else None,
        "maximize": getattr(node.metric, "maximize", None) if node.metric else None,
        "parser_analysis": parser_analysis,
        "decision_signals": node.decision_signals,
        "plan_excerpt": trim_long_string(node.plan or "", threshold=1200, k=600),
        "execution_tail": (node.term_out or "")[-1600:],
    }
    system_message = (
        "You write short UI-facing insights for AutoML search nodes. "
        "The parser facts are authoritative: do not change metrics, counts, validity, or warnings. "
        "Do not simply restate JSON fields or parser text; translate them into a useful human explanation. "
        "Use Chinese when the task/output is Chinese; otherwise use concise English. "
        "Return only the structured JSON field requested."
    )
    user_message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    try:
        response = cast(
            dict,
            query(
                system_message=system_message,
                user_message=user_message,
                func_spec=NODE_INSIGHT_FUNC_SPEC,
                model=agent.acfg.feedback.model,
                temperature=agent.acfg.feedback.temp,
                stage_name="feedback",
                cfg=agent.cfg,
            ),
        )
        insight = str(response.get("insight") or "").strip()
        node.llm_insight = insight or _fallback_human_insight(node, parser_analysis)
    except Exception as e:
        logger.warning("[parse] failed to generate human node insight for %s: %s", node.id, e)
        node.llm_insight = _fallback_human_insight(node, parser_analysis)
    setattr(node, "_llm_insight_fingerprint", fingerprint)


def refresh_human_node_insight(agent, node: SearchNode) -> SearchNode:
    """Refresh the UI-facing insight after later validator steps update parser facts."""
    _generate_human_node_insight(agent, node, parser_generated=True)
    return node


def _parse_bool_like(value) -> bool | None:
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


def _extract_decision_validation_summary(text: str) -> dict | None:
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


def _trusted_decision_score_source(summary: dict | None) -> bool:
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

    # AutoRealize optimization/RL tasks often define the official score directly
    # as a penalized-cost formula. Treat that formula as trusted even when the
    # generated code names the source by formula instead of by function name.
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


def _decision_summary_warnings(summary: dict | None) -> list[str]:
    """Decision summaries are diagnostic only; no generic warning fields."""
    return []


def _decision_summary_defects(summary: dict | None) -> list[str]:
    """Compatibility no-op: summary fields no longer block node acceptance."""
    return []


def decision_summary_is_scorable(summary: dict | None) -> bool:
    """Decision/RL acceptance is based on final score, not summary fields."""
    return True


# Keep the parser and standalone tests on the same decision-validation rules.
# The local helpers above are retained for backward compatibility with older
# imports, but runtime parsing should use the shared utility implementation.
_parse_bool_like = _dv_parse_bool_like
_extract_decision_validation_summary = _dv_extract_decision_validation_summary
_trusted_decision_score_source = _dv_trusted_decision_score_source
_decision_summary_defects = _dv_decision_summary_defects
decision_summary_is_scorable = _dv_decision_summary_is_scorable


def has_scorable_decision_run(agent, node: SearchNode) -> bool:
    """Whether this optimization/RL execution produced a scalar final score."""
    if not _is_optimization_or_rl_agent(agent):
        return False
    return _extract_final_validation_score(node.term_out) is not None


def _short_json(value, *, limit: int = 1200) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + (" ..." if len(text) > limit else "")


def _decision_summary_details(summary: dict | None) -> list[str]:
    """Extract actionable validation details for debug/improve prompts."""

    if not isinstance(summary, dict):
        return []

    details: list[str] = []
    signals = _dv_decision_signal_summary(summary)
    if signals:
        details.append(f"decision_signals: {_short_json(signals, limit=800)}")
    for key in [
        "notes",
        "error",
        "failure_reason",
        "validation_error",
        "final_score_source",
    ]:
        value = summary.get(key)
        if value not in (None, "", [], {}):
            details.append(f"{key}: {_short_json(value, limit=800)}")

    score_components = summary.get("score_components")
    if score_components not in (None, "", [], {}):
        details.append(f"score_components: {_short_json(score_components, limit=1200)}")

    for key in [
        "validation_report",
        "feasibility_report",
        "objective_report",
        "constraint_report",
        "violation_report",
        "schema_report",
    ]:
        value = summary.get(key)
        if value not in (None, "", [], {}):
            details.append(f"{key}: {_short_json(value, limit=1200)}")

    generic_diagnostic_suffixes = (
        "_report",
        "_reports",
        "_detail",
        "_details",
        "_example",
        "_examples",
        "_reason",
        "_reasons",
        "_status",
        "_warning",
        "_warnings",
        "_error",
        "_errors",
    )
    reserved_keys = {
        "score_components",
        "final_score_source",
        "notes",
        "error",
        "failure_reason",
        "validation_error",
        "evaluator_self_tests_passed",
        "is_feasible",
    }
    for key, value in summary.items():
        normalized = str(key).lower()
        if key in reserved_keys or value in (None, "", [], {}):
            continue
        if normalized.endswith(generic_diagnostic_suffixes):
            details.append(f"{key}: {_short_json(value, limit=1200)}")

    return details[:12]

metric_direction_func_spec = FunctionSpec(
    name="determine_metric_direction",
    json_schema={
        "type": "object",
        "properties": {
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE, RMSE, MAE, loss, error rate), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy, F1 score, AUC, precision, recall, Jaccard score, IoU).",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this metric direction is chosen based on the task's evaluation metric description.",
            },
        },
        "required": [
            "lower_is_better",
            "reasoning",
        ],
    },
    description="Determine whether the evaluation metric should be minimized or maximized based on the task description.",
)


def determine_metric_direction(agent) -> None:
    logger.info("=" * 80)
    logger.info("Starting pre-determination of metric optimization direction...")
    logger.info("=" * 80)

    prompt = """You are analyzing a machine learning competition task. Your task is to determine whether the evaluation metric should be minimized or maximized.

    **IMPORTANT: Focus on the EVALUATION section in the task description, which specifies the metric used to score submissions.**

    Based on the evaluation metric mentioned in the task description, determine:
    - If the metric should be MINIMIZED (lower is better), set lower_is_better to TRUE.
    Examples: MSE, RMSE, MAE, Cross-Entropy Loss, Log Loss, Error Rate
    - If the metric should be MAXIMIZED (higher is better), set lower_is_better to FALSE.
    Examples: Accuracy, F1 Score, AUC-ROC, Precision, Recall, Jaccard Score, IoU, mAP

    **Pay special attention to:**
    1. The "Evaluation" or "Metric" section in the task description
    2. Common metric conventions (e.g., accuracy is always maximized, MSE is always minimized)
    3. Whether the metric measures error/loss (minimize) or performance/quality (maximize)

    Provide clear reasoning based on the evaluation metric specified in the task.
    """
    user_prompt = task_section(agent.task_desc)

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                logger.info(f"Attempt {attempt}/{max_retries} to determine metric direction...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries} to determine metric direction...")
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=user_prompt,
                    func_spec=metric_direction_func_spec,
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    stage_name="feedback",
                    cfg=agent.cfg
                ),
            )

            lower_is_better = response["lower_is_better"]
            agent.metric_maximize = not lower_is_better
            reasoning = response.get("reasoning", "")
            agent.metric_maximize_reasoning = reasoning

            logger.info("=" * 80)
            logger.info("Pre-determination completed successfully:")
            logger.info(f"  - lower_is_better = {lower_is_better}")
            logger.info(f"  - maximize = {agent.metric_maximize}")
            logger.info(f"  - Reasoning: {reasoning}")
            logger.info("=" * 80)
            logger.info(f"All subsequent nodes MUST use maximize={agent.metric_maximize}, otherwise they will be marked as buggy")
            logger.info("=" * 80)
            return

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info("Retrying in a moment...")
                time.sleep(1)
            else:
                logger.error("=" * 80)
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                logger.error("Using default value maximize=True (assuming higher is better)")
                logger.error("=" * 80)
                agent.metric_maximize = True
                agent.metric_maximize_reasoning = "Default: assuming higher is better (most common case)"


def get_review_func_spec(use_memory: bool, optimization_rl: bool = False) -> FunctionSpec:
    bug_description = (
        "true if the execution failed, crashed, or did not report a usable scalar metric. "
        "For optimization/RL tasks, do not mark a run buggy merely because it lacks a "
        "Decision Validation Summary or optional evaluator diagnostics."
        if optimization_rl
        else "true if the output log shows that the execution failed or has some bug, otherwise false. "
             "Focus only on actual execution errors, exceptions, or crashes."
    )
    metric_description = (
        "For optimization/RL tasks, report the final scalar validation score printed by the program. "
        "Optional Decision Validation Summary fields are diagnostic and are not acceptance requirements."
        if optimization_rl
        else "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null."
    )
    properties = {
        "is_bug": {
            "type": "boolean",
            "description": bug_description,
        },
        "summary": {
            "type": "string",
            "description": "Provide a concise summary (2-3 sentences) of the execution outcome. "
                           "If successful, describe the key empirical results. "
                           "If failed, describe the error encountered. "
                           "Focus on observations only — do not include suggestions for improvement.",
        },
        "metric": {
            "type": "number",
            "description": metric_description,
        },
        "lower_is_better": {
            "type": "boolean",
            "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
        },
    }
    required = ["is_bug", "summary", "metric", "lower_is_better"]
    if use_memory:
        properties["code_summary"] = {
            "type": "string",
            "description": "Write a summary including the methods used in each stage of the code, such as data preprocessing, feature engineering, model architecture, etc.",
        }
        required.append("code_summary")
    return FunctionSpec(
        name="submit_review",
        json_schema={"type": "object", "properties": properties, "required": required},
        description="Submit a review evaluating the output of the training script.",
    )


def _build_introduction(agent) -> str:
    use_memory = getattr(agent.acfg, "use_global_memory", False)
    submission_required = getattr(agent.acfg, "generate_submission", True)
    optimization_rl = _is_optimization_or_rl_agent(agent)
    intro = (
        "You are a Kaggle grandmaster attending a competition. "
        "You have written code to solve this task and now need to evaluate the output of the code execution. "
        "You should determine if there were any bugs as well as report the empirical findings.\n\n"
        "You MUST respond with a JSON object containing ALL of the following fields:\n"
        "- \"is_bug\": (boolean) true if execution failed or has bugs, false otherwise. Must be a JSON boolean (true/false), NOT a string.\n"
        "- \"summary\": (string) A concise 2-3 sentence summary of the execution outcome.\n"
        "- \"metric\": (number or null) The validation metric value as a raw JSON number (e.g. 0.9995), NOT a string. If failed, use null.\n"
        "- \"lower_is_better\": (boolean) true if the metric should be minimized, false if maximized. Must be a JSON boolean (true/false), NOT a string.\n"
    )
    if not submission_required:
        intro += (
            "\nConfig note: final submission.csv generation is disabled for this run. "
            "Do NOT mark the execution as buggy merely because it did not create a submission file; "
            "judge success by execution correctness and the reported validation metric.\n"
        )
    if optimization_rl:
        intro += (
            "\nOptimization/RL/decision-task review rules:\n"
            "- Accept a run when it executes and reports a usable scalar `Final Validation Score`.\n"
            "- `Decision Validation Summary`, `score_components`, `final_score_source`, `evaluator_self_tests_passed`, and `is_feasible` are optional diagnostics, not acceptance requirements.\n"
            "- Do NOT require universal progress or violation fields; those are task-specific diagnostics.\n"
            "- A successful optimization/RL node may be partial, infeasible, or diagnostic if it still reports a scalar score that later nodes can improve.\n"
            "- If optional diagnostics exist, preserve their warnings, examples, infeasibility reasons, or objective-component details for later improvement.\n"
        )
    if use_memory:
        intro += (
            "- \"code_summary\": (string) A concise method summary of the code, covering key parts such as "
            "data preprocessing, feature engineering, model architecture/training, and validation strategy.\n"
        )
    intro += "\nDo NOT omit any field."
    return intro
    


def _check_submission_file(agent, node: SearchNode) -> bool:
    correct_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    if not correct_path.exists():
        wrong_path = agent.cfg.workspace_dir / f"submission_{node.id}.csv"
        if wrong_path.exists():
            correct_path.parent.mkdir(parents=True, exist_ok=True)
            wrong_path.rename(correct_path)
            logger.warning(f" {wrong_path} are moved to {correct_path}")

    return correct_path.exists()


def _save_code_summary(agent, node: SearchNode, response: dict):
    use_memory = getattr(agent.acfg, "use_global_memory", False)
    if not use_memory:
        node.code_summary = None
        return
    if "code_summary" in response and response["code_summary"]:
        node.code_summary = response["code_summary"]
        logger.info(f"Saved code summary for node {node.id}")
    else:
        logger.warning(f"Node {node.id} missing code_summary in response")
        node.code_summary = None


def _determine_buggy(
    node: SearchNode,
    response: dict,
    has_csv_submission: bool,
    requires_submission: bool = True,
    allow_missing_submission: bool = False,
):
    failure_reasons = []
    if response["is_bug"]:
        failure_reasons.append("execution error detected")
    if node.exc_type is not None:
        failure_reasons.append(f"exception raised: {node.exc_type}")
    if response["metric"] is None:
        failure_reasons.append("no metric value reported")
    if requires_submission and not has_csv_submission and not allow_missing_submission:
        failure_reasons.append("submission file not found")

    node.is_buggy = len(failure_reasons) > 0
    if node.is_buggy:
        logger.warning(f"Node {node.id} marked as buggy: {'; '.join(failure_reasons)}")


def _validate_format_with_retry(agent, node: SearchNode):
    exp_id = _resolve_exp_id(agent)
    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    status, res = _validate_submission_with_retry(
        exp_id=exp_id,
        submission_path=submission_path,
        cfg=agent.cfg,
        max_attempts=2,
        sample_path=None,
    )

    if status:
        if not res['is_valid']:
            logger.warning(f"[validate] node {node.id}: invalid after retry attempts.")
            node.is_valid = False
            node.is_buggy = True
            node._term_out.append(f"\n{res['result']}")
            _set_parser_analysis(
                node,
                f"FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\nDetails:\n{res['result']}",
            )
        else:
            _check_content_quality(agent, node, submission_path)
    else:
        logger.error(f"An unexpected error occurred: {res}, skip this stage.")
        logger.info(f"Node {node.id} format validation passed. Now checking content quality...")
        content_valid, content_error = validate_submission_content_quality(
                submission_path=submission_path,
                sample_path=None,
                constant_threshold=0.95,
            )

        if not content_valid:
            _mark_content_quality_failure(node, content_error)
        else:
            logger.info(f"[validate] node {node.id}: valid")
            node.is_valid = True


def _append_analysis_note(node: SearchNode, note: str) -> None:
    if not note:
        return
    if node.analysis:
        if note not in node.analysis:
            node.analysis = f"{node.analysis}\n\n[Non-fatal warning] {note}"
    else:
        node.analysis = f"[Non-fatal warning] {note}"
    node.parser_analysis = node.analysis


def _validate_format_simple(agent, node: SearchNode):
    exp_id = _resolve_exp_id(agent)
    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    status, res = call_validate(exp_id=exp_id, submission_path=submission_path)
    if status:
        if not res['is_valid']:
            logger.warning(f"[validate] node {node.id}: invalid.")
            node.is_valid = False
            node.is_buggy = True
            node._term_out.append(f"\n{res['result']}")
            _set_parser_analysis(
                node,
                f"FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\nDetails:\n{res['result']}",
            )
        else:
            _check_content_quality(agent, node, submission_path)
    else:
        logger.error(f"An unexpected error occurred: {res}, skip this stage.")


def _check_content_quality(agent, node: SearchNode, submission_path):
    logger.info(f"Node {node.id} format validation passed. Now checking content quality...")
    content_valid, content_error = validate_submission_content_quality(
            submission_path=submission_path,
            sample_path=None,
            constant_threshold=0.95,
        )

    if not content_valid:
        _mark_content_quality_failure(node, content_error)
    else:
        logger.info(f"✅ Node {node.id} passed both format and content quality checks.")
        node.is_valid = True


def _mark_content_quality_failure(node: SearchNode, content_error):
    logger.warning(f"Node {node.id} is marked as buggy due to content quality check failure.")
    node.is_valid = False
    node.is_buggy = True
    error_message = (
        "Submission format is correct, but content quality check FAILED:\n\n"
        f"{content_error}\n\n"
        "🚨 CRITICAL: All predictions must come from actual model inference.\n"
        "You must:\n"
        "1. Load each test sample\n"
        "2. Preprocess it with the same transformations as training\n"
        "3. Run model.predict() / model.forward() on the sample\n"
        "4. Use the model's output as the prediction\n\n"
        "Filling submissions with constants, placeholders, or dummy values is STRICTLY FORBIDDEN."
    )
    node._term_out.append(f"\n{error_message}")
    _set_parser_analysis(
        node,
        f"CONTENT_QUALITY_ERROR: This previous solution runs without bugs and has correct format, but failed content quality check.\n\nDetails:\n{content_error}",
    )


def _validate_metric_direction(agent, node: SearchNode, response: dict):
    returned_maximize = not response["lower_is_better"]
    if agent.metric_maximize is not None and returned_maximize != agent.metric_maximize:
        logger.error("=" * 80)
        logger.error(f"METRIC DIRECTION MISMATCH for Node {node.id}!")
        logger.error(f"  - Returned lower_is_better = {response['lower_is_better']} (maximize={returned_maximize})")
        logger.error(f"  - Pre-determined maximize = {agent.metric_maximize}")
        logger.error(f"  - Marking this node as BUGGY, will NOT update top candidates")
        logger.error("=" * 80)
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            f"{node.analysis}\n\n[ERROR] Metric direction mismatch detected:\n"
            f"- Returned lower_is_better={response['lower_is_better']} (maximize={returned_maximize})\n"
            f"- Expected maximize={agent.metric_maximize}\n"
            f"- Pre-determination reasoning: {agent.metric_maximize_reasoning or 'N/A'}\n"
            f"This node is marked as buggy and will not be considered for best/top candidates."
        )
        node.parser_analysis = node.analysis
    else:
        logger.info(f"Node {node.id} metric direction validated: maximize={agent.metric_maximize}")
        node.metric = MetricValue(
            response["metric"], maximize=agent.metric_maximize
        )


def _check_data_leakage(agent, node: SearchNode, response: dict):
    if not (agent.acfg.check_data_leakage and should_check_data_leakage(agent, node)):
        return

    logger.warning(
        f"Node {node.id} triggers data leakage check due to extreme metric value: {node.metric.value}"
    )

    leakage_result = data_leakage_agent.run(agent, node)

    if leakage_result["has_leakage"] and leakage_result["confidence"] in ["high", "medium"]:
        logger.error(
            f"⚠️  Node {node.id} detected data leakage with {leakage_result['confidence']} confidence. "
            f"Marking as buggy and resetting metric."
        )
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            f"⚠️ DATA LEAKAGE DETECTED (Confidence: {leakage_result['confidence'].upper()})\n\n"
            f"{leakage_result['reason']}\n\n"
            f"The validation metric was {response['metric']:.4f} which is unrealistically extreme due to data leakage. "
            f"To fix this issue, you need to:\n"
            f"1. Carefully review the train/validation split logic\n"
            f"2. Ensure no validation/test data is used during training\n"
            f"3. Check that feature engineering only uses training data statistics\n"
            f"4. Verify data augmentation doesn't leak validation samples\n"
            f"5. Ensure proper temporal/group separation if applicable"
        )
        node.parser_analysis = node.analysis
        logger.info(f"Updated node.analysis with leakage detection details for debugging")
    else:
        if leakage_result["has_leakage"]:
            logger.info(
                f"Node {node.id} has potential leakage but confidence is low. Not marking as buggy."
            )
        else:
            logger.info(
                f"Node {node.id} extreme value is justified: {leakage_result['reason']}"
            )


def _save_to_global_memory(agent, node: SearchNode):
    if agent.global_memory and not node.is_buggy and node.metric and node.metric.value is not None:
        try:
            parent_node = node.parent
            agent.global_memory.save_node(node, parent_node)
        except Exception as e:
            logger.warning(f"[AgentSearch] Failed to save node {node.id} to global memory: {e}")


def _extract_final_validation_score(text: str) -> float | None:
    matches = FINAL_SCORE_RE.findall(text or "")
    if not matches:
        return None
    try:
        value = float(matches[-1])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _compact_failure_summary(node: SearchNode) -> str:
    tail = (node.term_out or "").strip()[-1400:]
    exc = node.exc_type or "ExecutionError"
    if tail:
        return f"Execution failed with {exc}. Tail output:\n{tail}"
    return f"Execution failed with {exc}."


def _normalize_review_response(agent, response: dict) -> dict:
    response.setdefault("is_bug", True)
    response.setdefault("summary", "No summary returned by model.")
    response.setdefault("metric", None)
    response.setdefault(
        "lower_is_better",
        not agent.metric_maximize if agent.metric_maximize is not None else False,
    )

    metric_val = response.get("metric")
    if not isinstance(metric_val, (int, float)):
        try:
            response["metric"] = float(metric_val)
        except (TypeError, ValueError):
            response["metric"] = None

    for bool_field in ("is_bug", "lower_is_better"):
        v = response.get(bool_field)
        if isinstance(v, str):
            response[bool_field] = v.strip().lower() not in ("false", "0", "no", "")
    return response


def _apply_review_response(agent, node: SearchNode, response: dict) -> SearchNode:
    response = _normalize_review_response(agent, response)

    requires_submission = getattr(agent.acfg, "generate_submission", True)
    has_csv_submission = _check_submission_file(agent, node) if requires_submission else True
    scorable_decision_run = (
        response.get("is_bug") is False
        and response.get("metric") is not None
        and has_scorable_decision_run(agent, node)
    )

    parser_generated = bool(response.pop("_parser_generated", False))
    decision_summary = response.pop("_decision_summary", None)
    _set_parser_analysis(node, response["summary"])
    node.decision_signals = _decision_signals_for_node(decision_summary, response.get("metric"))
    _save_code_summary(agent, node, response)
    _determine_buggy(
        node,
        response,
        has_csv_submission,
        requires_submission=requires_submission,
        allow_missing_submission=scorable_decision_run,
    )

    if not node.is_buggy and requires_submission and scorable_decision_run:
        if has_csv_submission:
            node.is_valid = True
            _append_analysis_note(
                node,
                "Generic Kaggle-style submission format/content validation was skipped because "
                "this optimization/RL node reported a trusted penalized decision score. "
                "Task-specific quality issues remain visible in the Decision Validation Summary "
                "and should be improved in later nodes.",
            )
        else:
            node.is_valid = False
            _append_analysis_note(
                node,
                "No submission file was found, but the node produced a trusted penalized decision score. "
                "The node is retained for debug/improve instead of being treated as a runtime failure.",
            )
    elif not node.is_buggy and requires_submission:
        _validate_format_with_retry(agent, node)
    elif not node.is_buggy:
        node.is_valid = True

    if node.is_buggy:
        node.metric = WorstMetricValue()
    else:
        _validate_metric_direction(agent, node, response)
        _check_data_leakage(agent, node, response)
    node.parser_analysis = node.analysis
    _generate_human_node_insight(agent, node, parser_generated=parser_generated)

    status = "FAIL" if node.is_buggy else "PASS"
    metric_val = node.metric.value if node.metric else None
    logger.info(f"[parse] node {node.id}: {status} | metric={metric_val}")

    _save_to_global_memory(agent, node)

    return node


def _try_deterministic_parse(agent, node: SearchNode) -> SearchNode | None:
    """Avoid an LLM call when execution status and final score are unambiguous."""
    if node.exc_type is not None:
        node.is_buggy = True
        node.metric = WorstMetricValue()
        _set_parser_analysis(node, _compact_failure_summary(node))
        node.decision_signals = None
        _generate_human_node_insight(agent, node, parser_generated=True)
        logger.info("[parse] node %s: deterministic failure parse, generated human insight", node.id)
        return node

    score = _extract_final_validation_score(node.term_out)
    if score is None:
        return None

    optimization_rl = _is_optimization_or_rl_agent(agent)
    decision_summary = None
    if optimization_rl:
        decision_summary = _extract_decision_validation_summary(node.term_out)

    lower_is_better = not agent.metric_maximize if agent.metric_maximize is not None else False
    response = {
        "is_bug": False,
        "summary": (
            "Execution completed and printed a deterministic final validation score. "
            f"Parsed `Final Validation Score` = {score}."
        ),
        "metric": score,
        "lower_is_better": lower_is_better,
        "_parser_generated": True,
    }
    if optimization_rl:
        detail_lines = _decision_summary_details(decision_summary)
        detail_suffix = ""
        if detail_lines:
            detail_suffix = " Details: " + " | ".join(detail_lines)
        summary_status = (
            "with optional Decision Validation Summary"
            if isinstance(decision_summary, dict)
            else "without Decision Validation Summary"
        )
        response["summary"] = (
            f"Execution completed {summary_status}, and printed a final validation score. "
            f"Parsed `Final Validation Score` = {score}."
            + detail_suffix
        )
        response["_decision_summary"] = decision_summary
    if getattr(agent.acfg, "use_global_memory", False):
        response["code_summary"] = (node.plan or "Deterministically parsed successful execution.")[:800]
    logger.info("[parse] node %s: deterministic score parse, no LLM call", node.id)
    return _apply_review_response(agent, node, response)


def run(agent, node: SearchNode, exec_result: ExecutionResult) -> SearchNode:
    max_retries = 3
    for retry_idx in range(max_retries):
        try:
            logger.info(f"Agent is parsing execution results for node {node.id}")

            node.absorb_exec_result(exec_result)
            deterministic = _try_deterministic_parse(agent, node)
            if deterministic is not None:
                return deterministic

            introduction = _build_introduction(agent)
            prompt = {
                "Introduction": introduction,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
            }

            optimization_rl = _is_optimization_or_rl_agent(agent)
            response = cast(
                dict,
                query(
                    system_message={"Introduction": introduction},
                    user_message=(
                        f"{task_section(agent.task_desc)}\n"
                        f"# Implementation\n{prompt['Implementation']}\n\n"
                        f"# Execution output\n{prompt['Execution output']}"
                    ),
                    func_spec=get_review_func_spec(
                        getattr(agent.acfg, "use_global_memory", False),
                        optimization_rl=optimization_rl,
                    ),
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    stage_name="feedback",
                    cfg=agent.cfg
                ),
            )

            return _apply_review_response(agent, node, response)
        except Exception as e:
            logger.warning(f"[parse] tool call failed: {e}")
            continue

    logger.error(f"All {max_retries} parse attempts failed for node {node.id}, marking as buggy")
    node.is_buggy = True
    node.metric = WorstMetricValue()
    _set_parser_analysis(node, "Execution result parsing failed after multiple attempts.")
    _generate_human_node_insight(agent, node, parser_generated=True)
    return node
