"""Post-execution validation: validate_executed_node (csv existence, metric=0.0, register success)."""

import logging
import re

from engine.model_artifacts import find_model_artifacts
from engine.search_node import SearchNode
from utils.decision_validation import (
    is_optimization_or_rl_text,
)
from utils.metric import WorstMetricValue

logger = logging.getLogger("MLEvolve")

_ZERO_METRIC_ANALYSIS = (
    "Performance is 0.0 (complete failure). This indicates fundamental issues that need debugging:\n"
    "1. Model architecture may be incorrect or not learning\n"
    "2. Data preprocessing might be broken (wrong format, normalization issues)\n"
    "3. Loss function or evaluation metric calculation may be faulty\n"
    "4. Training loop might not be updating weights properly\n"
    "5. Input data might not be loaded correctly\n\n"
    "Please review the code carefully to identify the root cause."
)


def _append_nonfatal_decision_warning(node: SearchNode, message: str) -> None:
    node.is_valid = False
    if node.analysis:
        if message not in node.analysis:
            node.analysis = f"{node.analysis}\n\n[Non-fatal warning] {message}"
    else:
        node.analysis = f"[Non-fatal warning] {message}"
    node.parser_analysis = node.analysis


def _has_scorable_decision_run(agent, node: SearchNode) -> bool:
    if not is_optimization_or_rl_text(
        task_desc=getattr(agent, "task_desc", ""),
        coldstart_description=getattr(agent, "coldstart_description", ""),
    ):
        return False
    metric = getattr(node, "metric", None)
    return metric is not None and getattr(metric, "value", None) is not None


def validate_executed_node(agent, node: SearchNode):
    """Check submission.csv exists, metric=0.0 anomaly; register successful node to branch."""
    if node.is_buggy:
        return

    scorable_decision_run = _has_scorable_decision_run(agent, node)

    if not re.search(
        r"def\s+predict\s*\(\s*model_path(?:\s*:\s*[^,)=]+)?(?:\s*=\s*[^,)]*)?\s*,\s*data(?:\s*:\s*[^,)=]+)?(?:\s*=\s*[^,)]*)?\s*[,)]",
        node.code or "",
    ):
        if scorable_decision_run:
            _append_nonfatal_decision_warning(
                node,
                "The node produced a trusted penalized decision score, but it did not expose "
                "`predict(model_path, data)`. It is kept for debug/improve so a follow-up node can "
                "add the reusable API without discarding the partial solution.",
            )
            logger.info(
                "Node %s lacks predict(model_path, data), but is retained as a scorable decision candidate",
                node.id,
            )
            return
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            "The solution did not expose the required reusable inference API: "
            "`predict(model_path, data)`. Successful MLEvolve nodes must save a "
            "model artifact and provide this function so the best_solution can be reused."
        )
        node.parser_analysis = node.analysis
        logger.info(f"Node {node.id} did not define predict(model_path, data)")
        return

    model_artifacts = find_model_artifacts(agent.cfg.workspace_dir, str(node.id))
    if not model_artifacts:
        if scorable_decision_run:
            _append_nonfatal_decision_warning(
                node,
                "The node produced a trusted penalized decision score, but no node-specific model/solver "
                "artifact was found. It is kept for debug/improve; a follow-up node should save a lightweight "
                "artifact even for heuristic solvers.",
            )
            logger.info(
                "Node %s lacks a node-specific model artifact, but is retained as a scorable decision candidate",
                node.id,
            )
            return
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            "The solution did not save a node-specific model artifact under ./working, "
            "./models, ./artifacts, or ./checkpoints. Save the trained model and any "
            "required preprocessing state to a file such as `./working/model_artifact.pkl` "
            "or `./working/best_model.pt`, then load it inside `predict(model_path, data)`."
        )
        node.parser_analysis = node.analysis
        logger.info(f"Node {node.id} did not produce a model artifact")
        return

    if getattr(agent.acfg, "generate_submission", True):
        submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
        if not submission_path.exists():
            if scorable_decision_run:
                _append_nonfatal_decision_warning(
                    node,
                    "The node produced a trusted penalized decision score, but no node-specific submission file "
                    "was found. It is kept for debug/improve instead of being treated like a runtime failure.",
                )
                logger.info(
                    "Node %s lacks a submission file, but is retained as a scorable decision candidate",
                    node.id,
                )
                return
            node.is_buggy = True
            node.metric = WorstMetricValue()
            logger.info(f"Node {node.id} did not produce a submission.csv")
            return

    if node.metric.maximize and node.metric.value == 0.0:
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = _ZERO_METRIC_ANALYSIS
        node.parser_analysis = node.analysis
        logger.warning(
            f"Node {node.id} has metric=0.0 (maximize=True), marking as buggy for debugging."
        )
        return

    if hasattr(node, 'branch_id') and node.branch_id:
        if node.branch_id not in agent.branch_successful_nodes:
            agent.branch_successful_nodes[node.branch_id] = []
        agent.branch_successful_nodes[node.branch_id].append(node)
