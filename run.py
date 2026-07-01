import atexit
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from types import SimpleNamespace
from typing import Optional

import torch
from omegaconf import OmegaConf
from rich.status import Status

from config import load_cfg, load_task_desc, prep_agent_workspace, save_run
from engine.agent_search import AgentSearch as Agent
from engine.coldstart import build_guidance_description
from engine.executor import Interpreter
from engine.search_node import Journal
from agents.prompts import is_optimization_or_rl_task
from utils.seed import set_global_seed
from utils.logging_config import setup_logging
from utils.visualization import journal_to_string_tree
from utils.serialize import load_json


PENDING_NODES_FILE = "pending_nodes.json"
PENDING_DRAFT_STATUSES = {"generating", "pending_execution", "executing", "cancelled", "failed"}


def _node_attr(node, name: str, default=None):
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def _pending_node_row(node, status: str) -> dict:
    parent = _node_attr(node, "parent", None)
    metric = _node_attr(node, "metric", None)
    metric_value = getattr(metric, "value", None) if metric is not None else None
    metric_maximize = getattr(metric, "maximize", None) if metric is not None else None
    return {
        "id": str(_node_attr(node, "id", "")),
        "parent_id": _node_attr(node, "parent_id", None) or getattr(parent, "id", None),
        "stage": _node_attr(node, "stage", "draft"),
        "plan": _node_attr(node, "plan", None),
        "code": _node_attr(node, "code", None),
        "result": "",
        "insight": _node_attr(node, "llm_insight", None) or _node_attr(node, "analysis", None),
        "llm_insight": _node_attr(node, "llm_insight", None),
        "parser_analysis": _node_attr(node, "parser_analysis", None) or _node_attr(node, "analysis", None),
        "decision_signals": _node_attr(node, "decision_signals", None),
        "metric": metric_value,
        "maximize": metric_maximize if isinstance(metric_maximize, bool) else None,
        "is_buggy": _node_attr(node, "is_buggy", None),
        "is_valid": _node_attr(node, "is_valid", None),
        "visits": _node_attr(node, "visits", 0),
        "total_reward": _node_attr(node, "total_reward", 0.0),
        "uct": _node_attr(node, "_uct", None),
        "finish_time": _node_attr(node, "finish_time", None),
        "exec_time": _node_attr(node, "exec_time", None),
        "branch_id": _node_attr(node, "branch_id", None),
        "from_topk": _node_attr(node, "from_topk", None),
        "created_time": _node_attr(node, "created_time", None),
        "status": status,
        "pending_execution": status in {"generating", "pending_execution", "executing"},
        "label": {
            "generating": "Draft code is being generated",
            "pending_execution": "Draft generated, pending execution",
            "executing": "Draft execution is running",
            "cancelled": "Draft execution was cancelled before journal append",
            "failed": "Draft generation failed before execution",
        }.get(status, status),
    }


def _make_pending_draft_placeholder(draft_idx: int, draft_total: int, *, fast_first: bool) -> SimpleNamespace:
    mode = "fast_first_draft" if fast_first else "stepwise_draft"
    return SimpleNamespace(
        id=f"draft-{draft_idx + 1}-generating",
        parent=None,
        parent_id=None,
        stage="draft",
        plan=(
            f"Generating draft {draft_idx + 1}/{draft_total} via {mode}. "
            "This placeholder is shown before code generation finishes."
        ),
        code="",
        analysis=None,
        parser_analysis=None,
        decision_signals=None,
        llm_insight=None,
        metric=None,
        is_buggy=None,
        is_valid=None,
        visits=0,
        total_reward=0.0,
        _uct=None,
        finish_time=None,
        exec_time=None,
        branch_id=draft_idx + 1,
        from_topk=False,
        created_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def _write_pending_nodes_state(log_dir, nodes, status_by_id: dict[str, str], phase: str) -> None:
    path = log_dir / PENDING_NODES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for node in nodes:
        node_id = str(getattr(node, "id", ""))
        status = status_by_id.get(node_id)
        if status:
            rows.append(_pending_node_row(node, status))
    payload = {
        "schema_version": "mlevolve.pending_nodes.v1",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "phase": phase,
        "nodes": rows,
    }
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        tmp_path.unlink(missing_ok=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        if last_error is not None:
            raise last_error
        raise


def run():
    cfg = load_cfg()
    resume_run = os.environ.get("MLEVOLVE_RESUME_RUN", "").strip().lower() in {"1", "true", "yes", "on"}
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart:
        logger.info("Loading guidance from knowledge base")
        cfg.coldstart.description = build_guidance_description(cfg, task_desc=task_desc)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    global_step = 0

    def cleanup():
        if global_step == 0 and not resume_run:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    def _repair_journal_state(journal: Journal):
        if len(journal) == 0:
            return journal
        id2node = {n.id: n for n in journal.nodes}
        for n in journal.nodes:
            n.children = set()
            n.lock = False
            n.child_count_lock = threading.Lock()
        for n in journal.nodes:
            p = getattr(n, "parent", None)
            if p is None:
                continue
            if p.id in id2node:
                parent = id2node[p.id]
                n.parent = parent
                parent.children.add(n)
        for n in journal.nodes:
            n.expected_child_count = len(getattr(n, "children", set()) or set())
        return journal

    resume_path = cfg.log_dir / "journal.json"
    if resume_path.exists():
        try:
            journal = load_json(resume_path, Journal)
            journal = _repair_journal_state(journal)
            logger.info(f"Resuming from existing journal: {resume_path}")
        except Exception as e:
            logger.warning(f"Failed to load existing journal, starting fresh: {e}")
            journal = Journal()
    else:
        journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    completed = max(0, len(journal) - 1)
    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info("[step_task] Processing virtual root node.")
        return agent.step(exec_callback=exec_callback, node=node)

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.initial_drafts
    optimization_or_rl_task = is_optimization_or_rl_task(
        task_desc=str(getattr(agent, "task_desc", "") or task_desc or ""),
        coldstart_description=str(getattr(agent, "coldstart_description", "") or ""),
    )
    if optimization_or_rl_task and initial_draft_count > 1:
        logger.info(
            "Optimization/RL task detected; reducing Phase 1 initial_drafts from %s to 1 "
            "so the first executable search node appears sooner.",
            initial_draft_count,
        )
        initial_draft_count = 1

    time_limit_secs = int(getattr(cfg.agent, "time_limit", 0) or 0)
    run_deadline: Optional[float] = time.time() + time_limit_secs if time_limit_secs > 0 else None
    timed_out = False

    logger.info(f"ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"Initial draft count: {initial_draft_count} (executed sequentially for diversity)")
    logger.info("Phase 1 fast_first_draft is enabled: draft 1 uses single-call generation before stepwise drafts.")
    if run_deadline is not None:
        logger.info(f"Hard timeout enabled: {time_limit_secs}s")

    def is_timed_out() -> bool:
        return run_deadline is not None and time.time() >= run_deadline

    lock = threading.Lock()
    logger.info(f"Resume progress: completed={completed}/{total_steps} from journal nodes={len(journal)}")

    pending_draft_nodes = []
    pending_status_by_id: dict[str, str] = {}
    _write_pending_nodes_state(cfg.log_dir, pending_draft_nodes, pending_status_by_id, "initialized")

    def refresh_pending_nodes_state(phase: str) -> None:
        try:
            _write_pending_nodes_state(cfg.log_dir, pending_draft_nodes, pending_status_by_id, phase)
        except Exception as exc:
            logger.warning(f"Failed to write {PENDING_NODES_FILE}: {exc}")

    if initial_draft_count > 0 and completed == 0 and total_steps > 0:
        logger.info(f"Phase 1: Sequential draft generation (code only, {initial_draft_count} drafts)")

        def step_task_generate_only(*, fast_first_draft: bool = False):
            logger.info(
                "[step_task_generate_only] Generating draft from virtual root%s",
                " using fast_first_draft single-call route" if fast_first_draft else "",
            )
            previous_stepwise = getattr(agent, "use_stepwise_generation", True)
            if fast_first_draft:
                agent.use_stepwise_generation = False
            try:
                return agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)
            finally:
                agent.use_stepwise_generation = previous_stepwise

        draft_total = min(initial_draft_count, total_steps)
        for draft_idx in range(draft_total):
            if is_timed_out():
                timed_out = True
                logger.warning("Time limit reached during Phase 1 draft generation; stop creating new drafts.")
                break
            fast_first_draft = draft_idx == 0
            placeholder = _make_pending_draft_placeholder(
                draft_idx,
                draft_total,
                fast_first=fast_first_draft,
            )
            pending_draft_nodes.append(placeholder)
            pending_status_by_id[placeholder.id] = "generating"
            refresh_pending_nodes_state("phase1_draft_generation")
            try:
                logger.info(
                    f"Generating draft {draft_idx + 1}/{min(initial_draft_count, total_steps)} (code only)"
                )
                cur_node = step_task_generate_only(fast_first_draft=fast_first_draft)
                pending_draft_nodes[-1] = cur_node
                pending_status_by_id.pop(placeholder.id, None)
                pending_status_by_id[cur_node.id] = "pending_execution"
                refresh_pending_nodes_state("phase1_draft_generation")
                logger.info(f"Draft {draft_idx + 1} code generated: node.id={cur_node.id}")
            except Exception as e:
                pending_status_by_id[placeholder.id] = "failed"
                refresh_pending_nodes_state("phase1_draft_generation")
                logger.exception(f"Exception during draft {draft_idx + 1} generation: {e}")

        logger.info(f"Phase 1 complete: {len(pending_draft_nodes)} draft codes generated")
        refresh_pending_nodes_state("phase1_draft_generation_complete")

    if pending_draft_nodes or completed < total_steps:
        drafts_to_execute = [
            node for node in pending_draft_nodes
            if pending_status_by_id.get(str(_node_attr(node, "id", ""))) == "pending_execution"
        ]
        logger.info("Phase 2: Pipelined parallel execution")
        logger.info(f"  - Pending draft executions: {len(drafts_to_execute)}")
        logger.info(f"  - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node):
            try:
                executed_node = agent.execute_deferred_node(node, exec_callback)
                logger.info(f"Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except Exception as e:
                logger.exception(f"Exception during draft node {node.id} execution: {e}")
                return None

        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        fast_shutdown = False
        try:
            futures = set()
            pending_future_ids: dict = {}
            submitted_drafts = 0
            for i, node in enumerate(drafts_to_execute):
                if is_timed_out():
                    timed_out = True
                    logger.warning("Time limit reached before submitting pending draft executions.")
                    break
                pending_status_by_id[node.id] = "executing"
                refresh_pending_nodes_state("phase2_execution")
                fut = executor.submit(execute_draft_node, node)
                futures.add(fut)
                pending_future_ids[fut] = node.id
                submitted_drafts += 1
                logger.info(f"Submitted draft execution: {node.id}")
                if i < len(drafts_to_execute) - 1:
                    time.sleep(10)
                    if is_timed_out():
                        timed_out = True
                        logger.warning("Time limit reached while staggering draft submissions.")
                        break

            initial_step_tasks = min(max_workers, total_steps - completed) - submitted_drafts
            if initial_step_tasks > 0 and not timed_out:
                for _ in range(initial_step_tasks):
                    if is_timed_out():
                        timed_out = True
                        logger.warning("Time limit reached before initial step submission.")
                        break
                    futures.add(executor.submit(step_task))
                    logger.info("Submitted initial step_task to fill thread pool")

            while completed < total_steps and futures:
                if is_timed_out():
                    timed_out = True
                    logger.warning("Time limit reached in Phase 2 main loop.")
                    break

                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                if not done:
                    continue

                for fut in done:
                    futures.remove(fut)
                    pending_node_id = pending_future_ids.pop(fut, None)
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(
                                f"Task completed: node_id={cur_node.id}, step={cur_node.step}, "
                                f"is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}"
                            )
                        else:
                            logger.warning("Task returned None (execution failed)")
                    except Exception as e:
                        logger.exception(f"Exception during task execution: {e}")
                        cur_node = None

                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal) - 1
                        if completed == total_steps:
                            logger.info(journal_to_string_tree(journal))

                        if pending_node_id and cur_node:
                            pending_status_by_id.pop(pending_node_id, None)
                            refresh_pending_nodes_state("phase2_execution")
                        elif pending_node_id:
                            pending_status_by_id[pending_node_id] = "failed"
                            refresh_pending_nodes_state("phase2_execution")

                    if completed + len(futures) < total_steps and not timed_out:
                        if is_timed_out():
                            timed_out = True
                            logger.warning("Time limit reached before scheduling next task.")
                            continue
                        futures.add(executor.submit(step_task, cur_node))
                        logger.info(
                            f"Submitted next task based on node {cur_node.id if cur_node else 'None'}"
                        )
                    logger.info(f"Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")

            if timed_out:
                logger.error(
                    f"Time limit reached (configured={time_limit_secs}s). "
                    "Stop submitting new tasks and terminate running subprocesses."
                )
                interpreter.terminate_all_subprocesses()
                fast_shutdown = True
                with lock:
                    for node_id in list(pending_status_by_id):
                        pending_status_by_id[node_id] = "cancelled"
                    refresh_pending_nodes_state("timed_out")
                    save_run(cfg, journal)
            elif completed < total_steps and not futures:
                logger.warning(
                    f"Phase 2 exited with no active futures before reaching target steps: {completed}/{total_steps}"
                )
        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            for node_id in list(pending_status_by_id):
                pending_status_by_id[node_id] = "cancelled"
            refresh_pending_nodes_state("interrupted")
            interpreter.terminate_all_subprocesses()
            if sys.version_info >= (3, 9):
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=False)
            raise
        finally:
            if fast_shutdown:
                if sys.version_info >= (3, 9):
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=False)
            elif not interrupted:
                executor.shutdown(wait=True)
    else:
        logger.info(
            f"All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})"
        )

    if timed_out:
        logger.error(f"MLEvolve stopped by hard timeout: {time_limit_secs}s")

    if not pending_status_by_id:
        refresh_pending_nodes_state("complete")

    interpreter.cleanup_session(-1)


if __name__ == "__main__":
    run()
