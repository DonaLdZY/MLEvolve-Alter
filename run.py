import atexit
import logging
import os
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Optional

import torch
from omegaconf import OmegaConf
from rich.status import Status

from config import load_cfg, load_task_desc, prep_agent_workspace, save_run
from engine.agent_search import AgentSearch as Agent
from engine.coldstart import build_guidance_description
from engine.executor import Interpreter
from engine.search_node import Journal
from utils.seed import set_global_seed
from utils.logging_config import setup_logging
from utils.visualization import journal_to_string_tree
from utils.serialize import load_json


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

    time_limit_secs = int(getattr(cfg.agent, "time_limit", 0) or 0)
    run_deadline: Optional[float] = time.time() + time_limit_secs if time_limit_secs > 0 else None
    timed_out = False

    logger.info(f"ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"Initial draft count: {initial_draft_count} (executed sequentially for diversity)")
    if run_deadline is not None:
        logger.info(f"Hard timeout enabled: {time_limit_secs}s")

    def is_timed_out() -> bool:
        return run_deadline is not None and time.time() >= run_deadline

    lock = threading.Lock()
    logger.info(f"Resume progress: completed={completed}/{total_steps} from journal nodes={len(journal)}")

    pending_draft_nodes = []
    if initial_draft_count > 0 and completed == 0 and total_steps > 0:
        logger.info(f"Phase 1: Sequential draft generation (code only, {initial_draft_count} drafts)")

        def step_task_generate_only():
            logger.info("[step_task_generate_only] Generating draft from virtual root")
            return agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)

        for draft_idx in range(min(initial_draft_count, total_steps)):
            if is_timed_out():
                timed_out = True
                logger.warning("Time limit reached during Phase 1 draft generation; stop creating new drafts.")
                break
            try:
                logger.info(
                    f"Generating draft {draft_idx + 1}/{min(initial_draft_count, total_steps)} (code only)"
                )
                cur_node = step_task_generate_only()
                pending_draft_nodes.append(cur_node)
                logger.info(f"Draft {draft_idx + 1} code generated: node.id={cur_node.id}")
            except Exception as e:
                logger.exception(f"Exception during draft {draft_idx + 1} generation: {e}")

        logger.info(f"Phase 1 complete: {len(pending_draft_nodes)} draft codes generated")

    if pending_draft_nodes or completed < total_steps:
        logger.info("Phase 2: Pipelined parallel execution")
        logger.info(f"  - Pending draft executions: {len(pending_draft_nodes)}")
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
            submitted_drafts = 0
            for i, node in enumerate(pending_draft_nodes):
                if is_timed_out():
                    timed_out = True
                    logger.warning("Time limit reached before submitting pending draft executions.")
                    break
                futures.add(executor.submit(execute_draft_node, node))
                submitted_drafts += 1
                logger.info(f"Submitted draft execution: {node.id}")
                if i < len(pending_draft_nodes) - 1:
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
                    save_run(cfg, journal)
            elif completed < total_steps and not futures:
                logger.warning(
                    f"Phase 2 exited with no active futures before reaching target steps: {completed}/{total_steps}"
                )
        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
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

    interpreter.cleanup_session(-1)


if __name__ == "__main__":
    run()
