from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKDIR = str(ROOT_DIR)


def now_ts() -> float:
    return time.time()


def _is_interrupted_exit_code(exit_code: int | None) -> bool:
    # Windows CTRL_C_EVENT/CTRL_BREAK_EVENT is reported as 0xC000013A.
    return exit_code in {3221225786, -1073741510, 130, -2, -15}


class StartMLEvolveRequest(BaseModel):
    task_id: str
    python_executable: str = "python"
    working_dir: str = DEFAULT_WORKDIR
    env_overrides: dict[str, str] = Field(default_factory=dict)
    args: list[str] = Field(default_factory=list)
    log_dir: str
    workspace_dir: str
    resume: bool = False
    graceful_shutdown_buffer_secs: int = Field(default=600, ge=0, le=3600)


class StopRequest(BaseModel):
    job_id: str


class SnapshotRequest(BaseModel):
    log_dir: str = ""
    workspace_dir: str = ""
    run_dir: str = ""
    task_name: str = ""


class JobStatus(BaseModel):
    job_id: str
    task_id: str
    status: str
    started_at: float
    updated_at: float
    log_dir: str
    workspace_dir: str
    exit_code: int | None = None
    last_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""


@dataclass
class JobRuntime:
    job_id: str
    task_id: str
    log_dir: str
    workspace_dir: str
    process: subprocess.Popen[str] | None
    status: str
    started_at: float
    updated_at: float
    exit_code: int | None = None
    last_error: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    stop_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRuntime] = {}

    def create(self, task_id: str, log_dir: str, workspace_dir: str) -> JobRuntime:
        with self._lock:
            for job in self._jobs.values():
                if job.task_id != task_id or job.status != "running":
                    continue
                proc = job.process
                if proc is not None and proc.poll() is not None:
                    job.status = "failed" if (proc.returncode or 0) != 0 else "completed"
                    job.exit_code = proc.returncode
                    job.updated_at = now_ts()
                    continue
                raise HTTPException(status_code=400, detail="task already running in MLEvolve service")
            job_id = uuid.uuid4().hex
            ts = now_ts()
            runtime = JobRuntime(
                job_id=job_id,
                task_id=task_id,
                log_dir=log_dir,
                workspace_dir=workspace_dir,
                process=None,
                status="pending",
                started_at=ts,
                updated_at=ts,
            )
            self._jobs[job_id] = runtime
            return runtime

    def _get_unlocked(self, job_id: str) -> JobRuntime:
        runtime = self._jobs.get(job_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="job not found")
        return runtime

    def get(self, job_id: str) -> JobRuntime:
        with self._lock:
            return self._get_unlocked(job_id)

    def set_process(self, job_id: str, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            runtime = self._get_unlocked(job_id)
            runtime.process = proc
            runtime.status = "running"
            runtime.updated_at = now_ts()

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            runtime = self._get_unlocked(job_id)
            for key, value in kwargs.items():
                setattr(runtime, key, value)
            runtime.updated_at = now_ts()

    def status(self, job_id: str) -> JobStatus:
        runtime = self.get(job_id)
        return JobStatus(
            job_id=runtime.job_id,
            task_id=runtime.task_id,
            status=runtime.status,
            started_at=runtime.started_at,
            updated_at=runtime.updated_at,
            log_dir=runtime.log_dir,
            workspace_dir=runtime.workspace_dir,
            exit_code=runtime.exit_code,
            last_error=runtime.last_error,
            stdout_tail=runtime.stdout_tail[-60000:],
            stderr_tail=runtime.stderr_tail[-60000:],
        )


store = JobStore()
app = FastAPI(title="MLEvolve Service API", version="0.1.0")


def _tail_text(text: str, limit: int = 200000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _unquote_cli_value(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        try:
            return json.loads(text)
        except Exception:
            return text[1:-1]
    return text


def _extract_cli_override(args: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for item in args:
        if isinstance(item, str) and item.startswith(prefix):
            return _unquote_cli_value(item[len(prefix) :])
    return None


def _extract_time_limit_secs(args: list[str]) -> int | None:
    for key in ("agent.time_limit", "exec.timeout"):
        raw = _extract_cli_override(args, key)
        if raw is None:
            continue
        try:
            return max(1, int(float(raw)))
        except Exception:
            return None
    return None


def _resolve_run_layout(req: StartMLEvolveRequest, run_timestamp: str) -> tuple[Path, Path, str]:
    exp_name = (_extract_cli_override(req.args, "exp_name") or "").strip()
    if not exp_name and not req.resume:
        raise HTTPException(status_code=400, detail="missing exp_name in MLEvolve args")

    log_root = Path(req.log_dir).expanduser().resolve()
    workspace_root = Path(req.workspace_dir).expanduser().resolve()
    if req.resume:
        final_name = exp_name or log_root.name or workspace_root.name
        return log_root, workspace_root, final_name

    final_name = f"{run_timestamp}_{exp_name}"
    if log_root == workspace_root:
        per_run_root = (log_root / final_name).resolve()
        return (per_run_root / "logs").resolve(), (per_run_root / "workspace").resolve(), final_name
    return (log_root / final_name).resolve(), (workspace_root / final_name).resolve(), final_name


def _safe_read_text(path: Path, limit: int = 60000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        byte_limit = max(limit, limit * 4)
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - byte_limit))
            data = f.read()
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return text[-limit:]


def _safe_tail_lines(path: Path, limit: int = 400, byte_limit: int = 512_000) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - byte_limit))
            data = f.read()
        return data.decode("utf-8", errors="ignore").splitlines()[-limit:]
    except Exception:
        return []


def _safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _parse_metric_obj(metric_obj: Any) -> tuple[float | None, bool | None]:
    if not isinstance(metric_obj, dict):
        return None, None
    value = metric_obj.get("value")
    maximize = metric_obj.get("maximize")
    try:
        value = None if value is None else float(value)
    except Exception:
        value = None
    if not isinstance(maximize, bool):
        maximize = None
    return value, maximize


def _read_pending_nodes(log_dir: Path) -> list[dict[str, Any]]:
    payload = _safe_read_json(log_dir / "pending_nodes.json", {})
    if not isinstance(payload, dict):
        return []
    rows = payload.get("nodes")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id") or "").strip()
        if not node_id:
            continue
        row = dict(row)
        row["id"] = node_id
        row["pending_execution"] = bool(
            row.get("pending_execution") or row.get("status") in {"generating", "pending_execution", "executing"}
        )
        out.append(row)
    return out


def _resolve_best_node_id(log_dir: Path, workspace_dir: Path, nodes: list[dict[str, Any]]) -> str | None:
    best_file = workspace_dir / "best_solution" / "node_id.txt"
    if best_file.exists():
        try:
            best_id = best_file.read_text(encoding="utf-8", errors="ignore").strip()
            if best_id:
                return best_id
        except Exception:
            pass

    best_metric = None
    best_id = None
    best_maximize: bool | None = None
    for node in nodes:
        metric = node.get("metric")
        maximize = node.get("maximize")
        if metric is None:
            continue
        if best_metric is None:
            best_metric = metric
            best_id = node.get("id")
            best_maximize = maximize
            continue
        compare_maximize = True if best_maximize is None else best_maximize
        if compare_maximize:
            if metric > best_metric:
                best_metric = metric
                best_id = node.get("id")
        else:
            if metric < best_metric:
                best_metric = metric
                best_id = node.get("id")
    return best_id


def _parse_log_events(log_path: Path, limit: int = 400) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    pattern = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<level>[A-Z]+):\s+(?P<msg>.*)$")
    rows: list[dict[str, Any]] = []
    for line in _safe_tail_lines(log_path, limit=limit, byte_limit=768_000):
        text = line.strip()
        if not text:
            continue
        match = pattern.match(text)
        if match:
            rows.append(
                {
                    "ts": match.group("ts"),
                    "component": "mlevolve.log",
                    "event": match.group("level"),
                    "message": match.group("msg"),
                }
            )
        else:
            rows.append({"ts": "", "component": "mlevolve.log", "event": "INFO", "message": text})
    return rows


def _pick_dirs(req: SnapshotRequest) -> tuple[Path | None, Path | None]:
    log_dir = None
    workspace_dir = None

    raw_log = req.log_dir.strip()
    if raw_log:
        candidate = Path(raw_log).expanduser().resolve()
        if candidate.exists():
            log_dir = candidate

    raw_ws = req.workspace_dir.strip()
    if raw_ws:
        candidate = Path(raw_ws).expanduser().resolve()
        if candidate.exists():
            workspace_dir = candidate

    return log_dir, workspace_dir


def _build_snapshot(req: SnapshotRequest) -> dict[str, Any]:
    log_dir, workspace_dir = _pick_dirs(req)
    if log_dir is None:
        return {"engine": "mlevolve", "nodes": [], "events": []}

    if workspace_dir is None and log_dir.parent.name == "logs":
        sibling = log_dir.parent.parent / "workspace"
        if sibling.exists():
            workspace_dir = sibling.resolve()

    # filtered_journal.json is a best-path projection. UI snapshots need the
    # full search tree when the journal is reasonably sized.
    journal_source = ""
    journal = {}
    journal_path = log_dir / "journal.json"
    filtered_journal_path = log_dir / "filtered_journal.json"
    try:
        if journal_path.exists() and journal_path.stat().st_size <= 150 * 1024 * 1024:
            journal = _safe_read_json(journal_path, {})
            if isinstance(journal, dict) and journal:
                journal_source = "journal"
    except Exception:
        journal = {}
    if not isinstance(journal, dict) or not journal:
        journal = _safe_read_json(filtered_journal_path, {})
        if isinstance(journal, dict) and journal:
            journal_source = "filtered_journal"
    node_rows: list[dict[str, Any]] = []
    node2parent = {}
    if isinstance(journal, dict):
        node2parent = journal.get("node2parent", {}) or {}
        for node in journal.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            metric, maximize = _parse_metric_obj(node.get("metric"))
            term_out = node.get("_term_out")
            result = ""
            if isinstance(term_out, list):
                result = "".join(str(part) for part in term_out)
            elif term_out is not None:
                result = str(term_out)
            node_id = str(node.get("id") or "")
            node_rows.append(
                {
                    "id": node_id,
                    "parent_id": node2parent.get(node_id),
                    "stage": node.get("stage"),
                    "plan": node.get("plan"),
                    "code": node.get("code"),
                    "result": result,
                    "insight": node.get("llm_insight") or node.get("analysis"),
                    "llm_insight": node.get("llm_insight"),
                    "parser_analysis": node.get("parser_analysis") or node.get("analysis"),
                    "decision_signals": node.get("decision_signals"),
                    "metric": metric,
                    "maximize": maximize,
                    "is_buggy": node.get("is_buggy"),
                    "is_valid": node.get("is_valid"),
                    "visits": node.get("visits"),
                    "total_reward": node.get("total_reward"),
                    "uct": node.get("_uct"),
                    "finish_time": node.get("finish_time"),
                    "exec_time": node.get("exec_time"),
                    "branch_id": node.get("branch_id"),
                    "from_topk": node.get("from_topk"),
                }
            )

    journal_node_ids = {str(node.get("id")) for node in node_rows if node.get("id")}
    pending_nodes = [
        node for node in _read_pending_nodes(log_dir)
        if str(node.get("id")) not in journal_node_ids
    ]
    best_node_id = _resolve_best_node_id(log_dir, workspace_dir or Path("."), node_rows)
    best_solution_code = _safe_read_text((workspace_dir / "best_solution" / "solution.py") if workspace_dir else Path(""), limit=200000)
    best_metric_text = _safe_read_text((workspace_dir / "best_solution" / "metric.txt") if workspace_dir else Path(""), limit=20000)

    return {
        "engine": "mlevolve",
        "log_dir": str(log_dir),
        "workspace_dir": str(workspace_dir) if workspace_dir else "",
        "events": _parse_log_events(log_dir / "MLEvolve.log"),
        "nodes": node_rows,
        "pending_nodes": pending_nodes,
        "best_node_id": best_node_id,
        "journal_source": journal_source,
        "best_solution_code": best_solution_code,
        "best_metric_text": best_metric_text,
        "ml_log": _safe_read_text(log_dir / "MLEvolve.log"),
        "verbose_log": _safe_read_text(log_dir / "MLEvolve.verbose.log"),
        "frontend_stdout": _safe_read_text(log_dir / "_frontend_stdout.log", limit=200000),
        "frontend_stderr": _safe_read_text(log_dir / "_frontend_stderr.log", limit=200000),
        "service_stdout": _safe_read_text(log_dir / "_service_stdout.log", limit=200000),
        "service_stderr": _safe_read_text(log_dir / "_service_stderr.log", limit=200000),
    }


def _run_job(job_id: str, req: StartMLEvolveRequest, actual_log_dir: Path, actual_workspace_dir: Path, run_timestamp: str) -> None:
    cmd = [req.python_executable or "python", "run.py", *req.args]
    env = os.environ.copy()
    env.update(req.env_overrides or {})
    env["MLEVOLVE_RUN_TIMESTAMP"] = run_timestamp
    env["MLEVOLVE_RESUME_RUN"] = "1" if req.resume else "0"
    workdir = req.working_dir.strip() or DEFAULT_WORKDIR

    try:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except Exception as exc:
        store.update(job_id, status="failed", last_error=f"start failed: {exc}")
        return

    store.set_process(job_id, proc)
    time_limit_secs = _extract_time_limit_secs(req.args)
    timed_out = False
    if time_limit_secs is not None:
        total_timeout = time_limit_secs + int(req.graceful_shutdown_buffer_secs)
        try:
            out, err = proc.communicate(timeout=total_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[arg-type]
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
                out, err = proc.communicate(timeout=20)
            except Exception:
                try:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                out, err = proc.communicate()
    else:
        out, err = proc.communicate()

    exit_code = proc.returncode
    current_job = store.get(job_id)
    stop_requested = bool(current_job.stop_requested or current_job.status in {"stopping", "stopped"})
    if timed_out:
        status = "failed"
    elif stop_requested or _is_interrupted_exit_code(exit_code):
        status = "stopped"
    elif exit_code == 0:
        status = "completed"
    else:
        status = "failed"
    last_error = None
    if timed_out:
        last_error = (
            "MLEvolve exceeded service timeout and was terminated by service. "
            f"search_limit={time_limit_secs}s, grace={int(req.graceful_shutdown_buffer_secs)}s."
        )
    elif status == "stopped":
        if stop_requested:
            last_error = "MLEvolve stopped by user."
        else:
            last_error = f"MLEvolve interrupted by console/control signal (exit code {exit_code})."
    elif exit_code != 0:
        tail = (err or out or "").strip()
        last_error = tail.splitlines()[-1][:300] if tail else f"MLEvolve exited with code {exit_code}"

    try:
        actual_log_dir.mkdir(parents=True, exist_ok=True)
        if out:
            (actual_log_dir / "_service_stdout.log").write_text(_tail_text(out), encoding="utf-8", errors="ignore")
        if err:
            (actual_log_dir / "_service_stderr.log").write_text(_tail_text(err), encoding="utf-8", errors="ignore")
    except Exception:
        pass

    store.update(
        job_id,
        status=status,
        exit_code=exit_code,
        last_error=last_error,
        stdout_tail=_tail_text(out or ""),
        stderr_tail=_tail_text(err or ""),
        log_dir=str(actual_log_dir),
        workspace_dir=str(actual_workspace_dir),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs/start")
def start_job(req: StartMLEvolveRequest) -> dict[str, Any]:
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    actual_log_dir, actual_workspace_dir, final_run_name = _resolve_run_layout(req, run_timestamp)
    actual_log_dir.parent.mkdir(parents=True, exist_ok=True)
    actual_workspace_dir.parent.mkdir(parents=True, exist_ok=True)

    job = store.create(task_id=req.task_id, log_dir=str(actual_log_dir), workspace_dir=str(actual_workspace_dir))
    thread = threading.Thread(
        target=_run_job,
        args=(job.job_id, req, actual_log_dir, actual_workspace_dir, run_timestamp),
        daemon=True,
    )
    thread.start()
    return {
        "job_id": job.job_id,
        "status": "started",
        "engine": "mlevolve",
        "run_name": final_run_name,
        "log_dir": str(actual_log_dir),
        "workspace_dir": str(actual_workspace_dir),
    }


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return store.status(job_id).model_dump()


@app.post("/jobs/stop")
def stop_job(req: StopRequest) -> dict[str, Any]:
    job = store.get(req.job_id)
    proc = job.process
    if proc is None or proc.poll() is not None:
        return {"status": "not_running", "job_id": req.job_id}
    store.update(req.job_id, status="stopping", stop_requested=True, last_error="stop requested by user")
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[arg-type]
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            proc.terminate()
    store.update(req.job_id, status="stopped", last_error="stopped by user")
    return {"status": "stopping", "job_id": req.job_id}


@app.post("/snapshot")
def snapshot(req: SnapshotRequest) -> dict[str, Any]:
    try:
        return _build_snapshot(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"snapshot failed: {exc}")
