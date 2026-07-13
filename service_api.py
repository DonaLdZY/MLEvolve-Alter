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
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.resource_limits import (
    accelerator_visibility_env,
    apply_process_tree_cpu_affinity,
    choose_cpu_ids,
    cpu_enforcement_capabilities,
    cpu_limit_environment,
    create_process_tree_memory_limiter,
    detect_resource_inventory,
    format_bytes,
    memory_enforcement_capabilities,
    process_tree_memory_bytes,
    relieve_process_tree_memory_pressure,
    terminate_process_tree,
    validate_accelerator_selection,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKDIR = str(ROOT_DIR)


def now_ts() -> float:
    return time.time()


def _is_interrupted_exit_code(exit_code: int | None) -> bool:
    # Windows CTRL_C_EVENT/CTRL_BREAK_EVENT is reported as 0xC000013A.
    return exit_code in {3221225786, -1073741510, 130, -2, -15}


class TaskResourceLimits(BaseModel):
    cpu_cores: int = Field(default=4, ge=1, le=4096)
    memory_limit_gb: float = Field(default=8.0, ge=0, le=1048576)
    accelerator_mode: Literal["all", "selected", "none"] = "all"
    accelerator_device_ids: list[str] = Field(default_factory=list)
    monitor_interval_seconds: float = Field(default=0.5, ge=0.1, le=10.0)


class StartMLEvolveRequest(BaseModel):
    task_id: str
    python_executable: str = "python"
    working_dir: str = DEFAULT_WORKDIR
    env_overrides: dict[str, str] = Field(default_factory=dict)
    config_path: str = ""
    args: list[str] = Field(default_factory=list)
    log_dir: str
    workspace_dir: str
    resume: bool = False
    graceful_shutdown_buffer_secs: int | None = Field(default=None, ge=0, le=3600)
    resources: TaskResourceLimits | None = None


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
    resource_limits: dict[str, Any] = Field(default_factory=dict)
    assigned_cpu_ids: list[int] = Field(default_factory=list)
    cpu_enforcement: dict[str, Any] = Field(default_factory=dict)
    current_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    memory_enforcement: dict[str, Any] = Field(default_factory=dict)
    resource_violation: str | None = None
    resource_warning: str | None = None


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
    job_status_tail_chars: int = 60000
    stop_wait_seconds: float = 20.0
    resource_limits: dict[str, Any] = field(default_factory=dict)
    assigned_cpu_ids: list[int] = field(default_factory=list)
    cpu_enforcement: dict[str, Any] = field(default_factory=dict)
    current_memory_bytes: int = 0
    peak_memory_bytes: int = 0
    memory_enforcement: dict[str, Any] = field(default_factory=dict)
    resource_violation: str | None = None
    resource_warning: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRuntime] = {}

    def create(
        self,
        task_id: str,
        log_dir: str,
        workspace_dir: str,
        *,
        resource_limits: dict[str, Any] | None = None,
        assigned_cpu_ids: list[int] | None = None,
    ) -> JobRuntime:
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
                resource_limits=dict(resource_limits or {}),
                assigned_cpu_ids=list(assigned_cpu_ids or []),
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
            stdout_tail=_tail_text(runtime.stdout_tail, runtime.job_status_tail_chars),
            stderr_tail=_tail_text(runtime.stderr_tail, runtime.job_status_tail_chars),
            resource_limits=dict(runtime.resource_limits),
            assigned_cpu_ids=list(runtime.assigned_cpu_ids),
            cpu_enforcement=dict(runtime.cpu_enforcement),
            current_memory_bytes=int(runtime.current_memory_bytes),
            peak_memory_bytes=int(runtime.peak_memory_bytes),
            memory_enforcement=dict(runtime.memory_enforcement),
            resource_violation=runtime.resource_violation,
            resource_warning=runtime.resource_warning,
        )


store = JobStore()
app = FastAPI(title="MLEvolve Service API", version="0.1.0")


def _tail_text(text: str, limit: int = 200000) -> str:
    if limit <= 0:
        return ""
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


def _with_cli_override(args: list[str], key: str, value: Any) -> list[str]:
    if _extract_cli_override(args, key) is not None:
        return args
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    return [*args, f"{key}={rendered}"]


def _saved_config_value(log_dir: Path, dotted_key: str, default: Any) -> Any:
    path = log_dir / "config.yaml"
    if not path.exists():
        return default
    try:
        current: Any = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    except Exception:
        return default


def _base_config_value(dotted_key: str, default: Any) -> Any:
    path = ROOT_DIR / "config" / "config.yaml"
    if not path.exists():
        return default
    try:
        current: Any = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    except Exception:
        return default


def _resolve_resource_limits(req: StartMLEvolveRequest) -> TaskResourceLimits:
    if req.resources is not None:
        return req.resources

    payload: dict[str, Any] = {}
    config_path = Path(req.config_path).expanduser() if req.config_path.strip() else None
    if config_path is not None:
        if not config_path.is_absolute():
            config_path = Path(req.working_dir.strip() or DEFAULT_WORKDIR) / config_path
        try:
            config_data = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
            resources = config_data.get("resources") if isinstance(config_data, dict) else {}
            if isinstance(resources, dict):
                payload.update(resources)
        except Exception:
            pass

    defaults = TaskResourceLimits()
    for field_name in TaskResourceLimits.model_fields:
        raw = _extract_cli_override(req.args, f"resources.{field_name}")
        if raw is not None:
            try:
                payload[field_name] = yaml.safe_load(raw)
            except Exception:
                payload[field_name] = raw
        elif field_name not in payload:
            payload[field_name] = _base_config_value(
                f"resources.{field_name}",
                getattr(defaults, field_name),
            )
    return TaskResourceLimits.model_validate(payload)


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


def _safe_read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _is_search_budget_exhausted(log_dir: Path) -> bool:
    status_name = str(_saved_config_value(log_dir, "runtime.run_status_filename", "run_status.json"))
    status = _safe_read_json_dict(log_dir / status_name)
    reason = str(status.get("termination_reason") or "").strip().lower()
    if reason in {"time_limit_exhausted", "step_limit_exhausted", "steps_completed"}:
        return True
    brief_log_name = str(_saved_config_value(log_dir, "logging.brief_log_filename", "MLEvolve.log"))
    log_tail = _safe_read_text(log_dir / brief_log_name, limit=120000)
    return (
        "Search budget is exhausted" in log_tail
        or "MLEvolve search budget exhausted" in log_tail
        or "Time limit reached (configured=" in log_tail
    )


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
    filename = str(_saved_config_value(log_dir, "runtime.pending_nodes_filename", "pending_nodes.json"))
    payload = _safe_read_json(log_dir / filename, {})
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
    journal_max_bytes = max(
        1,
        int(_saved_config_value(log_dir, "runtime.snapshot_journal_max_bytes", 150 * 1024 * 1024)),
    )
    snapshot_event_limit = max(1, int(_saved_config_value(log_dir, "runtime.snapshot_event_limit", 400)))
    snapshot_text_limit = max(
        1000,
        int(_saved_config_value(log_dir, "runtime.snapshot_text_tail_chars", 200000)),
    )
    brief_log_name = str(_saved_config_value(log_dir, "logging.brief_log_filename", "MLEvolve.log"))
    verbose_log_name = str(
        _saved_config_value(log_dir, "logging.verbose_log_filename", "MLEvolve.verbose.log")
    )
    try:
        if journal_path.exists() and journal_path.stat().st_size <= journal_max_bytes:
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
    best_solution_code = _safe_read_text(
        (workspace_dir / "best_solution" / "solution.py") if workspace_dir else Path(""),
        limit=snapshot_text_limit,
    )
    best_metric_text = _safe_read_text((workspace_dir / "best_solution" / "metric.txt") if workspace_dir else Path(""), limit=20000)

    return {
        "engine": "mlevolve",
        "log_dir": str(log_dir),
        "workspace_dir": str(workspace_dir) if workspace_dir else "",
        "events": _parse_log_events(log_dir / brief_log_name, limit=snapshot_event_limit),
        "nodes": node_rows,
        "pending_nodes": pending_nodes,
        "best_node_id": best_node_id,
        "journal_source": journal_source,
        "best_solution_code": best_solution_code,
        "best_metric_text": best_metric_text,
        "ml_log": _safe_read_text(log_dir / brief_log_name, limit=snapshot_text_limit),
        "verbose_log": _safe_read_text(log_dir / verbose_log_name, limit=snapshot_text_limit),
        "frontend_stdout": _safe_read_text(log_dir / "_frontend_stdout.log", limit=snapshot_text_limit),
        "frontend_stderr": _safe_read_text(log_dir / "_frontend_stderr.log", limit=snapshot_text_limit),
        "service_stdout": _safe_read_text(log_dir / "_service_stdout.log", limit=snapshot_text_limit),
        "service_stderr": _safe_read_text(log_dir / "_service_stderr.log", limit=snapshot_text_limit),
        "resource_usage": _safe_read_json(log_dir / "resource_usage.json", {}),
    }


def _monitor_task_resources(
    job_id: str,
    proc: subprocess.Popen[str],
    limits: TaskResourceLimits,
    cpu_ids: list[int],
    stop_event: threading.Event,
    hard_memory_limit_active: bool = False,
) -> None:
    memory_limit_bytes = int(float(limits.memory_limit_gb) * (1024**3))
    interval = max(0.1, float(limits.monitor_interval_seconds))
    peak_memory = 0
    while not stop_event.is_set() and proc.poll() is None:
        apply_process_tree_cpu_affinity(proc.pid, cpu_ids)
        current_memory = process_tree_memory_bytes(proc.pid)
        peak_memory = max(peak_memory, current_memory)
        store.update(
            job_id,
            current_memory_bytes=current_memory,
            peak_memory_bytes=peak_memory,
        )
        if memory_limit_bytes > 0 and current_memory > memory_limit_bytes and not hard_memory_limit_active:
            action = relieve_process_tree_memory_pressure(proc.pid, memory_limit_bytes)
            if action.action == "terminated_child":
                warning = (
                    "memory_limit_child_guard: stopped memory-heavy execution child "
                    f"pid={action.child_pid} after task memory reached {format_bytes(action.observed_bytes)}; "
                    f"configured limit={format_bytes(action.limit_bytes)}. MLEvolve controller continues."
                )
            elif action.action == "controller_over_limit":
                warning = (
                    "memory_limit_child_guard: controller memory exceeded the configured limit, "
                    "but the controller was preserved because whole-task termination is disabled. "
                    f"observed={format_bytes(action.observed_bytes)}, limit={format_bytes(action.limit_bytes)}"
                )
            else:
                warning = None
            if warning:
                store.update(job_id, resource_warning=warning)
        stop_event.wait(interval)


def _run_job(job_id: str, req: StartMLEvolveRequest, actual_log_dir: Path, actual_workspace_dir: Path, run_timestamp: str) -> None:
    limits = req.resources or _resolve_resource_limits(req)
    run_args = list(req.args)
    run_args = _with_cli_override(run_args, "runtime.run_timestamp", run_timestamp)
    run_args = _with_cli_override(run_args, "runtime.resume_run", req.resume)

    def runtime_value(name: str, default: Any) -> Any:
        raw = _extract_cli_override(run_args, f"runtime.{name}")
        if raw is not None:
            return raw
        return _base_config_value(f"runtime.{name}", default)

    job_status_tail_chars = max(0, int(float(runtime_value("job_status_tail_chars", 60000))))
    service_log_tail_chars = max(0, int(float(runtime_value("service_log_tail_chars", 200000))))
    service_last_error_chars = max(1, int(float(runtime_value("service_last_error_chars", 300))))
    termination_wait_default = max(0.0, float(runtime_value("termination_wait_seconds", 20)))
    store.update(
        job_id,
        job_status_tail_chars=job_status_tail_chars,
        stop_wait_seconds=termination_wait_default,
    )
    cmd = [req.python_executable or "python", "run.py", *run_args]
    workdir = req.working_dir.strip() or DEFAULT_WORKDIR
    job_runtime = store.get(job_id)
    env = os.environ.copy()
    env.update(req.env_overrides or {})
    env.update(accelerator_visibility_env(limits.accelerator_mode, limits.accelerator_device_ids))
    parallel_workers_raw = _extract_cli_override(run_args, "agent.search.parallel_search_num")
    try:
        parallel_workers = max(1, int(float(parallel_workers_raw or 1)))
    except Exception:
        parallel_workers = 1
    cpu_enforcement = cpu_enforcement_capabilities()
    env.update(
        cpu_limit_environment(
            limits.cpu_cores,
            parallel_workers,
            capabilities=cpu_enforcement,
        )
    )
    env["MLEVOLVE_ASSIGNED_CPU_IDS"] = json.dumps(job_runtime.assigned_cpu_ids)
    env["MLEVOLVE_MEMORY_LIMIT_GB"] = str(limits.memory_limit_gb)
    config_path: Path | None = None
    if req.config_path.strip():
        config_path = Path(req.config_path).expanduser()
        if not config_path.is_absolute():
            config_path = Path(workdir) / config_path
    if config_path is not None:
        env["MLEVOLVE_CONFIG_PATH"] = str(config_path.resolve())
    else:
        env.pop("MLEVOLVE_CONFIG_PATH", None)
    env["MLEVOLVE_RUN_TIMESTAMP"] = run_timestamp
    env["MLEVOLVE_RESUME_RUN"] = "1" if req.resume else "0"

    memory_limit_bytes = int(float(limits.memory_limit_gb) * (1024**3))
    memory_limiter = None
    memory_setup_warning: str | None = None
    if memory_limit_bytes > 0:
        try:
            memory_limiter = create_process_tree_memory_limiter(memory_limit_bytes)
        except Exception as exc:
            fallback_name = "POSIX RLIMIT_AS plus child guard" if os.name == "posix" else "child-process guard"
            memory_setup_warning = f"hard memory limiter setup failed; using {fallback_name}: {exc}"
    if memory_limiter is not None:
        memory_enforcement = memory_limiter.describe()
    elif memory_limit_bytes > 0:
        memory_enforcement = {
            **memory_enforcement_capabilities(),
            "backend": "posix_rlimit_as_plus_child_guard" if os.name == "posix" else "process_tree_child_guard",
            "hard_limit": False,
            "hard_limit_supported": False,
            "over_limit_behavior": (
                "per_process_allocation_failure_then_child_guard"
                if os.name == "posix"
                else "terminate_memory_heavy_child_process"
            ),
            "per_process_address_space_limit": os.name == "posix",
            "limit_bytes": memory_limit_bytes,
        }
    else:
        memory_enforcement = {
            "backend": "disabled",
            "hard_limit": False,
            "total_process_tree": True,
            "over_limit_behavior": "unlimited",
            "whole_task_termination": False,
            "limit_bytes": 0,
        }
    store.update(
        job_id,
        cpu_enforcement=cpu_enforcement,
        memory_enforcement=memory_enforcement,
        resource_warning=memory_setup_warning,
    )
    env["MLEVOLVE_MEMORY_LIMIT_BYTES"] = str(memory_limit_bytes)
    env["MLEVOLVE_MEMORY_ENFORCEMENT_MODE"] = str(memory_enforcement.get("backend") or "disabled")

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
        if memory_limiter is not None:
            memory_limiter.close()
        store.update(job_id, status="failed", last_error=f"start failed: {exc}")
        return

    store.set_process(job_id, proc)
    if memory_limiter is not None:
        try:
            memory_limiter.attach(proc.pid)
        except Exception as exc:
            memory_limiter.close()
            memory_limiter = None
            memory_enforcement = {
                **memory_enforcement_capabilities(),
                "backend": "process_tree_child_guard",
                "hard_limit": False,
                "hard_limit_supported": False,
                "limit_bytes": memory_limit_bytes,
                "setup_error": str(exc),
            }
            store.update(
                job_id,
                memory_enforcement=memory_enforcement,
                resource_warning=f"hard memory limiter attach failed; using child-process guard: {exc}",
            )
    cpu_errors = apply_process_tree_cpu_affinity(proc.pid, job_runtime.assigned_cpu_ids)
    if cpu_errors:
        message = "resource_limit_setup_failed: " + "; ".join(cpu_errors[:5])
        terminate_process_tree(proc.pid)
        out, err = proc.communicate()
        store.update(
            job_id,
            status="failed",
            exit_code=proc.returncode,
            last_error=message,
            resource_violation=message,
            stdout_tail=out or "",
            stderr_tail=err or "",
        )
        if memory_limiter is not None:
            memory_limiter.close()
        return

    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_task_resources,
        args=(job_id, proc, limits, job_runtime.assigned_cpu_ids, monitor_stop, memory_limiter is not None),
        daemon=True,
    )
    monitor_thread.start()
    time_limit_secs = _extract_time_limit_secs(run_args)
    timed_out = False
    try:
        if time_limit_secs is not None:
            configured_buffer = _extract_cli_override(run_args, "runtime.graceful_shutdown_buffer_seconds")
            try:
                shutdown_buffer = (
                    int(float(configured_buffer))
                    if configured_buffer is not None
                    else int(
                        req.graceful_shutdown_buffer_secs
                        if req.graceful_shutdown_buffer_secs is not None
                        else runtime_value("graceful_shutdown_buffer_seconds", 600)
                    )
                )
            except Exception:
                shutdown_buffer = int(runtime_value("graceful_shutdown_buffer_seconds", 600))
            total_timeout = time_limit_secs + max(0, shutdown_buffer)
            try:
                out, err = proc.communicate(timeout=total_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    if os.name == "nt":
                        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[arg-type]
                    else:
                        os.killpg(proc.pid, signal.SIGTERM)
                    configured_wait = _extract_cli_override(run_args, "runtime.termination_wait_seconds")
                    try:
                        termination_wait = (
                            max(0.0, float(configured_wait))
                            if configured_wait is not None
                            else termination_wait_default
                        )
                    except Exception:
                        termination_wait = termination_wait_default
                    out, err = proc.communicate(timeout=termination_wait)
                except Exception:
                    terminate_process_tree(proc.pid)
                    out, err = proc.communicate()
        else:
            out, err = proc.communicate()
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=max(1.0, float(limits.monitor_interval_seconds) * 2.0))
        if memory_limiter is not None:
            limiter_peak = memory_limiter.peak_memory_bytes()
            final_state = store.get(job_id)
            if limiter_peak > final_state.peak_memory_bytes:
                store.update(job_id, peak_memory_bytes=limiter_peak)
            memory_limiter.close()

    exit_code = proc.returncode
    current_job = store.get(job_id)
    stop_requested = bool(current_job.stop_requested or current_job.status in {"stopping", "stopped"})
    resource_violation = str(current_job.resource_violation or "").strip()
    budget_exhausted = _is_search_budget_exhausted(actual_log_dir)
    if resource_violation:
        status = "failed"
    elif timed_out and budget_exhausted and not stop_requested:
        status = "completed"
    elif timed_out:
        status = "failed"
    elif stop_requested or _is_interrupted_exit_code(exit_code):
        status = "stopped"
    elif exit_code == 0:
        status = "completed"
    else:
        status = "failed"
    last_error = None
    if resource_violation:
        last_error = resource_violation
    elif timed_out and budget_exhausted and not stop_requested:
        last_error = (
            "MLEvolve search budget was exhausted and the service finalized the run "
            "with current best artifacts. "
            f"search_limit={time_limit_secs}s, grace={int(shutdown_buffer)}s."
        )
    elif timed_out:
        last_error = (
            "MLEvolve exceeded service timeout and was terminated by service. "
            f"search_limit={time_limit_secs}s, grace={int(shutdown_buffer)}s."
        )
    elif status == "stopped":
        if stop_requested:
            last_error = "MLEvolve stopped by user."
        else:
            last_error = f"MLEvolve interrupted by console/control signal (exit code {exit_code})."
    elif exit_code != 0:
        tail = (err or out or "").strip()
        last_error = tail.splitlines()[-1][:service_last_error_chars] if tail else f"MLEvolve exited with code {exit_code}"

    try:
        actual_log_dir.mkdir(parents=True, exist_ok=True)
        if out:
            (actual_log_dir / "_service_stdout.log").write_text(
                _tail_text(out, service_log_tail_chars),
                encoding="utf-8",
                errors="ignore",
            )
        if err:
            (actual_log_dir / "_service_stderr.log").write_text(
                _tail_text(err, service_log_tail_chars),
                encoding="utf-8",
                errors="ignore",
            )
        final_resource_state = store.get(job_id)
        (actual_log_dir / "resource_usage.json").write_text(
            json.dumps(
                {
                    "limits": final_resource_state.resource_limits,
                    "assigned_cpu_ids": final_resource_state.assigned_cpu_ids,
                    "cpu_enforcement": final_resource_state.cpu_enforcement,
                    "current_memory_bytes": final_resource_state.current_memory_bytes,
                    "peak_memory_bytes": final_resource_state.peak_memory_bytes,
                    "memory_enforcement": final_resource_state.memory_enforcement,
                    "resource_violation": final_resource_state.resource_violation,
                    "resource_warning": final_resource_state.resource_warning,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    store.update(
        job_id,
        status=status,
        exit_code=exit_code,
        last_error=last_error,
        stdout_tail=_tail_text(out or "", service_log_tail_chars),
        stderr_tail=_tail_text(err or "", service_log_tail_chars),
        log_dir=str(actual_log_dir),
        workspace_dir=str(actual_workspace_dir),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/resources/inventory")
def resource_inventory() -> dict[str, Any]:
    return detect_resource_inventory()


@app.post("/jobs/start")
def start_job(req: StartMLEvolveRequest) -> dict[str, Any]:
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    actual_log_dir, actual_workspace_dir, final_run_name = _resolve_run_layout(req, run_timestamp)
    actual_log_dir.parent.mkdir(parents=True, exist_ok=True)
    actual_workspace_dir.parent.mkdir(parents=True, exist_ok=True)

    resources = _resolve_resource_limits(req)
    req.resources = resources
    try:
        cpu_ids = choose_cpu_ids(resources.cpu_cores)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if resources.accelerator_mode == "selected":
        inventory = detect_resource_inventory()
        selection_errors = validate_accelerator_selection(
            resources.accelerator_mode,
            resources.accelerator_device_ids,
            inventory,
        )
        if selection_errors:
            raise HTTPException(status_code=400, detail=selection_errors)

    job = store.create(
        task_id=req.task_id,
        log_dir=str(actual_log_dir),
        workspace_dir=str(actual_workspace_dir),
        resource_limits=resources.model_dump(),
        assigned_cpu_ids=cpu_ids,
    )
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
        "resources": resources.model_dump(),
        "assigned_cpu_ids": cpu_ids,
        "cpu_enforcement": cpu_enforcement_capabilities(),
        "memory_enforcement": memory_enforcement_capabilities(),
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
                proc.wait(timeout=max(0.0, float(job.stop_wait_seconds)))
            except subprocess.TimeoutExpired:
                terminate_process_tree(proc.pid)
        else:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=max(0.0, float(job.stop_wait_seconds)))
            except subprocess.TimeoutExpired:
                terminate_process_tree(proc.pid)
    except Exception:
        terminate_process_tree(proc.pid)
    store.update(req.job_id, status="stopped", last_error="stopped by user")
    return {"status": "stopping", "job_id": req.job_id}


@app.post("/snapshot")
def snapshot(req: SnapshotRequest) -> dict[str, Any]:
    try:
        return _build_snapshot(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"snapshot failed: {exc}")
