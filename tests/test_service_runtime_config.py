from __future__ import annotations

from service_api import JobStore, StartMLEvolveRequest, _base_config_value, _tail_text


def test_service_reads_runtime_defaults_from_default_yaml() -> None:
    assert _base_config_value("runtime.job_status_tail_chars", None) == 60000
    assert _base_config_value("runtime.service_log_tail_chars", None) == 200000
    assert _base_config_value("runtime.termination_wait_seconds", None) == 20


def test_job_status_tail_and_request_default_are_config_driven() -> None:
    store = JobStore()
    job = store.create("task", "logs", "workspace")
    store.update(job.job_id, stdout_tail="abcdefgh", job_status_tail_chars=4)

    assert store.status(job.job_id).stdout_tail == "efgh"
    assert _tail_text("abcdefgh", 0) == ""
    request = StartMLEvolveRequest(
        task_id="task",
        log_dir="logs",
        workspace_dir="workspace",
        config_path="other/task-config.yaml",
    )
    assert request.graceful_shutdown_buffer_secs is None
    assert request.config_path == "other/task-config.yaml"
    assert request.resources is None
