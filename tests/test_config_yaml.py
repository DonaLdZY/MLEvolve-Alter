from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from config import Config, _load_cfg, _redacted_cfg, prep_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_commented_default_yaml_matches_runtime_schema() -> None:
    runtime = OmegaConf.load(REPO_ROOT / "config" / "config.yaml")
    merged = OmegaConf.merge(OmegaConf.structured(Config), runtime)

    assert merged.data_dir is None
    assert merged.exp_name is None
    assert merged.agent.steps == 50
    assert merged.agent.time_limit == 10800
    assert merged.agent.search.parallel_search_num == 4
    assert merged.agent.search.num_drafts == 8
    assert merged.agent.search.num_improves == 5
    assert merged.agent.draft.fast_first_draft is True
    assert merged.agent.code.request_timeout_seconds == 1200.0
    assert merged.agent.code.continuation_max_rounds == 2
    assert merged.agent.retries.result_parse_max_attempts == 3
    assert merged.resources.cpu_cores == 4
    assert merged.resources.memory_limit_gb == 8.0
    assert merged.resources.accelerator_mode == "all"
    assert merged.runtime.job_status_tail_chars == 60000
    assert merged.runtime.service_last_error_chars == 300


def test_environment_api_key_fallback(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    description = tmp_path / "description.md"
    description.write_text("demo", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-test-key")

    raw = OmegaConf.load(REPO_ROOT / "config" / "config.yaml")
    raw.data_dir = str(data_dir)
    raw.desc_file = str(description)
    raw.log_dir = str(tmp_path / "logs")
    raw.workspace_dir = str(tmp_path / "workspaces")
    raw.runtime.run_timestamp = "20260710_000000"
    raw.exp_name = "config-test"
    cfg = prep_cfg(raw)

    assert cfg.agent.code.api_key == "env-test-key"
    assert cfg.agent.feedback.api_key == "env-test-key"
    assert cfg.log_dir.name == "20260710_000000_config-test"
    redacted = _redacted_cfg(cfg)
    assert redacted.agent.code.api_key == ""
    assert redacted.agent.feedback.api_key == ""


def test_task_config_path_and_key_override_environment(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    description = tmp_path / "description.md"
    description.write_text("demo", encoding="utf-8")
    task_cfg = OmegaConf.load(REPO_ROOT / "config" / "config.yaml")
    task_cfg.data_dir = str(data_dir)
    task_cfg.desc_file = str(description)
    task_cfg.log_dir = str(tmp_path / "logs")
    task_cfg.workspace_dir = str(tmp_path / "workspaces")
    task_cfg.exp_name = "priority-test"
    task_cfg.agent.code.api_key = "config-key"
    config_path = tmp_path / "task.yaml"
    OmegaConf.save(task_cfg, config_path)

    monkeypatch.setenv("MLEVOLVE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    loaded = prep_cfg(_load_cfg(use_cli_args=False))

    assert loaded.agent.code.api_key == "config-key"
