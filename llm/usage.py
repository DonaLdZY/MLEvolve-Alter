from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("MLEvolve")

_LOCK = threading.Lock()
_SUMMARIES: dict[str, dict[str, Any]] = {}


def usage_paths(cfg: Any) -> tuple[Path | None, Path | None]:
    log_dir = getattr(cfg, "log_dir", None)
    if not log_dir:
        return None, None
    root = Path(log_dir)
    return root / "llm_usage.jsonl", root / "llm_usage_summary.json"


def infer_prompt_name(prefix: str) -> str:
    """Infer a useful prompt label without changing every agent call site."""
    try:
        for frame in inspect.stack()[2:12]:
            module = inspect.getmodule(frame.frame)
            mod_name = getattr(module, "__name__", "") if module else ""
            if mod_name.startswith(("llm", "openai", "gemini")):
                continue
            if mod_name:
                return f"{prefix}:{mod_name}.{frame.function}"
            return f"{prefix}:{frame.function}"
    except Exception:
        pass
    return prefix


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            data = usage.model_dump()
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    if isinstance(usage, dict):
        return dict(usage)
    out: dict[str, Any] = {}
    for key in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "cached_tokens",
    ]:
        value = getattr(usage, key, None)
        if value is not None:
            out[key] = value
    for key in ["prompt_tokens_details", "completion_tokens_details"]:
        details = getattr(usage, key, None)
        if details is None:
            continue
        if hasattr(details, "model_dump"):
            try:
                details = details.model_dump()
            except Exception:
                details = None
        if isinstance(details, dict):
            out[key] = details
    return out


def usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
    for key in keys:
        for pool in [usage, details, completion_details]:
            value = pool.get(key) if isinstance(pool, dict) else None
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
    return None


def usage_cache_tokens(usage: dict[str, Any]) -> tuple[int | None, int | None]:
    cached = usage_int(usage, "prompt_cache_hit_tokens", "cached_tokens", "cache_read_input_tokens")
    missed = usage_int(usage, "prompt_cache_miss_tokens", "cache_miss_input_tokens")
    prompt = usage_int(usage, "prompt_tokens")
    if missed is None and prompt is not None and cached is not None:
        missed = max(0, prompt - cached)
    return cached, missed


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = 0
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
    non_cjk = max(0, len(text) - cjk)
    return max(1, int(round(cjk + non_cjk / 4)))


def prompt_part_stats(
    prompt_parts: list[dict[str, Any]] | None,
    *,
    provider_prompt_tokens: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    if not prompt_parts:
        return rows, 0
    for idx, part in enumerate(prompt_parts):
        content = str(part.get("content", "") or "")
        if not content and part.get("estimated_tokens") is None:
            continue
        chars = len(content) if content else int(part.get("chars", 0) or 0)
        utf8_bytes = len(content.encode("utf-8")) if content else int(part.get("utf8_bytes", chars) or chars)
        estimated_tokens = (
            estimate_text_tokens(content)
            if content
            else int(part.get("estimated_tokens", max(1, chars // 4)) or max(1, chars // 4))
        )
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16] if content else str(part.get("sha256_16", "synthetic"))[:16]
        rows.append(
            {
                "index": idx,
                "name": str(part.get("name", f"part_{idx}") or f"part_{idx}"),
                "role": str(part.get("role", "") or ""),
                "chars": chars,
                "utf8_bytes": utf8_bytes,
                "estimated_tokens": estimated_tokens,
                "sha256_16": digest,
            }
        )
    total_estimated = sum(int(x.get("estimated_tokens", 0) or 0) for x in rows)
    for row in rows:
        est = int(row.get("estimated_tokens", 0) or 0)
        row["share_of_estimated_prompt"] = round(est / total_estimated, 6) if total_estimated else 0.0
        row["provider_prompt_tokens_estimate"] = (
            int(round(provider_prompt_tokens * est / total_estimated))
            if provider_prompt_tokens and total_estimated
            else 0
        )
    return rows, total_estimated


def prompt_parts_from_messages(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages or []):
        parts.append(
            {
                "name": f"{msg.get('role', 'message')}_{idx}",
                "role": str(msg.get("role", "")),
                "content": str(msg.get("content", "") or ""),
            }
        )
    return parts


def _new_summary() -> dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "provider_cache_known_prompt_tokens": 0,
        "provider_cache_unknown_prompt_tokens": 0,
        "provider_usage_missing_calls": 0,
        "estimated_prompt_tokens": 0,
        "estimated_completion_tokens": 0,
        "by_prompt_part": {},
        "by_prompt": {},
    }


def _summary_for(path: Path) -> dict[str, Any]:
    key = str(path.resolve())
    summary = _SUMMARIES.get(key)
    if summary is None:
        summary = _new_summary()
        _SUMMARIES[key] = summary
    return summary


def _accumulate_parts(target: dict[str, Any], prompt_name: str, part_rows: list[dict[str, Any]]) -> None:
    by_part = target.setdefault("by_prompt_part", {})
    for row in part_rows:
        key = f"{prompt_name}:{row.get('name', '')}"
        item = by_part.setdefault(
            key,
            {
                "prompt_name": prompt_name,
                "part_name": row.get("name", ""),
                "role": row.get("role", ""),
                "calls": 0,
                "chars": 0,
                "utf8_bytes": 0,
                "estimated_tokens": 0,
                "provider_prompt_tokens_estimate": 0,
            },
        )
        item["calls"] = int(item.get("calls", 0)) + 1
        for field in ["chars", "utf8_bytes", "estimated_tokens", "provider_prompt_tokens_estimate"]:
            item[field] = int(item.get(field, 0)) + int(row.get(field, 0) or 0)


def _accumulate_prompt_item_parts(prompt_item: dict[str, Any], part_rows: list[dict[str, Any]]) -> None:
    by_part = prompt_item.setdefault("by_part", {})
    for row in part_rows:
        key = str(row.get("name", ""))
        item = by_part.setdefault(
            key,
            {
                "role": row.get("role", ""),
                "calls": 0,
                "chars": 0,
                "utf8_bytes": 0,
                "estimated_tokens": 0,
                "provider_prompt_tokens_estimate": 0,
            },
        )
        item["calls"] = int(item.get("calls", 0)) + 1
        for field in ["chars", "utf8_bytes", "estimated_tokens", "provider_prompt_tokens_estimate"]:
            item[field] = int(item.get(field, 0)) + int(row.get(field, 0) or 0)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    out = dict(summary)
    prompt_tokens = int(out.get("prompt_tokens", 0) or 0)
    cached = int(out.get("prompt_cache_hit_tokens", 0) or 0)
    missed = int(out.get("prompt_cache_miss_tokens", 0) or 0)
    known_prompt_tokens = int(out.get("provider_cache_known_prompt_tokens", 0) or 0)
    estimated_prompt_tokens = int(out.get("estimated_prompt_tokens", 0) or 0)
    out["provider_cache_hit_ratio"] = round(cached / prompt_tokens, 6) if prompt_tokens else 0.0
    out["provider_cache_miss_ratio"] = round(missed / prompt_tokens, 6) if prompt_tokens else 0.0
    out["known_provider_cache_hit_ratio"] = round(cached / known_prompt_tokens, 6) if known_prompt_tokens else 0.0
    out["known_provider_cache_miss_ratio"] = round(missed / known_prompt_tokens, 6) if known_prompt_tokens else 0.0
    by_part = out.get("by_prompt_part", {})
    if isinstance(by_part, dict):
        ranked = sorted(by_part.values(), key=lambda x: int(x.get("estimated_tokens", 0) or 0), reverse=True)
        for row in ranked:
            est = int(row.get("estimated_tokens", 0) or 0)
            row["share_of_estimated_prompt"] = round(est / estimated_prompt_tokens, 6) if estimated_prompt_tokens else 0.0
        out["by_prompt_part_ranked"] = ranked
    by_prompt = out.get("by_prompt", {})
    if isinstance(by_prompt, dict):
        for item in by_prompt.values():
            prompt_est = int(item.get("estimated_prompt_tokens", 0) or 0)
            parts = item.get("by_part", {})
            if isinstance(parts, dict):
                ranked = sorted(parts.values(), key=lambda x: int(x.get("estimated_tokens", 0) or 0), reverse=True)
                for row in ranked:
                    est = int(row.get("estimated_tokens", 0) or 0)
                    row["share_of_estimated_prompt"] = round(est / prompt_est, 6) if prompt_est else 0.0
                item["by_part_ranked"] = ranked
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def log_llm_usage(
    *,
    cfg: Any,
    prompt_name: str,
    mode: str,
    provider: str,
    model: str,
    response: Any = None,
    usage: Any = None,
    seconds: float = 0.0,
    finish_reason: str = "",
    max_tokens: int | None = None,
    parsed_ok: bool | None = None,
    source: str = "provider",
    prompt_parts: list[dict[str, Any]] | None = None,
    estimated_completion_text: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage_path, summary_path = usage_paths(cfg)
    if usage_path is None or summary_path is None:
        return {}
    raw_usage = usage_to_dict(usage if usage is not None else getattr(response, "usage", None))
    usage_available = bool(raw_usage)
    prompt_tokens = usage_int(raw_usage, "prompt_tokens") or 0
    completion_tokens = usage_int(raw_usage, "completion_tokens") or 0
    total_tokens = usage_int(raw_usage, "total_tokens") or (prompt_tokens + completion_tokens)
    cached_tokens, miss_tokens = usage_cache_tokens(raw_usage)
    cache_known = cached_tokens is not None or miss_tokens is not None
    part_rows, estimated_prompt_tokens = prompt_part_stats(prompt_parts, provider_prompt_tokens=prompt_tokens)
    estimated_completion_tokens = estimate_text_tokens(estimated_completion_text or "")
    row = {
        "ts": time.time(),
        "prompt_name": prompt_name,
        "mode": mode,
        "provider": provider,
        "source": source,
        "model": model,
        "seconds": round(float(seconds or 0.0), 4),
        "finish_reason": finish_reason,
        "max_tokens": max_tokens,
        "parsed_ok": parsed_ok,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cache_hit_tokens": cached_tokens or 0,
        "prompt_cache_miss_tokens": miss_tokens or 0,
        "provider_cache_tokens_known": cache_known,
        "usage_available": usage_available,
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "estimated_completion_tokens": estimated_completion_tokens,
        "prompt_parts": part_rows,
        "raw_usage": raw_usage,
        "extra": extra or {},
    }
    with _LOCK:
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        with usage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        summary = _summary_for(summary_path)
        summary["calls"] = int(summary.get("calls", 0)) + 1
        if not usage_available:
            summary["provider_usage_missing_calls"] = int(summary.get("provider_usage_missing_calls", 0)) + 1
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"]:
            summary[key] = int(summary.get(key, 0)) + int(row.get(key, 0) or 0)
        summary["estimated_prompt_tokens"] = int(summary.get("estimated_prompt_tokens", 0)) + estimated_prompt_tokens
        summary["estimated_completion_tokens"] = int(summary.get("estimated_completion_tokens", 0)) + estimated_completion_tokens
        bucket = "provider_cache_known_prompt_tokens" if cache_known else "provider_cache_unknown_prompt_tokens"
        summary[bucket] = int(summary.get(bucket, 0)) + prompt_tokens
        _accumulate_parts(summary, prompt_name, part_rows)
        by_prompt = summary.setdefault("by_prompt", {})
        item = by_prompt.setdefault(
            prompt_name,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
                "provider_cache_known_prompt_tokens": 0,
                "provider_cache_unknown_prompt_tokens": 0,
                "provider_usage_missing_calls": 0,
                "estimated_prompt_tokens": 0,
                "estimated_completion_tokens": 0,
                "by_part": {},
            },
        )
        item["calls"] = int(item.get("calls", 0)) + 1
        if not usage_available:
            item["provider_usage_missing_calls"] = int(item.get("provider_usage_missing_calls", 0)) + 1
        for key in ["prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"]:
            item[key] = int(item.get(key, 0)) + int(row.get(key, 0) or 0)
        item[bucket] = int(item.get(bucket, 0)) + prompt_tokens
        item["estimated_prompt_tokens"] = int(item.get("estimated_prompt_tokens", 0)) + estimated_prompt_tokens
        item["estimated_completion_tokens"] = int(item.get("estimated_completion_tokens", 0)) + estimated_completion_tokens
        _accumulate_prompt_item_parts(item, part_rows)
        _write_summary(summary_path, summary)
    logger.info(
        "[llm_usage] prompt=%s mode=%s provider=%s input=%s cached=%s miss=%s output=%s total=%s est_input=%s usage_available=%s",
        prompt_name,
        mode,
        provider,
        prompt_tokens,
        cached_tokens or 0,
        miss_tokens or 0,
        completion_tokens,
        total_tokens,
        estimated_prompt_tokens,
        usage_available,
    )
    return row
