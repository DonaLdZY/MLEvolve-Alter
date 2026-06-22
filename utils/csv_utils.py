from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CsvDialectHint:
    sep: str
    engine: str | None = None
    inferred: bool = False
    reason: str = ""


def _decode_sample(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_sample_lines(path: Path, *, max_bytes: int = 65536, max_lines: int = 60) -> list[str]:
    with path.open("rb") as f:
        text = _decode_sample(f.read(max_bytes))
    return [line.rstrip("\r\n") for line in text.splitlines() if line.strip()][:max_lines]


def _looks_like_whitespace_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    header_parts = lines[0].strip().split()
    if len(header_parts) < 2:
        return False
    if any("," in part for part in header_parts):
        return False

    checked = 0
    stable = 0
    for line in lines[1:16]:
        parts = line.strip().split()
        if not parts:
            continue
        checked += 1
        if len(parts) == len(header_parts):
            stable += 1
    if checked == 0:
        return False
    comma_counts = [line.count(",") for line in lines[1 : min(len(lines), 16)]]
    comma_unstable = bool(comma_counts) and (max(comma_counts) - min(comma_counts) > 2 or max(comma_counts) >= len(header_parts) * 2)
    return stable / checked >= 0.8 and (comma_unstable or lines[0].count(",") == 0)


def infer_csv_dialect(path: Path) -> CsvDialectHint:
    try:
        lines = _read_sample_lines(path)
    except Exception:
        return CsvDialectHint(sep=",", inferred=False, reason="sample_unavailable")
    if _looks_like_whitespace_table(lines):
        return CsvDialectHint(sep=r"\s+", engine="python", inferred=True, reason="whitespace_columns_with_comma_lists")
    return CsvDialectHint(sep=",", inferred=False, reason="default_comma")


def read_csv_auto(path: Path | str, *args: Any, **kwargs: Any) -> pd.DataFrame:
    path_obj = Path(path)
    if "sep" not in kwargs and "delimiter" not in kwargs:
        hint = infer_csv_dialect(path_obj)
        kwargs["sep"] = hint.sep
        if hint.engine:
            kwargs.setdefault("engine", hint.engine)
    try:
        return pd.read_csv(path_obj, *args, encoding=kwargs.pop("encoding", "utf-8-sig"), **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path_obj, *args, encoding="gb18030", **kwargs)

