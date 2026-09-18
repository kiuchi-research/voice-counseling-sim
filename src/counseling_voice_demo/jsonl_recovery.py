from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonlRecoveryResult:
    recovered_records: list[dict[str, Any]]
    broken_line_number: int | None
    quarantine_path: Path | None


def recover_jsonl(path: Path | str, *, errors_jsonl: Path | str | None = None) -> JsonlRecoveryResult:
    jsonl_path = Path(path)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines(keepends=True)
    recovered_records: list[dict[str, Any]] = []
    broken_line_index: int | None = None

    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            broken_line_index = index
            break
        if isinstance(record, dict):
            recovered_records.append(record)
        else:
            broken_line_index = index
            break

    if broken_line_index is None:
        return JsonlRecoveryResult(
            recovered_records=recovered_records,
            broken_line_number=None,
            quarantine_path=None,
        )

    quarantine_path = jsonl_path.with_suffix(jsonl_path.suffix + ".broken")
    quarantine_path.write_text("".join(lines[broken_line_index:]), encoding="utf-8")
    jsonl_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in recovered_records),
        encoding="utf-8",
    )

    if errors_jsonl is not None:
        _append_recovery_event(
            Path(errors_jsonl),
            source_path=jsonl_path,
            broken_line_number=broken_line_index + 1,
            quarantine_path=quarantine_path,
        )

    return JsonlRecoveryResult(
        recovered_records=recovered_records,
        broken_line_number=broken_line_index + 1,
        quarantine_path=quarantine_path,
    )


def _append_recovery_event(
    errors_jsonl: Path,
    *,
    source_path: Path,
    broken_line_number: int,
    quarantine_path: Path,
) -> None:
    record = {
        "event_type": "jsonl_recovery",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "broken_line_number": broken_line_number,
        "quarantine_path": str(quarantine_path),
    }
    with errors_jsonl.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
