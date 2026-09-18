from __future__ import annotations

import json

from counseling_voice_demo.jsonl_recovery import recover_jsonl
from counseling_voice_demo.log_writer import create_session_log_dirs


def test_recover_jsonl_keeps_readable_lines_and_quarantines_broken_tail(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_recovery_test")
    broken_jsonl = paths.internal_dir / "turn_events.jsonl"
    broken_jsonl.write_text(
        "\n".join(
            [
                json.dumps({"turn_id": 1, "event_type": "text_ready"}, ensure_ascii=False),
                "{broken json",
                json.dumps({"turn_id": 2, "event_type": "audio_ready"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = recover_jsonl(broken_jsonl, errors_jsonl=paths.errors_jsonl)

    assert result.recovered_records == [{"turn_id": 1, "event_type": "text_ready"}]
    assert result.broken_line_number == 2
    assert result.quarantine_path is not None
    assert "{broken json" in result.quarantine_path.read_text(encoding="utf-8")
    assert "audio_ready" in result.quarantine_path.read_text(encoding="utf-8")
    assert _read_jsonl(paths.errors_jsonl)[0]["event_type"] == "jsonl_recovery"


def test_recover_jsonl_reports_clean_file_without_quarantine(tmp_path) -> None:
    paths = create_session_log_dirs(tmp_path, "session_clean_recovery_test")
    clean_jsonl = paths.internal_dir / "turn_events.jsonl"
    clean_jsonl.write_text(
        json.dumps({"turn_id": 1, "event_type": "text_ready"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    result = recover_jsonl(clean_jsonl, errors_jsonl=paths.errors_jsonl)

    assert result.recovered_records == [{"turn_id": 1, "event_type": "text_ready"}]
    assert result.broken_line_number is None
    assert result.quarantine_path is None
    assert paths.errors_jsonl.read_text(encoding="utf-8") == ""


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
