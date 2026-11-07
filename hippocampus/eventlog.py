"""Append-only JSONL event log: the source of truth. Lines are never rewritten or deleted."""
from __future__ import annotations

import json
from pathlib import Path


class EventLog:
    """One JSON object per line. `path=None` keeps the log in memory only.

    A crash can leave a half-written last line. On load, any line that does not
    parse is skipped (and counted in `corrupt_lines`). Before the next append, a
    newline is written if the file did not end with one, so new events never
    glue onto the broken fragment.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.events: list[dict] = []
        self.corrupt_lines = 0
        self._needs_newline = False
        self._next_seq = 0
        if self.path is not None and self.path.exists():
            self.reload()

    def reload(self) -> list[dict]:
        """Re-read the file from disk (in-memory logs just keep their events)."""
        if self.path is None:
            return self.events
        events, corrupt = [], 0
        data = self.path.read_bytes() if self.path.exists() else b""
        self._needs_newline = bool(data) and not data.endswith(b"\n")
        for raw in data.split(b"\n"):
            if not raw.strip():
                continue
            try:
                ev = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                corrupt += 1
                continue
            if not isinstance(ev, dict) or "type" not in ev:
                corrupt += 1
                continue
            events.append(ev)
        self.events, self.corrupt_lines = events, corrupt
        self._next_seq = max((e.get("seq", -1) for e in events), default=-1) + 1
        return events

    def append(self, ev: dict) -> dict:
        ev = dict(ev)
        ev["seq"] = self._next_seq
        self._next_seq += 1
        if self.path is not None:
            line = json.dumps(ev, sort_keys=True, ensure_ascii=False) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "ab") as f:
                if self._needs_newline:
                    f.write(b"\n")
                    self._needs_newline = False
                f.write(line.encode("utf-8"))
        self.events.append(ev)
        return ev

    def __len__(self) -> int:
        return len(self.events)
