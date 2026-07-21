from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .models import DEFAULT_SETTINGS


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data if data is not None else {}, sort_keys=True, separators=(",", ":"))


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {key: row[key] for key in row.keys()}


_DEDUP_SCOPE_KEYS = ("instrumentType", "region", "universe", "delay", "neutralization", "decay", "truncation")


def _canonical_expression(expression: str) -> str:
    return "".join(str(expression or "").split())


def _canonical_scope(settings: Dict[str, Any]) -> tuple[tuple[str, str], ...]:
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings or {})
    return tuple((key, _canonical_scope_value(merged.get(key))) for key in _DEDUP_SCOPE_KEYS)


def _canonical_scope_value(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            number = float(text)
        except ValueError:
            return text.upper()
        return f"{number:.12g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.12g}"
    return str(value if value is not None else "").strip().upper()


class AlphaStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expression TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    alpha_id TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    checks_json TEXT NOT NULL DEFAULT '{}',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id INTEGER,
                    event_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(candidate_id) REFERENCES candidates(id)
                );

                CREATE TABLE IF NOT EXISTS run_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS archived_candidates (
                    original_candidate_id INTEGER PRIMARY KEY,
                    expression TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL,
                    alpha_id TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    checks_json TEXT NOT NULL DEFAULT '{}',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived_at TEXT NOT NULL,
                    prune_reason TEXT NOT NULL,
                    archive_metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS archived_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_event_id INTEGER NOT NULL UNIQUE,
                    candidate_id INTEGER,
                    event_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    archived_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_status_created_id
                ON candidates(status, created_at, id);

                CREATE INDEX IF NOT EXISTS idx_candidates_created_id
                ON candidates(created_at, id);

                CREATE INDEX IF NOT EXISTS idx_events_candidate_id_id
                ON events(candidate_id, id);

                CREATE INDEX IF NOT EXISTS idx_events_candidate_type_id
                ON events(candidate_id, event_type, id);

                CREATE INDEX IF NOT EXISTS idx_archived_candidates_scope
                ON archived_candidates(original_candidate_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_archived_events_candidate_id
                ON archived_events(candidate_id, original_event_id);
                """
            )

    def insert_candidate(self, expression: str, settings: Dict[str, Any], source: str) -> int:
        now = utc_now()
        with self.connection() as conn:
            cur = conn.execute(
                """
                INSERT INTO candidates (
                    expression, settings_json, source, status, metrics_json,
                    checks_json, retry_count, created_at, updated_at
                )
                VALUES (?, ?, ?, 'generated', '{}', '{}', 0, ?, ?)
                """,
                (expression, _json(settings), source, now, now),
            )
            candidate_id = int(cur.lastrowid)
        return candidate_id

    def find_duplicate_candidate(self, expression: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        target_expression = _canonical_expression(expression)
        target_scope = _canonical_scope(settings)
        for candidate in reversed(self.list_candidates()):
            if _canonical_expression(str(candidate.get("expression") or "")) != target_expression:
                continue
            try:
                candidate_settings = json.loads(str(candidate.get("settings_json") or "{}"))
            except json.JSONDecodeError:
                candidate_settings = {}
            if _canonical_scope(candidate_settings if isinstance(candidate_settings, dict) else {}) == target_scope:
                return candidate
        archived = self._find_archived_duplicate_candidate(target_expression, target_scope)
        if archived is not None:
            return archived
        return None

    def _find_archived_duplicate_candidate(
        self,
        target_expression: str,
        target_scope: tuple[tuple[str, str], ...],
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM archived_candidates ORDER BY original_candidate_id DESC").fetchall()
        for row in rows:
            candidate = _row_to_dict(row)
            if _canonical_expression(str(candidate.get("expression") or "")) != target_expression:
                continue
            try:
                candidate_settings = json.loads(str(candidate.get("settings_json") or "{}"))
            except json.JSONDecodeError:
                candidate_settings = {}
            if _canonical_scope(candidate_settings if isinstance(candidate_settings, dict) else {}) != target_scope:
                continue
            candidate["id"] = int(candidate["original_candidate_id"])
            candidate["archived"] = True
            return candidate
        return None

    def get_candidate(self, candidate_id: int) -> Dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(f"candidate not found: {candidate_id}")
        return _row_to_dict(row)

    def list_candidates(self, status: Optional[str] = None, created_since: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            where = []
            params: List[Any] = []
            if status is not None:
                where.append("status = ?")
                params.append(status)
            if created_since:
                where.append("created_at >= ?")
                params.append(created_since)
            clause = f" WHERE {' AND '.join(where)}" if where else ""
            rows = conn.execute(f"SELECT * FROM candidates{clause} ORDER BY id", params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_recent_candidates(self, limit: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
        limit = max(0, int(limit))
        if limit <= 0:
            return []
        with self.connection() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM candidates ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM candidates WHERE status = ? ORDER BY id DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_recent_archived_candidates(
        self,
        limit: int,
        status: Optional[str] = None,
        created_since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(0, int(limit))
        if limit <= 0:
            return []
        where = []
        params: List[Any] = []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if created_since:
            where.append("created_at >= ?")
            params.append(created_since)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM archived_candidates
                {clause}
                ORDER BY original_candidate_id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            candidate = _row_to_dict(row)
            candidate["id"] = int(candidate["original_candidate_id"])
            candidate["archived"] = True
            candidates.append(candidate)
        return candidates

    def update_candidate(self, candidate_id: int, **fields: Any) -> None:
        if not fields:
            return
        allowed = {"status", "alpha_id", "metrics_json", "checks_json", "retry_count"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown candidate fields: {sorted(unknown)}")
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [candidate_id]
        with self.connection() as conn:
            conn.execute(f"UPDATE candidates SET {assignments} WHERE id = ?", values)

    def transition(self, candidate_id: int, status: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # Atomic: status update and the status event are written in one transaction so a
        # crash can never leave a changed status without its event (or vice versa).
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                "UPDATE candidates SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, candidate_id),
            )
            conn.execute(
                """
                INSERT INTO events (candidate_id, event_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (candidate_id, f"status:{status}", _json(metadata or {}), now),
            )

    def fail_preflight_passed_candidates(
        self,
        *,
        created_since: Optional[str] = None,
        reason: str = "interrupted_after_preflight",
    ) -> int:
        where = ["status = 'preflight_passed'"]
        params: List[Any] = []
        if created_since:
            where.append("created_at >= ?")
            params.append(created_since)
        clause = " AND ".join(where)
        now = utc_now()
        metadata = {"reason": reason, "errors": ["INTERRUPTED_AFTER_PREFLIGHT"]}
        with self.connection() as conn:
            rows = conn.execute(f"SELECT id FROM candidates WHERE {clause} ORDER BY id", params).fetchall()
            for row in rows:
                candidate_id = int(row["id"])
                conn.execute(
                    "UPDATE candidates SET status = 'failed', updated_at = ? WHERE id = ?",
                    (now, candidate_id),
                )
                conn.execute(
                    """
                    INSERT INTO events (candidate_id, event_type, metadata_json, created_at)
                    VALUES (?, 'status:failed', ?, ?)
                    """,
                    (candidate_id, _json(metadata), now),
                )
        return len(rows)

    def count_preflight_passed_candidates(self, *, created_since: Optional[str] = None) -> int:
        where = ["status = 'preflight_passed'"]
        params: List[Any] = []
        if created_since:
            where.append("created_at >= ?")
            params.append(created_since)
        clause = " AND ".join(where)
        with self.connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM candidates WHERE {clause}", params).fetchone()
        return int(row["count"] or 0)

    def increment_retry(self, candidate_id: int) -> int:
        candidate = self.get_candidate(candidate_id)
        retry_count = int(candidate["retry_count"]) + 1
        self.update_candidate(candidate_id, retry_count=retry_count)
        return retry_count

    def record_event(self, candidate_id: Optional[int], event_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO events (candidate_id, event_type, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (candidate_id, event_type, _json(metadata), utc_now()),
            )

    def events_for_candidate(self, candidate_id: Optional[int]) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            if candidate_id is None:
                rows = conn.execute("SELECT * FROM events WHERE candidate_id IS NULL ORDER BY id").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events WHERE candidate_id = ? ORDER BY id",
                    (candidate_id,),
                ).fetchall()
                if not rows:
                    rows = conn.execute(
                        """
                        SELECT *
                        FROM archived_events
                        WHERE candidate_id = ?
                        ORDER BY original_event_id
                        """,
                        (candidate_id,),
                    ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def archive_candidates(
        self,
        candidate_ids: Iterable[int],
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        ids = [int(candidate_id) for candidate_id in candidate_ids]
        if not ids:
            return 0
        archived_at = utc_now()
        metadata_json = _json(metadata or {})
        archived_count = 0
        with self.connection() as conn:
            for candidate_id in ids:
                row = conn.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
                if row is None:
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO archived_candidates (
                        original_candidate_id, expression, settings_json, source, status, alpha_id,
                        metrics_json, checks_json, retry_count, created_at, updated_at,
                        archived_at, prune_reason, archive_metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(row["id"]),
                        row["expression"],
                        row["settings_json"],
                        row["source"],
                        row["status"],
                        row["alpha_id"],
                        row["metrics_json"],
                        row["checks_json"],
                        row["retry_count"],
                        row["created_at"],
                        row["updated_at"],
                        archived_at,
                        reason,
                        metadata_json,
                    ),
                )
                event_rows = conn.execute("SELECT * FROM events WHERE candidate_id = ? ORDER BY id", (candidate_id,)).fetchall()
                for event in event_rows:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO archived_events (
                            original_event_id, candidate_id, event_type, metadata_json, created_at, archived_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(event["id"]),
                            event["candidate_id"],
                            event["event_type"],
                            event["metadata_json"],
                            event["created_at"],
                            archived_at,
                        ),
                    )
                conn.execute("DELETE FROM events WHERE candidate_id = ?", (candidate_id,))
                conn.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
                archived_count += 1
            if archived_count:
                conn.execute(
                    """
                    INSERT INTO events (candidate_id, event_type, metadata_json, created_at)
                    VALUES (NULL, 'history_pruned', ?, ?)
                    """,
                    (
                        _json(
                            {
                                "reason": reason,
                                "archived_count": archived_count,
                                "candidate_ids": ids[:100],
                                "metadata": metadata or {},
                            }
                        ),
                        archived_at,
                    ),
                )
        return archived_count

    def status_counts(self, created_since: Optional[str] = None) -> Dict[str, int]:
        with self.connection() as conn:
            if created_since:
                rows = conn.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM candidates
                    WHERE created_at >= ?
                    GROUP BY status
                    ORDER BY status
                    """,
                    (created_since,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT status, COUNT(*) AS count FROM candidates GROUP BY status ORDER BY status").fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def set_run_state(self, key: str, value: Dict[str, Any]) -> None:
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO run_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, _json(value), now),
            )

    def get_run_state(self, key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute("SELECT value FROM run_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return dict(default or {})
        try:
            value = json.loads(str(row["value"]))
        except json.JSONDecodeError:
            return dict(default or {})
        return value if isinstance(value, dict) else dict(default or {})
