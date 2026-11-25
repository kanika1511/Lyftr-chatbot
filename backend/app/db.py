from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """SQLite persistence layer for sessions and multi-turn message history."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._memory_conn: Optional[sqlite3.Connection] = None

        if self.db_path != ":memory:":
            parent = Path(self.db_path).parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
        else:
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            with self._lock:
                yield self._memory_conn
            return

        conn = sqlite3.connect(self.db_path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
            yield conn
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create required tables and indexes if they do not already exist."""
        with self._connection() as conn:
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id    TEXT PRIMARY KEY,
                  system_prompt TEXT NOT NULL,
                  created_at    TEXT NOT NULL,
                  updated_at    TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                  id          INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id  TEXT NOT NULL,
                  role        TEXT NOT NULL,
                  content     TEXT NOT NULL,
                  metadata    TEXT,
                  created_at  TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);
                """
            )

    def ping(self) -> bool:
        """Return True if SQLite is reachable and responds to a simple query."""
        with self._connection() as conn:
            row = conn.execute("SELECT 1").fetchone()
            return row is not None and row[0] == 1

    def get_session_context(
        self, session_id: str
    ) -> tuple[Optional[str], list[dict[str, str]]]:
        """Load the stored system prompt and ordered message history for a session."""
        with self._connection() as conn:
            session_row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            stored_prompt = session_row["system_prompt"] if session_row else None

            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

            history = [
                {"role": row["role"], "content": row["content"]}
                for row in rows
            ]
            return stored_prompt, history

    def save_turn(
        self,
        session_id: str,
        system_prompt: str,
        user_message: str,
        assistant_reply: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist session prompt and both user/assistant messages atomically."""
        now = _utc_now_iso()
        metadata_json = json.dumps(metadata)

        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                      system_prompt = excluded.system_prompt,
                      updated_at = excluded.updated_at
                    """,
                    (session_id, system_prompt, now, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'user', ?, NULL, ?)
                    """,
                    (session_id, user_message, now),
                )
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, metadata, created_at)
                    VALUES (?, 'assistant', ?, ?, ?)
                    """,
                    (session_id, assistant_reply, metadata_json, now),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def get_session_messages(self, session_id: str) -> Optional[list[dict[str, Any]]]:
        """Return oldest-first message history for a session, or None if unknown."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

            if not rows:
                return None

            messages: list[dict[str, Any]] = []
            for row in rows:
                item: dict[str, Any] = {
                    "role": row["role"],
                    "content": row["content"],
                }
                if row["role"] == "assistant" and row["metadata"]:
                    item["metadata"] = json.loads(row["metadata"])
                messages.append(item)
            return messages
