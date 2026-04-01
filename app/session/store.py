import sqlite3
import json
from pathlib import Path
from loguru import logger

DB_PATH = Path("./pharma_iq.db")


class SessionStore:
    """
    SQLite-backed conversation history.

    Decision: SQLite over in-memory dict.
    Reason: in-memory state dies on every restart — unusable in production.
    SQLite is built into Python, zero extra dependency, persists to disk.
    Upgrade path: swap sqlite3 for asyncpg + PostgreSQL when horizontal
    scaling is needed. Interface stays identical — only the driver changes.
    """

    def __init__(self):
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)"
            )
            conn.commit()
        logger.info(f"Session store initialised at {DB_PATH}")

    def get_history(self, session_id: str, max_turns: int = 10) -> list[dict]:
        """
        Retrieve last N turns for a session.

        Why limit to 10 turns?
        Mistral 7B context window is ~8k tokens. A full conversation
        history could overflow it. 10 turns is a safe default that
        preserves enough context without hitting limits.
        At scale: summarise older turns rather than truncating.
        """
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("""
                SELECT role, content FROM conversations
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, max_turns * 2)).fetchall()

        # Reverse to chronological order
        history = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
        return history

    def add_turn(self, session_id: str, user_message: str, assistant_message: str):
        """Persist one conversation turn — both user and assistant messages."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany(
                "INSERT INTO conversations (session_id, role, content) VALUES (?, ?, ?)",
                [
                    (session_id, "user", user_message),
                    (session_id, "assistant", assistant_message),
                ]
            )
            conn.commit()

    def get_sessions(self) -> list[str]:
        """List all unique session IDs."""
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM conversations ORDER BY session_id"
            ).fetchall()
        return [row[0] for row in rows]