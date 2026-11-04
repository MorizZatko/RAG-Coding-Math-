"""Persistent chat history storage based on SQLite.

This module provides the ChatStore class, which saves chats and their
messages (including image references and the retrieved context sources)
in a local SQLite database, so that conversations survive restarts of
the application and can be reopened and continued later.
"""

import json
import re
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime

DB_PATH = "chat_history.db"
TITLE_MAX_CHARS = 40


def _now():
    """Returns the current local time as ISO timestamp string (millisecond precision, keeps sorting stable)."""
    return datetime.now().isoformat(timespec="milliseconds")


def _make_title(text: str, max_chars: int = TITLE_MAX_CHARS):
    """Builds a short chat title from the first user question.

    Args:
        text (str): The first user message of a chat.
        max_chars (int): Maximum title length before it gets truncated.

    Returns:
        str: The shortened single-line title.
    """
    title = re.sub(r"\s+", " ", text or "").strip()
    if len(title) > max_chars:
        title = title[:max_chars].rstrip() + "…"
    return title


class ChatStore:
    """Stores chats and chat messages in a local SQLite database.

    A new connection is opened for every call, because Streamlit executes
    scripts in changing threads and SQLite connections must not be shared
    between threads by default.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        """Opens a new SQLite connection with dict-like rows and foreign keys enabled."""
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _init_db(self):
        """Creates the database tables if they do not exist yet."""
        with closing(self._connect()) as con, con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL DEFAULT '',
                    model       TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    images_json TEXT NOT NULL DEFAULT '[]',
                    context     TEXT NOT NULL DEFAULT '',
                    created_at  TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
                """
            )

    def create_chat(self, title: str = "", model: str = ""):
        """Creates a new, empty chat.

        Args:
            title (str): Optional initial title. If empty, it is derived from the first question.
            model (str): Optional name of the model used when the chat was started.

        Returns:
            str: The id of the new chat.
        """
        chat_id = uuid.uuid4().hex
        now = _now()
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT INTO chats (id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (chat_id, title, model or "", now, now),
            )
        return chat_id

    def list_chats(self):
        """Returns all chats, most recently used first.

        Returns:
            list: A list of dictionaries with id, title, model, created_at and updated_at.
        """
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT id, title, model, created_at, updated_at FROM chats "
                "ORDER BY updated_at DESC, rowid DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chat(self, chat_id: str):
        """Returns a single chat or None if it does not exist.

        Args:
            chat_id (str): The id of the chat.

        Returns:
            dict | None: The chat metadata dictionary.
        """
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT id, title, model, created_at, updated_at FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
        return dict(row) if row else None

    def load_messages(self, chat_id: str):
        """Loads all messages of a chat in chronological order.

        Args:
            chat_id (str): The id of the chat.

        Returns:
            list: Message dictionaries with role, content, images and context,
                in the same format that is used in st.session_state.messages.
        """
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT role, content, images_json, context FROM messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()

        messages = []
        for row in rows:
            try:
                images = json.loads(row["images_json"] or "[]")
            except ValueError:
                images = []
            messages.append({
                "role": row["role"],
                "content": row["content"],
                "images": images,
                "context": row["context"] or "",
            })
        return messages

    def _insert_message(self, con, chat_id, role, content, images, context):
        """Inserts a single message row using an already opened connection."""
        con.execute(
            "INSERT INTO messages (chat_id, role, content, images_json, context, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, json.dumps(images or [], ensure_ascii=False), context or "", _now()),
        )

    def _touch_chat(self, con, chat_id, first_user_text=None):
        """Updates the last-used timestamp and sets an automatic title if none exists yet."""
        con.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id))
        if first_user_text:
            con.execute(
                "UPDATE chats SET title = ? WHERE id = ? AND title = ''",
                (_make_title(first_user_text), chat_id),
            )

    def append_message(self, chat_id: str, role: str, content: str, images: list = None, context: str = ""):
        """Appends a single message to a chat.

        Args:
            chat_id (str): The id of the chat.
            role (str): 'user' or 'assistant'.
            content (str): The message text.
            images (list, optional): Image info dictionaries (path and caption).
            context (str): The raw context sources used for an assistant answer.
        """
        with closing(self._connect()) as con, con:
            self._insert_message(con, chat_id, role, content, images, context)
            self._touch_chat(con, chat_id, content if role == "user" else None)

    def append_exchange(self, chat_id: str, user_content: str, assistant_content: str, images: list = None, context: str = ""):
        """Appends a question and its answer in one transaction.

        Saving both together avoids orphaned user messages when answer
        generation fails, so the history never contains two user messages in a row.

        Args:
            chat_id (str): The id of the chat.
            user_content (str): The user's question.
            assistant_content (str): The generated answer.
            images (list, optional): Image info dictionaries shown with the answer.
            context (str): The raw context sources used for the answer.
        """
        with closing(self._connect()) as con, con:
            self._insert_message(con, chat_id, "user", user_content, None, "")
            self._insert_message(con, chat_id, "assistant", assistant_content, images, context)
            self._touch_chat(con, chat_id, user_content)

    def rename_chat(self, chat_id: str, title: str):
        """Renames a chat.

        Args:
            chat_id (str): The id of the chat.
            title (str): The new title (ignored if empty).
        """
        title = _make_title(title, max_chars=80)
        if not title:
            return
        with closing(self._connect()) as con, con:
            con.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))

    def delete_chat(self, chat_id: str):
        """Deletes a chat including all of its messages.

        Args:
            chat_id (str): The id of the chat.
        """
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def delete_all_chats(self):
        """Deletes all chats and messages."""
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM chats")
