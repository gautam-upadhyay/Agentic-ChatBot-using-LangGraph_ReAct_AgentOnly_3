import csv
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from typing import List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
try:
    from langchain.agents import create_react_agent  # type: ignore[import-not-found]
except ModuleNotFoundError:
    import warnings

    try:
        from langgraph.utils import LangGraphDeprecatedSinceV10

        warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)
    except Exception:
        warnings.filterwarnings(
            "ignore",
            message="create_react_agent has been moved to `langchain.agents`",
        )

    from langgraph.prebuilt import create_react_agent

CSV_PATH = os.path.join(os.path.dirname(__file__), "imbd_movie.csv")
TABLE_NAME = "movies"
MODEL_NAME = "qwen2:7b"

PERSONA = (
    "You are MovieAnalyst, a helpful data analyst for the IMDB movies dataset. "
    "You only answer questions that can be answered by the imbd_movie.csv dataset. "
    "If a question is not about the movie dataset, respond with a brief refusal "
    "and ask the user to rephrase using the dataset."
)

AGENT_SYSTEM = (
    PERSONA
    + "\n\nYou are a ReAct agent. Use tools to answer questions about imbd_movie.csv.\n"
    + "Database schema:\n{schema}\n\n"
    + "Rules:\n"
    + "- Always call get_schema before your first SQL query if unsure.\n"
    + "- Use run_sql_query to answer ALL database questions.\n"
    + "- Only SELECT queries are allowed.\n"
    + "- Never ask the user to confirm schema or table names.\n"
    + "- If the question is not about the dataset, politely refuse and ask for a dataset question.\n"
    + "- If run_sql_query returns NO_ROWS, say no matching data was found and ask a clarifying question.\n"
    + "- If run_sql_query returns an ERROR, ask the user to rephrase.\n"
    + "- Respond with a concise natural-language answer, not SQL or tool output.\n"
)


class GraphState(TypedDict):
    messages: List[BaseMessage]


_DB_CONN: sqlite3.Connection | None = None
_DB_ERROR: str | None = None
_DB_LOCK = threading.Lock()
_LAST_SQL_STATUS: str | None = None
_LAST_SQL_ERROR: str | None = None


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = re.sub(r"[^\d]", "", value)
    return int(cleaned) if cleaned else None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _load_movies_csv(csv_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE movies (
            Poster_Link TEXT,
            Series_Title TEXT,
            Released_Year INTEGER,
            Certificate TEXT,
            Runtime INTEGER,
            Genre TEXT,
            IMDB_Rating REAL,
            Overview TEXT,
            Meta_score INTEGER,
            Director TEXT,
            Star1 TEXT,
            Star2 TEXT,
            Star3 TEXT,
            Star4 TEXT,
            No_of_Votes INTEGER,
            Gross INTEGER
        )
        """
    )
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            rows.append(
                (
                    row.get("Poster_Link"),
                    row.get("Series_Title"),
                    _to_int(row.get("Released_Year")),
                    row.get("Certificate"),
                    _to_int(row.get("Runtime")),
                    row.get("Genre"),
                    _to_float(row.get("IMDB_Rating")),
                    row.get("Overview"),
                    _to_int(row.get("Meta_score")),
                    row.get("Director"),
                    row.get("Star1"),
                    row.get("Star2"),
                    row.get("Star3"),
                    row.get("Star4"),
                    _to_int(row.get("No_of_Votes")),
                    _to_int(row.get("Gross")),
                )
            )
    cur.executemany(
        """
        INSERT INTO movies (
            Poster_Link, Series_Title, Released_Year, Certificate, Runtime, Genre,
            IMDB_Rating, Overview, Meta_score, Director, Star1, Star2, Star3, Star4,
            No_of_Votes, Gross
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return conn


def _init_db() -> str | None:
    global _DB_CONN, _DB_ERROR
    if _DB_CONN or _DB_ERROR:
        return _DB_ERROR
    if not os.path.exists(CSV_PATH):
        _DB_ERROR = "ERROR: imbd_movie.csv not found."
        return _DB_ERROR
    try:
        _DB_CONN = _load_movies_csv(CSV_PATH)
    except Exception as exc:
        _DB_ERROR = f"ERROR: Failed to load imbd_movie.csv ({exc})"
    return _DB_ERROR


def _get_schema_description() -> str:
    db_error = _init_db()
    if db_error:
        return db_error
    assert _DB_CONN is not None
    cur = _DB_CONN.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    tables = [row[0] for row in cur.fetchall()]
    lines = []
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [f"{col[1]} ({col[2]})" for col in cur.fetchall()]
        lines.append(f"- {table}: " + ", ".join(cols))
    return "\n".join(lines)


def _is_select_only(query: str) -> bool:
    q = query.strip().strip(";")
    if not q.lower().startswith("select"):
        return False
    banned = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "pragma",
        "attach",
        "detach",
        "replace",
    ]
    for kw in banned:
        if re.search(rf"\\b{kw}\\b", q, flags=re.IGNORECASE):
            return False
    return True


@tool("get_schema")
def get_schema() -> str:
    """Return the available tables and columns in the movies database."""
    return _get_schema_description()


@tool("run_sql_query")
def run_sql_query(query: str) -> str:
    """Execute a read-only SQL query against the IMDB movies dataset."""
    global _LAST_SQL_STATUS, _LAST_SQL_ERROR
    if not _is_select_only(query):
        _LAST_SQL_STATUS = "ERROR"
        _LAST_SQL_ERROR = "Only SELECT queries are allowed."
        return "ERROR: Only SELECT queries are allowed."
    db_error = _init_db()
    if db_error:
        _LAST_SQL_STATUS = "ERROR"
        _LAST_SQL_ERROR = db_error
        return db_error
    assert _DB_CONN is not None
    with _DB_LOCK:
        cur = _DB_CONN.cursor()
        try:
            cur.execute(query)
        except sqlite3.Error as exc:
            _LAST_SQL_STATUS = "ERROR"
            _LAST_SQL_ERROR = str(exc)
            return f"ERROR: {exc}"
        rows = cur.fetchmany(21)
    if not rows:
        _LAST_SQL_STATUS = "NO_ROWS"
        _LAST_SQL_ERROR = None
        return "NO_ROWS"
    _LAST_SQL_STATUS = "OK"
    _LAST_SQL_ERROR = None
    columns = rows[0].keys()
    display_rows = rows[:20]
    header = " | ".join(columns)
    separator = "-+-".join(["-" * len(col) for col in columns])
    body = []
    for row in display_rows:
        body.append(" | ".join(str(row[col]) for col in columns))
    suffix = ""
    if len(rows) > 20:
        suffix = "\n... showing first 20 rows"
    return "\n".join([header, separator] + body) + suffix


def _ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL") or os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"
    )


def _ollama_server_check() -> str | None:
    base_url = _ollama_base_url().rstrip("/")
    url = f"{base_url}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return (
            "ERROR: Ollama server is not reachable.\n"
            "Start it with: ollama serve\n"
            f"Then ensure the model is pulled: ollama pull {MODEL_NAME}\n"
            f"If Ollama is remote, set OLLAMA_HOST or OLLAMA_BASE_URL to {base_url}"
        )

    models = {m.get("name") for m in payload.get("models", []) if m.get("name")}
    if MODEL_NAME not in models:
        return (
            f"ERROR: Ollama model '{MODEL_NAME}' not found.\n"
            f"Run: ollama pull {MODEL_NAME}"
        )
    return None


def _build_model() -> ChatOllama:
    return ChatOllama(model=MODEL_NAME, base_url=_ollama_base_url())


def build_agent():
    model = _build_model()
    tools = [get_schema, run_sql_query]
    schema = _get_schema_description()
    return create_react_agent(model, tools, prompt=AGENT_SYSTEM.format(schema=schema))


def _is_refusal(text: str) -> bool:
    lowered = text.lower()
    return "please ask a question" in lowered or "dataset" in lowered and "please" in lowered


def _accuracy_score(sql_status: str | None, response: str) -> float:
    if sql_status == "OK":
        return 1.0
    if sql_status == "NO_ROWS":
        return 0.5
    if sql_status == "ERROR":
        return 0.0
    if _is_refusal(response):
        return 0.0
    return 0.5


def main() -> None:
    db_error = _init_db()
    if db_error:
        print(db_error)
        return
    ollama_error = _ollama_server_check()
    if ollama_error:
        print(ollama_error)
        return
    app = build_agent()
    state: GraphState = {"messages": []}
    print("IMDB movie chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        global _LAST_SQL_STATUS, _LAST_SQL_ERROR
        _LAST_SQL_STATUS = None
        _LAST_SQL_ERROR = None
        start_time = time.perf_counter()
        state["messages"].append(HumanMessage(content=user_input))
        state = app.invoke(state)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response_text = state["messages"][-1].content
        print(response_text)
        success = _LAST_SQL_STATUS != "ERROR"
        accuracy = _accuracy_score(_LAST_SQL_STATUS, response_text)
        print(
            f"[metrics] latency_ms={latency_ms:.0f} "
            f"success={success} accuracy={accuracy:.2f}"
        )


if __name__ == "__main__":
    main()
