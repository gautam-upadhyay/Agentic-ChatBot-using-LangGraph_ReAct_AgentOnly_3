# LangGraph IMDB Movies Chatbot

Simple CLI chatbot using LangGraph + a ReAct agent that answers questions by generating and executing SQL against `imbd_movie.csv`.

## Requirements
- Python 3.10+
- Ollama running locally
- Ollama model: `qwen2:7b`
  - Start server: `ollama serve`
  - Pull model: `ollama pull qwen2:7b`

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python chatbot.py
```

### Remote Ollama
If Ollama is not local, set one of:
- `OLLAMA_HOST` (e.g. `http://10.0.0.5:11434`)
- `OLLAMA_BASE_URL` (same value)

## What It Includes
- **Persona**: `MovieAnalyst` (helpful data analyst for the IMDB movies dataset)
- **Knowledge Base**: CSV `imbd_movie.csv` loaded into an in-memory SQLite table
- **Agent**: LangGraph ReAct agent created in `build_agent()` in `chatbot.py`
- **Tools**: `get_schema` and `run_sql_query`
- **Memory**: conversation history stored in graph state

## Tools
The database tools are defined in `chatbot.py`:
- `get_schema()` — list tables and columns
- `run_sql_query(query: str)` — read-only SELECT queries against the IMDB movies table

## Notes
- DB-only behavior: non-database questions are refused and the user is asked to rephrase.
- Result limit: first 20 rows are returned if more are available.

## Example Prompts
- "Top 5 movies by IMDB rating"
- "Which director has the most movies?"
- "Highest grossing movie in the dataset"

DEMO:

![ALT]()