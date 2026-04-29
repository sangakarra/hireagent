# HireAgent — Claude Code Context

## What this project is

HireAgent is an AI-powered hiring assistant. It uses a multi-agent architecture built with LangChain, a ChromaDB-backed RAG pipeline for semantic search over resumes and job descriptions, and a Streamlit frontend for recruiters.

## Architecture

- **`src/hireagent/agents/`** — LangChain agents, each with a focused responsibility (e.g., resume screener, job matcher, candidate evaluator). Add new agents here as separate modules.
- **`src/hireagent/rag/`** — RAG pipeline: document loaders, chunking, embedding, and ChromaDB retrieval. This is the data layer for all agents.
- **`src/hireagent/ui/`** — Streamlit app. Entry point is `app.py`. Keeps UI logic separate from agent/RAG logic.
- **`tests/`** — Pytest-based tests. Mirror the `src/hireagent/` structure.

## Key conventions

- Use `python-dotenv` to load `ANTHROPIC_API_KEY` from `.env` — never hardcode keys.
- Agents should be stateless where possible; pass context explicitly.
- RAG retrieval returns LangChain `Document` objects — keep that interface consistent.
- Use Claude (Anthropic) as the LLM backend via LangChain's Anthropic integration.

## Running the app

```bash
streamlit run src/hireagent/ui/app.py
```

## Running tests

```bash
pytest tests/
```
