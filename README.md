# HireAgent

An AI-powered hiring assistant that uses a multi-agent architecture and RAG (Retrieval-Augmented Generation) to streamline the recruitment process.

## Features

- **Multi-Agent System**: Specialized agents for resume screening, job matching, and candidate evaluation
- **RAG Pipeline**: Indexes resumes and job descriptions for semantic search and retrieval
- **Streamlit UI**: Interactive frontend for recruiters to manage candidates and view insights

## Project Structure

```
src/hireagent/
├── agents/   # AI agents (screening, matching, evaluation)
├── rag/      # RAG pipeline (indexing, retrieval, embeddings)
└── ui/       # Streamlit frontend
tests/        # Unit and integration tests
```

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your API key:
   ```bash
   cp .env.example .env
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run src/hireagent/ui/app.py
   ```

## Requirements

- Python 3.10+
- Anthropic API key
