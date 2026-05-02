# HireAgent

An AI-powered hiring assistant that parses job postings, scores resume fit, drafts cover letters, and identifies skill gaps — all orchestrated through a LangGraph state machine with a ChromaDB-backed RAG pipeline.

---

## Architecture

```mermaid
flowchart LR
    subgraph RAG["RAG Layer (ChromaDB + MiniLM)"]
        R[(Resume\nCorpus)]
    end

    subgraph Graph["LangGraph State Machine"]
        direction LR
        A[parse_job] --> B[match_resume]
        B -->|score ≥ 6| C[write_cover_letter]
        B -->|score < 6| D[analyze_gaps]
        C --> D
        D --> E([END])
    end

    JD[Job Description] --> A
    R -->|RAG: all skills query| B
    R -->|RAG: matched skills query| C

    style RAG fill:#f0f4ff,stroke:#9aabff
    style Graph fill:#fff8f0,stroke:#ffb347
```

**State flows left to right through four nodes.** The conditional edge after `match_resume` is the only branch point: strong candidates (score ≥ 6) get a cover letter drafted before gap analysis runs; weaker candidates skip straight to gap analysis.

---

## How it works

### RAG Pipeline

Resumes are loaded from PDF, split into overlapping chunks, and embedded with `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are stored in a local ChromaDB collection. At query time, relevant chunks are retrieved by cosine similarity and passed as context to Claude.

Two separate retrievals happen inside the graph:
- **`match_resume`** queries with all required skills to get the broadest coverage for scoring.
- **`write_cover_letter`** queries with only the *matched* skills to surface achievement evidence — the query is targeted at what the candidate already has, not what they lack.

### Multi-Agent Design

Each LangGraph node is a single-responsibility agent:

| Node | Input | Output | LLM? |
|---|---|---|---|
| `parse_job` | raw job text | structured `requirements` dict | Yes — forced `tool_use` |
| `match_resume` | requirements + resume chunks | `match_score`, `matched_skills` | Yes — scored 1–10 |
| `write_cover_letter` | requirements + matched skills + chunks | `cover_letter` prose | Yes — free text |
| `analyze_gaps` | requirements + matched_skills | `gaps` list with severity + suggestions | Partial — set difference is deterministic; LLM only for suggestions |

The graph compiles once per process and is reused across calls. Results are cached on disk keyed by `sha256(resume_hash + job_text)` — identical inputs return instantly with no API calls.

---

## Tech stack

- **LangGraph** — state machine orchestration, conditional routing, node isolation
- **LangChain** — document loaders, text splitters, ChromaDB integration
- **ChromaDB** — local vector store for resume embeddings
- **`sentence-transformers/all-MiniLM-L6-v2`** — fast, local embedding model (no API cost)
- **Anthropic Claude (Sonnet 4.6)** — LLM backend for all four nodes
- **Pydantic v2** — structured output schemas (tool_use input_schema)
- **Streamlit** — recruiter-facing UI
- **`python-dotenv`** — API key management

---

## Setup

```bash
# 1. Clone and install in editable mode
git clone https://github.com/your-username/hireagent.git
cd hireagent
pip install -e ".[dev]"

# 2. Add your API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the app
streamlit run src/hireagent/ui/app.py
```

---

## Screenshot

> _Screenshot coming soon — upload your resume and paste a job description to see the live analysis._

---

## Design decisions

### Why LangGraph over a plain ReAct loop

ReAct loops decide the next action dynamically at runtime — the LLM chooses what to do next based on the previous step's output. That's powerful for open-ended agents, but every call in this pipeline is predetermined: parse → match → (maybe) write → analyze. There is no decision to make dynamically; the only branch is a threshold check.

LangGraph expresses that as a compiled graph with an explicit conditional edge. The routing logic (`score >= 6`) lives in one place in the graph definition and is visible and modifiable without touching node code. It also validates at compile time that all nodes are reachable and END is always accessible — a class of wiring bugs that ReAct never catches.

### Why TypedDict over dataclass for state

LangGraph merges state between nodes by calling `dict.update(node_output)`. Each node returns only the keys it changed — a partial dict. TypedDict is a plain dict at runtime, so this merge works without any special handling. A dataclass is not a dict: LangGraph would have to reconstruct it after every node, requiring every node to return the full object and copy fields it didn't touch. TypedDict annotations are also erased at runtime — zero overhead, no `__init__`, no descriptor machinery.

### Why per-skill RAG probing for gap analysis

Gap identification itself is deterministic: `missing = required_skills − matched_skills`. We don't send this question to an LLM because set difference is not a language task — there's no accuracy benefit, and there's hallucination risk (e.g. "the candidate might have this skill implicitly"). The LLM only handles remediation suggestions, where world knowledge and phrasing actually matter. Separating the two makes each step independently auditable.

### Why different retrieval queries for matcher vs writer

The matcher needs broad coverage to score fairly — it queries with *all* required skills so no relevant resume section is missed. The writer needs targeted evidence to ground the cover letter — it queries with only *matched* skills, nudging the retriever toward achievement-oriented sections (metrics, project outcomes, impact statements) rather than skill-list sections. Same vector store, different semantic intent: the query controls what gets surfaced.

### Why temperature=0

Every Claude call in this pipeline uses `temperature=0`. The parser, scorer, gap analyzer, and cover letter writer all produce outputs that feed downstream logic or are displayed to a recruiter. Determinism here is a feature: the same resume + job description produces the same result every run, which is required for the disk cache to be meaningful and for results to be reproducible during debugging.

---

## Future improvements

- **Batch analysis** — process multiple resumes against one job description in parallel; surface a ranked shortlist
- **Persistent checkpointing** — swap `MemorySaver` in for `SqliteSaver` so long-running pipelines can resume after failure
- **Structured gap tracking** — store gap history per candidate over time to surface progress across applications
- **Multi-collection RAG** — separate ChromaDB collections per recruiter or per job family, not one global corpus
- **Streaming UI** — use LangGraph's `stream()` instead of `invoke()` to show node-by-node progress in the Streamlit app
- **Kubernetes / cloud deployment** — containerize with Docker Compose (app + ChromaDB), add a Helm chart for production
