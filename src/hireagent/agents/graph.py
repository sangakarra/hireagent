"""
LangGraph StateGraph wiring all job-analysis nodes into a runnable pipeline.

Graph topology
───────────────

  parse_job → match_resume ──(score >= 6)──► write_cover_letter ──► analyze_gaps → END
                           └─(score < 6)──────────────────────────► analyze_gaps → END

Why LangGraph instead of a plain function chain
─────────────────────────────────────────────────

1. Routing as a first-class concept.
   The score threshold (>= 6) is expressed as an *edge condition* that you can
   see and modify in the graph definition — not buried in an if-statement inside
   a node. This separation means changing the threshold or adding a new branch
   doesn't touch any node's code.

2. Explicit, inspectable state at every boundary.
   Between every node call, the full state dict is available. With a compiled
   graph you can use a checkpointer (MemorySaver, SqliteSaver) to snapshot state
   after each node — useful for debugging expensive pipelines, resuming after
   failure, or replaying from a midpoint.

3. Nodes are independently testable.
   Because each node receives the full state and returns a partial update, you
   can unit-test parse_job by calling parse_job({"job_text": "..."}) directly —
   no graph wiring required. The graph is a composition layer, not a hard dependency.

4. Topology validation at compile time.
   graph.compile() validates that every node is reachable, all edges point to
   existing nodes, and END is reachable from all terminal nodes. This catches
   wiring bugs before any API calls are made.

StateGraph mechanics (summary for reference)
──────────────────────────────────────────────
  StateGraph(Schema)         — registers the TypedDict as the state schema;
                               LangGraph uses it to validate node update keys
  add_node(name, fn)         — fn signature: (state: dict) → dict (partial update)
  set_entry_point(name)      — shorthand for add_edge(START, name)
  add_edge(src, dst)         — unconditional transition
  add_conditional_edges(     — routing_fn inspects state, returns a key from
    src, routing_fn, mapping)  the mapping dict to resolve the next node
  compile()                  — validates topology, builds the Pregel execution plan;
                               returns a CompiledStateGraph (Runnable)
  invoke(initial_state)      — runs synchronously, returns the final state dict
  stream(initial_state)      — yields state snapshots after each node (async-friendly)
"""

import hashlib
import json
from pathlib import Path

from langgraph.graph import END, StateGraph

from .gap_analyzer import analyze_gaps
from .matcher import match_resume
from .parser import parse_job
from .state import JobAnalysisState
from .writer import write_cover_letter

_CACHE_FILE = Path("./data/cache/analysis_cache.json")
_RESUME_HASH_FILE = Path("./data/resume.hash")


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))


def _cache_key(job_text: str) -> str:
    resume_hash = _RESUME_HASH_FILE.read_text().strip() if _RESUME_HASH_FILE.exists() else ""
    return hashlib.sha256(f"{resume_hash}|{job_text}".encode()).hexdigest()


def _route_after_matcher(state: JobAnalysisState) -> str:
    """
    Routing function for the conditional edge after match_resume.

    Returns a key from the edge mapping in build_graph(). LangGraph resolves
    that key → node name via the mapping dict — here the keys equal the node
    names directly (identity mapping) for readability.

    Error short-circuit: if an upstream node failed, skip the writer.
    The gap_analyzer will also short-circuit when it sees state["error"].
    """
    if state.get("error"):
        return "analyze_gaps"

    return "write_cover_letter" if (state.get("match_score") or 0.0) >= 6.0 else "analyze_gaps"


def build_graph():
    """
    Construct and compile the job-analysis StateGraph.

    Returns a CompiledStateGraph (a LangChain Runnable). Separating build
    from invoke lets callers compile once and call many times — compilation
    is not free (topology validation + Pregel plan construction).
    """
    graph = StateGraph(JobAnalysisState)

    # Register nodes. Each name is an arbitrary string; it must match exactly
    # in edge definitions below.
    graph.add_node("parse_job", parse_job)
    graph.add_node("match_resume", match_resume)
    graph.add_node("analyze_gaps", analyze_gaps)
    graph.add_node("write_cover_letter", write_cover_letter)

    # Entry point — equivalent to add_edge(START, "parse_job")
    graph.set_entry_point("parse_job")

    # Unconditional edge: parsing always feeds into matching
    graph.add_edge("parse_job", "match_resume")

    # Conditional edge: routing function decides the branch after scoring.
    # The mapping dict translates the routing function's return value → node name.
    # Using an identity mapping (key == value) avoids a separate lookup table and
    # makes the branch destinations immediately obvious at the call site.
    graph.add_conditional_edges(
        "match_resume",
        _route_after_matcher,
        {
            "write_cover_letter": "write_cover_letter",
            "analyze_gaps": "analyze_gaps",
        },
    )

    # write_cover_letter always flows into analyze_gaps so gap analysis runs
    # regardless of whether a letter was drafted. This means the caller always
    # gets both outputs (letter + gaps) when score >= 6.
    graph.add_edge("write_cover_letter", "analyze_gaps")
    graph.add_edge("analyze_gaps", END)

    return graph.compile()


# Module-level singleton: compile once per process, reuse across run_analysis() calls.
# LangGraph's compile() is safe to call multiple times, but not free. Caching it here
# means the first import of this module pays the compilation cost; subsequent calls
# to run_analysis() are just invoke() calls on the pre-compiled graph.
_compiled_graph = build_graph()


def run_analysis(job_text: str) -> JobAnalysisState:
    """
    Entry point: run the full job-analysis pipeline against the ingested resume corpus.

    Results are cached on disk keyed by sha256(resume_hash + job_text). An
    identical job+resume pair returns the cached result immediately — no API
    calls, no embedding, deterministic output.  The cache is invalidated
    automatically whenever the resume changes (ingest_resume writes a new hash).

    Args:
        job_text: Raw text of the job posting to analyse.

    Returns:
        The final JobAnalysisState after all nodes have executed (or the cached
        state from a previous identical run).  Always check state["error"] first.
    """
    key = _cache_key(job_text)
    cache = _load_cache()
    if key in cache:
        cached: JobAnalysisState = cache[key]
        cached["_cached"] = True  # type: ignore[typeddict-unknown-key]
        return cached

    initial_state: JobAnalysisState = {
        "job_text": job_text,
        "requirements": {},
        "match_score": 0.0,
        "matched_skills": [],
        "gaps": [],
        "cover_letter": "",
        "error": None,
    }

    result = _compiled_graph.invoke(initial_state)

    if not result.get("error"):
        cache[key] = result
        _save_cache(cache)

    return result
