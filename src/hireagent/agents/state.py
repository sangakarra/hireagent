"""
LangGraph state schema for the HireAgent job-analysis pipeline.

Why TypedDict over dataclass
─────────────────────────────
LangGraph moves data between nodes by *merging partial dicts* into shared state.
Each node receives the full state dict and returns only the keys it changed —
LangGraph calls dict.update(node_output) under the hood.

This makes TypedDict the natural fit for three reasons:

  1. TypedDict IS a plain dict at runtime.
     LangGraph can merge {"match_score": 7.5} into a TypedDict instance without
     knowing anything about the class structure. A dataclass instance is NOT a
     dict — LangGraph would have to special-case it.

  2. Partial updates are safe and explicit.
     A node that only sets match_score returns {"match_score": 7.5}.
     With a dataclass you'd need to return the *full* object, copying every field
     you didn't touch — fragile, and it obscures which fields the node owns.

  3. Zero runtime overhead.
     TypedDict annotations are erased at runtime. There's no __init__, no
     __slots__, no attribute descriptor machinery. The state dict is just a dict.

Why not Pydantic BaseModel
───────────────────────────
Pydantic models ARE dicts-compatible (via model_dump()), but they add validation
on assignment and aren't dict subclasses — LangGraph can't merge into them
directly. You'd have to re-construct the model from the merged dict after every
node, which is both wasteful and easy to get wrong.

The error field convention
───────────────────────────
Any node that fails sets state["error"] and returns early instead of raising.
Downstream nodes check this field at the top and short-circuit (return {}) if
it's set. This gives you:
  - A full state snapshot at the point of failure for debugging
  - No exception propagation that would abort the entire graph run
  - A single exit path: always call run_analysis(), inspect state["error"] first
"""

from typing import Optional, TypedDict


class JobAnalysisState(TypedDict):
    """
    Shared state that flows through every node in the job-analysis graph.

    Field lifecycle (which node populates each field):
      job_text        → caller, via run_analysis()
      requirements    → parse_job node
      match_score     → match_resume node
      matched_skills  → match_resume node
      gaps            → analyze_gaps node
      cover_letter    → write_cover_letter node (only when match_score >= 6)
      error           → any node that encounters a recoverable failure
    """

    job_text: str
    requirements: dict          # Serialized JobRequirements — plain dict keeps state JSON-serializable
    match_score: float
    matched_skills: list[str]
    gaps: list[dict]            # Each entry: {"skill": str, "severity": str, "suggestion": str}
    cover_letter: str
    error: Optional[str]
