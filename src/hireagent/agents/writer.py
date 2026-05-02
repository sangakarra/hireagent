"""
write_cover_letter — LangGraph node: draft a tailored cover letter from resume evidence.

Key design decisions
─────────────────────

1. This node does NOT check match_score.
   The score gate (>= 6.0) is an *edge condition* in the graph, not a guard
   clause in this node. This is intentional separation of concerns:
     - Nodes own transformation logic ("given these inputs, produce this output")
     - The graph owns routing logic ("given this state, go here or there")
   If the writer also checked the score, you'd have the same logic in two places —
   and a future change to the threshold would require updating both.

2. RAG context is re-retrieved here rather than threaded through state.
   We could have the matcher store raw resume chunks in state["resume_context"]
   and reuse them here. That would save one embedding + similarity-search call.
   Trade-off: it bloats the state object (chunks can be 2-5 KB each), makes the
   state schema less clean, and couples the matcher's retrieval strategy to the
   writer's needs. The cover letter needs a *different* query — focused on
   achievements with matched skills, not on all required skills. Re-querying
   with a better-targeted prompt produces better cover letter evidence.

3. The query targets matched_skills, not all_skills.
   The cover letter should cite what the candidate *has*, not what they lack.
   Querying with matched skills surfaces the resume evidence most relevant to
   leading with strengths.

4. Critical gaps are computed inline, not read from state["gaps"].
   The graph runs write_cover_letter BEFORE analyze_gaps (see graph.py).
   state["gaps"] is empty when this node executes. Rather than reorder the
   graph (which would delay the routing decision), we compute the missing
   critical skills inline using the same set-difference logic as gap_analyzer.
   This is a small, cheap computation — no duplication concern.

5. Free text output — no tool_use.
   Cover letters are inherently prose. Imposing a Pydantic schema (e.g.,
   {"opening": str, "body": str, "closing": str}) would add complexity
   with no downstream benefit — nothing parses the letter structurally.
   We request the full text and extract the first text content block.
"""

import anthropic
from dotenv import load_dotenv

from .state import JobAnalysisState
from ..rag.embeddings import get_embedding_model
from ..rag.vector_store import load_vector_store, query_vector_store

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are an expert career coach writing a compelling cover letter on behalf of a candidate. "
    "Write in first person. Be concise: 3–4 short paragraphs. "
    "Reference specific skills and experiences from the provided resume excerpts. "
    "Do not fabricate information that is not present in the resume context. "
    "If addressing a skill gap, frame it as a growth area, not a deficiency."
)


def write_cover_letter(state: JobAnalysisState) -> dict:
    """
    Draft a tailored cover letter grounded in retrieved resume evidence.

    LangGraph node contract:
      Input:  full JobAnalysisState (uses requirements, matched_skills)
      Output: {"cover_letter": str}  — on success
              {"error": str}         — on failure
              {}                     — upstream error (skip silently)

    Precondition (enforced by graph routing, never checked here):
      state["match_score"] >= 6.0
    """
    if state.get("error"):
        return {}

    requirements = state.get("requirements", {})
    matched_skills = state.get("matched_skills", [])

    try:
        # ── Step 1: Retrieve resume chunks relevant to matched skills ─────────
        embeddings = get_embedding_model()
        store = load_vector_store(embeddings, "resumes")

        # Query on matched skills to surface achievement evidence, not gaps.
        # "Experience and achievements" in the query nudges the retriever toward
        # result-oriented resume sections (projects, metrics, impact statements).
        query = (
            f"Experience and achievements with {', '.join(matched_skills)}"
            if matched_skills
            else requirements.get("title", "relevant technical experience")
        )
        chunks = sorted(
            query_vector_store(store, query, top_k=4),
            key=lambda c: c.page_content,
        )
        resume_context = "\n\n---\n\n".join(c.page_content for c in chunks)

        # ── Step 2: Identify critical gaps to acknowledge in the letter ───────
        # We can't read state["gaps"] — analyze_gaps hasn't run yet.
        # Recompute the set difference inline (cheap, no LLM call).
        matched_lower = {s.lower() for s in matched_skills}
        must_have = requirements.get("must_have_skills", [])
        critical_missing = [s for s in must_have if s.lower() not in matched_lower]

        # Acknowledge at most 2 gaps — more than that makes the letter defensive
        gap_note = (
            f"\nBriefly acknowledge (1 sentence) that the candidate is actively "
            f"developing: {', '.join(critical_missing[:2])}."
            if critical_missing
            else ""
        )

        # ── Step 3: Build the prompt ──────────────────────────────────────────
        responsibilities = requirements.get("responsibilities", [])
        responsibilities_text = (
            "\n".join(f"  - {r}" for r in responsibilities[:5])  # cap at 5 to avoid prompt bloat
            if responsibilities
            else "  Not specified"
        )

        prompt = (
            f"Write a cover letter for this role:\n"
            f"  Title: {requirements.get('title')}\n"
            f"  Company: {requirements.get('company')}\n"
            f"  Key responsibilities:\n{responsibilities_text}\n\n"
            f"The candidate's relevant matched skills: {matched_skills}\n\n"
            f"Resume excerpts to draw specific evidence from:\n{resume_context}"
            f"{gap_note}"
        )

        # ── Step 4: Generate the cover letter ─────────────────────────────────
        client = anthropic.Anthropic()

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )

        cover_letter = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        return {"cover_letter": cover_letter}

    except Exception as exc:
        return {"error": f"write_cover_letter failed: {exc}"}
