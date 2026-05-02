"""
match_resume — LangGraph node: parsed requirements + resume corpus → match score.

Key design decisions
─────────────────────

1. Single aggregated RAG query, not N per-skill queries.
   One option is to query the vector store separately for each required skill
   and merge the results. That's N network/disk round-trips and N embedding
   computations. Instead, we build one sentence-style query from all skills
   and retrieve the top-5 most relevant chunks in one shot.

   Why sentence-style ("Experience with Python, AWS, Kubernetes") rather than
   a bare comma list? Sentence-transformers were fine-tuned on sentence pairs —
   they embed sentence-shaped inputs more accurately than keyword lists.
   Reconstructing a sentence from the skill list shifts the query distribution
   into the model's training domain.

2. Claude scores the match — we don't threshold keywords.
   Keyword matching (skill in resume text) has high false-negative rate: a
   candidate who "designed distributed systems on EC2" has AWS experience even
   if the word "AWS" never appears. Claude can read between those lines; a
   keyword grep cannot. The trade-off is an LLM call cost per analysis.

3. Structured output via tool_use for score + matched_skills.
   We need both a float and a list back from Claude. Free text would require
   parsing ("Score: 7.5 out of 10") which is fragile. tool_use gives us a
   clean dict we can validate through MatchAssessment.

4. Explicit calibration instruction in the system prompt.
   Without guidance, Claude tends to cluster scores in the 7–8 range.
   The routing gate is at 6.0 — a miscalibrated scorer that awards 7 to poor
   matches would defeat the gating logic entirely. We explicitly ask for the
   full 1–10 scale with anchors.
"""

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .state import JobAnalysisState
from ..rag.embeddings import get_embedding_model
from ..rag.vector_store import load_vector_store, query_vector_store

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are an expert technical recruiter assessing resume fit against a job description. "
    "Use the full 1–10 scale with discipline: "
    "1–3 = poor fit (missing most requirements), "
    "4–5 = below average (meets some but lacks core skills), "
    "6–7 = solid match (meets most requirements with minor gaps), "
    "8–9 = strong match (exceeds most requirements), "
    "10 = exceptional (perfect fit, exceeds all requirements). "
    "Do not inflate scores. Only list a skill as matched if the resume provides clear evidence."
)


class MatchAssessment(BaseModel):
    score: float = Field(
        description="Overall fit score from 1.0 (no fit) to 10.0 (exceptional fit)"
    )
    matched_skills: list[str] = Field(
        description=(
            "Skills from the job requirements that are clearly evidenced in the resume. "
            "Use the same names as they appear in the requirements."
        )
    )
    justification: str = Field(
        description="2–3 sentence explanation of the score citing specific resume evidence"
    )


def match_resume(state: JobAnalysisState) -> dict:
    """
    Score resume fit against parsed job requirements using RAG + Claude.

    LangGraph node contract:
      Input:  full JobAnalysisState (uses state["requirements"])
      Output: {"match_score": float, "matched_skills": list[str]}  — on success
              {"error": str}                                         — on failure
              {}                                                     — upstream error (skip silently)
    """
    # Short-circuit if an upstream node already set an error — return an empty
    # dict so LangGraph doesn't overwrite the existing error message.
    if state.get("error"):
        return {}

    requirements = state.get("requirements", {})
    if not requirements:
        return {"error": "match_resume: requirements missing — did parse_job run?"}

    try:
        # ── Step 1: Retrieve relevant resume chunks via RAG ───────────────────
        embeddings = get_embedding_model()

        # load_vector_store reconnects to the on-disk ChromaDB collection without
        # re-embedding anything — fast, just a file open + index load.
        store = load_vector_store(embeddings, "resumes")

        all_skills = (
            requirements.get("must_have_skills", [])
            + requirements.get("nice_to_have_skills", [])
        )

        # Sentence-style query outperforms bare keyword lists with MiniLM
        skills_query = (
            f"Experience with {', '.join(all_skills)}"
            if all_skills
            else requirements.get("title", "software engineering experience")
        )

        # top_k=5 gives broader resume coverage than the default 3.
        # Sorted by content so chunk order is stable across calls — HNSW tie-breaking
        # is non-deterministic and different orderings shift the score even at temp=0.
        chunks = sorted(
            query_vector_store(store, skills_query, top_k=5),
            key=lambda c: c.page_content,
        )
        context = "\n\n---\n\n".join(c.page_content for c in chunks)

        # ── Step 2: Ask Claude to assess the match ────────────────────────────
        client = anthropic.Anthropic()

        prompt = (
            f"Job Requirements:\n"
            f"  Title: {requirements.get('title')}\n"
            f"  Company: {requirements.get('company')}\n"
            f"  Must-have skills: {requirements.get('must_have_skills', [])}\n"
            f"  Nice-to-have skills: {requirements.get('nice_to_have_skills', [])}\n"
            f"  Years of experience required: {requirements.get('years_experience', 'not specified')}\n\n"
            f"Resume excerpts (most relevant to the required skills):\n{context}\n\n"
            "Assess the candidate's fit for this role."
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[
                {
                    "name": "assess_match",
                    "description": "Score how well the resume matches the job requirements",
                    "input_schema": MatchAssessment.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": "assess_match"},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        assessment = MatchAssessment(**tool_use_block.input)

        # Return only the keys this node owns — LangGraph merges, not replaces
        return {
            "match_score": assessment.score,
            "matched_skills": assessment.matched_skills,
        }

    except Exception as exc:
        return {"error": f"match_resume failed: {exc}"}
