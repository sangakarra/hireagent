"""
analyze_gaps — LangGraph node: identify missing skills and suggest remediation.

Key design decisions
─────────────────────

1. Deterministic gap identification — no LLM for the "what" step.
   We compute which skills are missing using set difference:
     missing = required_skills − matched_skills
   This is deterministic, free, and auditable. Sending this question to an
   LLM would introduce hallucination risk ("the candidate might have this
   skill but not explicitly listed it") and waste tokens on a task that is
   unambiguously computable.

2. LLM for remediation suggestions only — the "how" step.
   Generating actionable, role-specific suggestions ("Build a Kubernetes
   homelab and study for CKA" vs "Learn Kubernetes") requires language
   understanding and world knowledge. That's where the LLM adds value.
   Separating "what's missing" (deterministic) from "how to address it"
   (generative) makes each step independently testable and auditable.

3. Severity derived from requirements structure, not inferred by Claude.
   A skill on must_have_skills is "critical" — full stop. We don't ask
   Claude to decide severity because that introduces inconsistency and
   the risk of demoting a hard requirement to "nice-to-have" based on
   how the resume reads. The source of truth is the requirements dict.

4. Single batched LLM call for all suggestions.
   One call for all missing skills is cheaper than N per-skill calls.
   It also gives Claude the full gap profile at once, which produces
   more coherent, non-repetitive suggestions — Claude can vary its advice
   across similar skills rather than suggesting "build a project" for each.

5. Order-preserving suggestion assignment.
   We send skills to Claude in a numbered list and ask it to respond in
   the same order. This avoids any name-matching logic when zipping skills
   with suggestions — we rely on positional alignment.
"""

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .state import JobAnalysisState

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a technical career coach helping a job seeker address skill gaps. "
    "For each missing skill, provide one concrete, actionable suggestion (1–2 sentences). "
    "Be specific — name certifications, project types, or learning resources where helpful. "
    "Vary your advice; do not repeat the same suggestion for similar skills."
)


class GapSuggestions(BaseModel):
    suggestions: list[str] = Field(
        description=(
            "One actionable suggestion per missing skill, "
            "in the exact same order as the numbered input list."
        )
    )


def analyze_gaps(state: JobAnalysisState) -> dict:
    """
    Identify skill gaps between job requirements and matched resume skills.
    Generate role-specific remediation suggestions for each gap.

    LangGraph node contract:
      Input:  full JobAnalysisState (uses state["requirements"] and state["matched_skills"])
      Output: {"gaps": list[dict]}  — on success (empty list = no gaps)
              {"error": str}        — on failure
              {}                    — upstream error (skip silently)

    Each gap dict shape:
      {"skill": str, "severity": "critical" | "nice-to-have", "suggestion": str}
    """
    if state.get("error"):
        return {}

    requirements = state.get("requirements", {})
    matched_skills = state.get("matched_skills", [])

    # Normalise to lowercase for case-insensitive comparison.
    # "Python" and "python" in different sources should not produce a false gap.
    # We preserve original casing in the output for readability.
    matched_lower = {s.lower() for s in matched_skills}

    must_have = requirements.get("must_have_skills", [])
    nice_to_have = requirements.get("nice_to_have_skills", [])

    # Deterministic gap identification — no LLM needed here
    missing_critical = [s for s in must_have if s.lower() not in matched_lower]
    missing_nice = [s for s in nice_to_have if s.lower() not in matched_lower]

    all_missing = missing_critical + missing_nice
    if not all_missing:
        return {"gaps": []}

    try:
        client = anthropic.Anthropic()

        # Number the skills so Claude can return suggestions in the same order
        skills_list = "\n".join(
            f"  {i + 1}. {skill}" for i, skill in enumerate(all_missing)
        )
        prompt = (
            f"The candidate is applying for: {requirements.get('title', 'this role')} "
            f"at {requirements.get('company', 'this company')}.\n\n"
            f"Missing skills (provide one suggestion per skill, in this order):\n"
            f"{skills_list}"
        )

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
            tools=[
                {
                    "name": "provide_suggestions",
                    "description": "Provide one remediation suggestion per missing skill",
                    "input_schema": GapSuggestions.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": "provide_suggestions"},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        result = GapSuggestions(**tool_use_block.input)
        suggestions = result.suggestions

        # Zip skills with suggestions using positional alignment.
        # Severity is assigned from the requirements structure — not inferred.
        gaps: list[dict] = []

        for i, skill in enumerate(missing_critical):
            gaps.append({
                "skill": skill,
                "severity": "critical",
                "suggestion": suggestions[i] if i < len(suggestions) else "",
            })

        offset = len(missing_critical)
        for i, skill in enumerate(missing_nice):
            idx = offset + i
            gaps.append({
                "skill": skill,
                "severity": "nice-to-have",
                "suggestion": suggestions[idx] if idx < len(suggestions) else "",
            })

        return {"gaps": gaps}

    except Exception as exc:
        return {"error": f"analyze_gaps failed: {exc}"}
