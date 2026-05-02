"""
parse_job — LangGraph node: raw job text → structured requirements dict.

Key design decisions
─────────────────────

1. Pydantic model as a single source of truth for the schema.
   JobRequirements.model_json_schema() generates the JSON Schema that Claude's
   tool_use API accepts as input_schema. Defining the schema once in Pydantic
   means it drives the API call, the response validation, and IDE type-checking —
   no duplication, no drift between "what we asked for" and "what we parse".

2. Forced tool_use for deterministic structured output.
   Claude has no dedicated "JSON mode". The idiomatic workaround: define a tool
   whose input_schema IS the structure you want, then set:
     tool_choice={"type": "tool", "name": "extract_requirements"}
   This forces Claude to always emit a tool_use block rather than free text.
   The result is machine-readable every time — no JSON extraction regex needed.

3. Prompt caching on the system prompt.
   The system prompt is identical across every job posting. Adding
   cache_control={"type": "ephemeral"} tells Anthropic to cache that prefix
   for up to 5 minutes. On a pipeline that parses many postings in a session,
   this cuts the input-token cost of the system prompt to ~10% of normal.

4. Node return contract: partial dict, not full state.
   LangGraph merges the returned dict into the shared state. Returning only
   {"requirements": ...} means this node can never accidentally overwrite
   match_score, matched_skills, or any other field it doesn't own.
   On failure, {"error": str} is returned so the graph continues gracefully
   rather than crashing with an unhandled exception.
"""

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

from .state import JobAnalysisState

load_dotenv()

# Sonnet 4.6 is accurate enough for structured extraction and substantially
# cheaper than Opus — the right trade-off for a high-frequency parsing node.
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a senior technical recruiter. Extract structured information "
    "from job postings accurately and completely. If a field is not mentioned "
    "in the posting, return an empty list or null as appropriate — never invent."
)


class JobRequirements(BaseModel):
    """Structured representation of a parsed job posting."""

    title: str = Field(description="Job title, e.g. 'Senior Backend Engineer'")
    company: str = Field(description="Company name; use 'Unknown' if not stated")
    must_have_skills: list[str] = Field(
        description="Required/mandatory technical skills explicitly stated (e.g. 'Python', 'AWS')"
    )
    nice_to_have_skills: list[str] = Field(
        description="Optional or preferred skills listed as a bonus or 'nice to have'"
    )
    years_experience: Optional[int] = Field(
        None,
        description="Minimum years of experience required as an integer; null if not stated",
    )
    responsibilities: list[str] = Field(
        description="Key job responsibilities or day-to-day duties — one item per bullet"
    )


def parse_job(state: JobAnalysisState) -> dict:
    """
    Extract structured requirements from raw job posting text.

    LangGraph node contract:
      Input:  full JobAnalysisState (uses only state["job_text"])
      Output: {"requirements": dict}  — on success
              {"error": str}          — on failure (downstream nodes will skip)
    """
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            temperature=0,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Cache this stable prefix — billed at ~0.1× on cache hits
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[
                {
                    "name": "extract_requirements",
                    "description": "Extract structured requirements from a job posting",
                    # model_json_schema() is the Pydantic v2 API (replaces schema())
                    "input_schema": JobRequirements.model_json_schema(),
                }
            ],
            # Force this specific tool — eliminates any free-text fallback
            tool_choice={"type": "tool", "name": "extract_requirements"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract the requirements from the following job posting:\n\n"
                        f"{state['job_text']}"
                    ),
                }
            ],
        )

        # Because we forced tool_use, there will always be exactly one tool_use block.
        # tool_use_block.input is already a dict matching JobRequirements' field names.
        tool_use_block = next(b for b in response.content if b.type == "tool_use")

        # Validate through Pydantic — catches type mismatches and missing required fields
        requirements = JobRequirements(**tool_use_block.input)

        # model_dump() serializes to a plain dict so state stays JSON-serializable.
        # Storing a Pydantic model instance in state would break checkpointing.
        return {"requirements": requirements.model_dump()}

    except Exception as exc:
        # Return an error key rather than raising — lets the graph capture the failure
        # in state["error"] and surface it to the caller cleanly via run_analysis().
        return {"error": f"parse_job failed: {exc}"}
