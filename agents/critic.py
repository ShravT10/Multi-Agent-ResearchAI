from agents.base import BaseAgent
from graph.state import ResearchState
from core.schemas import CriticOutput
import re, json


class CriticAgent(BaseAgent):
    def run(self, state: ResearchState) -> dict:
        question = state["question"]
        analysis = state["analysis"]
        documents = state["documents"]

        # Flatten evidence content
        evidence_text = ""
        for task_id, docs in documents.items():
            for doc in docs:
                evidence_text += f"[Task {task_id}] {doc.content}\n"

        prompt = f"""
You are a research critic.

Evaluate whether the analysis is:

1. Grounded in provided evidence.
2. Free of unsupported claims.
3. Covers all research tasks.

Classify failure type as:
- "reasoning" → If analysis logic is weak or incorrect.
- "retrieval" → If evidence is insufficient or weak.
- "coverage" → If some tasks are not addressed.
- "none" → If analysis is valid.

Research Question:
{question}

Analysis:
{analysis}

Evidence:
{evidence_text}

Return ONLY valid JSON:

{{
  "is_valid": true or false,
  "failure_type": "reasoning" | "retrieval" | "coverage" | "none",
  "feedback": ["list specific issues if any"]
}}
"""

        response = self.llm.invoke(prompt).content

        # Extract JSON safely
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON returned by critic")

        json_str = json_match.group()

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON from critic")

        return {"critic": CriticOutput(**parsed)}