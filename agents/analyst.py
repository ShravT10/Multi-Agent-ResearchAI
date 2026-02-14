import json
import re

from agents.base import BaseAgent
from graph.state import ResearchState
from core.schemas import AnalystOutput

class AnalystAgent(BaseAgent):
    def run(self, state: ResearchState) -> dict:
        question = state["question"]
        documents_by_task = state["documents"]

        all_docs = []

        for task_id, docs in documents_by_task.items():
            for doc in docs:
                all_docs.append(
                    f"[Task {task_id} | Score {doc.score}] {doc.content}"
                )

        context = "\n\n".join(all_docs)


        prompt = f"""
You are a research analyst.

Using ONLY the provided evidence, analyze the research question.

Research Question:
{question}

Evidence:
{context}

Return ONLY valid JSON in this format:

{{
  "verified_facts": ["..."],
  "uncertain_claims": ["..."],
  "key_insights": ["..."]
}}

Do not include explanations outside JSON.
"""

        response = self.llm.invoke(prompt).content
        
        # Extract JSON block using regex
        json_match = re.search(r"\{.*\}", response, re.DOTALL)

        if not json_match:
            raise ValueError("No JSON object found in LLM response")

        json_str = json_match.group()

        try:
            parsed = json.loads(json_str)

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format returned by LLM")

        return {
            "analysis": AnalystOutput(**parsed)
        }
