import json
import re

from agents.base import BaseAgent
from graph.state import ResearchState
from core.schemas import AnalystOutput , TaskAnalysis

class AnalystAgent(BaseAgent):
    def run(self, state: ResearchState) -> dict:
        question = state["question"]
        documents_by_task = state["documents"]

        per_task_analysis = {}

        for task_id, docs in documents_by_task.items():

            context = "\n\n".join(
                [f"[Score {doc.score}] {doc.content}" for doc in docs]
            )

            prompt = f"""
    You are a research analyst.

    Using ONLY the provided evidence, analyze the research question for Task {task_id}.

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

            # Extract JSON
            import re, json

            def extract_json(text):
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if not json_match:
                    return None
                return json_match.group()

            json_str = extract_json(response)

            if not json_str:
                raise ValueError("No JSON found in LLM response")

            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # Retry once with correction prompt
                fix_prompt = f"""
            The following JSON is invalid. Fix it and return ONLY valid JSON.

            Invalid JSON:
            {json_str}
            """
                fixed_response = self.llm.invoke(fix_prompt).content
                fixed_json = extract_json(fixed_response)

                if not fixed_json:
                    raise ValueError("Retry failed: No JSON found")

                parsed = json.loads(fixed_json)

            per_task_analysis[task_id] = TaskAnalysis(**parsed)

        return {
            "analysis": AnalystOutput(
                per_task_analysis=per_task_analysis
            )
        }
