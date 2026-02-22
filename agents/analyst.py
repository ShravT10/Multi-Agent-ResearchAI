import json
from agents.base import BaseAgent
from graph.state import ResearchState
from core.schemas import AnalystOutput, TaskAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed


class AnalystAgent(BaseAgent):

    def run(self, state: ResearchState) -> dict:
        question = state["question"]
        documents_by_task = state["documents"]

        def safe_json_extract(text):
            start = text.find("{")
            if start == -1:
                return None

            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]

            return None

        def analyze_single_task(task_id, docs):
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

Return ONLY valid JSON:

{{
  "verified_facts": ["..."],
  "uncertain_claims": ["..."],
  "key_insights": ["..."]
}}

Do not include explanations outside JSON.
"""

            response = self.llm.invoke(prompt).content

            json_str = safe_json_extract(response)
            if not json_str:
                raise ValueError("No JSON found in LLM response")

            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                # Retry once with correction instruction
                fix_prompt = f"""
The following JSON is invalid. Fix it and return ONLY valid JSON.

Invalid JSON:
{json_str}
"""
                fixed_response = self.llm.invoke(fix_prompt).content
                fixed_json = safe_json_extract(fixed_response)

                if not fixed_json:
                    raise ValueError("Retry failed: No JSON found")

                parsed = json.loads(fixed_json)

            return task_id, TaskAnalysis(**parsed)

        per_task_analysis = {}

        # Parallel execution
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(analyze_single_task, task_id, docs)
                for task_id, docs in documents_by_task.items()
            ]

            for future in as_completed(futures):
                task_id, result = future.result()
                per_task_analysis[task_id] = result

        return {
            "analysis": AnalystOutput(
                per_task_analysis=per_task_analysis
            )
        }