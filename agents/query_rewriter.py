from agents.base import BaseAgent
from graph.state import ResearchState
from core.schemas import RewriterOutput
import re, json


class QueryRewriterAgent(BaseAgent):
    def run(self, state: ResearchState) -> dict:
        tasks = state["tasks"]
        critic = state["critic"]
        question = state["question"]

        rewritten = {}

        for task in tasks:
            prompt = f"""
You are a query rewriting specialist.

The retriever failed due to weak evidence.

Research Question:
{question}

Original Task:
{task.description}

Critic Feedback:
{critic.feedback}

Rewrite the task into a more specific, retrieval-optimized query.
Add missing keywords if necessary.
Be precise and semantically rich.

Return ONLY valid JSON:

{{
  "rewritten_query": "improved query here"
}}
"""

            response = self.llm.invoke(prompt).content

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON returned by query rewriter")

            json_str = json_match.group()

            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON from query rewriter")

            rewritten[task.task_id] = parsed["rewritten_query"]

        return {"rewritten_tasks": RewriterOutput(rewritten_tasks=rewritten)}