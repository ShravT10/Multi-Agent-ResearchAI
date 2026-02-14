from agents.base import BaseAgent
from graph.state import ResearchState

class WriterAgent(BaseAgent):
    def run(self, state: ResearchState) -> dict:
        question = state["question"]
        analysis = state["analysis"]
        documents_by_task = state["documents"]

        all_sources = set()

        for task_id, docs in documents_by_task.items():
            for doc in docs:
                all_sources.add(doc.source)

        sources = list(all_sources)


        prompt = f"""
You are a research report writer.

Research Question:
{question}

Verified Facts:
{analysis.verified_facts}

Key Insights:
{analysis.key_insights}

Sources:
{sources}

Write a professional research report in Markdown format with:

1. Title
2. Introduction
3. Key Findings
4. Insights
5. Conclusion
6. References section listing the sources

Be concise and structured.
"""


        report = self.llm.invoke(prompt).content

        return {"report": report}
