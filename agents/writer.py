from agents.base import BaseAgent
from graph.state import ResearchState

class WriterAgent(BaseAgent):
    def run(self, state: ResearchState) -> dict:
        question = state["question"]
        analysis = state["analysis"]
        documents = state["documents"]

        # Collect unique sources
        sources = list(set([doc.source for doc in documents]))

        prompt = f"""
You are a research report writer.

Research Question:
{question}

Verified Facts:
{analysis.verified_facts}

Key Insights:
{analysis.key_insights}

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
