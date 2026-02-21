from agents.base import BaseAgent
from graph.state import ResearchState

class WriterAgent(BaseAgent):
    def run(self, state: ResearchState) -> dict:
        question = state["question"]
        analysis = state["analysis"]
        documents_by_task = state["documents"]

        per_task_analysis = analysis.per_task_analysis

        # Combine insights across tasks
        all_verified = []
        all_uncertain = []
        all_insights = []

        for task_id, task_analysis in per_task_analysis.items():
            all_verified.extend(task_analysis.verified_facts)
            all_uncertain.extend(task_analysis.uncertain_claims)
            all_insights.extend(task_analysis.key_insights)

        # Collect unique sources
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
    {all_verified}

    Key Insights:
    {all_insights}

    Uncertain Claims:
    {all_uncertain}

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
