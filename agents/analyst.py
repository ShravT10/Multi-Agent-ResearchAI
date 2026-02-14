from agents.base import BaseAgent
from graph.state import ResearchState
from core.schemas import AnalystOutput

class AnalystAgent(BaseAgent):
    def run(self, state: ResearchState) -> dict:
        question = state["question"]
        documents = state["documents"]

        context = "\n\n".join(
            [doc.content for doc in documents]
        )

        prompt = f"""
You are a research analyst.

Using ONLY the provided evidence, analyze the research question.

Research Question:
{question}

Evidence:
{context}

Instructions:
1. Extract verified facts directly supported by evidence.
2. Identify uncertain or weakly supported claims.
3. Provide 3-5 key insights.
4. Be concise and structured.

Return response in this format:

Verified Facts:
- ...

Uncertain Claims:
- ...

Key Insights:
- ...
"""

        response = self.llm.invoke(prompt).content

        # Simple parsing (improve later)
        verified = []
        uncertain = []
        insights = []

        section = None
        for line in response.split("\n"):
            if "Verified Facts" in line:
                section = "verified"
            elif "Uncertain Claims" in line:
                section = "uncertain"
            elif "Key Insights" in line:
                section = "insights"
            elif line.strip().startswith("-"):
                if section == "verified":
                    verified.append(line.strip("- ").strip())
                elif section == "uncertain":
                    uncertain.append(line.strip("- ").strip())
                elif section == "insights":
                    insights.append(line.strip("- ").strip())

        return {
            "analysis": AnalystOutput(
                verified_facts=verified,
                uncertain_claims=uncertain,
                key_insights=insights
            )
        }
