from agents.base import BaseAgent
from core.schemas import PlannerInput, PlannerOutput, ResearchTask

class PlannerAgent(BaseAgent):
    def run(self, input: PlannerInput) -> PlannerOutput:
        prompt = f"""
You are a research planning agent.

Break the following research question into 4–6 clear, non-overlapping research tasks.

Rules:
- Each task should be specific
- Avoid generic tasks
- Tasks should be executable independently

Question:
{input.question}

Return ONLY a numbered list of tasks.
"""

        response = self.llm.invoke(prompt).content

        tasks = []
        for i, line in enumerate(response.split("\n")):
            if line.strip():
                tasks.append(
                    ResearchTask(
                        task_id=i + 1,
                        description=line.strip("0123456789. ")
                    )
                )

        return PlannerOutput(tasks=tasks)