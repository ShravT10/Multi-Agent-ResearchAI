from agents.planner import PlannerAgent
from core.schemas import PlannerInput

planner = PlannerAgent()

result = planner.run(
    PlannerInput(
        question="What is the impact of generative AI on agriculture in India?"
    )
)

for task in result.tasks:
    print(task)