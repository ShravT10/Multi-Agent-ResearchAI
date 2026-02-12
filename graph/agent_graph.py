from langgraph.graph import StateGraph, END
from graph.state import ResearchState

from agents.planner import PlannerAgent
from core.schemas import PlannerInput

planner = PlannerAgent()

def planner_node(state: ResearchState):
    result = planner.run(
        PlannerInput(question=state["question"])
    )
    return {"tasks": result.tasks}

def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.set_entry_point("planner")

    workflow.add_edge("planner", END)

    return workflow.compile()
