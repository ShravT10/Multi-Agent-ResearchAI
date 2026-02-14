from langgraph.graph import StateGraph, END
from graph.state import ResearchState

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.analyst import AnalystAgent

from core.schemas import PlannerInput

sample_texts = [
    "Generative AI helps optimize crop yield through predictive analytics.",
    "AI adoption in India is increasing in agriculture sector.",
    "Challenges include lack of infrastructure in rural areas.",
    "Government policies are encouraging agri-tech startups."
]

planner = PlannerAgent()
retriever = RetrieverAgent(sample_texts)
analyst = AnalystAgent()

def planner_node(state: ResearchState):
    result = planner.run(
        PlannerInput(question=state["question"])
    )
    return {"tasks": result.tasks}

def retriever_node(state: ResearchState):
    return retriever.run(state)

def analyst_node(state: ResearchState):
    return analyst.run(state)



def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("analyst", analyst_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever","analyst")
    workflow.add_edge("analyst",END)

    return workflow.compile()
