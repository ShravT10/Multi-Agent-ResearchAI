from langgraph.graph import StateGraph, END
from graph.state import ResearchState

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent

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
writer = WriterAgent()

def planner_node(state: ResearchState):
    result = planner.run(
        PlannerInput(question=state["question"])
    )
    return {"tasks": result.tasks}

def retriever_node(state: ResearchState):
    return retriever.run(state)

def analyst_node(state: ResearchState):
    return analyst.run(state)

def merge_node(state: ResearchState):
    documents_by_task = state["documents"]

    merged_docs = {}
    content_map = {}

    # Flatten and deduplicate
    for task_id, docs in documents_by_task.items():
        for doc in docs:
            content_key = doc.content.strip()

            if content_key not in content_map:
                content_map[content_key] = doc
            else:
                # Keep higher score
                if doc.score > content_map[content_key].score:
                    content_map[content_key] = doc

    # Convert to list (global cleaned list)
    cleaned_docs = list(content_map.values())

    # Optional: sort by score descending
    cleaned_docs.sort(key=lambda x: x.score, reverse=True)

    return {"documents": {0: cleaned_docs}}

def writer_node(state: ResearchState):
    return writer.run(state)


def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "merge")
    workflow.add_edge("merge", "analyst")
    workflow.add_edge("analyst","writer")
    workflow.add_edge("writer",END)

    return workflow.compile()
