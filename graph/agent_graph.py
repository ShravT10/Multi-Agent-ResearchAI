from langgraph.graph import StateGraph, END
from graph.state import ResearchState

from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent
from agents.critic import CriticAgent
from agents.query_rewriter import QueryRewriterAgent


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
critic = CriticAgent()
query_rewriter = QueryRewriterAgent()


def planner_node(state: ResearchState):
    result = planner.run(
        PlannerInput(question=state["question"])
    )
    return {"tasks": result.tasks}

def retriever_node(state: ResearchState):
    return retriever.run(state)

def analyst_node(state: ResearchState):
    result = analyst.run(state)
    state_update = {"analysis": result["analysis"]}

    # increment retry count only if coming from critic
    if state.get("critic") and not state["critic"].is_valid:
        state_update["retry_count"] = state["retry_count"] + 1

    return state_update

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


def critic_node(state: ResearchState):
    result = critic.run(state)

    update = {"critic": result["critic"]}

    if not result["critic"].is_valid:
        update["retry_count"] = state["retry_count"] + 1

    return update

def critic_router(state: ResearchState):
    critic = state["critic"]

    if critic.is_valid:
        return "writer"

    if state["retry_count"] >= 2:
        return "writer"

    if critic.failure_type == "reasoning":
        return "analyst"

    if critic.failure_type == "retrieval":
        # First retry → increase k
        if state["retry_count"] == 0:
            return "retriever"
        # Second retry → rewrite query
        else:
            return "query_rewriter"

    if critic.failure_type == "coverage":
        return "planner"

    return "writer"

def query_rewriter_node(state: ResearchState):
    return query_rewriter.run(state)


def writer_node(state: ResearchState):
    return writer.run(state)


def build_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("merge", merge_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "merge")
    workflow.add_edge("merge", "analyst")
    workflow.add_edge("analyst", "critic")
    workflow.add_conditional_edges("critic", critic_router)
    workflow.add_edge("query_rewriter", "retriever")
    workflow.add_edge("writer", END)

    return workflow.compile()
