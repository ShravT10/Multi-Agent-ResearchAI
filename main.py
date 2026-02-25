from fastapi import FastAPI
from pydantic import BaseModel
from graph.agent_graph import build_graph

app = FastAPI(title="Multi-Agent Research Assistant")

graph = build_graph()


class ResearchRequest(BaseModel):
    question: str


@app.post("/research")
def run_research(request: ResearchRequest):

    initial_state = {
        "question": request.question,
        "tasks": [],
        "documents": {},
        "web_documents": {},
        "analysis": None,
        "critic": None,
        "report": None,
        "retry_count": 0,
        "rewritten_tasks": None
    }

    result = graph.invoke(initial_state)

    return {
        "report": result.get("report"),
        "analysis": result.get("analysis"),
        "critic": result.get("critic")
    }