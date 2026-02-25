from fastapi import FastAPI
from pydantic import BaseModel
from graph.agent_graph import build_graph

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multi-Agent Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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