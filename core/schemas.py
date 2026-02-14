from pydantic import BaseModel
from typing import List, Optional

class ResearchTask(BaseModel):
    task_id: int
    description: str

class PlannerInput(BaseModel):
    question: str

class PlannerOutput(BaseModel):
    tasks: List[ResearchTask]

class RetrievedDocument(BaseModel):
    task_id: int
    source: str
    score: float
    content: str


class RetrieverOutput(BaseModel):
    documents: List[RetrievedDocument]

class AnalystOutput(BaseModel):
    verified_facts: List[str]
    uncertain_claims: List[str]
    key_insights: List[str]

class WriterOutput(BaseModel):
    report_markdown: str