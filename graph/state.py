from typing import TypedDict, List
from core.schemas import ResearchTask, RetrievedDocument, AnalystOutput

class ResearchState(TypedDict):
    question: str
    tasks: List[ResearchTask]
    documents: List[RetrievedDocument]
    analysis: AnalystOutput
    report: str
