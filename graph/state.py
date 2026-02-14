from typing import TypedDict, List , Dict

from core.schemas import ResearchTask, RetrievedDocument, AnalystOutput

class ResearchState(TypedDict):
    question: str
    tasks: List[ResearchTask]
    documents: Dict[int ,List[RetrievedDocument]]
    analysis: AnalystOutput
    report: str
