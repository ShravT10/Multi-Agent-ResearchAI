from typing import TypedDict, List , Dict

from core.schemas import ResearchTask, RetrievedDocument, AnalystOutput , CriticOutput , RewriterOutput

class ResearchState(TypedDict):
    question: str
    tasks: List[ResearchTask]
    rewritten_tasks: RewriterOutput
    documents: Dict[int ,List[RetrievedDocument]]
    analysis: AnalystOutput
    critic: CriticOutput
    retry_count: int
    report: str
