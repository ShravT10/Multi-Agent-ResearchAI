from agents.base import BaseAgent
from core.schemas import RetrieverOutput, RetrievedDocument
from graph.state import ResearchState
from core.vector_store import create_vector_store

class RetrieverAgent(BaseAgent):
    def __init__(self, texts):
        super().__init__()
        self.vector_store = create_vector_store(texts) #Create a vectore store with given text(parameter)

    def run(self, state: ResearchState) -> dict:
        tasks = state["tasks"]
        documents_by_task = {}

        for task in tasks:
            # similarity_search_with_score returns (doc, score)
            results = self.vector_store.similarity_search_with_score(
                task.description,
                k=2
            )

            task_docs = []

            for doc, score in results:
                task_docs.append(
                    RetrievedDocument(
                        task_id=task.task_id,
                        source="local_vector_store",
                        score=float(score),
                        content=doc.page_content
                    )
                )

            documents_by_task[task.task_id] = task_docs

        return {"documents": documents_by_task}

