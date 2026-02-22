from agents.base import BaseAgent
from core.schemas import RetrieverOutput, RetrievedDocument
from graph.state import ResearchState
from core.vector_store import VectorStoreManager

vector_manager = VectorStoreManager()
class RetrieverAgent(BaseAgent):
    def __init__(self, texts):
        super().__init__()
        self.vector_store = vector_manager.get_vector_store() #Create a vectore store with given text(parameter)

    def run(self, state: ResearchState) -> dict:
        tasks = state["tasks"]
        documents_by_task = {}

        for task in tasks:
            rewritten_tasks = state.get("rewritten_tasks")

            ### RETRY LOGIC
            if (
                rewritten_tasks
                and rewritten_tasks.rewritten_tasks
                and task.task_id in rewritten_tasks.rewritten_tasks
            ):
                query = rewritten_tasks.rewritten_tasks[task.task_id]
            else:
                query = task.description
            ################

            base_k = 2
            retry_count = state.get("retry_count", 0)

            adaptive_k = base_k + (retry_count * 3)

            results = self.vector_store.similarity_search_with_score(
                query,
                k=adaptive_k
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

