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
        all_docs = []

        for task in tasks:
            results = self.vector_store.similarity_search(
                task.description,
                k=2
            ) #Get docments

            for doc in results:
                all_docs.append(
                    RetrievedDocument(
                        source="local_vector_store",
                        content=doc.page_content
                    )
                ) #iterate over documents and add page_content metadata of doc in all_docs 

        return {"documents": all_docs}
