from agents.base import BaseAgent
from graph.state import ResearchState
from core.schemas import RetrievedDocument
from duckduckgo_search import DDGS
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


class WebRetrieverAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    def run(self, state: ResearchState) -> dict:
        tasks = state["tasks"]
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        documents_by_task = {}

        # Embed research question once
        question_embedding = self.embedding_model.embed_query(question)

        max_results = 5 + retry_count  # slight increase on retry
        similarity_threshold = 0.4

        for task in tasks:
            task_docs = []

            with DDGS() as ddgs:
                results = ddgs.text(task.description, max_results=max_results)

                for r in results:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    content = f"{title}. {body}"
                    source = r.get("href", "web_search")

                    # Basic English filter
                    if sum(1 for c in content if ord(c) > 127) > 10:
                        continue

                    # Embed snippet
                    snippet_embedding = self.embedding_model.embed_query(content)

                    # Compute similarity
                    similarity = self.cosine_similarity(
                        question_embedding,
                        snippet_embedding
                    )

                    if similarity < similarity_threshold:
                        continue

                    task_docs.append(
                        RetrievedDocument(
                            task_id=task.task_id,
                            source=source,
                            score=similarity,
                            content=content
                        )
                    )

            documents_by_task[task.task_id] = task_docs

        return {"web_documents": documents_by_task}