import numpy as np
from agents.base import BaseAgent
from graph.state import ResearchState
from langchain_huggingface import HuggingFaceEmbeddings
from core.vector_store import VectorStoreManager


class KnowledgeConsolidatorAgent(BaseAgent):

    def __init__(self):
        super().__init__()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_manager = VectorStoreManager()
        self.vector_store = self.vector_manager.get_vector_store()

    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    def run(self, state: ResearchState) -> dict:
        critic = state["critic"]

        if not critic.is_valid:
            return {}

        analysis = state["analysis"].per_task_analysis
        documents = state["documents"]

        similarity_threshold = 0.7
        new_texts = []

        for task_id, task_analysis in analysis.items():
            verified_facts = task_analysis.verified_facts
            task_docs = documents.get(task_id, [])

            for fact in verified_facts:
                fact_embedding = self.embedding_model.embed_query(fact)

                for doc in task_docs:
                    if not doc.source.startswith("http"):
                        continue  # Only persist web docs

                    doc_embedding = self.embedding_model.embed_query(doc.content)

                    similarity = self.cosine_similarity(
                        fact_embedding,
                        doc_embedding
                    )

                    if similarity >= similarity_threshold:
                        new_texts.append(doc.content)

        if new_texts:
            print(f"Persisting {len(new_texts)} new web documents...")
            self.vector_store.add_texts(new_texts)
            self.vector_store.save_local("faiss_index")

        return {}