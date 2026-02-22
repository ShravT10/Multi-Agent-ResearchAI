import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index"


class VectorStoreManager:

    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = self._load_index()

    def _load_index(self):
        if not os.path.exists(INDEX_PATH):
            raise ValueError(
                "FAISS index not found. Run ingest.py first."
            )

        print("Loading FAISS index from disk...")
        return FAISS.load_local(
            INDEX_PATH,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def get_vector_store(self):
        return self.vector_store