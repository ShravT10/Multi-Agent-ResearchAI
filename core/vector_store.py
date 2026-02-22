import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = "faiss_index"


class VectorStoreManager:

    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(INDEX_PATH):
            print("Loading FAISS index from disk...")
            return FAISS.load_local(
                INDEX_PATH,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )

        print("Building new FAISS index...")

        documents = [
            "AI adoption in India is increasing in agriculture sector.",
            "Generative AI helps optimize crop yield through predictive analytics.",
            "Government policies are encouraging agri-tech startups."
        ]

        vector_store = FAISS.from_texts(
            documents,
            self.embedding_model
        )

        vector_store.save_local(INDEX_PATH)

        return vector_store

    def get_vector_store(self):
        return self.vector_store