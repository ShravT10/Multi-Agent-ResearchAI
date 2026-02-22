import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from docx import Document as DocxDocument

INDEX_PATH = "faiss_index"
DATA_PATH = "data"


def load_docx(file_path):
    doc = DocxDocument(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return full_text


def load_documents():
    documents = []

    for filename in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

        elif filename.endswith(".txt"):
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

        elif filename.endswith(".docx"):
            text = load_docx(file_path)
            documents.append({"page_content": text})

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = []

    for doc in documents:
        if isinstance(doc, dict):
            texts.extend(splitter.split_text(doc["page_content"]))
        else:
            texts.extend(splitter.split_text(doc.page_content))

    return texts


def main():
    print("Loading documents...")
    documents = load_documents()

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index...")
        vector_store = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("Adding new chunks to index...")
        vector_store.add_texts(chunks)
    else:
        print("Creating new FAISS index...")
        vector_store = FAISS.from_texts(chunks, embeddings)

    print("Saving FAISS index...")
    vector_store.save_local(INDEX_PATH)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()