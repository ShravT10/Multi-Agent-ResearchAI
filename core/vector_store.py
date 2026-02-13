from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vector_store(texts: list[str]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    documents = []
    for text in texts:
        chunks = splitter.split_text(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk)) 
            #Documentis a LangChain Data Structure with text data + meta data. 
            #We assign the metadata page_content to chunk we created

    vector_store = FAISS.from_documents(documents, embedding_model) # Embedds each document and store in vector_store
    return vector_store
