from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize the Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def create_vector_store(chunks):
    # Store embeddings into the vector store
    vector_store = FAISS.from_documents(
        documents=chunks,  # Input chunks to the vector store
        embedding=embeddings  # Use the initialized embeddings model
    )
    return vector_store
