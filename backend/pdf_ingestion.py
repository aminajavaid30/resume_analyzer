from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load and split the PDF document and return the documents and text chunks
def load_split_pdf(file_path):
    # Load the PDF document and split it into chunks
    loader = PyPDFLoader(file_path)  # Initialize the PDF loader with the file path
    documents = loader.load()  # Load the PDF document 

    # Initialize the recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Set the maximum chunk size
        chunk_overlap=50,  # Set the number of overlapping characters between chunks
        separators=["\n\n", "\n", " ", ""],  # Define resume-specific separators for splitting
    )   

    # Split the loaded documents into chunks
    chunks = text_splitter.split_documents(documents)
    return documents, chunks
