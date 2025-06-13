from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")


# Initialize QdrantVectorStore with environment variables
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    
)
# Check if the environment variables are set
if not qdrant_url:
    raise ValueError("QDRANT_URL environment variable is not set.")
if not qdrant_api_key:
    raise ValueError("QDRANT_API_KEY environment variable is not set.")

collection_name = "knowledge_base"
model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

try:
    create_db = client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=384,
        distance=models.Distance.COSINE
        )
    )
    print(f"Collection {collection_name} created successfully.")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Collection {collection_name} already exists. Using existing collection.")
    else:
        print(f"error creating collection: {e}")


# Initialize the QdrantVectorStore
def vector_store():
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,  # You can specify an embedding function if needed

    )
    return vector_store




# Function to add a file to the vector store
def add_file_to_vector_store(file_path):

    # load the PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the documents into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128 # Adjust overlap as needed
    )
    chunks = text_splitter.split_documents(documents)

    vector_store_instance = vector_store()
    vector_store_instance.add_documents(chunks)
    print(f"Added {len(chunks)} chunks from {file_path} to the vector store.")
   
        
   

# Test file
test_file = "test_file.pdf"
add_file_to_vector_store(test_file)



