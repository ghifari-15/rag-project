# RAG Project - Retrieval-Augmented Generation

A Python-based RAG (Retrieval-Augmented Generation) system that allows you to query documents using natural language. The system uses Qdrant as a vector database, HuggingFace embeddings, and Groq's LLM API for generating responses.

## Features

- **Document Processing**: Load and process PDF documents
- **Vector Storage**: Store document embeddings in Qdrant vector database
- **Semantic Search**: Find relevant document chunks using similarity search
- **AI-Powered Responses**: Generate contextual answers using Groq's Llama model
- **Interactive Chat**: Command-line interface for asking questions

## Architecture

```
PDF Document → Text Chunks → Embeddings → Qdrant DB
                                            ↓
User Query → Similarity Search → Relevant Chunks → LLM → Response
```

## Prerequisites

- Python 3.8+
- Qdrant Cloud account (or local Qdrant instance)
- Groq API key

## Installation

1. **Clone the repository**g
```bash
git clone https://github.com/ghifari-15/rag-project.git
cd Rag-project
```

2. **Install dependencies**
```bash
pip install langchain-qdrant langchain-openai langchain-huggingface langchain-community
pip install qdrant-client python-dotenv pypdf
```

3. **Set up environment variables**
Create a `.env` file in the project root:
```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
```

## Configuration

### Vector Database Settings
- **Collection Name**: `knowledge_base`
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (384 dimensions)
- **Distance Metric**: Cosine similarity
- **Chunk Size**: 1024 characters
- **Chunk Overlap**: 128 characters

### LLM Settings
- **Model**: `meta-llama/llama-4-scout-17b-16e-instruct`
- **API Provider**: Groq
- **Max Tokens**: 1024
- **Temperature**: 1.0

## Usage

### 1. Add Documents to Vector Store

```python
from vector_store import add_file_to_vector_store

# Add a PDF file to the vector database
add_file_to_vector_store("your_document.pdf")
```

### 2. Run the Interactive Chat

```bash
python main.py
```

### 3. Ask Questions

```
Enter the questions: What is artificial intelligence?
```

Type `exit` to quit the application.

## File Structure

```
Rag-project/
├── main.py              # Main chat interface
├── vector_store.py      # Vector database operations
├── .env                 # Environment variables
├── README.md           # Project documentation
└── test_file.pdf       # Sample document (optional)
```

## Code Overview

### `vector_store.py`
- Initializes Qdrant client and vector store
- Creates document embeddings using HuggingFace
- Handles PDF loading and text chunking
- Provides functions to add documents to the database

### `main.py`
- Sets up the Groq LLM client
- Implements the chat loop
- Retrieves relevant documents based on user queries
- Combines context with questions for the LLM

## Key Functions

### Vector Store Operations
```python
# Initialize vector store
vector_store()

# Add PDF to database
add_file_to_vector_store(file_path)
```

### Document Retrieval
```python
# Get relevant documents
retriever = vector_store().as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
docs = retriever.get_relevant_documents(query)
```

## Customization

### Adjust Chunk Size
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,    # Modify chunk size
    chunk_overlap=128   # Modify overlap
)
```

### Change Number of Retrieved Documents
```python
search_kwargs={"k": 5}  # Retrieve top 5 similar chunks
```

### Modify LLM Parameters
```python
llm = ChatOpenAI(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=1,          # Adjust creativity (0-2)
    max_completion_tokens=1024,  # Max response length
)
```

## Troubleshooting

### Common Issues

1. **Collection Already Exists Error**
   - The code handles this automatically
   - Collection will be reused if it exists

2. **API Key Issues**
   - Ensure `.env` file is in the project root
   - Verify API keys are correct and active

3. **PDF Loading Issues**
   - Ensure PDF file exists in the specified path
   - Check if PDF is not password-protected

4. **Memory Issues**
   - Reduce `chunk_size` for large documents
   - Process documents in smaller batches

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `QDRANT_URL` | Qdrant database URL | Yes |
| `QDRANT_API_KEY` | Qdrant API key | Yes |
| `GROQ_API_KEY` | Groq API key | Yes |

## Dependencies

```
langchain-qdrant==0.1.0
langchain-openai==0.1.0
langchain-huggingface==0.1.0
langchain-community==0.2.0
qdrant-client==1.7.0
python-dotenv==1.0.0
pypdf==4.0.0
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Create an issue in the repository

---

**Note**: Make sure to keep your API keys secure and never commit them to version control.