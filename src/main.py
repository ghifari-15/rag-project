from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from src.vector_store import vector_store
# Load environment variables from .env file
load_dotenv()
api_key = "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" or os.getenv("GROQ_API_KEY")


llm = ChatOpenAI(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


while True:
    print("------------------------------")
    user_input = input(str("Enter the questions: "))
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    print(f"You asked: {user_input[0:20]}...")
    print("Processing your question...")

    reviews = vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Adjust the number of documents to retrieve
    ).get_relevant_documents(user_input)

    
    # Combine the content of the retrieved documents into a single context
    context = "\n".join([doc.page_content for doc in reviews])

    # Create prompt with context and questions
    prompt = f"""
    Context from documents:
    {context}
    
    Question: {user_input}
    
    Please answer the question based on the provided context. If the context doesn't contain relevant information, please say so.
    """
    
    # Send combined prompt to LLM<
    response = llm.invoke(prompt)
    print(response.content)