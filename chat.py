from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Embeddings (same as indexing)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector store (Qdrant)
client = QdrantClient(url="http://localhost:6333")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="learning_rag",
    embedding=embedding_model,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Gemini LLM (Free Tier)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.
Answer the question ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
)

# RAG Chain (LCEL)
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Chat loop
print("\n💬 Gemini RAG Chatbot (type 'exit' to quit)\n")

while True:
    query = input("You: ")
    if query.lower() in {"exit", "quit"}:
        print("👋 Bye!")
        break

    answer = rag_chain.invoke(query)

    print("\n🤖 Answer:")
    print(answer)