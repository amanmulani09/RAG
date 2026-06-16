from langchain_chroma import Chroma
from lanchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

persist_directory = "db/vector_store"

# load the embedding model 
# note ; (keep the same embedding model as the one used to create the embeddings for the documents)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# load the vector store

db = Chroma(
    collection_metadata={"hnsw:space":"cosine"},
    embedding_function=embedding_model,
    persist_directory=persist_directory,
    )

# search for relevant documents given a query

query = "What is RAG architecture?"

# the retriever will return the top 3 most relevant documents to the query
retriever = db.as_retriever(search_kwargs={"k":5})

relevant_docs = retriever.invoke(query)


for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}:")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content}")
    print("\n")


combined_input = f"""Based on the following documents, please answer this question: {query}

Documents: {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. 
If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
