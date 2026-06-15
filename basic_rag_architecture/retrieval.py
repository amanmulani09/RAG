from langchain_chroma import Chroma
from lanchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

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


