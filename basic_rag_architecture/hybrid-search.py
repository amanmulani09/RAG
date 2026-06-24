from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
# ──────────────────────────────────────────────────────────────────
# SETUP: Create our sample company data
# ──────────────────────────────────────────────────────────────────

chunks = [
    "Microsoft acquired GitHub for 7.5 billion dollars in 2018.",
    "Tesla Cybertruck production ramp begins in 2024.",
    "Google is a large technology company with global operations.",
    "Tesla reported strong quarterly results. Tesla continues to lead in electric vehicles. Tesla announced new manufacturing facilities.",
    "SpaceX develops Starship rockets for Mars missions.",
    "The tech giant acquired the code repository platform for software development.",
    "NVIDIA designs Starship architecture for their new GPUs.",
    "Tesla Tesla Tesla financial quarterly results improved significantly.",
    "Cybertruck reservations exceeded company expectations.",
    "Microsoft is a large technology company with global operations.", 
    "Apple announced new iPhone features for developers.",
    "The apple orchard harvest was excellent this year.",
    "Python programming language is widely used in AI.",
    "The python snake can grow up to 20 feet long.",
    "Java coffee beans are imported from Indonesia.", 
    "Java programming requires understanding of object-oriented concepts.",
    "Orange juice sales increased during winter months.",
    "Orange County reported new housing developments."
]

# Convert to Document objects for LangChain
documents = [Document(page_content=chunk, metadata={"source": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]

print("Sample Data:")
for i, chunk in enumerate(chunks, 1):
    print(f"{i}. {chunk}")

print("\n" + "="*80)

print("Setting up Vector Retriever...")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Test semantic search
test_query = "space exploration company" #works in vector search but wouldn't work with keyword search

print(f"Testing: '{test_query}'")
test_docs = vector_retriever.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")

# 2. BM25 Retriever (Keyword Search)
print("Setting up BM25 Retriever...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3

# Test exact keyword matching
# test_query = "space exploration company"
test_query = "Cybertruck"
# test_query = "Tesla"

print(f"Testing: '{test_query}'")
test_docs = bm25_retriever.invoke(test_query)
for doc in test_docs:
    print(f"Found: {doc.page_content}")

#  3. Hybrid Retriever (Combination)
print("Setting up Hybrid Retriever...")
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Equal weight to vector and keyword search
)

print("Setup complete!\n")


# Query 1: Mixed semantic and exact terms

# Vector search understands "purchase cost" semantically
# BM25 search finds exact "7.5 billion" 
# Hybrid should combine both strengths for best result
test_query = "purchase cost 7.5 billion"

retrieved_chunks = hybrid_retriever.invoke(test_query)
for i, doc in enumerate(retrieved_chunks, 1):
    print(f"{i}. {doc.page_content}")

print("Query 1 shows how hybrid finds exact financial info using both semantic understanding and keyword matching")