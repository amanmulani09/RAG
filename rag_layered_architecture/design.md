## requirements

functional

- admin can upload documents
- system indexes the documents
- user ask questions
- system retrives relevant chunks
- LLM generates answer

non functional

- scalable
- low latency(<2 sec )
- accurate retrieval
- secure
- observable

2. Define APIs

## Injection APIs

- upload documents

POST /documents

Request

    {
    "document_name": "insurance.pdf",
    "file": "<binary>"
    }

Response

    {
    "document_id": "123",
    "status": "processing"
    }

Check Processing Status

GET /documents/{id}

Response:

```
    {
    "document_id": "123",
    "status": "completed"
    }
```

## Query APIs

Ask questions

POST /chat

Request

    {
    "query": "What is claim settlement process?"
    }

Response

    {
    "answer": "...",
    "sources": ["page_10"]
    }

## High Level Architecture

           User
            |
        API Gateway
            |
    |       |       |
            |
    injection      query flow
    fllow

## Injection Pipeline

- upload document
  - Storage (s3 / blob storage)
  - document processor (PDFs, docs, texts, )
  - text extraction (why and how ? )
  - chunking service (chunk1, chunk2, chunk3)
  - embedding service (text -> embedding )
  - vector database (chunk text, embedding, metadata)

## Query Pipeline

- user query
  - API
  - Embedding Service(embede the query)
  - Vector Search (cosine similarity with q & vector)
  - Top K Chunks (retrived chunks )
  - prompt builder
  - LLM
  - Answer

## Folder Structure

app/
│
├── api/
│ ├── chat.py
│ └── documents.py
│
├── services/
│ ├── ingestion_service.py
│ ├── retrieval_service.py
│ ├── embedding_service.py
│ ├── llm_service.py
│ └── chunking_service.py
│
├── repositories/
│ ├── document_repository.py
│ └── vector_repository.py
│
├── models/
│ ├── requests.py
│ └── responses.py
│
├── core/
│ ├── config.py
│ └── dependencies.py
│
├── main.py

Notice the separation:

API → HTTP only
Services → Business logic
Repositories → Database/Vector DB
Models → DTOs
Core → Config & Dependency Injection
