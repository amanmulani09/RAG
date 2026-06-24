1. requirements

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

req:
{
"document_name": "insurance.pdf",
"file": "<binary>"
}

res:

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

Response:

{
"answer": "...",
"sources": ["page_10"]
}
